from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .benchmark import benchmark_full_step, save_benchmark
from .checkpoint import CheckpointManager
from .config import KernelBackend, load_config
from .data import (
    PackedClimbMixDataset,
    SuperBPETokenizer,
    load_validation_cache,
    materialize_validation_cache,
)
from .model import K3MiniForCausalLM, estimate_parameter_counts
from .training import build_optimizer, setup_distributed, train, validate


def _common_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default="configs/primary.json",
        help="JSON file containing model/data/train sections",
    )
    parser.add_argument(
        "--backend",
        choices=[backend.value for backend in KernelBackend],
        default=None,
        help="override model.kernel_backend",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="k3-mini")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="run a staged pretraining job")
    _common_config(train_parser)
    train_parser.add_argument("--resume", default=None, help="checkpoint directory or `latest`")
    train_parser.add_argument("--stage", choices=["overfit", "10m", "1b"], default="1b")
    train_parser.add_argument("--synthetic", action="store_true", help="use deterministic random tokens")

    validation_parser = subparsers.add_parser(
        "validate", help="evaluate a checkpoint on the materialized cache"
    )
    _common_config(validation_parser)
    validation_parser.add_argument("--checkpoint", required=True)

    benchmark_parser = subparsers.add_parser(
        "kernel-benchmark", help="profile a full forward/backward step and selected backend"
    )
    _common_config(benchmark_parser)
    benchmark_parser.add_argument("--sequence-length", type=int, default=None)
    benchmark_parser.add_argument("--batch-size", type=int, default=1)
    benchmark_parser.add_argument("--warmup", type=int, default=2)
    benchmark_parser.add_argument("--iterations", type=int, default=5)
    benchmark_parser.add_argument("--trace", default=None)
    benchmark_parser.add_argument("--output", default=None)

    inspect_parser = subparsers.add_parser(
        "data-inspection", help="stream and inspect packed ClimbMix samples"
    )
    _common_config(inspect_parser)
    inspect_parser.add_argument("--samples", type=int, default=2)
    inspect_parser.add_argument("--rank", type=int, default=0)
    inspect_parser.add_argument("--world-size", type=int, default=1)

    cache_parser = subparsers.add_parser(
        "make-validation", help="materialize the deterministic 1M-token validation cache"
    )
    _common_config(cache_parser)
    cache_parser.add_argument("--output", default=None)
    cache_parser.add_argument("--tokens", type=int, default=None)

    dry_parser = subparsers.add_parser(
        "dry-run", help="validate configuration and estimate parameters without allocating weights"
    )
    _common_config(dry_parser)
    return parser


def _load(args: argparse.Namespace):
    model_cfg, data_cfg, train_cfg = load_config(args.config)
    if args.backend is not None:
        model_cfg.kernel_backend = KernelBackend(args.backend)
        model_cfg.validate()
    return model_cfg, data_cfg, train_cfg


def _run_validation(args: argparse.Namespace) -> None:
    model_cfg, data_cfg, train_cfg = _load(args)
    context = setup_distributed()
    model = K3MiniForCausalLM(model_cfg).to(context.device)
    optimizer = build_optimizer(model, train_cfg)
    manager = CheckpointManager(train_cfg.output_dir, context.rank, context.world_size)
    state = manager.load(
        args.checkpoint,
        model=model,
        optimizer=optimizer,
        scaler=None,
        data_stream=None,
    )
    samples = load_validation_cache(data_cfg.validation_cache)
    samples = samples[context.rank :: context.world_size]
    metrics = validate(model, samples, context.device, train_cfg.precision)
    if context.is_main:
        print(json.dumps({"checkpoint": state, **metrics}, indent=2))
    if context.distributed:
        torch.distributed.destroy_process_group()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    model_cfg, data_cfg, train_cfg = _load(args)

    if args.command == "train":
        fixed_batch = args.stage == "overfit"
        if args.stage == "overfit":
            train_cfg.target_tokens = train_cfg.global_batch_tokens * 100
            train_cfg.validate_every_tokens = train_cfg.target_tokens
            train_cfg.checkpoint_every_tokens = train_cfg.target_tokens
        elif args.stage == "10m":
            train_cfg.target_tokens = 10_000_000
        result = train(
            model_cfg,
            data_cfg,
            train_cfg,
            resume=args.resume,
            synthetic=args.synthetic,
            fixed_batch=fixed_batch,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "validate":
        _run_validation(args)
    elif args.command == "kernel-benchmark":
        result = benchmark_full_step(
            model_cfg,
            sequence_length=args.sequence_length or data_cfg.sequence_length,
            batch_size=args.batch_size,
            warmup=args.warmup,
            iterations=args.iterations,
            trace_path=args.trace,
        )
        save_benchmark(result, args.output)
    elif args.command == "data-inspection":
        tokenizer = SuperBPETokenizer(
            data_cfg.tokenizer_name,
            revision=data_cfg.tokenizer_revision,
            eod_token_id=data_cfg.eod_token_id,
        )
        dataset = PackedClimbMixDataset(
            data_cfg,
            rank=args.rank,
            world_size=args.world_size,
            tokenizer=tokenizer,
        )
        iterator = iter(dataset)
        samples = [next(iterator) for _ in range(args.samples)]
        print(
            json.dumps(
                {
                    "tokenizer_revision": tokenizer.revision,
                    "sample_shapes": [
                        {key: list(value.shape) for key, value in sample.items()} for sample in samples
                    ],
                    "first_tokens": samples[0]["input_ids"][:32].tolist(),
                    "stream": dataset.diagnostics(),
                },
                indent=2,
            )
        )
    elif args.command == "make-validation":
        dataset = PackedClimbMixDataset(data_cfg, validation=True)
        result = materialize_validation_cache(
            dataset,
            args.output or data_cfg.validation_cache,
            args.tokens or data_cfg.validation_tokens,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "dry-run":
        print(
            json.dumps(
                {
                    "model": asdict(model_cfg),
                    "data": asdict(data_cfg),
                    "train": asdict(train_cfg),
                    "parameters": estimate_parameter_counts(model_cfg),
                    "config_path": str(Path(args.config).resolve()),
                },
                indent=2,
                default=str,
            )
        )


if __name__ == "__main__":
    main()
