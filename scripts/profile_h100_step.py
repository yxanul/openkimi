from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Iterator
from contextlib import contextmanager

import torch

from k3mini.config import load_config
from k3mini.model import K3MiniForCausalLM
from k3mini.training import build_optimizer


@contextmanager
def _nvtx_range(name: str) -> Iterator[None]:
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile one warmed H100 optimizer update.")
    parser.add_argument("--config", default="configs/h100-batch64-compiled.json")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--microbatch-sequences",
        type=int,
        default=None,
        help="override sequences per microstep; accumulation is derived from the global token batch",
    )
    parser.add_argument("--compile", action="store_true", help="compile the model with torch.compile")
    parser.add_argument(
        "--checkpoint-attention",
        choices=("config", "on", "off"),
        default="config",
        help="override outer attention checkpointing",
    )
    parser.add_argument(
        "--checkpoint-ffn",
        choices=("config", "on", "off"),
        default="config",
        help="override outer FFN checkpointing",
    )
    parser.add_argument(
        "--attnres-checkpoint-level",
        type=int,
        choices=(0, 1),
        default=None,
        help="override FLA AttnRes checkpoint level",
    )
    parser.add_argument(
        "--fp8-lm-head-chunk-size",
        type=int,
        default=None,
        help="override FP8 LM-head rows per chunk; omit to use the config/automatic heuristic",
    )
    parser.add_argument(
        "--transformer-engine-experts",
        action="store_true",
        help="replace MegaBlocks routed experts with experimental TE 2.16 GroupedLinear",
    )
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="delimit the measured update with cudaProfilerStart/Stop for Nsight Systems",
    )
    parser.add_argument(
        "--kda-intra-experiment",
        choices=("none", "blocked", "exact", "fused"),
        default="none",
        help="install a process-local experimental KDA intra backward provider",
    )
    parser.add_argument(
        "--kda-intra-num-warps",
        type=int,
        choices=(2, 4, 8),
        default=4,
        help="warp count for the experimental coarsened KDA intra provider",
    )
    args = parser.parse_args()

    if args.kda_intra_experiment != "none":
        if args.kda_intra_experiment == "fused":
            from experiments.kda_sm90.fused_wy_intra_triton import (
                install_fused_wy_intra_experiment,
            )

            install_fused_wy_intra_experiment(
                num_warps=args.kda_intra_num_warps,
            )
        else:
            from experiments.kda_sm90.fused_wy_intra_triton import (
                install_intra_chunk_experiment,
            )

            install_intra_chunk_experiment(
                args.kda_intra_experiment,
                num_warps=args.kda_intra_num_warps,
            )

    model_cfg, data_cfg, train_cfg = load_config(args.config)
    if args.microbatch_sequences is not None:
        train_cfg.microbatch_sequences = args.microbatch_sequences
    if args.checkpoint_attention != "config":
        model_cfg.checkpoint_attention = args.checkpoint_attention == "on"
    if args.checkpoint_ffn != "config":
        model_cfg.checkpoint_ffn = args.checkpoint_ffn == "on"
    if args.attnres_checkpoint_level is not None:
        model_cfg.attnres_checkpoint_level = args.attnres_checkpoint_level
    if args.fp8_lm_head_chunk_size is not None:
        model_cfg.fp8_lm_head_chunk_size = args.fp8_lm_head_chunk_size
    model_cfg.validate()
    train_cfg.compile_model = False
    train_cfg.validate(data_cfg, world_size=1)
    if not torch.cuda.is_available():
        raise RuntimeError("this profiler requires CUDA")

    torch.manual_seed(train_cfg.seed)
    torch.cuda.manual_seed_all(train_cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    raw_model = K3MiniForCausalLM(model_cfg).to(device).train()
    if args.transformer_engine_experts:
        from benchmark_expert_backends import replace_model_experts_with_transformer_engine

        replace_model_experts_with_transformer_engine(raw_model, model_cfg)
    optimizer = build_optimizer(raw_model, train_cfg)
    model = torch.compile(raw_model) if args.compile else raw_model
    input_ids = torch.randint(
        model_cfg.vocab_size,
        (train_cfg.microbatch_sequences, data_cfg.sequence_length),
        device=device,
    )
    labels = torch.randint(
        model_cfg.vocab_size,
        (train_cfg.microbatch_sequences, data_cfg.sequence_length),
        device=device,
    )
    accumulation_steps = train_cfg.gradient_accumulation(data_cfg, 1)
    effective_lm_head_chunk_size: int | None = None
    lm_head_chunks_per_microstep: int | None = None
    lm_head_chunks_per_update: int | None = None
    if model_cfg.linear_precision.value == "fp8_current":
        from k3mini.fp8 import resolve_fp8_lm_head_chunk_size

        microbatch_tokens = train_cfg.microbatch_sequences * data_cfg.sequence_length
        effective_lm_head_chunk_size = resolve_fp8_lm_head_chunk_size(
            microbatch_tokens,
            model_cfg.physical_vocab_size,
            model_cfg.d_model,
            model_cfg.fp8_lm_head_chunk_size,
        )
        lm_head_chunks_per_microstep = (
            microbatch_tokens + effective_lm_head_chunk_size - 1
        ) // effective_lm_head_chunk_size
        lm_head_chunks_per_update = lm_head_chunks_per_microstep * accumulation_steps

    def optimizer_update() -> None:
        for microstep in range(accumulation_steps):
            with _nvtx_range(f"forward.microstep_{microstep}"):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(
                        input_ids,
                        labels,
                        is_first_microbatch=microstep == 0,
                    )
            assert output.loss is not None
            with _nvtx_range(f"backward.microstep_{microstep}"):
                (output.loss / accumulation_steps).backward()
        with _nvtx_range("gradient_clip"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.gradient_clip)
        with _nvtx_range("adamw"):
            optimizer.step()
        with _nvtx_range("zero_grad"):
            optimizer.zero_grad(set_to_none=True)

    for _ in range(args.warmup):
        optimizer_update()
        torch.cuda.synchronize()

    elapsed_samples: list[float] = []
    peak_allocated_gib = 0.0
    peak_reserved_gib = 0.0
    if args.cuda_profiler_range:
        torch.cuda.cudart().cudaProfilerStart()
    for _ in range(args.repeats):
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with _nvtx_range("optimizer_update"):
            optimizer_update()
        end.record()
        torch.cuda.synchronize()
        elapsed_samples.append(start.elapsed_time(end))
        peak_allocated_gib = max(
            peak_allocated_gib,
            torch.cuda.max_memory_allocated() / 2**30,
        )
        peak_reserved_gib = max(
            peak_reserved_gib,
            torch.cuda.max_memory_reserved() / 2**30,
        )
    if args.cuda_profiler_range:
        torch.cuda.cudart().cudaProfilerStop()

    elapsed_ms = statistics.median(elapsed_samples)
    tokens = train_cfg.global_batch_tokens
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "torch_compile": args.compile,
                "transformer_engine_experts": args.transformer_engine_experts,
                "checkpoint_attention": model_cfg.checkpoint_attention_enabled,
                "checkpoint_ffn": model_cfg.checkpoint_ffn_enabled,
                "attnres_checkpoint_level": model_cfg.attnres_checkpoint_level,
                "kda_disable_recompute": model_cfg.kda_disable_recompute,
                "kda_intra_experiment": args.kda_intra_experiment,
                "kda_intra_num_warps": args.kda_intra_num_warps,
                "configured_fp8_lm_head_chunk_size": model_cfg.fp8_lm_head_chunk_size,
                "effective_fp8_lm_head_chunk_size": effective_lm_head_chunk_size,
                "lm_head_chunks_per_microstep": lm_head_chunks_per_microstep,
                "lm_head_chunks_per_update": lm_head_chunks_per_update,
                "microbatch_sequences": train_cfg.microbatch_sequences,
                "sequence_length": data_cfg.sequence_length,
                "gradient_accumulation_steps": accumulation_steps,
                "microbatch_tokens": train_cfg.microbatch_sequences * data_cfg.sequence_length,
                "tokens": tokens,
                "elapsed_ms": elapsed_ms,
                "elapsed_samples_ms": elapsed_samples,
                "tokens_per_second": tokens / (elapsed_ms / 1000),
                "peak_allocated_gib": peak_allocated_gib,
                "peak_reserved_gib": peak_reserved_gib,
                "backend": raw_model.backend.as_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
