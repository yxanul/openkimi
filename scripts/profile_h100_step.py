from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--compile", action="store_true", help="compile the model with torch.compile")
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="delimit the measured update with cudaProfilerStart/Stop for Nsight Systems",
    )
    args = parser.parse_args()

    model_cfg, data_cfg, train_cfg = load_config(args.config)
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

    def optimizer_update() -> None:
        with _nvtx_range("forward"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(input_ids, labels)
        assert output.loss is not None
        with _nvtx_range("backward"):
            output.loss.backward()
        with _nvtx_range("gradient_clip"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.gradient_clip)
        with _nvtx_range("adamw"):
            optimizer.step()
        with _nvtx_range("zero_grad"):
            optimizer.zero_grad(set_to_none=True)

    for _ in range(args.warmup):
        optimizer_update()
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if args.cuda_profiler_range:
        torch.cuda.cudart().cudaProfilerStart()
    start.record()
    with _nvtx_range("optimizer_update"):
        optimizer_update()
    end.record()
    torch.cuda.synchronize()
    if args.cuda_profiler_range:
        torch.cuda.cudart().cudaProfilerStop()

    elapsed_ms = start.elapsed_time(end)
    tokens = train_cfg.microbatch_sequences * data_cfg.sequence_length
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "torch_compile": args.compile,
                "microbatch_sequences": train_cfg.microbatch_sequences,
                "sequence_length": data_cfg.sequence_length,
                "gradient_accumulation_steps": train_cfg.gradient_accumulation(data_cfg, 1),
                "tokens": tokens,
                "elapsed_ms": elapsed_ms,
                "tokens_per_second": tokens / (elapsed_ms / 1000),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "backend": raw_model.backend.as_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
