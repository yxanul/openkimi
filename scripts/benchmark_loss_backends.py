from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
from collections.abc import Callable

import torch


def _measure(
    name: str,
    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, float | int | str]:
    times: list[float] = []
    loss_value = math.nan
    for iteration in range(warmup + repeats):
        hidden.grad = None
        weight.grad = None
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = fn(hidden, weight, labels)
        loss.backward()
        end.record()
        torch.cuda.synchronize()
        loss_value = float(loss.detach())
        if iteration >= warmup:
            times.append(start.elapsed_time(end))

    return {
        "provider": name,
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "loss": loss_value,
        "hidden_grad_l2": float(hidden.grad.float().norm()),
        "weight_grad_l2": float(weight.grad.float().norm()),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fused LM-head cross-entropy backends.")
    parser.add_argument("--tokens", type=int, default=262_144)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--vocab", type=int, default=128_001)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--provider", action="append", choices=("fla", "liger"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")

    from fla.modules import FusedLinearCrossEntropyLoss
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cuda.matmul.allow_tf32 = True
    hidden = torch.randn(
        args.tokens,
        args.hidden,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    weight = torch.randn(
        args.vocab,
        args.hidden,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    labels = torch.randint(args.vocab, (args.tokens,), device="cuda")

    fla = FusedLinearCrossEntropyLoss(
        ignore_index=-100,
        num_chunks=8,
        accumulate_grad_in_fp32=True,
    )
    liger = LigerFusedLinearCrossEntropyLoss(
        ignore_index=-100,
        reduction="mean",
        accum_dtype=torch.float32,
    )
    providers = args.provider or ["fla", "liger"]
    functions = {
        "fla": lambda x, w, y: fla(x, y, w),
        "liger": lambda x, w, y: liger(w, x, y),
    }

    liger_increase = math.ceil(args.vocab / args.hidden)
    liger_chunk_size = 1 << (math.ceil(args.tokens / liger_increase) - 1).bit_length()
    metadata = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "tokens": args.tokens,
        "hidden": args.hidden,
        "vocab": args.vocab,
        "fla_chunks": 8,
        "liger_chunk_size": liger_chunk_size,
        "liger_chunks": math.ceil(args.tokens / liger_chunk_size),
    }
    print(json.dumps(metadata))
    for provider in providers:
        result = _measure(
            provider,
            functions[provider],
            hidden,
            weight,
            labels,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        print(json.dumps(result))
        hidden.grad = None
        weight.grad = None
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
