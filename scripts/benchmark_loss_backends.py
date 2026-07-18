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
    logical_vocab_size: int,
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
        "dummy_weight_grad_nonzero": int(
            torch.count_nonzero(weight.grad[logical_vocab_size:])
        ),
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
    parser.add_argument(
        "--provider",
        action="append",
        choices=("fla", "liger", "fp8_liger", "fp8_quack"),
    )
    parser.add_argument("--chunk-size", type=int, default=16_384)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")

    from fla.modules import FusedLinearCrossEntropyLoss
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

    from k3mini.fp8 import CurrentScalingFusedLinearCrossEntropyLoss

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
    physical_vocab = ((args.vocab + 15) // 16) * 16
    weight = torch.randn(
        physical_vocab,
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
    fp8_liger = CurrentScalingFusedLinearCrossEntropyLoss(
        args.vocab,
        chunk_size=args.chunk_size,
        ce_backend="liger",
    )
    fp8_quack = CurrentScalingFusedLinearCrossEntropyLoss(
        args.vocab,
        chunk_size=args.chunk_size,
        ce_backend="quack",
    )
    providers = args.provider or ["fla", "liger"]
    functions = {
        "fla": lambda x, w, y: fla(x, y, w[: args.vocab]),
        "liger": lambda x, w, y: liger(w[: args.vocab], x, y),
        "fp8_liger": lambda x, w, y: fp8_liger(w, x, y),
        "fp8_quack": lambda x, w, y: fp8_quack(w, x, y),
    }

    liger_increase = math.ceil(args.vocab / args.hidden)
    liger_chunk_size = 1 << (math.ceil(args.tokens / liger_increase) - 1).bit_length()
    metadata = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "tokens": args.tokens,
        "hidden": args.hidden,
        "vocab": args.vocab,
        "physical_vocab": physical_vocab,
        "fla_chunks": 8,
        "liger_chunk_size": liger_chunk_size,
        "liger_chunks": math.ceil(args.tokens / liger_chunk_size),
        "fp8_chunk_size": args.chunk_size,
        "fp8_chunks": math.ceil(args.tokens / args.chunk_size),
    }
    print(json.dumps(metadata))
    for provider in providers:
        result = _measure(
            provider,
            functions[provider],
            hidden,
            weight,
            labels,
            logical_vocab_size=args.vocab,
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
