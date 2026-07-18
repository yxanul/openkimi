from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch


def _pointer(tensor: torch.Tensor):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(tensor.detach(), assumed_align=16).iterator


def _measure(function, warmup: int, repetitions: int) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=196608)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError("this benchmark requires an SM90+ CUDA GPU")

    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cutlass.cutlass_dsl import Int32

    working = Path(__file__).resolve().parent / "working"
    sys.path.insert(0, str(working))
    from tf32_tile_cute import Tf32TileSm90

    torch.manual_seed(71)
    a = torch.randn(args.blocks, 16, 16, device="cuda", dtype=torch.float32)
    b = torch.randn(args.blocks, 16, 32, device="cuda", dtype=torch.float32)
    scale = torch.randn(args.blocks, 16, 32, device="cuda", dtype=torch.float32)
    output = torch.empty_like(scale)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compile_args = (
        _pointer(a),
        _pointer(b),
        _pointer(scale),
        _pointer(output),
        Int32(args.blocks),
        stream,
    )
    cold_start = time.perf_counter()
    compiled = cute.compile(Tf32TileSm90(), *compile_args)
    compile_seconds = time.perf_counter() - cold_start

    def run() -> None:
        compiled(*compile_args)

    run()
    torch.cuda.synchronize()
    reference = torch.bmm(a, b) * scale
    relative_error = float(
        torch.linalg.vector_norm(output - reference)
        / torch.linalg.vector_norm(reference).clamp_min(1e-12)
    )
    maximum_absolute_error = float((output - reference).abs().max())
    samples = _measure(run, args.warmup, args.repetitions)
    print(
        json.dumps(
            {
                "blocks": args.blocks,
                "compile_seconds": compile_seconds,
                "median_ms": statistics.median(samples),
                "minimum_ms": min(samples),
                "relative_error": relative_error,
                "maximum_absolute_error": maximum_absolute_error,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
