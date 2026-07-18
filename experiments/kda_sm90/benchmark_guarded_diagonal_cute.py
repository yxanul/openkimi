from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from functools import partial
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


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(actual.float() - expected.float())
        / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=196608)
    parser.add_argument("--mean-log2-decay", type=float, default=8.0)
    parser.add_argument("--threshold", type=float, default=248.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument(
        "--swizzle-bits",
        default="0,1,2,3",
        help="Comma-separated CuTe XOR swizzle widths; 0 is the plain-layout control.",
    )
    parser.add_argument(
        "--negative-factors",
        default="exp2",
        help="Comma-separated negative-factor implementations: exp2 and/or reciprocal.",
    )
    parser.add_argument(
        "--gate-cache-modes",
        default="uncached",
        help="Comma-separated gate sources: uncached and/or cached.",
    )
    parser.add_argument(
        "--lower-epilogue-modes",
        default="shared",
        help="Comma-separated lower-product epilogues: shared and/or direct.",
    )
    parser.add_argument(
        "--warp-assignment-modes",
        default="product",
        help="Comma-separated warp assignments: product and/or channel-half.",
    )
    parser.add_argument(
        "--fallback-diagonal-modes",
        default="loop",
        help="Comma-separated exact-fallback diagonal policies: loop and/or explicit.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError("this benchmark requires an SM90+ CUDA GPU")
    swizzle_bits = [int(value) for value in args.swizzle_bits.split(",")]
    if not swizzle_bits or any(value not in (0, 1, 2, 3) for value in swizzle_bits):
        raise ValueError("swizzle-bits must contain values from 0 through 3")
    negative_factors = args.negative_factors.split(",")
    if not negative_factors or any(
        value not in ("exp2", "reciprocal") for value in negative_factors
    ):
        raise ValueError("negative-factors must contain exp2 and/or reciprocal")
    gate_cache_modes = args.gate_cache_modes.split(",")
    if not gate_cache_modes or any(
        value not in ("uncached", "cached") for value in gate_cache_modes
    ):
        raise ValueError("gate-cache-modes must contain uncached and/or cached")
    lower_epilogue_modes = args.lower_epilogue_modes.split(",")
    if not lower_epilogue_modes or any(
        value not in ("shared", "direct") for value in lower_epilogue_modes
    ):
        raise ValueError("lower-epilogue-modes must contain shared and/or direct")
    warp_assignment_modes = args.warp_assignment_modes.split(",")
    if not warp_assignment_modes or any(
        value not in ("product", "channel-half")
        for value in warp_assignment_modes
    ):
        raise ValueError("warp-assignment-modes must contain product and/or channel-half")
    fallback_diagonal_modes = args.fallback_diagonal_modes.split(",")
    if not fallback_diagonal_modes or any(
        value not in ("loop", "explicit") for value in fallback_diagonal_modes
    ):
        raise ValueError("fallback-diagonal-modes must contain loop and/or explicit")

    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cutlass.cutlass_dsl import Float32, Int32

    experiment = Path(__file__).resolve().parent
    sys.path.insert(0, str(experiment / "working"))
    sys.path.insert(0, str(experiment))
    from benchmark_guarded_diagonal import _exact_diagonal_kernel
    from guarded_diagonal_cute import GuardedDiagonalSm90

    torch.manual_seed(1907)
    shape = (args.blocks, 16, 32)
    increments = torch.empty(shape, device="cuda", dtype=torch.float32).exponential_()
    increments *= args.mean_log2_decay
    increments[:, 0] = 0.0
    gate = -increments.cumsum(dim=1)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    beta = torch.sigmoid(
        torch.randn(args.blocks, 16, device="cuda", dtype=torch.bfloat16)
    )
    da_qk = torch.randn(args.blocks, 16, 16, device="cuda", dtype=torch.float32)
    da_kk = torch.randn_like(da_qk)
    exact_outputs = [torch.empty_like(gate) for _ in range(3)]
    candidate_outputs = [torch.empty_like(gate) for _ in range(3)]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compile_args = (
        *[
            _pointer(tensor)
            for tensor in (
                gate,
                q,
                k,
                beta,
                da_qk,
                da_kk,
                *candidate_outputs,
            )
        ],
        Int32(args.blocks),
        Float32(args.threshold),
        stream,
    )

    def run_exact() -> None:
        _exact_diagonal_kernel[(args.blocks,)](
            gate,
            q,
            k,
            beta,
            da_qk,
            da_kk,
            *exact_outputs,
            BC=16,
            BK=32,
            num_warps=4,
        )

    run_exact()
    torch.cuda.synchronize()
    exact_samples = _measure(run_exact, args.warmup, args.repetitions)
    spans = (gate[:, :1] - gate).amax(dim=(1, 2))
    exact_median = statistics.median(exact_samples)
    candidates = []
    for bits in swizzle_bits:
        for negative_factor in negative_factors:
            for gate_cache_mode in gate_cache_modes:
                for lower_epilogue_mode in lower_epilogue_modes:
                    for warp_assignment_mode in warp_assignment_modes:
                        if (
                            lower_epilogue_mode == "direct"
                            and warp_assignment_mode == "channel-half"
                        ):
                            continue
                        for fallback_diagonal_mode in fallback_diagonal_modes:
                            cold_start = time.perf_counter()
                            compiled = cute.compile(
                                GuardedDiagonalSm90(
                                    b_operand_swizzle_bits=bits,
                                    reciprocal_negative=(
                                        negative_factor == "reciprocal"
                                    ),
                                    cache_gate=gate_cache_mode == "cached",
                                    direct_lower_epilogue=(
                                        lower_epilogue_mode == "direct"
                                    ),
                                    channel_half_warps=(
                                        warp_assignment_mode == "channel-half"
                                    ),
                                    explicit_fallback_diagonal=(
                                        fallback_diagonal_mode == "explicit"
                                    ),
                                ),
                                *compile_args,
                            )
                            compile_seconds = time.perf_counter() - cold_start

                            run_candidate = partial(compiled, *compile_args)

                            run_candidate()
                            torch.cuda.synchronize()
                            relative_errors = [
                                _relative_error(candidate, exact)
                                for candidate, exact in zip(
                                    candidate_outputs,
                                    exact_outputs,
                                    strict=True,
                                )
                            ]
                            candidate_samples = _measure(
                                run_candidate,
                                args.warmup,
                                args.repetitions,
                            )
                            candidate_median = statistics.median(candidate_samples)
                            candidates.append(
                                {
                                    "swizzle_bits": bits,
                                    "negative_factor": negative_factor,
                                    "gate_cache": gate_cache_mode,
                                    "lower_epilogue": lower_epilogue_mode,
                                    "warp_assignment": warp_assignment_mode,
                                    "fallback_diagonal": fallback_diagonal_mode,
                                    "expected_max_store_conflict": 16 // (2**bits),
                                    "compile_seconds": compile_seconds,
                                    "samples_ms": candidate_samples,
                                    "median_ms": candidate_median,
                                    "speedup_vs_exact": (
                                        exact_median / candidate_median
                                    ),
                                    "relative_errors": relative_errors,
                                    "maximum_absolute_errors": [
                                        float((candidate - exact).abs().max())
                                        for candidate, exact in zip(
                                            candidate_outputs,
                                            exact_outputs,
                                            strict=True,
                                        )
                                    ],
                                }
                            )
    payload = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "blocks": args.blocks,
        "mean_log2_decay": args.mean_log2_decay,
        "threshold": args.threshold,
        "guard_hit_rate": float((spans <= args.threshold).float().mean()),
        "exact_samples_ms": exact_samples,
        "exact_median_ms": exact_median,
        "candidates": candidates,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
