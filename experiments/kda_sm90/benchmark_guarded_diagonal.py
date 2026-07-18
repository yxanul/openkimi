"""H100 microbenchmark for the guarded KDA diagonal-backward factorization.

This isolates the two diagonal phases in FLA's
``chunk_kda_bwd_kernel_intra``. It is intentionally not wired into training:
the next H100 run must establish a safe guard threshold and a real speedup
before the same schedule is lowered as an ABI-compatible CuTe kernel.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

BLOCK: int = 16
CHANNELS: int = 32


@triton.jit
def _exact_diagonal_kernel(
    gate,
    q,
    k,
    beta,
    da_qk,
    da_kk,
    dq,
    dk,
    dkt,
    BC: tl.constexpr,
    BK: tl.constexpr,
):
    block = tl.program_id(0)
    rows = tl.arange(0, BC)
    channels = tl.arange(0, BK)
    gate_offsets = block * BC * BK + rows[:, None] * BK + channels[None, :]

    b_gate = tl.load(gate + gate_offsets).to(tl.float32)

    b_dq = tl.zeros((BC, BK), dtype=tl.float32)
    b_dk = tl.zeros((BC, BK), dtype=tl.float32)
    b_dkt = tl.zeros((BC, BK), dtype=tl.float32)
    for partner in tl.static_range(BC):
        lower = rows[:, None] >= partner
        upper = rows[:, None] <= partner
        partner_offsets = block * BC * BK + partner * BK + channels
        partner_gate = tl.load(gate + partner_offsets).to(tl.float32)[None, :]
        partner_k = tl.load(k + partner_offsets).to(tl.float32)[None, :]
        partner_q = tl.load(q + partner_offsets).to(tl.float32)[None, :]
        partner_beta = tl.load(beta + block * BC + partner).to(tl.float32)
        partner_matrix_offsets = block * BC * BC + rows * BC + partner
        partner_da_qk = tl.load(da_qk + partner_matrix_offsets).to(tl.float32)
        partner_da_kk = tl.load(da_kk + partner_matrix_offsets).to(tl.float32)
        lower_decay = tl.exp2(b_gate - partner_gate)
        upper_decay = tl.exp2(partner_gate - b_gate)
        b_dq += tl.where(
            lower,
            partner_da_qk[:, None] * partner_k * lower_decay,
            0.0,
        )
        b_dk += tl.where(
            lower,
            partner_da_kk[:, None] * partner_k * lower_decay,
            0.0,
        )
        b_dkt += tl.where(
            upper,
            partner_da_qk[:, None] * partner_q * upper_decay,
            0.0,
        )
        b_dkt += tl.where(
            upper,
            partner_da_kk[:, None]
            * partner_k
            * partner_beta
            * upper_decay,
            0.0,
        )

    tl.store(dq + gate_offsets, b_dq)
    tl.store(dk + gate_offsets, b_dk)
    tl.store(dkt + gate_offsets, b_dkt)


@triton.jit
def _guarded_diagonal_kernel(
    gate,
    q,
    k,
    beta,
    da_qk,
    da_kk,
    dq,
    dk,
    dkt,
    max_log2_span,
    BC: tl.constexpr,
    BK: tl.constexpr,
    CENTER_REFERENCE: tl.constexpr,
):
    block = tl.program_id(0)
    rows = tl.arange(0, BC)
    channels = tl.arange(0, BK)
    gate_offsets = block * BC * BK + rows[:, None] * BK + channels[None, :]
    matrix_offsets = block * BC * BC + rows[:, None] * BC + rows[None, :]
    beta_offsets = block * BC + rows

    b_gate = tl.load(gate + gate_offsets).to(tl.float32)
    b_q = tl.load(q + gate_offsets).to(tl.float32)
    b_k = tl.load(k + gate_offsets).to(tl.float32)
    b_beta = tl.load(beta + beta_offsets).to(tl.float32)
    b_da_qk = tl.load(da_qk + matrix_offsets).to(tl.float32)
    b_da_kk = tl.load(da_kk + matrix_offsets).to(tl.float32)

    maximum_gate = tl.max(b_gate, axis=0)
    minimum_gate = tl.min(b_gate, axis=0)
    if CENTER_REFERENCE:
        # Centering minimizes the maximum absolute exponent. A total span of
        # 250 keeps both factors in the FP32 normal range (approximately
        # 2**[-125, 125]); larger tiles take the exact pairwise fallback.
        reference = ((maximum_gate + minimum_gate) * 0.5)[None, :]
    else:
        # The first row is the maximum cumulative gate for faithful KDA. This
        # keeps exp2(g-reference) <= 1 but only supports much smaller spans.
        reference_offsets = block * BC * BK + channels
        reference = tl.load(gate + reference_offsets).to(tl.float32)[None, :]
    channel_spans = maximum_gate - minimum_gate
    block_span = tl.max(channel_spans, axis=0)
    previous = tl.load(
        gate + gate_offsets - BK,
        mask=rows[:, None] > 0,
        other=0.0,
    ).to(tl.float32)
    monotonic = (rows[:, None] == 0) | (b_gate <= previous)
    finite = (b_gate == b_gate) & (tl.abs(b_gate) < 3.402823e38)
    valid_entries = tl.sum(
        tl.sum((monotonic & finite).to(tl.int32), axis=1),
        axis=0,
    )
    guard_valid = valid_entries == BC * BK

    if guard_valid & (block_span <= max_log2_span):
        # Do not evaluate the positive exponent until after the guard. The
        # fallback must remain safe even when the rejected span would overflow.
        positive = tl.exp2(b_gate - reference)
        negative = tl.exp2(reference - b_gate)
        lower_matrix = rows[:, None] >= rows[None, :]
        upper_matrix = rows[:, None] <= rows[None, :]
        lower_qk = tl.where(lower_matrix, b_da_qk, 0.0)
        lower_kk = tl.where(lower_matrix, b_da_kk, 0.0)
        upper_qk = tl.where(upper_matrix, b_da_qk, 0.0)
        upper_kk = tl.where(upper_matrix, b_da_kk, 0.0)
        b_dq = (
            tl.dot(lower_qk, b_k * negative, input_precision="tf32")
            * positive
        )
        b_dk = (
            tl.dot(lower_kk, b_k * negative, input_precision="tf32")
            * positive
        )
        b_dkt = (
            tl.dot(upper_qk, b_q * positive, input_precision="tf32")
            * negative
        )
        b_dkt += (
            tl.dot(
                upper_kk,
                b_k * b_beta[:, None] * positive,
                input_precision="tf32",
            )
            * negative
        )
    else:
        b_dq = tl.zeros((BC, BK), dtype=tl.float32)
        b_dk = tl.zeros((BC, BK), dtype=tl.float32)
        b_dkt = tl.zeros((BC, BK), dtype=tl.float32)
        for partner in tl.static_range(BC):
            lower_rows = rows[:, None] >= partner
            upper_rows = rows[:, None] <= partner
            partner_offsets = block * BC * BK + partner * BK + channels
            partner_gate = tl.load(gate + partner_offsets).to(tl.float32)[None, :]
            partner_k = tl.load(k + partner_offsets).to(tl.float32)[None, :]
            partner_q = tl.load(q + partner_offsets).to(tl.float32)[None, :]
            partner_beta = tl.load(beta + block * BC + partner).to(tl.float32)
            partner_matrix_offsets = block * BC * BC + rows * BC + partner
            partner_da_qk = tl.load(da_qk + partner_matrix_offsets).to(tl.float32)
            partner_da_kk = tl.load(da_kk + partner_matrix_offsets).to(tl.float32)
            lower_decay = tl.exp2(b_gate - partner_gate)
            upper_decay = tl.exp2(partner_gate - b_gate)
            b_dq += tl.where(
                lower_rows,
                partner_da_qk[:, None] * partner_k * lower_decay,
                0.0,
            )
            b_dk += tl.where(
                lower_rows,
                partner_da_kk[:, None] * partner_k * lower_decay,
                0.0,
            )
            b_dkt += tl.where(
                upper_rows,
                partner_da_qk[:, None] * partner_q * upper_decay,
                0.0,
            )
            b_dkt += tl.where(
                upper_rows,
                partner_da_kk[:, None]
                * partner_k
                * partner_beta
                * upper_decay,
                0.0,
            )

    tl.store(dq + gate_offsets, b_dq)
    tl.store(dk + gate_offsets, b_dk)
    tl.store(dkt + gate_offsets, b_dkt)


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(actual.float() - expected.float())
    denominator = torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
    return float(numerator / denominator)


def _measure(
    function,
    *,
    warmup: int,
    repetitions: int,
) -> tuple[list[float], float]:
    cold_start = time.perf_counter()
    function()
    torch.cuda.synchronize()
    cold_seconds = time.perf_counter() - cold_start
    for _ in range(max(0, warmup - 1)):
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
    return samples, cold_seconds


def _distribution(samples: list[float]) -> dict[str, float]:
    values = torch.tensor(samples)
    return {
        "minimum_ms": min(samples),
        "p10_ms": float(torch.quantile(values, 0.1)),
        "median_ms": statistics.median(samples),
        "p90_ms": float(torch.quantile(values, 0.9)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=32768)
    parser.add_argument("--mean-log2-decay", type=float, default=0.5)
    parser.add_argument(
        "--thresholds",
        default="4,8,12,16,20,24,30",
        help="Comma-separated maximum within-block log2 spans.",
    )
    parser.add_argument(
        "--reference-policy",
        choices=("first", "midpoint"),
        default="first",
        help="Factorization reference; midpoint supports much larger safe spans.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--relative-tolerance", type=float, default=5e-3)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON result path.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError("this benchmark requires an SM90+ CUDA GPU")
    if args.blocks <= 0 or args.mean_log2_decay <= 0:
        raise ValueError("blocks and mean-log2-decay must be positive")

    torch.manual_seed(1907)
    shape = (args.blocks, BLOCK, CHANNELS)
    increments = torch.empty(shape, device="cuda", dtype=torch.float32).exponential_()
    increments *= args.mean_log2_decay
    increments[:, 0] = 0.0
    gate = -increments.cumsum(dim=1)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    beta = torch.sigmoid(
        torch.randn(args.blocks, BLOCK, device="cuda", dtype=torch.bfloat16)
    )
    da_qk = torch.randn(
        args.blocks,
        BLOCK,
        BLOCK,
        device="cuda",
        dtype=torch.float32,
    )
    da_kk = torch.randn_like(da_qk)
    exact_outputs = [torch.empty_like(gate) for _ in range(3)]
    candidate_outputs = [torch.empty_like(gate) for _ in range(3)]

    def run_exact() -> None:
        _exact_diagonal_kernel[(args.blocks,)](
            gate,
            q,
            k,
            beta,
            da_qk,
            da_kk,
            *exact_outputs,
            BC=BLOCK,
            BK=CHANNELS,
            num_warps=4,
        )

    exact_samples, exact_cold_seconds = _measure(
        run_exact,
        warmup=args.warmup,
        repetitions=args.repetitions,
    )
    exact_median = statistics.median(exact_samples)
    thresholds = [float(value) for value in args.thresholds.split(",")]
    maximum_threshold = 252.0 if args.reference_policy == "midpoint" else 127.0
    if not thresholds or any(
        not 0.0 < value < maximum_threshold for value in thresholds
    ):
        raise ValueError(
            f"thresholds must be finite values between 0 and {maximum_threshold:g}"
        )
    results = []
    block_spans = (gate[:, :1] - gate).amax(dim=(1, 2))
    monotonic = torch.all(gate[:, 1:, :] <= gate[:, :-1, :], dim=(1, 2))
    finite = torch.all(torch.isfinite(gate), dim=(1, 2))
    for threshold in thresholds:

        def run_candidate(threshold: float = threshold) -> None:
            _guarded_diagonal_kernel[(args.blocks,)](
                gate,
                q,
                k,
                beta,
                da_qk,
                da_kk,
                *candidate_outputs,
                threshold,
                BC=BLOCK,
                BK=CHANNELS,
                CENTER_REFERENCE=args.reference_policy == "midpoint",
                num_warps=4,
            )

        candidate_samples, candidate_cold_seconds = _measure(
            run_candidate,
            warmup=args.warmup,
            repetitions=args.repetitions,
        )
        errors = [
            _relative_error(candidate, exact)
            for candidate, exact in zip(
                candidate_outputs,
                exact_outputs,
                strict=True,
            )
        ]
        maximum_errors = [
            float((candidate.float() - exact.float()).abs().max())
            for candidate, exact in zip(
                candidate_outputs,
                exact_outputs,
                strict=True,
            )
        ]
        hit_rate = float(
            (monotonic & finite & (block_spans <= threshold)).float().mean()
        )
        result = {
            "threshold": threshold,
            "guard_hit_rate": hit_rate,
            "cold_seconds": candidate_cold_seconds,
            **_distribution(candidate_samples),
            "speedup": exact_median / statistics.median(candidate_samples),
            "relative_errors": errors,
            "maximum_absolute_errors": maximum_errors,
        }
        results.append(result)
        if hit_rate > 0.0 and max(errors) > args.relative_tolerance:
            raise RuntimeError(
                f"threshold {threshold} failed parity: {max(errors):.6g} "
                f"> {args.relative_tolerance:.6g}"
            )

    payload = {
        "device": torch.cuda.get_device_name(),
        "blocks": args.blocks,
        "mean_log2_decay": args.mean_log2_decay,
        "reference_policy": args.reference_policy,
        "exact_cold_seconds": exact_cold_seconds,
        "exact": _distribution(exact_samples),
        "results": results,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
