from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from fused_wy_intra_triton import (  # noqa: E402
    install_fused_wy_intra_experiment,
    install_intra_chunk_experiment,
)


def _make_inputs(
    batch: int,
    sequence_length: int,
    heads: int = 6,
    head_dim: int = 128,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    shape = (batch, sequence_length, heads, head_dim)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    raw_decay = torch.randn_like(q, requires_grad=True)
    beta_logits = torch.randn(
        batch,
        sequence_length,
        heads,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    a_log = torch.log(
        torch.empty(heads, device="cuda", dtype=torch.float32).uniform_(1.0, 16.0)
    ).requires_grad_()
    dt_bias = torch.zeros(
        heads * head_dim,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    gradient = torch.randn_like(v)
    return [q, k, v, raw_decay, beta_logits, a_log, dt_bias], gradient


def _forward(inputs: list[torch.Tensor]) -> torch.Tensor:
    from fla.ops.kda import chunk_kda

    q, k, v, raw_decay, beta_logits, a_log, dt_bias = inputs
    output, _ = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=raw_decay,
        beta=beta_logits,
        A_log=a_log,
        dt_bias=dt_bias,
        scale=q.shape[-1] ** -0.5,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=False,
        output_final_state=False,
        state_v_first=True,
        disable_recompute=True,
    )
    return output


def _measure(
    inputs: list[torch.Tensor],
    gradient: torch.Tensor,
    *,
    warmup: int,
    repetitions: int,
) -> tuple[dict[str, object], torch.Tensor, list[torch.Tensor]]:
    samples: list[float] = []
    snapshot: torch.Tensor | None = None
    gradients: list[torch.Tensor] = []
    for iteration in range(1 + warmup + repetitions):
        for tensor in inputs:
            tensor.grad = None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = _forward(inputs)
        output.backward(gradient)
        end.record()
        torch.cuda.synchronize()
        if iteration > warmup:
            samples.append(start.elapsed_time(end))
        if iteration == warmup + repetitions:
            snapshot = output.detach().clone()
            gradients = [tensor.grad.detach().clone() for tensor in inputs]
    assert snapshot is not None
    return (
        {
            "samples_ms": samples,
            "median_ms": statistics.median(samples),
        },
        snapshot,
        gradients,
    )


def _metric(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = actual.float() - expected.float()
    denominator = expected.float().norm().clamp_min(1e-12)
    return {
        "relative_l2": float(difference.norm() / denominator),
        "max_absolute_error": float(difference.abs().max()),
        "reference_l2": float(denominator),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--candidate",
        choices=("blocked", "exact", "fused"),
        default="blocked",
    )
    parser.add_argument("--num-warps", type=int, choices=(2, 4, 8), default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.manual_seed(240719)
    inputs, gradient = _make_inputs(args.batch, args.sequence_length)
    from fla.ops.kda import chunk_intra

    packaged_threshold = chunk_intra._OPENKIMI_GUARD_MAX_LOG2_SPAN
    packaged, packaged_output, packaged_gradients = _measure(
        inputs,
        gradient,
        warmup=args.warmup,
        repetitions=args.repetitions,
    )
    chunk_intra._OPENKIMI_GUARD_MAX_LOG2_SPAN = -1.0
    exact, exact_output, exact_gradients = _measure(
        inputs,
        gradient,
        warmup=args.warmup,
        repetitions=args.repetitions,
    )
    chunk_intra._OPENKIMI_GUARD_MAX_LOG2_SPAN = packaged_threshold
    if args.candidate == "fused":
        install_fused_wy_intra_experiment(num_warps=args.num_warps)
    else:
        install_intra_chunk_experiment(args.candidate, num_warps=args.num_warps)
    candidate, candidate_output, candidate_gradients = _measure(
        inputs,
        gradient,
        warmup=args.warmup,
        repetitions=args.repetitions,
    )

    names = ("output", "q", "k", "v", "raw_decay", "beta", "A_log", "dt_bias")
    candidate_metrics = {
        name: _metric(actual, expected)
        for name, actual, expected in zip(
            names,
            [candidate_output, *candidate_gradients],
            [exact_output, *exact_gradients],
            strict=True,
        )
    }
    packaged_metrics = {
        name: _metric(actual, expected)
        for name, actual, expected in zip(
            names,
            [packaged_output, *packaged_gradients],
            [exact_output, *exact_gradients],
            strict=True,
        )
    }
    payload = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "shape": [args.batch, args.sequence_length, 6, 128],
        "candidate": args.candidate,
        "candidate_num_warps": args.num_warps,
        "packaged_tf32x3_span232": packaged,
        "exact_fla": exact,
        "candidate_result": candidate,
        "speedup_vs_packaged": packaged["median_ms"] / candidate["median_ms"],
        "candidate_metrics_vs_exact": candidate_metrics,
        "packaged_metrics_vs_exact": packaged_metrics,
        "candidate_maximum_relative_l2": max(
            value["relative_l2"] for value in candidate_metrics.values()
        ),
    }
    print(json.dumps(payload, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
