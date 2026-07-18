from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def _json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(actual.float() - expected.float())
    denominator = torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
    return float(numerator / denominator)


def _make_inputs(
    batch: int,
    sequence_length: int,
    heads: int,
    head_dim: int,
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


def _fla_forward(
    inputs: list[torch.Tensor],
    *,
    scale: float,
    disable_recompute: bool,
    safe_gate: bool,
) -> torch.Tensor:
    from fla.ops.kda import chunk_kda

    q, k, v, raw_decay, beta_logits, a_log, dt_bias = inputs
    gate = raw_decay
    use_gate_in_kernel = True
    if safe_gate:
        # Preserve the primary KDA gate equation while selecting FLA's
        # midpoint-factorized intra-chunk path. FLA's fused safe-gate
        # activation instead uses a bounded sigmoid and is not equivalent.
        gate = -a_log.exp().view(1, 1, -1, 1) * F.softplus(
            raw_decay.float() + dt_bias.view(1, 1, q.shape[2], q.shape[3])
        )
        use_gate_in_kernel = False
    output, _ = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=gate,
        beta=beta_logits,
        A_log=a_log,
        dt_bias=dt_bias,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=safe_gate,
        output_final_state=False,
        state_v_first=True,
        disable_recompute=disable_recompute,
    )
    return output


def _measure_fla(
    name: str,
    inputs: list[torch.Tensor],
    gradient: torch.Tensor,
    *,
    scale: float,
    disable_recompute: bool,
    safe_gate: bool,
    warmup: int,
    repeats: int,
) -> tuple[dict[str, object], torch.Tensor, list[torch.Tensor]]:
    forward_samples: list[float] = []
    backward_samples: list[float] = []
    total_samples: list[float] = []
    cold_start_seconds = 0.0
    output_snapshot: torch.Tensor | None = None
    gradient_snapshots: list[torch.Tensor] = []
    for iteration in range(1 + warmup + repeats):
        for value in inputs:
            value.grad = None
        if iteration == 1 + warmup:
            torch.cuda.reset_peak_memory_stats()
        wall_start = time.perf_counter()
        start = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)
        backward_end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = _fla_forward(
            inputs,
            scale=scale,
            disable_recompute=disable_recompute,
            safe_gate=safe_gate,
        )
        forward_end.record()
        output.backward(gradient)
        backward_end.record()
        torch.cuda.synchronize()
        if iteration == 0:
            cold_start_seconds = time.perf_counter() - wall_start
        elif iteration > warmup:
            forward_ms = start.elapsed_time(forward_end)
            backward_ms = forward_end.elapsed_time(backward_end)
            forward_samples.append(forward_ms)
            backward_samples.append(backward_ms)
            total_samples.append(forward_ms + backward_ms)
        if iteration == warmup + repeats:
            output_snapshot = output.detach().clone()
            gradient_snapshots = [
                value.grad.detach().clone()
                for value in inputs
            ]
    assert output_snapshot is not None
    return (
        {
            "provider": name,
            "disable_recompute": disable_recompute,
            "safe_gate": safe_gate,
            "cold_start_seconds": cold_start_seconds,
            "forward_samples_ms": forward_samples,
            "backward_samples_ms": backward_samples,
            "total_samples_ms": total_samples,
            "forward_median_ms": statistics.median(forward_samples),
            "backward_median_ms": statistics.median(backward_samples),
            "total_median_ms": statistics.median(total_samples),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "output_l2": float(output_snapshot.float().norm()),
            "gradient_l2": [
                float(value.float().norm())
                for value in gradient_snapshots
            ],
        },
        output_snapshot,
        gradient_snapshots,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark exact-shape KDA forward and backward providers."
    )
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--include-safe-gate",
        action="store_true",
        help=(
            "also measure FLA's midpoint-factorized path with the faithful "
            "gate equation precomputed outside the kernel"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        help=(
            "save the last provider's output and input gradients separately "
            "for cross-process exact/experimental parity checks"
        ),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    if torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError("the optimized KDA benchmark requires SM90+")

    torch.manual_seed(240719)
    inputs, gradient = _make_inputs(
        args.batch,
        args.sequence_length,
        args.heads,
        args.head_dim,
    )
    scale = args.head_dim**-0.5
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "shape": [
                    args.batch,
                    args.sequence_length,
                    args.heads,
                    args.head_dim,
                ],
                "warmup": args.warmup,
                "repeats": args.repeats,
            }
        )
    )

    results: list[tuple[dict[str, object], torch.Tensor, list[torch.Tensor]]] = []
    providers = [
        ("fla_recompute", False, False),
        ("fla_saved_intermediates", True, False),
    ]
    if args.include_safe_gate:
        providers.append(("fla_safe_gate_saved_intermediates", True, True))
    for name, disable_recompute, safe_gate in providers:
        result = _measure_fla(
            name,
            inputs,
            gradient,
            scale=scale,
            disable_recompute=disable_recompute,
            safe_gate=safe_gate,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        results.append(result)
        print(json.dumps(_json_safe(result[0])))

    expected_result, expected_output, expected_gradients = results[0]
    comparisons: list[dict[str, object]] = []
    for result, output, gradients in results[1:]:
        comparison = {
            "comparison": f"{result['provider']}_vs_{expected_result['provider']}",
            "output_relative_error": _relative_error(output, expected_output),
            "gradient_relative_errors": [
                _relative_error(actual, expected)
                for actual, expected in zip(
                    gradients,
                    expected_gradients,
                    strict=True,
                )
            ],
        }
        comparisons.append(comparison)
        print(json.dumps(_json_safe(comparison)))

    if args.snapshot_dir is not None:
        _, snapshot_output, snapshot_gradients = results[-1]
        args.snapshot_dir.mkdir(parents=True, exist_ok=True)
        torch.save(snapshot_output.cpu(), args.snapshot_dir / "output.pt")
        for index, value in enumerate(snapshot_gradients):
            torch.save(value.cpu(), args.snapshot_dir / f"gradient_{index}.pt")
        snapshot_metadata = {
            "provider": results[-1][0]["provider"],
            "shape": list(snapshot_output.shape),
            "files": [
                "output.pt",
                *[
                    f"gradient_{index}.pt"
                    for index in range(len(snapshot_gradients))
                ],
            ],
        }
        (args.snapshot_dir / "metadata.json").write_text(
            json.dumps(snapshot_metadata, indent=2) + "\n"
        )

    if args.output is not None:
        payload = {
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "shape": [
                args.batch,
                args.sequence_length,
                args.heads,
                args.head_dim,
            ],
            "warmup": args.warmup,
            "repeats": args.repeats,
            "providers": [result[0] for result in results],
            "comparisons": comparisons,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(_json_safe(payload), indent=2) + "\n")


if __name__ == "__main__":
    main()
