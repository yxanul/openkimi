from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def _pointer(tensor: torch.Tensor):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(tensor.detach(), assumed_align=16).iterator


def _time_cuda(function, warmup: int, repetitions: int) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples: list[float] = []
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
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    args = parser.parse_args()

    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cutlass.cutlass_dsl import Float32, Int32
    from fla.modules.l2norm import l2norm_fwd
    from fla.ops.kda.gate import kda_gate_chunk_cumsum
    from fla.ops.utils.constant import RCP_LN2

    working = Path(__file__).resolve().parent / "working"
    sys.path.insert(0, str(working))
    from preprocess_cute import DIM, KdaPreprocessSm90

    torch.manual_seed(123)
    shape = (args.batch, args.sequence_length, args.heads, DIM)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    raw_gate = torch.randn_like(q)
    beta_logits = torch.randn(
        shape[:-1],
        device="cuda",
        dtype=torch.bfloat16,
    )
    a_log = torch.log(
        torch.empty(args.heads, device="cuda", dtype=torch.float32).uniform_(
            1.0,
            16.0,
        )
    )
    dt_bias = torch.empty(
        args.heads,
        DIM,
        device="cuda",
        dtype=torch.float32,
    ).uniform_(-4.0, -1.0)
    q_norm = torch.empty_like(q)
    k_norm = torch.empty_like(k)
    q_rstd = torch.empty(shape[:-1], device="cuda", dtype=torch.float32)
    k_rstd = torch.empty_like(q_rstd)
    gate_cumsum = torch.empty_like(raw_gate, dtype=torch.float32)
    beta = torch.empty_like(beta_logits)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compile_args = (
        *[
            _pointer(tensor)
            for tensor in (
                q,
                k,
                raw_gate,
                beta_logits,
                a_log,
                dt_bias,
                q_norm,
                k_norm,
                q_rstd,
                k_rstd,
                gate_cumsum,
                beta,
            )
        ],
        Int32(args.batch),
        Int32(args.sequence_length),
        Int32(args.heads),
        Float32(1e-6),
        stream,
    )
    compiled = cute.compile(KdaPreprocessSm90(), *compile_args)

    def run_cute() -> None:
        compiled(*compile_args)

    def run_fla():
        q_ref, q_rstd_ref = l2norm_fwd(q)
        k_ref, k_rstd_ref = l2norm_fwd(k)
        gate_ref = kda_gate_chunk_cumsum(
            g=raw_gate,
            A_log=a_log,
            dt_bias=dt_bias,
            chunk_size=64,
            scale=RCP_LN2,
        )
        beta_ref = beta_logits.sigmoid()
        return (
            q_ref,
            k_ref,
            q_rstd_ref,
            k_rstd_ref,
            gate_ref,
            beta_ref,
        )

    references = run_fla()
    run_cute()
    torch.cuda.synchronize()
    candidates = (
        q_norm,
        k_norm,
        q_rstd,
        k_rstd,
        gate_cumsum,
        beta,
    )
    comparisons = {}
    for name, candidate, reference in zip(
        ("q", "k", "q_rstd", "k_rstd", "gate", "beta"),
        candidates,
        references,
        strict=True,
    ):
        delta = (candidate.float() - reference.float()).abs()
        comparisons[name] = {
            "max_abs": delta.max().item(),
            "mean_abs": delta.mean().item(),
        }

    cute_samples = _time_cuda(run_cute, args.warmup, args.repetitions)
    fla_samples = _time_cuda(
        lambda: run_fla(),
        args.warmup,
        args.repetitions,
    )
    cute_median = torch.tensor(cute_samples).median().item()
    fla_median = torch.tensor(fla_samples).median().item()
    print(
        json.dumps(
            {
                "shape": shape,
                "cute_ms": cute_median,
                "fla_ms": fla_median,
                "speedup": fla_median / cute_median,
                "comparisons": comparisons,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
