from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _cute_pointer(tensor: torch.Tensor):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(tensor.detach(), assumed_align=16).iterator


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile the supplied SM90 KDA prepare draft.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--heads", type=int, default=1)
    args = parser.parse_args()

    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cutlass.cutlass_dsl import Int32

    working_dir = Path(__file__).resolve().parent / "working"
    sys.path.insert(0, str(working_dir))
    from sm90_cute_dsl import (  # noqa: PLC0415
        CHUNK,
        DIM,
        KdaPrepareFwdSm90,
        WorkspaceShapes,
    )

    if args.sequence_length % CHUNK:
        raise ValueError("the initial smoke test requires a multiple of 16 tokens")
    shape = (args.batch, args.sequence_length, args.heads, DIM)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    raw_decay = torch.randn_like(q)
    beta_logits = torch.randn(
        args.batch,
        args.sequence_length,
        args.heads,
        device="cuda",
        dtype=torch.bfloat16,
    )
    a_log = torch.log(
        torch.empty(args.heads, device="cuda", dtype=torch.float32).uniform_(1.0, 16.0)
    )
    dt_bias = torch.zeros(args.heads, DIM, device="cuda", dtype=torch.float32)
    workspace_shapes = WorkspaceShapes(
        args.batch,
        args.sequence_length,
        args.heads,
    ).as_dict()
    qd = torch.empty(workspace_shapes["qd"], device="cuda", dtype=torch.bfloat16)
    kd = torch.empty_like(qd)
    kc = torch.empty_like(qd)
    e = torch.empty(workspace_shapes["e"], device="cuda", dtype=torch.float32)
    ainv = torch.empty(workspace_shapes["ainv"], device="cuda", dtype=torch.bfloat16)
    mqk = torch.empty_like(ainv)
    local_l = torch.empty_like(ainv)
    beta = torch.empty(workspace_shapes["beta"], device="cuda", dtype=torch.bfloat16)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compile_args = (
        *[
            _cute_pointer(tensor)
            for tensor in (
                q,
                k,
                raw_decay,
                beta_logits,
                a_log,
                dt_bias,
                qd,
                kd,
                kc,
                e,
                ainv,
                mqk,
                local_l,
                beta,
            )
        ],
        Int32(args.batch),
        Int32(args.sequence_length),
        Int32(args.heads),
        stream,
    )
    compiled = cute.compile(KdaPrepareFwdSm90(), *compile_args)
    compiled(*compile_args)
    torch.cuda.synchronize()
    print(
        {
            "shape": shape,
            "qd_finite": bool(torch.isfinite(qd).all()),
            "ainv_finite": bool(torch.isfinite(ainv).all()),
            "mqk_finite": bool(torch.isfinite(mqk).all()),
        }
    )


if __name__ == "__main__":
    main()
