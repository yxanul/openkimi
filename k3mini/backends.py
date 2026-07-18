from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec

import torch

from .config import (
    KernelBackend,
    LinearPrecision,
    LossBackend,
    RoutedExpertBackend,
)


@dataclass(frozen=True, slots=True)
class BackendStatus:
    requested: KernelBackend
    selected: KernelBackend
    kda: str
    short_conv: str
    attnres: str
    routed_expert_backend: RoutedExpertBackend
    expert_mlp: str
    linear_precision: LinearPrecision
    dense_ffn: str
    loss_backend: LossBackend
    loss: str

    def as_dict(self) -> dict[str, str]:
        return {
            "requested": self.requested.value,
            "selected": self.selected.value,
            "kda": self.kda,
            "short_conv": self.short_conv,
            "attnres": self.attnres,
            "routed_expert_backend": self.routed_expert_backend.value,
            "expert_mlp": self.expert_mlp,
            "linear_precision": self.linear_precision.value,
            "dense_ffn": self.dense_ffn,
            "loss_backend": self.loss_backend.value,
            "loss": self.loss,
        }


def _h100_capable() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


def _cuda_dependencies() -> tuple[bool, list[str]]:
    missing: list[str] = []
    if find_spec("fla") is None:
        missing.append("flash-linear-attention")
    if find_spec("megablocks") is None or find_spec("grouped_gemm") is None:
        missing.append("megablocks/grouped_gemm")
    return not missing, missing


def _resolve_loss_backend(requested: LossBackend, kernel_backend: KernelBackend) -> LossBackend:
    requested = LossBackend(requested)
    if kernel_backend is KernelBackend.REFERENCE:
        if requested not in {LossBackend.AUTO, LossBackend.TORCH}:
            raise RuntimeError(f"loss_backend={requested.value} requires the H100 kernel backend")
        return LossBackend.TORCH
    if requested is LossBackend.AUTO:
        return LossBackend.LIGER if find_spec("liger_kernel") is not None else LossBackend.FLA
    if requested is LossBackend.LIGER and find_spec("liger_kernel") is None:
        raise RuntimeError(
            "loss_backend=liger requires liger-kernel; install with `uv sync --extra cuda`"
        )
    if requested is LossBackend.QUACK and find_spec("quack") is None:
        raise RuntimeError(
            "loss_backend=quack requires quack-kernels 0.6.1 in a Python 3.12, "
            "CUDA 12.9+ environment"
        )
    return requested


def _resolve_routed_expert_backend(
    requested: RoutedExpertBackend,
    kernel_backend: KernelBackend,
) -> RoutedExpertBackend:
    requested = RoutedExpertBackend(requested)
    if kernel_backend is KernelBackend.REFERENCE:
        if requested not in {RoutedExpertBackend.AUTO, RoutedExpertBackend.REFERENCE}:
            raise RuntimeError(
                f"routed_expert_backend={requested.value} requires the H100 kernel backend"
            )
        return RoutedExpertBackend.REFERENCE
    if requested is RoutedExpertBackend.AUTO:
        return RoutedExpertBackend.MEGABLOCKS
    if requested is RoutedExpertBackend.REFERENCE:
        raise RuntimeError("routed_expert_backend=reference requires kernel_backend=reference")
    if requested is RoutedExpertBackend.SONIC and find_spec("sonicmoe") is None:
        raise RuntimeError(
            "routed_expert_backend=sonic requires the isolated SonicMoE environment; "
            "run scripts/install_sonic_isolated.sh"
        )
    return requested


def resolve_backend(
    requested: KernelBackend,
    loss_backend: LossBackend = LossBackend.AUTO,
    linear_precision: LinearPrecision = LinearPrecision.BF16,
    routed_expert_backend: RoutedExpertBackend = RoutedExpertBackend.AUTO,
) -> BackendStatus:
    requested = KernelBackend(requested)
    linear_precision = LinearPrecision(linear_precision)
    dependencies_ok, missing = _cuda_dependencies()
    if requested is KernelBackend.H100:
        if not _h100_capable():
            raise RuntimeError("kernel_backend=h100 requires an NVIDIA SM90+ GPU")
        if not dependencies_ok:
            raise RuntimeError(
                "kernel_backend=h100 is missing CUDA extras: "
                + ", ".join(missing)
                + "; install with `uv sync --extra cuda`"
            )
        selected = KernelBackend.H100
    elif requested is KernelBackend.AUTO and _h100_capable() and dependencies_ok:
        selected = KernelBackend.H100
    else:
        selected = KernelBackend.REFERENCE

    selected_loss = _resolve_loss_backend(loss_backend, selected)
    selected_experts = _resolve_routed_expert_backend(routed_expert_backend, selected)
    if linear_precision is LinearPrecision.FP8_CURRENT:
        if selected is not KernelBackend.H100:
            raise RuntimeError("linear_precision=fp8_current requires an NVIDIA SM90+ GPU")
        if find_spec("transformer_engine") is None:
            raise RuntimeError(
                "linear_precision=fp8_current requires Transformer Engine 2.16; "
                "install with `uv sync --extra cuda`"
            )
        if selected_loss not in {LossBackend.LIGER, LossBackend.QUACK}:
            raise RuntimeError(
                "linear_precision=fp8_current requires loss_backend=liger or quack "
                "for the chunked FP8 LM head"
            )
    elif selected_loss is LossBackend.QUACK:
        raise RuntimeError("loss_backend=quack currently requires linear_precision=fp8_current")
    if selected is KernelBackend.H100:
        loss_name = {
            LossBackend.TORCH: "torch.cross_entropy",
            LossBackend.FLA: "fla.modules.FusedLinearCrossEntropyLoss",
            LossBackend.LIGER: "liger_kernel.LigerFusedLinearCrossEntropyLoss",
            LossBackend.QUACK: "quack.cross_entropy_fwd_out",
        }[selected_loss]
        return BackendStatus(
            requested=requested,
            selected=selected,
            kda="fla.ops.kda.chunk_kda",
            short_conv="fla.modules.ShortConvolution",
            attnres="fla.ops.attnres.fused_attnres(configurable checkpoint_level)",
            routed_expert_backend=selected_experts,
            expert_mlp=(
                "sonicmoe.bitmatrix+quack_grouped_gemm+fused_swiglu"
                if selected_experts is RoutedExpertBackend.SONIC
                else "megablocks.permute+device_counts+grouped_gemm"
            ),
            linear_precision=linear_precision,
            dense_ffn=(
                "transformer_engine.Linear(Float8CurrentScaling)"
                if linear_precision is LinearPrecision.FP8_CURRENT
                else "torch.nn.Linear(bf16 autocast)"
            ),
            loss_backend=selected_loss,
            loss=(
                f"transformer_engine.BasicLinear(fp8_current)+{selected_loss.value}_cross_entropy"
                if linear_precision is LinearPrecision.FP8_CURRENT
                else loss_name
            ),
        )
    return BackendStatus(
        requested=requested,
        selected=selected,
        kda="torch.recurrent_fp32",
        short_conv="torch.depthwise_conv1d",
        attnres="torch.depth_softmax",
        routed_expert_backend=selected_experts,
        expert_mlp="torch.expert_loop",
        linear_precision=linear_precision,
        dense_ffn="torch.nn.Linear",
        loss_backend=selected_loss,
        loss="torch.cross_entropy",
    )
