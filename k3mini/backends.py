from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec

import torch

from .config import KernelBackend


@dataclass(frozen=True, slots=True)
class BackendStatus:
    requested: KernelBackend
    selected: KernelBackend
    kda: str
    short_conv: str
    attnres: str
    expert_mlp: str
    loss: str

    def as_dict(self) -> dict[str, str]:
        return {
            "requested": self.requested.value,
            "selected": self.selected.value,
            "kda": self.kda,
            "short_conv": self.short_conv,
            "attnres": self.attnres,
            "expert_mlp": self.expert_mlp,
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


def resolve_backend(requested: KernelBackend) -> BackendStatus:
    requested = KernelBackend(requested)
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

    if selected is KernelBackend.H100:
        return BackendStatus(
            requested=requested,
            selected=selected,
            kda="fla.ops.kda.chunk_kda",
            short_conv="fla.modules.ShortConvolution",
            attnres="fla.ops.attnres.fused_attnres(checkpoint_level=1)",
            expert_mlp="megablocks.grouped_gemm",
            loss="fla.modules.FusedLinearCrossEntropyLoss",
        )
    return BackendStatus(
        requested=requested,
        selected=selected,
        kda="torch.recurrent_fp32",
        short_conv="torch.depthwise_conv1d",
        attnres="torch.depth_softmax",
        expert_mlp="torch.expert_loop",
        loss="torch.cross_entropy",
    )
