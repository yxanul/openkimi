from __future__ import annotations

from typing import Any

import torch


def _import_sonic_fixed_topk_ops() -> tuple[Any, Any, Any, Any]:
    """Import the pinned SonicMoE orchestration primitives."""
    if not hasattr(torch, "float4_e2m1fn_x2"):
        torch.float4_e2m1fn_x2 = object()  # type: ignore[attr-defined]

    from sonicmoe.enums import ActivationType
    from sonicmoe.functional import _DownProjection, _UpProjection
    from sonicmoe.functional.triton_kernels import TC_topk_router_metadata_triton

    return _UpProjection, _DownProjection, TC_topk_router_metadata_triton, ActivationType


def sonic_fixed_topk_experts(
    latent: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """Run SonicMoE with OpenKimi's externally computed fixed top-k routing."""
    if latent.ndim != 2 or indices.ndim != 2 or weights.shape != indices.shape:
        raise ValueError("SonicMoE expects latent [T,H] and matching indices/weights [T,K]")
    if latent.dtype not in {torch.bfloat16, torch.float16}:
        raise ValueError("SonicMoE routed experts require BF16 or FP16 activations")

    token_count, top_k = indices.shape
    n_experts, gate_up_dim, hidden_dim = gate_up_weight.shape
    if latent.shape != (token_count, hidden_dim):
        raise ValueError("latent and gate/up weight dimensions do not match")
    if down_weight.shape[0] != n_experts or down_weight.shape[2] != hidden_dim:
        raise ValueError("down-projection weight dimensions do not match")
    if gate_up_dim != 2 * down_weight.shape[1]:
        raise ValueError("gate/up and down-projection intermediate dimensions do not match")

    up_projection, down_projection, metadata_kernel, activation_type = (
        _import_sonic_fixed_topk_ops()
    )
    flat_routes = token_count * top_k
    device = latent.device
    sonic_indices = indices.to(torch.int32).contiguous()
    # Sonic/QuACK's weighted-down epilogue consumes FP32 routing scores. Keeping
    # them in FP32 also avoids the SM90 async-copy alignment path for BF16
    # score vectors during backward.
    sonic_weights = weights.float().contiguous()
    scatter_indices = torch.empty(flat_routes, dtype=torch.int32, device=device)
    reverse_scatter_indices = torch.empty_like(scatter_indices)
    expert_frequency = torch.empty(n_experts, dtype=torch.int32, device=device)
    expert_offsets = torch.empty(n_experts + 1, dtype=torch.int32, device=device)
    gather_indices = torch.empty_like(scatter_indices)

    metadata_kernel(
        sonic_indices,
        n_experts,
        expert_frequency,
        expert_offsets,
        gather_indices,
        scatter_indices,
        reverse_scatter_indices,
    )

    # The permutations are zero-copy views after converting FP32 master weights
    # to the activation dtype expected by Sonic's BF16/FP16 grouped GEMMs.
    sonic_gate_up = gate_up_weight.to(latent.dtype).permute(1, 2, 0)
    sonic_down = down_weight.to(latent.dtype).permute(2, 1, 0)
    activated, preactivation = up_projection.apply(
        latent,
        sonic_gate_up,
        None,
        expert_offsets,
        flat_routes,
        top_k,
        gather_indices,
        scatter_indices,
        reverse_scatter_indices,
        None,
        False,
        activation_type.SWIGLU,
        False,
        True,
    )
    return down_projection.apply(
        activated,
        preactivation,
        sonic_down,
        None,
        sonic_weights,
        expert_offsets,
        token_count,
        top_k,
        gather_indices,
        scatter_indices,
        reverse_scatter_indices,
        None,
        False,
        activation_type.SWIGLU,
    )
