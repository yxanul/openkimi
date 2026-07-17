from __future__ import annotations

import copy
import os

import pytest
import torch

from k3mini.backends import resolve_backend
from k3mini.config import KernelBackend, ModelConfig
from k3mini.model import BlockAttnResRead, KimiDeltaAttention, StackedRoutedExperts

H100_ENABLED = (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] >= 9
    and os.environ.get("K3MINI_RUN_GPU_TESTS") == "1"
)
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not H100_ENABLED,
        reason="set K3MINI_RUN_GPU_TESTS=1 on SM90+ with CUDA extras",
    ),
]


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        ((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-8)).item()
    )


def _kernel_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=257,
        d_model=128,
        n_layers=4,
        n_heads=1,
        kda_head_dim=128,
        mla_qk_head_dim=128,
        mla_v_head_dim=128,
        mla_kv_lora_rank=64,
        latent_dim=128,
        n_routed_experts=4,
        top_k=2,
        expert_ffn_dim=256,
        shared_ffn_dim=256,
        dense_ffn_dim=256,
        activation_checkpointing=False,
    )


def test_fla_kda_forward_and_gradient_parity() -> None:
    reference_cfg = _kernel_config()
    reference_cfg.kernel_backend = KernelBackend.REFERENCE
    fused_cfg = copy.deepcopy(reference_cfg)
    fused_cfg.kernel_backend = KernelBackend.H100
    reference = KimiDeltaAttention(reference_cfg, resolve_backend(KernelBackend.REFERENCE)).cuda()
    fused = KimiDeltaAttention(fused_cfg, resolve_backend(KernelBackend.H100)).cuda()
    fused.load_state_dict(reference.state_dict())
    x_reference = torch.randn(1, 64, 128, device="cuda", requires_grad=True)
    x_fused = x_reference.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_reference = reference(x_reference)
        output_fused = fused(x_fused)
        loss_reference = output_reference.float().square().mean()
        loss_fused = output_fused.float().square().mean()
    loss_reference.backward()
    loss_fused.backward()
    assert _relative_error(output_fused, output_reference) < 5e-3
    assert _relative_error(x_fused.grad, x_reference.grad) < 5e-3
    assert _relative_error(fused.q_proj.weight.grad, reference.q_proj.weight.grad) < 5e-3


def test_fused_attnres_forward_and_gradient_parity() -> None:
    cfg = _kernel_config()
    reference = BlockAttnResRead(
        cfg.d_model, cfg.rms_norm_eps, resolve_backend(KernelBackend.REFERENCE)
    ).cuda()
    fused = BlockAttnResRead(cfg.d_model, cfg.rms_norm_eps, resolve_backend(KernelBackend.H100)).cuda()
    fused.load_state_dict(reference.state_dict())
    reference_sources = [
        torch.randn(2, 64, cfg.d_model, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(9)
    ]
    fused_sources = [source.detach().clone().requires_grad_(True) for source in reference_sources]
    reference_norm = torch.ones(cfg.d_model, device="cuda", requires_grad=True)
    fused_norm = reference_norm.detach().clone().requires_grad_(True)
    output_reference, _ = reference(
        reference_sources, output_norm_weight=reference_norm, return_weights=False
    )
    output_fused, _ = fused(fused_sources, output_norm_weight=fused_norm, return_weights=False)
    gradient = torch.randn_like(output_reference)
    output_reference.backward(gradient)
    output_fused.backward(gradient)
    assert _relative_error(output_fused, output_reference) < 5e-3
    assert _relative_error(fused_sources[0].grad, reference_sources[0].grad) < 5e-3
    assert _relative_error(fused.pseudo_query.grad, reference.pseudo_query.grad) < 5e-3


def test_grouped_gemm_empty_and_heavy_expert_parity() -> None:
    cfg = _kernel_config()
    reference = StackedRoutedExperts(cfg, resolve_backend(KernelBackend.REFERENCE)).cuda()
    grouped = StackedRoutedExperts(cfg, resolve_backend(KernelBackend.H100)).cuda()
    grouped.load_state_dict(reference.state_dict())
    latent_reference = torch.randn(
        64, cfg.latent_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    latent_grouped = latent_reference.detach().clone().requires_grad_(True)
    indices = torch.zeros(64, cfg.top_k, device="cuda", dtype=torch.long)
    indices[:, 1] = torch.arange(64, device="cuda") % 2 + 1
    weights = torch.tensor([0.8, 0.2], device="cuda", dtype=torch.bfloat16).expand(64, -1)
    output_reference = reference(latent_reference, indices, weights)
    output_grouped = grouped(latent_grouped, indices, weights)
    gradient = torch.randn_like(output_reference)
    output_reference.backward(gradient)
    output_grouped.backward(gradient)
    assert _relative_error(output_grouped, output_reference) < 5e-3
    assert _relative_error(latent_grouped.grad, latent_reference.grad) < 5e-3
    assert _relative_error(grouped.gate_weight.grad, reference.gate_weight.grad) < 5e-3
