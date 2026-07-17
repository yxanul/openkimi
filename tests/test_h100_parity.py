from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

from k3mini.backends import resolve_backend
from k3mini.config import KernelBackend, ModelConfig
from k3mini.model import (
    BlockAttnResRead,
    SoftmaxTopKRouter,
    StackedRoutedExperts,
    kda_recurrent_reference,
)

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
    from fla.ops.kda import chunk_kda

    torch.manual_seed(123)
    batch, time, heads, dim = 1, 64, 1, 128
    q = torch.randn(batch, time, heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    raw_gate = torch.randn_like(q, requires_grad=True)
    beta_logits = torch.randn(batch, time, heads, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    a_log = torch.log(torch.empty(heads, device="cuda").uniform_(1, 16)).requires_grad_()
    dt_bias = torch.zeros(heads, dim, device="cuda", requires_grad=True)
    reference_inputs = [q, k, v, raw_gate, beta_logits, a_log, dt_bias]
    fused_inputs = [value.detach().clone().requires_grad_(True) for value in reference_inputs]

    q_reference = F.normalize(q.float(), dim=-1, eps=1e-6).to(q.dtype)
    k_reference = F.normalize(k.float(), dim=-1, eps=1e-6).to(k.dtype)
    log_alpha = -a_log.exp().view(1, 1, heads, 1) * F.softplus(
        raw_gate.float() + dt_bias.view(1, 1, heads, dim)
    )
    output_reference = kda_recurrent_reference(
        q_reference,
        k_reference,
        v,
        log_alpha,
        beta_logits.float().sigmoid(),
        scale=dim**-0.5,
    )
    q_fused, k_fused, v_fused, gate_fused, beta_fused, a_fused, dt_fused = fused_inputs
    output_fused, _ = chunk_kda(
        q_fused,
        k_fused,
        v_fused,
        gate_fused,
        beta_fused,
        A_log=a_fused,
        dt_bias=dt_fused.reshape(-1),
        scale=dim**-0.5,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=False,
        state_v_first=True,
    )
    output_reference.float().square().mean().backward()
    output_fused.float().square().mean().backward()
    assert _relative_error(output_fused, output_reference) < 5e-3
    gradient_errors = [
        _relative_error(fused_value.grad, reference_value.grad)
        for fused_value, reference_value in zip(fused_inputs, reference_inputs, strict=True)
    ]
    assert max(gradient_errors) < 7e-3


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


def test_fused_weighted_swiglu_forward_and_gradient_parity() -> None:
    from k3mini.cuda_kernels import fused_weighted_swiglu

    torch.manual_seed(321)
    gate_up_reference = torch.randn(
        137, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    route_reference = torch.rand(137, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    gate_up_fused = gate_up_reference.detach().clone().requires_grad_(True)
    route_fused = route_reference.detach().clone().requires_grad_(True)
    gate, up = gate_up_reference.chunk(2, dim=-1)
    output_reference = F.silu(gate) * up * route_reference.unsqueeze(-1)
    output_fused = fused_weighted_swiglu(gate_up_fused, route_fused)
    gradient = torch.randn_like(output_reference)
    output_reference.backward(gradient)
    output_fused.backward(gradient)
    assert _relative_error(output_fused, output_reference) < 5e-3
    assert _relative_error(gate_up_fused.grad, gate_up_reference.grad) < 5e-3
    assert _relative_error(route_fused.grad, route_reference.grad) < 5e-3


def test_router_device_histogram_parity() -> None:
    cfg = _kernel_config()
    router = SoftmaxTopKRouter(cfg, resolve_backend(KernelBackend.H100)).cuda()
    hidden = torch.randn(511, cfg.d_model, device="cuda", dtype=torch.bfloat16)
    routing = router(hidden, collect_diagnostics=False)
    expected = torch.bincount(routing.indices.flatten(), minlength=cfg.n_routed_experts).float()
    torch.testing.assert_close(routing.load, expected)
    assert routing.entropy.item() == 0.0
    assert routing.max_load_violation.item() == 0.0


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
    assert _relative_error(grouped.gate_up_weight.grad, reference.gate_up_weight.grad) < 5e-3
