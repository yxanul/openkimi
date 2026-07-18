from __future__ import annotations

import os
from importlib.util import find_spec

import pytest
import torch
import torch.nn.functional as F

from k3mini.backends import resolve_backend
from k3mini.config import (
    KernelBackend,
    LinearPrecision,
    LossBackend,
    ModelConfig,
    RoutedExpertBackend,
)
from k3mini.model import (
    BlockAttnResRead,
    K3MiniForCausalLM,
    SoftmaxTopKRouter,
    StackedRoutedExperts,
    kda_recurrent_reference,
)

H100_ENABLED = (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] >= 9
    and os.environ.get("K3MINI_RUN_GPU_TESTS") == "1"
)
QUACK_ENABLED = find_spec("quack") is not None
SONIC_ENABLED = find_spec("sonicmoe") is not None
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


def test_liger_fused_linear_cross_entropy_gradient_parity() -> None:
    from fla.modules import FusedLinearCrossEntropyLoss
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

    torch.manual_seed(456)
    hidden_fla = torch.randn(512, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    hidden_liger = hidden_fla.detach().clone().requires_grad_(True)
    weight_fla = torch.randn(257, 128, device="cuda", requires_grad=True)
    weight_liger = weight_fla.detach().clone().requires_grad_(True)
    labels = torch.randint(257, (512,), device="cuda")
    fla = FusedLinearCrossEntropyLoss(
        ignore_index=-100,
        num_chunks=8,
        accumulate_grad_in_fp32=True,
    )
    liger = LigerFusedLinearCrossEntropyLoss(
        ignore_index=-100,
        reduction="mean",
        accum_dtype=torch.float32,
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss_fla = fla(hidden_fla, labels, weight_fla)
        loss_liger = liger(weight_liger, hidden_liger, labels)
    loss_fla.backward()
    loss_liger.backward()
    torch.testing.assert_close(loss_liger, loss_fla, atol=5e-3, rtol=5e-3)
    assert _relative_error(hidden_liger.grad, hidden_fla.grad) < 5e-3
    assert _relative_error(weight_liger.grad, weight_fla.grad) < 5e-3


@pytest.mark.parametrize("chunk_size", [16, 32, 64])
def test_current_scaling_fp8_loss_padding_and_gradient_parity(chunk_size: int) -> None:
    from k3mini.fp8 import CurrentScalingFusedLinearCrossEntropyLoss

    torch.manual_seed(654)
    logical_vocab = 257
    physical_vocab = 272
    hidden_reference = torch.randn(
        64,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    hidden_fp8 = hidden_reference.detach().clone().requires_grad_(True)
    weight_reference = torch.randn(
        physical_vocab,
        128,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    weight_fp8 = weight_reference.detach().clone().requires_grad_(True)
    labels = torch.randint(logical_vocab, (64,), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        reference_logits = F.linear(hidden_reference, weight_reference)[:, :logical_vocab]
        reference_loss = F.cross_entropy(reference_logits, labels)
        fp8_loss = CurrentScalingFusedLinearCrossEntropyLoss(
            logical_vocab,
            chunk_size=chunk_size,
        )(
            weight_fp8,
            hidden_fp8,
            labels,
        )
    reference_loss.backward()
    fp8_loss.backward()
    torch.testing.assert_close(fp8_loss, reference_loss, atol=7e-2, rtol=2e-2)
    assert _relative_error(hidden_fp8.grad, hidden_reference.grad) < 0.2
    assert _relative_error(
        weight_fp8.grad[:logical_vocab],
        weight_reference.grad[:logical_vocab],
    ) < 0.2
    assert torch.count_nonzero(weight_fp8.grad[logical_vocab:]) == 0


@pytest.mark.skipif(not QUACK_ENABLED, reason="quack-kernels is not installed")
def test_quack_cross_entropy_logical_vocabulary_and_edge_cases() -> None:
    from k3mini.fp8 import _import_quack_cross_entropy_fwd_out

    torch.manual_seed(260718)
    logical_vocab = 257
    physical_vocab = 272
    row_count = 8
    logical_logits = torch.randn(
        row_count,
        logical_vocab,
        device="cuda",
        dtype=torch.bfloat16,
    )
    logical_logits[1].fill_(10)
    logical_logits[2].fill_(-10)
    logical_logits[3, 0] = 30
    logical_logits[4, -1] = 30
    labels = torch.tensor(
        [0, logical_vocab - 1, 7, 0, logical_vocab - 1, 7, 7, -100],
        device="cuda",
    )

    logits = torch.full(
        (row_count, physical_vocab),
        -torch.inf,
        device="cuda",
        dtype=torch.bfloat16,
    )
    logits[:, :logical_vocab].copy_(logical_logits)
    loss = torch.empty(row_count, device="cuda", dtype=torch.float32)
    _import_quack_cross_entropy_fwd_out()(
        logits,
        labels,
        None,
        loss,
        None,
        logits,
        None,
        -100,
    )

    reference_logits = logical_logits.float().requires_grad_(True)
    reference_loss = F.cross_entropy(
        reference_logits,
        labels,
        reduction="none",
        ignore_index=-100,
    )
    reference_loss.sum().backward()
    torch.testing.assert_close(loss, reference_loss, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(
        logits[:, :logical_vocab],
        reference_logits.grad.to(torch.bfloat16),
        atol=2e-3,
        rtol=2e-3,
    )
    assert torch.count_nonzero(logits[:, logical_vocab:]) == 0
    assert loss[-1] == 0
    assert torch.count_nonzero(logits[-1]) == 0


@pytest.mark.skipif(not QUACK_ENABLED, reason="quack-kernels is not installed")
def test_current_scaling_quack_mean_and_ignored_labels() -> None:
    from k3mini.fp8 import CurrentScalingFusedLinearCrossEntropyLoss

    torch.manual_seed(260719)
    logical_vocab = 257
    physical_vocab = 272
    token_count = 64
    hidden_dim = 128
    hidden_reference = torch.randn(
        token_count,
        hidden_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    hidden_quack = hidden_reference.detach().clone().requires_grad_(True)
    weight_reference = torch.randn(
        physical_vocab,
        hidden_dim,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    weight_quack = weight_reference.detach().clone().requires_grad_(True)
    labels = torch.randint(logical_vocab, (token_count,), device="cuda")
    labels[::5] = -100
    with torch.autocast("cuda", dtype=torch.bfloat16):
        reference_logits = F.linear(
            hidden_reference,
            weight_reference,
        )[:, :logical_vocab]
        reference_loss = F.cross_entropy(
            reference_logits,
            labels,
            ignore_index=-100,
        )
        quack_loss = CurrentScalingFusedLinearCrossEntropyLoss(
            logical_vocab,
            chunk_size=32,
            ce_backend="quack",
        )(weight_quack, hidden_quack, labels)
    reference_loss.backward()
    quack_loss.backward()
    torch.testing.assert_close(quack_loss, reference_loss, atol=7e-2, rtol=2e-2)
    assert _relative_error(hidden_quack.grad, hidden_reference.grad) < 0.2
    assert _relative_error(
        weight_quack.grad[:logical_vocab],
        weight_reference.grad[:logical_vocab],
    ) < 0.2
    assert torch.count_nonzero(weight_quack.grad[logical_vocab:]) == 0

    hidden_ignored = hidden_quack.detach().clone().requires_grad_(True)
    weight_ignored = weight_quack.detach().clone().requires_grad_(True)
    all_ignored = torch.full_like(labels, -100)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ignored_loss = CurrentScalingFusedLinearCrossEntropyLoss(
            logical_vocab,
            chunk_size=32,
            ce_backend="quack",
        )(weight_ignored, hidden_ignored, all_ignored)
    ignored_loss.backward()
    assert torch.isfinite(ignored_loss) and ignored_loss == 0
    assert torch.count_nonzero(hidden_ignored.grad) == 0
    assert torch.count_nonzero(weight_ignored.grad) == 0


def test_current_scaling_fp8_chunk_parity_at_model_shape() -> None:
    from k3mini.fp8 import CurrentScalingFusedLinearCrossEntropyLoss

    torch.manual_seed(20260718)
    logical_vocab = 128_001
    physical_vocab = 128_016
    token_count = 16_384
    hidden_dim = 768
    base_hidden = torch.randn(
        token_count,
        hidden_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    base_weight = torch.randn(
        physical_vocab,
        hidden_dim,
        device="cuda",
        dtype=torch.float32,
    ) * 0.02
    labels = torch.randint(logical_vocab, (token_count,), device="cuda")

    def run_fp8(
        chunk_size: int,
        ce_backend: str = "liger",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = base_hidden.detach().clone().requires_grad_(True)
        weight = base_weight.detach().clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = CurrentScalingFusedLinearCrossEntropyLoss(
                logical_vocab,
                chunk_size=chunk_size,
                ce_backend=ce_backend,
            )(weight, hidden, labels)
        loss.backward()
        return loss.detach(), hidden.grad.detach(), weight.grad.detach()

    loss_2k, hidden_grad_2k, weight_grad_2k = run_fp8(2_048)
    loss_16k, hidden_grad_16k, weight_grad_16k = run_fp8(16_384)

    hidden_reference = base_hidden.detach().clone().requires_grad_(True)
    weight_reference = base_weight.detach().clone().requires_grad_(True)
    loss_reference = torch.zeros((), device="cuda")
    for start in range(0, token_count, 2_048):
        end = start + 2_048
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = F.linear(
                hidden_reference[start:end],
                weight_reference,
            )[:, :logical_vocab]
            chunk_loss = (
                F.cross_entropy(logits, labels[start:end], reduction="sum") / token_count
            )
        chunk_loss.backward()
        loss_reference += chunk_loss.detach()

    torch.testing.assert_close(loss_2k, loss_reference, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(loss_16k, loss_reference, atol=2e-3, rtol=2e-3)
    assert _relative_error(hidden_grad_2k, hidden_reference.grad) < 0.04
    assert _relative_error(hidden_grad_16k, hidden_reference.grad) < 0.04
    assert _relative_error(
        weight_grad_2k[:logical_vocab],
        weight_reference.grad[:logical_vocab],
    ) < 0.04
    assert _relative_error(
        weight_grad_16k[:logical_vocab],
        weight_reference.grad[:logical_vocab],
    ) < 0.04
    assert torch.count_nonzero(weight_grad_2k[logical_vocab:]) == 0
    assert torch.count_nonzero(weight_grad_16k[logical_vocab:]) == 0
    if QUACK_ENABLED:
        for chunk_size in (2_048, 4_096, 8_192, 16_384):
            loss_quack, hidden_grad_quack, weight_grad_quack = run_fp8(
                chunk_size,
                ce_backend="quack",
            )
            torch.testing.assert_close(loss_quack, loss_reference, atol=2e-3, rtol=2e-3)
            assert _relative_error(hidden_grad_quack, hidden_reference.grad) < 0.04
            assert _relative_error(
                weight_grad_quack[:logical_vocab],
                weight_reference.grad[:logical_vocab],
            ) < 0.04
            assert torch.count_nonzero(weight_grad_quack[logical_vocab:]) == 0


def test_full_model_selects_current_scaling_without_amax_history() -> None:
    cfg = _kernel_config()
    cfg.kernel_backend = KernelBackend.H100
    cfg.loss_backend = LossBackend.LIGER
    cfg.linear_precision = LinearPrecision.FP8_CURRENT
    cfg.activation_checkpointing = True
    model = K3MiniForCausalLM(cfg).cuda()
    tokens = torch.randint(cfg.vocab_size, (1, 16), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(tokens, tokens, is_first_microbatch=True)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    assert model.token_embedding.weight.grad is not None
    assert model.token_embedding.num_embeddings == 272
    assert model.backend.linear_precision is LinearPrecision.FP8_CURRENT
    assert model.fp8_recipe.__class__.__name__ == "Float8CurrentScaling"
    assert not any("amax_history" in name for name, _ in model.named_buffers())
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(tokens, return_logits=True).logits
    assert logits is not None and logits.shape[-1] == cfg.vocab_size


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


@pytest.mark.skipif(not SONIC_ENABLED, reason="the isolated SonicMoE stack is not installed")
@pytest.mark.parametrize(
    ("latent_dim", "expert_ffn_dim", "top_k"),
    [(192, 512, 4), (256, 768, 2), (256, 768, 4)],
)
@pytest.mark.parametrize("routing_pattern", ["random", "skewed"])
def test_sonic_fixed_topk_routed_expert_parity(
    routing_pattern: str,
    latent_dim: int,
    expert_ffn_dim: int,
    top_k: int,
) -> None:
    torch.manual_seed(260720)
    cfg = ModelConfig(
        latent_dim=latent_dim,
        n_routed_experts=64,
        top_k=top_k,
        expert_ffn_dim=expert_ffn_dim,
        kernel_backend=KernelBackend.H100,
    )
    megablocks = StackedRoutedExperts(
        cfg,
        resolve_backend(
            KernelBackend.H100,
            routed_expert_backend=RoutedExpertBackend.MEGABLOCKS,
        ),
    ).cuda()
    sonic = StackedRoutedExperts(
        cfg,
        resolve_backend(
            KernelBackend.H100,
            routed_expert_backend=RoutedExpertBackend.SONIC,
        ),
    ).cuda()
    sonic.load_state_dict(megablocks.state_dict())

    token_count = 4_096
    base_latent = torch.randn(
        token_count,
        cfg.latent_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    if routing_pattern == "random":
        indices = torch.randn(
            token_count,
            cfg.n_routed_experts,
            device="cuda",
        ).topk(cfg.top_k, dim=-1).indices
    else:
        indices = torch.arange(cfg.top_k, device="cuda").expand(token_count, -1)
    base_weights = torch.rand(token_count, cfg.top_k, device="cuda")
    base_weights /= base_weights.sum(-1, keepdim=True)
    gradient = torch.randn_like(base_latent)

    def run(
        module: StackedRoutedExperts,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = base_latent.detach().clone().requires_grad_(True)
        route_weights = base_weights.detach().clone().requires_grad_(True)
        output = module(
            latent,
            indices,
            route_weights.to(torch.bfloat16),
        )
        output.backward(gradient)
        return (
            output.detach(),
            latent.grad.detach(),
            route_weights.grad.detach(),
            module.gate_up_weight.grad.detach(),
            module.down_weight.grad.detach(),
        )

    expected = run(megablocks)
    actual = run(sonic)
    assert all(torch.isfinite(value).all() for value in actual)
    for sonic_value, expected_value in zip(actual, expected, strict=True):
        assert _relative_error(sonic_value, expected_value) < 6e-3
    if routing_pattern == "skewed":
        assert torch.count_nonzero(actual[3][cfg.top_k :]) == 0
        assert torch.count_nonzero(actual[4][cfg.top_k :]) == 0
