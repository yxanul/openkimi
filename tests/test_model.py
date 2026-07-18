from __future__ import annotations

import copy

import pytest
import torch

from k3mini.backends import resolve_backend
from k3mini.config import (
    KernelBackend,
    LinearPrecision,
    LossBackend,
    ModelConfig,
    RoutedExpertBackend,
    load_config,
)
from k3mini.fp8 import resolve_fp8_lm_head_chunk_size
from k3mini.model import (
    BlockAttnResRead,
    K3MiniForCausalLM,
    KimiDeltaAttention,
    LatentMoE,
    NoPELatentAttention,
    estimate_parameter_counts,
    kda_recurrent_reference,
)
from k3mini.training import build_optimizer


def test_kda_recurrence_forward_backward_and_causality(tiny_model_config) -> None:
    backend = resolve_backend(KernelBackend.REFERENCE)
    module = KimiDeltaAttention(tiny_model_config, backend)
    x = torch.randn(2, 7, tiny_model_config.d_model, requires_grad=True)
    changed = x.detach().clone()
    changed[:, 5:] += 10
    output = module(x)
    changed_output = module(changed)
    torch.testing.assert_close(output[:, :5], changed_output[:, :5])
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    q = torch.nn.functional.normalize(torch.randn(1, 4, 2, 3), dim=-1).requires_grad_()
    k = torch.nn.functional.normalize(torch.randn(1, 4, 2, 3), dim=-1)
    v = torch.randn_like(q)
    decay = -torch.rand_like(q)
    beta = torch.rand(1, 4, 2)
    recurrent = kda_recurrent_reference(q, k, v, decay, beta)
    recurrent.sum().backward()
    assert recurrent.shape == q.shape and q.grad is not None


def test_nope_mla_is_causal(tiny_model_config) -> None:
    module = NoPELatentAttention(tiny_model_config).eval()
    x = torch.randn(1, 8, tiny_model_config.d_model)
    changed = x.clone()
    changed[:, 6:] = torch.randn_like(changed[:, 6:]) * 20
    with torch.no_grad():
        original = module(x)
        future_changed = module(changed)
    torch.testing.assert_close(original[:, :6], future_changed[:, :6], atol=1e-5, rtol=1e-5)


def test_attnres_zero_query_is_average_and_gradients(tiny_model_config) -> None:
    backend = resolve_backend(KernelBackend.REFERENCE)
    read = BlockAttnResRead(tiny_model_config.d_model, 1e-6, backend)
    sources = [torch.randn(2, 3, tiny_model_config.d_model, requires_grad=True) for _ in range(3)]
    output_weight = torch.ones(tiny_model_config.d_model, requires_grad=True)
    output, weights = read(sources, output_norm_weight=output_weight, return_weights=True)
    expected_pre_norm = torch.stack(sources).mean(0)
    expected = expected_pre_norm * torch.rsqrt(
        expected_pre_norm.float().square().mean(-1, keepdim=True) + 1e-6
    )
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(weights, torch.full((3,), 1 / 3))
    output.sum().backward()
    assert all(source.grad is not None for source in sources)


def test_architecture_schedule_topology_and_first_dense(tiny_model_config) -> None:
    model = K3MiniForCausalLM(tiny_model_config)
    assert model.lm_head.weight is model.token_embedding.weight
    assert [layer.mixer_kind for layer in model.layers] == [
        "kda",
        "kda",
        "kda",
        "nope_mla",
    ]
    assert not isinstance(model.layers[0].ffn, LatentMoE)
    assert all(isinstance(layer.ffn, LatentMoE) for layer in model.layers[1:])
    tokens = torch.randint(0, tiny_model_config.vocab_size, (1, 6))
    output = model(tokens, tokens, return_logits=True, return_diagnostics=True)
    assert output.logits is not None
    assert output.diagnostics["attnres_sources"] == 5
    assert output.loss is not None and torch.isfinite(output.loss)


def test_primary_parameter_count_and_tied_embeddings() -> None:
    cfg, _, _ = load_config("configs/primary.json")
    counts = estimate_parameter_counts(cfg)
    assert 440_000_000 <= counts["total"] <= 450_000_000
    assert 175_000_000 <= counts["active_estimate"] <= 185_000_000
    assert cfg.tie_embeddings


def test_fp8_current_configuration_pads_only_the_physical_vocabulary() -> None:
    cfg = ModelConfig(linear_precision=LinearPrecision.FP8_CURRENT)
    cfg.validate()
    assert cfg.vocab_size == 128_001
    assert cfg.physical_vocab_size == 128_016

    reference_cfg = ModelConfig(
        kernel_backend=KernelBackend.REFERENCE,
        linear_precision=LinearPrecision.FP8_CURRENT,
    )
    with pytest.raises(ValueError, match="requires the H100"):
        reference_cfg.validate()

    assert resolve_fp8_lm_head_chunk_size(131_072, 128_016, 768, None) == 1_024
    assert resolve_fp8_lm_head_chunk_size(131_072, 128_016, 768, 8_192) == 8_192

    cfg.fp8_lm_head_chunk_size = 2_047
    with pytest.raises(ValueError, match="positive multiple of 16"):
        cfg.validate()

    quack_without_fp8 = ModelConfig(loss_backend=LossBackend.QUACK)
    with pytest.raises(ValueError, match="requires linear_precision=fp8_current"):
        quack_without_fp8.validate()

    quack_cfg, data_cfg, train_cfg = load_config("configs/h100-fp8-current-quack.json")
    assert quack_cfg.loss_backend is LossBackend.QUACK
    assert quack_cfg.fp8_lm_head_chunk_size == 16_384
    train_cfg.validate(data_cfg)

    sonic_reference = ModelConfig(
        kernel_backend=KernelBackend.REFERENCE,
        routed_expert_backend=RoutedExpertBackend.SONIC,
    )
    with pytest.raises(ValueError, match="requires the H100"):
        sonic_reference.validate()

    sonic_cfg, data_cfg, train_cfg = load_config(
        "configs/h100-fp8-current-quack-sonic.json"
    )
    assert sonic_cfg.routed_expert_backend is RoutedExpertBackend.SONIC
    assert sonic_cfg.loss_backend is LossBackend.QUACK
    train_cfg.validate(data_cfg)

    kda_saved_cfg, data_cfg, train_cfg = load_config(
        "configs/h100-fp8-current-quack-sonic-kda-saved.json"
    )
    assert kda_saved_cfg.kda_disable_recompute
    train_cfg.validate(data_cfg)


def test_checkpoint_policy_defaults_and_overrides() -> None:
    cfg = ModelConfig(activation_checkpointing=True)
    assert not cfg.kda_disable_recompute
    assert cfg.checkpoint_attention_enabled
    assert cfg.checkpoint_ffn_enabled

    cfg.checkpoint_attention = False
    assert not cfg.checkpoint_attention_enabled
    assert cfg.checkpoint_ffn_enabled

    cfg.checkpoint_ffn = False
    assert not cfg.checkpoint_ffn_enabled

    cfg.attnres_checkpoint_level = 2
    with pytest.raises(ValueError, match="must be 0 or 1"):
        cfg.validate()


def test_full_optimizer_step_and_no_weight_decay_groups(tiny_model_config, tiny_train_config) -> None:
    model = K3MiniForCausalLM(tiny_model_config)
    optimizer = build_optimizer(model, tiny_train_config)
    assert {group["weight_decay"] for group in optimizer.param_groups} == {0.0, 0.1}
    before = copy.deepcopy(model.token_embedding.weight.detach())
    tokens = torch.randint(0, tiny_model_config.vocab_size, (2, 6))
    output = model(tokens, tokens)
    assert output.loss is not None
    output.loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.token_embedding.weight)
