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
from k3mini.optim import partition_muon_parameters
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


def test_qk_clip_scales_only_heads_above_threshold(tiny_model_config) -> None:
    module = NoPELatentAttention(tiny_model_config)
    module.configure_qk_clip(enabled=True, query_chunk_size=4)
    with torch.no_grad():
        module.observed_max_attention_logits.copy_(
            torch.tensor([200.0, 100.0, 50.0, 25.0])
        )
        query_before = module.q_proj.weight.clone()
        kv_before = module.kv_b_proj.weight.clone()
    diagnostics = module.apply_qk_clip(100.0)
    query_after = module.q_proj.weight.view(4, 8, -1)
    query_before = query_before.view(4, 8, -1)
    torch.testing.assert_close(query_after[0], query_before[0] * 2**-0.5)
    torch.testing.assert_close(query_after[1:], query_before[1:])
    kv_after = module.kv_b_proj.weight.view(4, 16, -1)
    kv_before = kv_before.view(4, 16, -1)
    torch.testing.assert_close(kv_after[0, :8], kv_before[0, :8] * 2**-0.5)
    torch.testing.assert_close(kv_after[0, 8:], kv_before[0, 8:])
    torch.testing.assert_close(kv_after[1:], kv_before[1:])
    assert diagnostics == {
        "maximum_attention_logit": 200.0,
        "clipped_heads": 1,
        "minimum_scale": pytest.approx(2**-0.5),
    }


def test_qk_clip_observes_forward_logits(tiny_model_config) -> None:
    module = NoPELatentAttention(tiny_model_config).train()
    module.configure_qk_clip(enabled=True, query_chunk_size=3)
    module(torch.randn(2, 7, tiny_model_config.d_model))
    assert torch.isfinite(module.observed_max_attention_logits).all()


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


def test_mtp_is_training_only_and_backpropagates_to_backbone(tiny_model_config) -> None:
    cfg = copy.deepcopy(tiny_model_config)
    cfg.mtp_depth = 2
    cfg.mtp_loss_weight = 0.1
    model = K3MiniForCausalLM(cfg).train()
    tokens = torch.randint(0, cfg.vocab_size, (1, 7))
    output = model(tokens, tokens)
    assert output.mtp_loss is not None and torch.isfinite(output.mtp_loss)
    assert output.loss is not None
    output.loss.backward()
    assert model.layers[0].mixer.q_proj.weight.grad is not None
    assert model.mtp_stages[0].input_projection.weight.grad is not None
    assert model.mtp_stages[1].input_projection.weight.grad is not None

    model.eval()
    with torch.no_grad():
        evaluation_output = model(tokens, tokens)
    assert evaluation_output.mtp_loss is None
    assert evaluation_output.lm_loss is not None


def test_sigmoid_checkpoint_replay_does_not_double_router_load(
    tiny_model_config,
) -> None:
    cfg = copy.deepcopy(tiny_model_config)
    cfg.router_type = "sigmoid_noaux"
    cfg.activation_checkpointing = True
    cfg.checkpoint_attention = True
    cfg.checkpoint_ffn = True
    model = K3MiniForCausalLM(cfg).train()
    tokens = torch.randint(0, cfg.vocab_size, (1, 6))
    output = model(tokens, tokens)
    loads_after_forward = [
        module.pending_load.clone()
        for module in model.modules()
        if module.__class__.__name__ == "SigmoidNoAuxTopKRouter"
    ]
    assert output.loss is not None
    output.loss.backward()
    loads_after_backward = [
        module.pending_load
        for module in model.modules()
        if module.__class__.__name__ == "SigmoidNoAuxTopKRouter"
    ]
    for forward_load, backward_load in zip(
        loads_after_forward,
        loads_after_backward,
        strict=True,
    ):
        torch.testing.assert_close(forward_load, backward_load)
        assert int(forward_load.sum().item()) == tokens.numel() * cfg.top_k


def test_muon_partition_excludes_tied_embedding(tiny_model_config) -> None:
    model = K3MiniForCausalLM(tiny_model_config)
    muon, adam_decay, adam_no_decay = partition_muon_parameters(model)
    assert id(model.token_embedding.weight) not in {id(parameter) for parameter in muon}
    assert id(model.token_embedding.weight) in {
        id(parameter) for parameter in adam_decay
    }
    router_weight_ids = {
        id(module.router.weight)
        for module in model.modules()
        if isinstance(module, LatentMoE)
    }
    assert not router_weight_ids.intersection(id(parameter) for parameter in muon)
    assert router_weight_ids.issubset(id(parameter) for parameter in adam_decay)
    assert all(parameter.ndim >= 2 for parameter in muon)
    assert all(parameter.ndim < 2 for parameter in adam_no_decay)
