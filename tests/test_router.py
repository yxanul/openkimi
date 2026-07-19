from __future__ import annotations

import torch

from k3mini.config import RouterBiasUpdateRule, RouterType
from k3mini.model import SigmoidNoAuxTopKRouter, SoftmaxTopKRouter


def test_softmax_router_weights_losses_and_diagnostics(tiny_model_config) -> None:
    router = SoftmaxTopKRouter(tiny_model_config)
    output = router(torch.randn(19, tiny_model_config.d_model))
    torch.testing.assert_close(output.weights.sum(-1), torch.ones(19))
    assert output.indices.shape == (19, tiny_model_config.top_k)
    assert output.auxiliary_loss.item() > 0
    assert output.z_loss.item() >= 0
    assert int(output.load.sum()) == 19 * tiny_model_config.top_k
    assert torch.isfinite(output.entropy)


def test_sigmoid_noaux_bias_update(tiny_model_config) -> None:
    tiny_model_config.router_type = RouterType.SIGMOID_NOAUX
    router = SigmoidNoAuxTopKRouter(tiny_model_config)
    with torch.no_grad():
        router.weight.zero_()
        router.weight[0].fill_(2.0)
        router.weight[1].fill_(1.0)
    router(torch.ones(32, tiny_model_config.d_model))
    before = router.e_score_correction_bias.clone()
    router.update_bias()
    after = router.e_score_correction_bias
    assert not torch.equal(before, after)
    assert after[0] < before[0]


def test_sigmoid_noaux_proportional_bias_update_is_centered(tiny_model_config) -> None:
    tiny_model_config.router_type = RouterType.SIGMOID_NOAUX
    tiny_model_config.router_bias_update_rule = RouterBiasUpdateRule.PROPORTIONAL
    tiny_model_config.router_bias_update_rate = 0.01
    router = SigmoidNoAuxTopKRouter(tiny_model_config)
    with torch.no_grad():
        router.pending_load.copy_(torch.tensor([80.0, 40.0, 20.0, 20.0]))

    router.update_bias()

    torch.testing.assert_close(router.e_score_correction_bias.mean(), torch.zeros(()))
    assert router.e_score_correction_bias[0] < 0
    assert router.e_score_correction_bias[2] > 0
    assert router.e_score_correction_bias.abs().max() <= tiny_model_config.router_bias_update_rate
