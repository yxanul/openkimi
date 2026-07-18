from __future__ import annotations

import pytest
import torch

from experiments.kda_sm90.guarded_diagonal_oracle import (
    exact_diagonal_backward,
    guarded_diagonal_backward,
)


def _inputs(spans: list[float]):
    torch.manual_seed(19)
    blocks = len(spans)
    block_size = 16
    channels = 32
    increments = torch.rand(blocks, block_size, channels)
    increments[:, 0] = 0.0
    for block, span in enumerate(spans):
        increments[block] *= span / increments[block].sum(dim=0).amax().clamp_min(1e-6)
    gate = -increments.cumsum(dim=1)
    q = torch.randn_like(gate)
    k = torch.randn_like(gate)
    beta = torch.sigmoid(torch.randn(blocks, block_size))
    da_qk = torch.randn(blocks, block_size, block_size)
    da_kk = torch.randn_like(da_qk)
    return gate, q, k, beta, da_qk, da_kk


def test_guarded_factorization_matches_pairwise_oracle() -> None:
    inputs = _inputs([2.0, 8.0, 16.0])
    exact = exact_diagonal_backward(*inputs)
    guarded = guarded_diagonal_backward(*inputs, max_log2_span=20.0)

    assert guarded.guard_hits.tolist() == [True, True, True]
    torch.testing.assert_close(guarded.dq, exact.dq, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(guarded.dk, exact.dk, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(guarded.dkt, exact.dkt, atol=2e-4, rtol=2e-4)


def test_guard_rejects_large_or_nonmonotonic_blocks() -> None:
    inputs = list(_inputs([4.0, 200.0, 4.0]))
    inputs[0][2, 5, 0] = inputs[0][2, 4, 0] + 1.0
    exact = exact_diagonal_backward(*inputs)
    guarded = guarded_diagonal_backward(*inputs, max_log2_span=20.0)

    assert guarded.guard_hits.tolist() == [True, False, False]
    torch.testing.assert_close(guarded.dq[1:], exact.dq[1:], atol=0.0, rtol=0.0)
    torch.testing.assert_close(guarded.dk[1:], exact.dk[1:], atol=0.0, rtol=0.0)
    torch.testing.assert_close(guarded.dkt[1:], exact.dkt[1:], atol=0.0, rtol=0.0)


def test_midpoint_reference_supports_large_spans() -> None:
    inputs = _inputs([100.0, 200.0, 251.0])
    exact = exact_diagonal_backward(*inputs)
    guarded = guarded_diagonal_backward(
        *inputs,
        max_log2_span=240.0,
        reference_policy="midpoint",
    )

    assert guarded.guard_hits.tolist() == [True, True, False]
    torch.testing.assert_close(guarded.dq, exact.dq, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(guarded.dk, exact.dk, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(guarded.dkt, exact.dkt, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("threshold", [-1.0, 0.0, 127.0, float("inf")])
def test_guard_threshold_validation(threshold: float) -> None:
    with pytest.raises(ValueError):
        guarded_diagonal_backward(*_inputs([1.0]), max_log2_span=threshold)
