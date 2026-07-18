from __future__ import annotations

import torch

from experiments.kda_sm90.validate_fla_guarded_integration import (
    ABSOLUTE_ERROR_LIMIT,
    RELATIVE_ERROR_LIMIT,
    TensorMetrics,
    tensor_metrics,
)


def test_tensor_metrics_accept_exact_values() -> None:
    reference = torch.tensor([1.0, -2.0, 3.0])
    metrics = tensor_metrics(reference.clone(), reference)

    assert metrics.relative_l2 == 0.0
    assert metrics.max_absolute_error == 0.0
    assert metrics.passed


def test_tensor_metrics_enforce_relative_error_limit() -> None:
    reference = torch.full((32,), 100.0)
    accepted = tensor_metrics(
        reference * (1.0 + RELATIVE_ERROR_LIMIT / 2.0),
        reference,
    )
    rejected = tensor_metrics(
        reference * (1.0 + RELATIVE_ERROR_LIMIT * 2.0),
        reference,
    )

    assert accepted.passed
    assert not rejected.passed


def test_tensor_metrics_accept_negligible_absolute_error_for_near_zero() -> None:
    reference = torch.full((6,), 1e-8)
    actual = reference + ABSOLUTE_ERROR_LIMIT / 10.0
    metrics = tensor_metrics(actual, reference)

    assert metrics.relative_l2 > RELATIVE_ERROR_LIMIT
    assert metrics.passed


def test_tensor_metrics_reject_nonfinite_values() -> None:
    metrics = TensorMetrics(
        relative_l2=0.0,
        max_absolute_error=0.0,
        reference_l2=1.0,
        actual_l2=1.0,
        reference_nonfinite=0,
        actual_nonfinite=1,
    )

    assert not metrics.passed
