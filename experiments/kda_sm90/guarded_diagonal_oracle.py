"""Numerical oracle for a guarded tensor-core KDA diagonal backward path.

FLA's faithful ``safe_gate=False`` branch evaluates every pairwise decay as
``exp2(g_i - g_j)``. That is stable because KDA's cumulative gate is
non-increasing, but it prevents the diagonal 16x16 blocks from using matrix
multiplication.

For a block reference ``r = g_0`` the same decay factors as

    exp2(g_i - g_j) = exp2(g_i - r) * exp2(r - g_j).

The first factor is at most one. The second factor is safe only while the
within-block log2 span is bounded, so the optimized implementation must retain
an exact pairwise fallback. This module is the device-independent contract for
that runtime choice; it is not a training backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GuardedDiagonalResult:
    dq: torch.Tensor
    dk: torch.Tensor
    dkt: torch.Tensor
    guard_hits: torch.Tensor


def _validate(
    gate: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    beta: torch.Tensor,
    da_qk: torch.Tensor,
    da_kk: torch.Tensor,
) -> None:
    if gate.ndim != 3:
        raise ValueError("gate must have shape [blocks, block_size, channels]")
    blocks, block_size, channels = gate.shape
    expected_token_shape = (blocks, block_size, channels)
    expected_beta_shape = (blocks, block_size)
    expected_matrix_shape = (blocks, block_size, block_size)
    if q.shape != expected_token_shape or k.shape != expected_token_shape:
        raise ValueError("q and k must have the same shape as gate")
    if beta.shape != expected_beta_shape:
        raise ValueError(f"beta must have shape {expected_beta_shape}")
    if da_qk.shape != expected_matrix_shape or da_kk.shape != expected_matrix_shape:
        raise ValueError(f"dA tensors must have shape {expected_matrix_shape}")
    if channels == 0 or block_size == 0:
        raise ValueError("block size and channel count must be positive")


def exact_diagonal_backward(
    gate: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    beta: torch.Tensor,
    da_qk: torch.Tensor,
    da_kk: torch.Tensor,
) -> GuardedDiagonalResult:
    """Evaluate the two FLA diagonal phases with stable pairwise exponentials."""

    _validate(gate, q, k, beta, da_qk, da_kk)
    block_size = gate.shape[1]
    rows = torch.arange(block_size, device=gate.device)
    lower = rows[:, None] >= rows[None, :]
    upper = rows[:, None] <= rows[None, :]
    pairwise_difference = gate[:, :, None, :] - gate[:, None, :, :]
    lower_decay = torch.exp2(
        torch.where(lower[None, :, :, None], pairwise_difference, 0.0)
    )
    upper_decay = torch.exp2(
        torch.where(upper[None, :, :, None], -pairwise_difference, 0.0)
    )

    da_qk_lower = da_qk.masked_fill(~lower, 0.0)
    da_kk_lower = da_kk.masked_fill(~lower, 0.0)
    dq = torch.einsum("bij,bjc,bijc->bic", da_qk_lower, k, lower_decay)
    dk = torch.einsum("bij,bjc,bijc->bic", da_kk_lower, k, lower_decay)

    da_qk_upper = da_qk.masked_fill(~upper, 0.0)
    da_kk_upper = da_kk.masked_fill(~upper, 0.0)
    dkt = torch.einsum("bij,bjc,bijc->bic", da_qk_upper, q, upper_decay)
    dkt += torch.einsum(
        "bij,bjc,bijc->bic",
        da_kk_upper,
        k * beta[..., None],
        upper_decay,
    )
    return GuardedDiagonalResult(
        dq=dq,
        dk=dk,
        dkt=dkt,
        guard_hits=torch.zeros(
            gate.shape[0],
            dtype=torch.bool,
            device=gate.device,
        ),
    )


def guarded_diagonal_backward(
    gate: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    beta: torch.Tensor,
    da_qk: torch.Tensor,
    da_kk: torch.Tensor,
    *,
    max_log2_span: float,
    reference_policy: str = "first",
) -> GuardedDiagonalResult:
    """Use factored matrix products only for blocks passing a runtime span guard."""

    if reference_policy not in {"first", "midpoint"}:
        raise ValueError("reference_policy must be 'first' or 'midpoint'")
    maximum_span = 252.0 if reference_policy == "midpoint" else 127.0
    if not 0.0 < max_log2_span < maximum_span:
        raise ValueError(
            f"max_log2_span must be finite and between 0 and {maximum_span:g}"
        )
    _validate(gate, q, k, beta, da_qk, da_kk)
    exact = exact_diagonal_backward(gate, q, k, beta, da_qk, da_kk)
    maximum_gate = gate.amax(dim=1, keepdim=True)
    minimum_gate = gate.amin(dim=1, keepdim=True)
    spans = (maximum_gate - minimum_gate).amax(dim=(1, 2))
    reference = (
        (maximum_gate + minimum_gate) * 0.5
        if reference_policy == "midpoint"
        else gate[:, :1, :]
    )
    monotonic = torch.all(gate[:, 1:, :] <= gate[:, :-1, :], dim=(1, 2))
    guard_hits = monotonic & torch.isfinite(spans) & (spans <= max_log2_span)

    block_size = gate.shape[1]
    rows = torch.arange(block_size, device=gate.device)
    lower = rows[:, None] >= rows[None, :]
    upper = rows[:, None] <= rows[None, :]
    da_qk_lower = da_qk.masked_fill(~lower, 0.0)
    da_kk_lower = da_kk.masked_fill(~lower, 0.0)
    da_qk_upper = da_qk.masked_fill(~upper, 0.0)
    da_kk_upper = da_kk.masked_fill(~upper, 0.0)

    dq = exact.dq.clone()
    dk = exact.dk.clone()
    dkt = exact.dkt.clone()
    if torch.any(guard_hits):
        selected_gate = gate[guard_hits]
        selected_reference = reference[guard_hits]
        positive = torch.exp2(selected_gate - selected_reference)
        negative = torch.exp2(selected_reference - selected_gate)
        dq[guard_hits] = (
            torch.bmm(da_qk_lower[guard_hits], k[guard_hits] * negative)
            * positive
        )
        dk[guard_hits] = (
            torch.bmm(da_kk_lower[guard_hits], k[guard_hits] * negative)
            * positive
        )
        dkt[guard_hits] = (
            torch.bmm(da_qk_upper[guard_hits], q[guard_hits] * positive)
            * negative
        )
        dkt[guard_hits] += (
            torch.bmm(
                da_kk_upper[guard_hits],
                k[guard_hits] * beta[guard_hits, :, None] * positive,
            )
            * negative
        )
    return GuardedDiagonalResult(
        dq=dq,
        dk=dk,
        dkt=dkt,
        guard_hits=guard_hits,
    )
