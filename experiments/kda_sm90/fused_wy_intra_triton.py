"""Experimental KDA backward fusion building blocks.

The first candidate coarsens FLA's intra-chunk backward from one CTA per
``(16-token sub-block, 32-channel slice, chunk, batch, value-head)`` to one CTA
per ``(32-channel slice, chunk, batch, value-head)``.  It processes all 64
tokens in one program, updates dq/dk/dg in place, and atomically accumulates the
four channel-slice contributions into the existing beta gradient.

This is deliberately an experiment, not a production backend.  The direct
pairwise loop is the exact numerical control for deciding whether CTA
coarsening, in-place output, and removal of ``db2.sum(0)`` are useful before
those operations are fused into ``chunk_kda_bwd_kernel_wy_dqkg_fused``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import triton
import triton.language as tl

_BLOCKED_NUM_WARPS = 4
_FUSED_NUM_WARPS = 4
_PENDING_DAQK: torch.Tensor | None = None


@triton.jit(do_not_specialize=["T"])
def chunk_kda_bwd_kernel_intra_chunk_exact(
    q,
    k,
    g,
    beta,
    dAqk,
    dAkk,
    dq,
    dk,
    dg,
    db,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    bos = i_b * T
    o_t = i_t * BT + tl.arange(0, BT)
    o_k = i_k * BK + tl.arange(0, BK)
    m_t = o_t < T
    m_k = o_k < K

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    beta += bos * HV + i_hv
    dAqk += (bos * HV + i_hv) * BT
    dAkk += (bos * HV + i_hv) * BT
    dq += (bos * HV + i_hv) * K
    dk += (bos * HV + i_hv) * K
    dg += (bos * HV + i_hv) * K
    db += bos * HV + i_hv

    p_q = tl.make_block_ptr(
        q,
        (T, K),
        (H * K, 1),
        (i_t * BT, i_k * BK),
        (BT, BK),
        (1, 0),
    )
    p_k = tl.make_block_ptr(
        k,
        (T, K),
        (H * K, 1),
        (i_t * BT, i_k * BK),
        (BT, BK),
        (1, 0),
    )
    p_g = tl.make_block_ptr(
        g,
        (T, K),
        (HV * K, 1),
        (i_t * BT, i_k * BK),
        (BT, BK),
        (1, 0),
    )
    p_beta = tl.make_block_ptr(
        beta,
        (T,),
        (HV,),
        (i_t * BT,),
        (BT,),
        (0,),
    )
    p_dq = tl.make_block_ptr(
        dq,
        (T, K),
        (HV * K, 1),
        (i_t * BT, i_k * BK),
        (BT, BK),
        (1, 0),
    )
    p_dk = tl.make_block_ptr(
        dk,
        (T, K),
        (HV * K, 1),
        (i_t * BT, i_k * BK),
        (BT, BK),
        (1, 0),
    )
    p_dg = tl.make_block_ptr(
        dg,
        (T, K),
        (HV * K, 1),
        (i_t * BT, i_k * BK),
        (BT, BK),
        (1, 0),
    )

    b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
    b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    b_beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)
    b_dq_base = tl.load(p_dq, boundary_check=(0, 1)).to(tl.float32)
    b_dk_base = tl.load(p_dk, boundary_check=(0, 1)).to(tl.float32)
    b_dg_base = tl.load(p_dg, boundary_check=(0, 1)).to(tl.float32)

    b_dq_intra = tl.zeros((BT, BK), dtype=tl.float32)
    b_dk_lower_raw = tl.zeros((BT, BK), dtype=tl.float32)
    b_dkt = tl.zeros((BT, BK), dtype=tl.float32)
    row = tl.arange(0, BT)
    chunk_start = i_t * BT

    # Device loop keeps compile time bounded. Every exponential is evaluated
    # only for its stable triangular half, so the faithful unbounded KDA gate
    # never forms a positive large-span exponent.
    for j in tl.range(0, BT, num_stages=1):
        j_valid = chunk_start + j < T

        p_kj = k + (chunk_start + j) * H * K + o_k
        p_gj = g + (chunk_start + j) * HV * K + o_k
        b_kj = tl.load(p_kj, mask=j_valid & m_k, other=0.0).to(tl.float32)
        b_gj = tl.load(p_gj, mask=j_valid & m_k, other=0.0).to(tl.float32)

        lower = (row >= j) & m_t & j_valid
        lower_delta = tl.where(lower[:, None] & m_k[None, :], b_g - b_gj, 0.0)
        lower_decay = tl.exp2(lower_delta)
        lower_decay = tl.where(lower[:, None] & m_k[None, :], lower_decay, 0.0)

        lower_offsets = o_t * HV * BT + j
        b_dAqk_lower = tl.load(
            dAqk + lower_offsets,
            mask=m_t & j_valid,
            other=0.0,
        ).to(tl.float32)
        b_dAkk_lower = tl.load(
            dAkk + lower_offsets,
            mask=m_t & j_valid,
            other=0.0,
        ).to(tl.float32)
        b_dq_intra += b_dAqk_lower[:, None] * b_kj[None, :] * lower_decay
        b_dk_lower_raw += b_dAkk_lower[:, None] * b_kj[None, :] * lower_decay

        p_qj = q + (chunk_start + j) * H * K + o_k
        p_bj = beta + (chunk_start + j) * HV
        b_qj = tl.load(p_qj, mask=j_valid & m_k, other=0.0).to(tl.float32)
        b_bj = tl.load(p_bj, mask=j_valid, other=0.0).to(tl.float32)

        upper = (row <= j) & m_t & j_valid
        upper_delta = tl.where(upper[:, None] & m_k[None, :], b_gj - b_g, 0.0)
        upper_decay = tl.exp2(upper_delta)
        upper_decay = tl.where(upper[:, None] & m_k[None, :], upper_decay, 0.0)

        upper_offsets = (chunk_start + j) * HV * BT + row
        b_dAqk_upper = tl.load(
            dAqk + upper_offsets,
            mask=m_t & j_valid,
            other=0.0,
        ).to(tl.float32)
        b_dAkk_upper = tl.load(
            dAkk + upper_offsets,
            mask=m_t & j_valid,
            other=0.0,
        ).to(tl.float32)
        b_dkt += (
            b_dAqk_upper[:, None] * b_qj[None, :]
            + b_dAkk_upper[:, None] * (b_kj * b_bj)[None, :]
        ) * upper_decay

    b_db = tl.sum(b_dk_lower_raw * b_k, axis=1)
    b_dk_lower = b_dk_lower_raw * b_beta[:, None]

    b_dq_out = b_dq_base + b_dq_intra
    b_dk_out = b_dk_base + b_dk_lower + b_dkt
    b_dg_out = (
        b_dg_base
        + b_q * b_dq_intra
        + (b_dk_lower - b_dkt) * b_k
    )

    tl.store(p_dq, b_dq_out, boundary_check=(0, 1))
    tl.store(p_dk, b_dk_out, boundary_check=(0, 1))
    tl.store(p_dg, b_dg_out, boundary_check=(0, 1))
    tl.atomic_add(db + o_t * HV, b_db, mask=m_t, sem="relaxed")


@triton.jit
def _chunk_kda_bwd_intra_chunk_blocked_body(
    i_k,
    i_t,
    i_bh,
    q,
    k,
    g,
    beta,
    dAqk,
    dAkk,
    dq,
    dk,
    dg,
    db,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    GUARD_MAX_LOG2_SPAN,
    GUARD_DOT_PRECISION: tl.constexpr,
):
    """Body shared by the standalone coarsened and fused kernels.

    The algebra and guarded diagonal policy intentionally mirror the pinned
    FLA implementation.
    """

    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)
    bos = i_b * T

    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    beta += bos * HV + i_hv
    dAqk += (bos * HV + i_hv) * BT
    dAkk += (bos * HV + i_hv) * BT
    dq += (bos * HV + i_hv) * K
    dk += (bos * HV + i_hv) * K
    dg += (bos * HV + i_hv) * K
    db += bos * HV + i_hv

    o_i = tl.arange(0, BC)
    chunk_start = i_t * BT

    for i_i in tl.range(0, NC, num_stages=1):
        i_ti = chunk_start + i_i * BC
        row_valid = i_ti + o_i < T

        p_g = tl.make_block_ptr(
            g,
            (T, K),
            (HV * K, 1),
            (i_ti, i_k * BK),
            (BC, BK),
            (1, 0),
        )
        p_q = tl.make_block_ptr(
            q,
            (T, K),
            (H * K, 1),
            (i_ti, i_k * BK),
            (BC, BK),
            (1, 0),
        )
        p_k = tl.make_block_ptr(
            k,
            (T, K),
            (H * K, 1),
            (i_ti, i_k * BK),
            (BC, BK),
            (1, 0),
        )
        p_beta = tl.make_block_ptr(
            beta,
            (T,),
            (HV,),
            (i_ti,),
            (BC,),
            (0,),
        )
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_beta = tl.load(p_beta, boundary_check=(0,))

        b_dq2 = tl.zeros([BC, BK], dtype=tl.float32)
        b_dk2 = tl.zeros([BC, BK], dtype=tl.float32)

        if i_i > 0:
            b_gn = tl.load(
                g + i_ti * HV * K + o_k,
                mask=m_k,
                other=0.0,
            ).to(tl.float32)[None, :]
            for i_j in tl.range(0, i_i, num_stages=1):
                p_kj = tl.make_block_ptr(
                    k,
                    (T, K),
                    (H * K, 1),
                    (chunk_start + i_j * BC, i_k * BK),
                    (BC, BK),
                    (1, 0),
                )
                p_gj = tl.make_block_ptr(
                    g,
                    (T, K),
                    (HV * K, 1),
                    (chunk_start + i_j * BC, i_k * BK),
                    (BC, BK),
                    (1, 0),
                )
                p_dAqk = tl.make_block_ptr(
                    dAqk,
                    (T, BT),
                    (HV * BT, 1),
                    (i_ti, i_j * BC),
                    (BC, BC),
                    (1, 0),
                )
                p_dAkk = tl.make_block_ptr(
                    dAkk,
                    (T, BT),
                    (HV * BT, 1),
                    (i_ti, i_j * BC),
                    (BC, BC),
                    (1, 0),
                )
                b_kj = tl.load(p_kj, boundary_check=(0, 1))
                b_gj = tl.load(p_gj, boundary_check=(0, 1)).to(tl.float32)
                b_kg = b_kj * tl.exp2(b_gn - b_gj)
                b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1))
                b_dAkk = tl.load(p_dAkk, boundary_check=(0, 1))
                b_dq2 += tl.dot(b_dAqk, b_kg)
                b_dk2 += tl.dot(b_dAkk, b_kg)
            b_dest_scale = tl.exp2(b_g - b_gn)
            b_dq2 *= b_dest_scale
            b_dk2 *= b_dest_scale

        guard_rows = o_i[:, None]
        m_guard = row_valid[:, None] & m_k[None, :]
        guard_max = tl.max(tl.where(m_guard, b_g, -3.402823e38), axis=0)
        guard_min = tl.min(tl.where(m_guard, b_g, 3.402823e38), axis=0)
        guard_span = tl.max(
            tl.where(m_k, guard_max - guard_min, 0.0),
            axis=0,
        )
        guard_previous = tl.load(
            g + (i_ti + guard_rows - 1) * HV * K + o_k[None, :],
            mask=(guard_rows > 0) & m_guard,
            other=0.0,
        ).to(tl.float32)
        guard_monotonic = (
            (guard_rows == 0) | ~m_guard | (b_g <= guard_previous)
        )
        guard_finite = (
            ~m_guard | ((b_g == b_g) & (tl.abs(b_g) < 3.402823e38))
        )
        guard_valid_count = tl.sum(
            tl.sum((guard_monotonic & guard_finite).to(tl.int32), axis=1),
            axis=0,
        )
        guard_fast = (
            (guard_valid_count == BC * BK)
            & (guard_span <= GUARD_MAX_LOG2_SPAN)
        )
        guard_ref = ((guard_max + guard_min) * 0.5)[None, :]

        if guard_fast:
            p_dAqk_diag = tl.make_block_ptr(
                dAqk,
                (T, BT),
                (HV * BT, 1),
                (i_ti, i_i * BC),
                (BC, BC),
                (1, 0),
            )
            p_dAkk_diag = tl.make_block_ptr(
                dAkk,
                (T, BT),
                (HV * BT, 1),
                (i_ti, i_i * BC),
                (BC, BC),
                (1, 0),
            )
            b_dAqk_diag = tl.load(
                p_dAqk_diag,
                boundary_check=(0, 1),
            ).to(tl.float32)
            b_dAkk_diag = tl.load(
                p_dAkk_diag,
                boundary_check=(0, 1),
            ).to(tl.float32)
            lower_mask = (
                (o_i[:, None] >= o_i[None, :])
                & row_valid[:, None]
                & row_valid[None, :]
            )
            b_dAqk_diag = tl.where(lower_mask, b_dAqk_diag, 0.0)
            b_dAkk_diag = tl.where(lower_mask, b_dAkk_diag, 0.0)
            b_g_centered = tl.where(
                row_valid[:, None],
                b_g - guard_ref,
                0.0,
            )
            b_pos = tl.where(
                row_valid[:, None],
                tl.exp2(b_g_centered),
                0.0,
            )
            b_neg = tl.where(
                row_valid[:, None],
                tl.exp2(-b_g_centered),
                0.0,
            )
            b_k_neg = b_k * b_neg
            b_dq2 += (
                tl.dot(
                    b_dAqk_diag,
                    b_k_neg,
                    input_precision=GUARD_DOT_PRECISION,
                )
                * b_pos
            )
            b_dk2 += (
                tl.dot(
                    b_dAkk_diag,
                    b_k_neg,
                    input_precision=GUARD_DOT_PRECISION,
                )
                * b_pos
            )
        else:
            for j in tl.range(0, BC, num_stages=1):
                j_valid = i_ti + j < T
                lower = (o_i >= j) & row_valid & j_valid
                offsets = (i_ti + o_i) * HV * BT + i_i * BC + j
                b_dAqk_j = tl.load(
                    dAqk + offsets,
                    mask=row_valid & j_valid,
                    other=0.0,
                )
                b_dAkk_j = tl.load(
                    dAkk + offsets,
                    mask=row_valid & j_valid,
                    other=0.0,
                )
                b_kj = tl.load(
                    k + (i_ti + j) * H * K + o_k,
                    mask=j_valid & m_k,
                    other=0.0,
                ).to(tl.float32)
                b_gj = tl.load(
                    g + (i_ti + j) * HV * K + o_k,
                    mask=j_valid & m_k,
                    other=0.0,
                ).to(tl.float32)
                delta = tl.where(
                    lower[:, None] & m_k[None, :],
                    b_g - b_gj,
                    0.0,
                )
                decay = tl.where(
                    lower[:, None] & m_k[None, :],
                    tl.exp2(delta),
                    0.0,
                )
                b_dq2 += b_dAqk_j[:, None] * b_kj[None, :] * decay
                b_dk2 += b_dAkk_j[:, None] * b_kj[None, :] * decay

        b_db = tl.sum(b_dk2 * b_k, axis=1)
        b_dk2 *= b_beta[:, None]
        b_dg2 = b_q * b_dq2

        b_dkt = tl.zeros([BC, BK], dtype=tl.float32)
        if i_i < NC - 1:
            last_row = min(i_ti + BC, T) - 1
            b_gn = tl.load(
                g + last_row * HV * K + o_k,
                mask=m_k,
                other=0.0,
            ).to(tl.float32)[None, :]
            for i_j in tl.range(i_i + 1, NC, num_stages=1):
                source_start = chunk_start + i_j * BC
                p_qj = tl.make_block_ptr(
                    q,
                    (T, K),
                    (H * K, 1),
                    (source_start, i_k * BK),
                    (BC, BK),
                    (1, 0),
                )
                p_kj = tl.make_block_ptr(
                    k,
                    (T, K),
                    (H * K, 1),
                    (source_start, i_k * BK),
                    (BC, BK),
                    (1, 0),
                )
                p_gj = tl.make_block_ptr(
                    g,
                    (T, K),
                    (HV * K, 1),
                    (source_start, i_k * BK),
                    (BC, BK),
                    (1, 0),
                )
                p_bj = tl.make_block_ptr(
                    beta,
                    (T,),
                    (HV,),
                    (source_start,),
                    (BC,),
                    (0,),
                )
                p_dAqk = tl.make_block_ptr(
                    dAqk,
                    (BT, T),
                    (1, HV * BT),
                    (i_i * BC, source_start),
                    (BC, BC),
                    (0, 1),
                )
                p_dAkk = tl.make_block_ptr(
                    dAkk,
                    (BT, T),
                    (1, HV * BT),
                    (i_i * BC, source_start),
                    (BC, BC),
                    (0, 1),
                )
                b_qj = tl.load(p_qj, boundary_check=(0, 1))
                b_kj = tl.load(p_kj, boundary_check=(0, 1))
                b_gj = tl.load(p_gj, boundary_check=(0, 1)).to(tl.float32)
                b_bj = tl.load(p_bj, boundary_check=(0,))
                source_valid = source_start + o_i < T
                b_source_scale = tl.where(
                    source_valid[:, None],
                    tl.exp2(b_gj - b_gn),
                    0.0,
                )
                b_qg = b_qj * b_source_scale
                b_kbg = b_kj * b_bj[:, None] * b_source_scale
                b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1))
                b_dAkk = tl.load(p_dAkk, boundary_check=(0, 1))
                b_dkt += tl.dot(b_dAqk, b_qg)
                b_dkt += tl.dot(b_dAkk, b_kbg)
            b_dkt *= tl.exp2(b_gn - b_g)

        if guard_fast:
            p_dAqk_diag = tl.make_block_ptr(
                dAqk,
                (BT, T),
                (1, HV * BT),
                (i_i * BC, i_ti),
                (BC, BC),
                (0, 1),
            )
            p_dAkk_diag = tl.make_block_ptr(
                dAkk,
                (BT, T),
                (1, HV * BT),
                (i_i * BC, i_ti),
                (BC, BC),
                (0, 1),
            )
            b_dAqk_diag = tl.load(
                p_dAqk_diag,
                boundary_check=(0, 1),
            ).to(tl.float32)
            b_dAkk_diag = tl.load(
                p_dAkk_diag,
                boundary_check=(0, 1),
            ).to(tl.float32)
            upper_mask = (
                (o_i[:, None] <= o_i[None, :])
                & row_valid[:, None]
                & row_valid[None, :]
            )
            b_dAqk_diag = tl.where(upper_mask, b_dAqk_diag, 0.0)
            b_dAkk_diag = tl.where(upper_mask, b_dAkk_diag, 0.0)
            b_g_centered = tl.where(
                row_valid[:, None],
                b_g - guard_ref,
                0.0,
            )
            b_pos = tl.where(
                row_valid[:, None],
                tl.exp2(b_g_centered),
                0.0,
            )
            b_neg = tl.where(
                row_valid[:, None],
                tl.exp2(-b_g_centered),
                0.0,
            )
            b_q_pos = b_q * b_pos
            b_kb_pos = b_k * b_beta[:, None] * b_pos
            b_dkt += (
                tl.dot(
                    b_dAqk_diag,
                    b_q_pos,
                    input_precision=GUARD_DOT_PRECISION,
                )
                * b_neg
            )
            b_dkt += (
                tl.dot(
                    b_dAkk_diag,
                    b_kb_pos,
                    input_precision=GUARD_DOT_PRECISION,
                )
                * b_neg
            )
        else:
            for j in tl.range(0, BC, num_stages=1):
                j_valid = i_ti + j < T
                upper = (o_i <= j) & row_valid & j_valid
                offsets = (i_ti + j) * HV * BT + i_i * BC + o_i
                b_dAqk_j = tl.load(
                    dAqk + offsets,
                    mask=row_valid & j_valid,
                    other=0.0,
                )
                b_dAkk_j = tl.load(
                    dAkk + offsets,
                    mask=row_valid & j_valid,
                    other=0.0,
                )
                b_qj = tl.load(
                    q + (i_ti + j) * H * K + o_k,
                    mask=j_valid & m_k,
                    other=0.0,
                ).to(tl.float32)
                b_kj = tl.load(
                    k + (i_ti + j) * H * K + o_k,
                    mask=j_valid & m_k,
                    other=0.0,
                ).to(tl.float32)
                b_gj = tl.load(
                    g + (i_ti + j) * HV * K + o_k,
                    mask=j_valid & m_k,
                    other=0.0,
                ).to(tl.float32)
                b_bj = tl.load(
                    beta + (i_ti + j) * HV,
                    mask=j_valid,
                    other=0.0,
                ).to(tl.float32)
                delta = tl.where(
                    upper[:, None] & m_k[None, :],
                    b_gj - b_g,
                    0.0,
                )
                decay = tl.where(
                    upper[:, None] & m_k[None, :],
                    tl.exp2(delta),
                    0.0,
                )
                b_dkt += (
                    b_dAqk_j[:, None] * b_qj[None, :]
                    + b_dAkk_j[:, None] * (b_kj * b_bj)[None, :]
                ) * decay

        p_dq = tl.make_block_ptr(
            dq,
            (T, K),
            (HV * K, 1),
            (i_ti, i_k * BK),
            (BC, BK),
            (1, 0),
        )
        p_dk = tl.make_block_ptr(
            dk,
            (T, K),
            (HV * K, 1),
            (i_ti, i_k * BK),
            (BC, BK),
            (1, 0),
        )
        p_dg = tl.make_block_ptr(
            dg,
            (T, K),
            (HV * K, 1),
            (i_ti, i_k * BK),
            (BC, BK),
            (1, 0),
        )
        b_dq_out = tl.load(p_dq, boundary_check=(0, 1)) + b_dq2
        b_dk_out = tl.load(p_dk, boundary_check=(0, 1)) + b_dk2 + b_dkt
        b_dg_out = (
            tl.load(p_dg, boundary_check=(0, 1))
            + b_dg2
            + (b_dk2 - b_dkt) * b_k
        )
        tl.store(p_dq, b_dq_out, boundary_check=(0, 1))
        tl.store(p_dk, b_dk_out, boundary_check=(0, 1))
        tl.store(p_dg, b_dg_out, boundary_check=(0, 1))
        tl.atomic_add(
            db + (i_ti + o_i) * HV,
            b_db,
            mask=row_valid,
            sem="relaxed",
        )


@triton.jit(do_not_specialize=["T", "GUARD_MAX_LOG2_SPAN"])
def chunk_kda_bwd_kernel_intra_chunk_blocked(
    q,
    k,
    g,
    beta,
    dAqk,
    dAkk,
    dq,
    dk,
    dg,
    db,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    GUARD_MAX_LOG2_SPAN,
    GUARD_DOT_PRECISION: tl.constexpr,
):
    _chunk_kda_bwd_intra_chunk_blocked_body(
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
        q,
        k,
        g,
        beta,
        dAqk,
        dAkk,
        dq,
        dk,
        dg,
        db,
        T,
        H,
        HV,
        K,
        BT,
        BC,
        BK,
        NC,
        GUARD_MAX_LOG2_SPAN,
        GUARD_DOT_PRECISION,
    )


def _unwrap_wy_kernel():
    from fla.ops.kda.chunk_bwd import chunk_kda_bwd_kernel_wy_dqkg_fused

    kernel = chunk_kda_bwd_kernel_wy_dqkg_fused
    while not isinstance(kernel, triton.runtime.JITFunction):
        kernel = kernel.fn
    return kernel


_WY_DQKG_KERNEL = _unwrap_wy_kernel()


@triton.jit(do_not_specialize=["T", "GUARD_MAX_LOG2_SPAN"])
def chunk_kda_bwd_kernel_wy_intra_fused(
    q,
    k,
    v,
    v_new,
    g,
    beta,
    A,
    h,
    do,
    dh,
    dq,
    dk,
    dv,
    dv2,
    dg,
    db,
    dAqk,
    dAkk,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK_WY: tl.constexpr,
    BV: tl.constexpr,
    BK_INTRA: tl.constexpr,
    BC: tl.constexpr,
    NC: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    GUARD_MAX_LOG2_SPAN,
    GUARD_DOT_PRECISION: tl.constexpr,
):
    # The original FLA body computes the WY/state gradients and writes its
    # preliminary dq/dk/dg plus dAkk. Keeping it as a JIT subroutine makes this
    # experiment track the exact pinned implementation.
    _WY_DQKG_KERNEL(
        q,
        k,
        v,
        v_new,
        g,
        beta,
        A,
        h,
        do,
        dh,
        dq,
        dk,
        dv,
        dv2,
        dg,
        db,
        dAkk,
        cu_seqlens,
        chunk_indices,
        scale,
        T,
        H,
        HV,
        K,
        V,
        BT,
        BK_WY,
        BV,
        STATE_V_FIRST,
        IS_VARLEN,
    )

    # The same CTA consumes dAkk after a block barrier, processes every
    # 32-channel intra slice, and updates the preliminary gradients in place.
    tl.debug_barrier()
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    for i_k in range(0, K // BK_INTRA):
        _chunk_kda_bwd_intra_chunk_blocked_body(
            i_k,
            i_t,
            i_bh,
            q,
            k,
            g,
            beta,
            dAqk,
            dAkk,
            dq,
            dk,
            dg,
            db,
            T,
            H,
            HV,
            K,
            BT,
            BC,
            BK_INTRA,
            NC,
            GUARD_MAX_LOG2_SPAN,
            GUARD_DOT_PRECISION,
        )


def chunk_kda_bwd_intra_chunk_exact(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dAqk: torch.Tensor,
    dAkk: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    db: torch.Tensor,
    dg: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if cu_seqlens is not None or chunk_indices is not None:
        raise NotImplementedError("the first fused-intra experiment is fixed-length only")
    if safe_gate:
        raise NotImplementedError("the first fused-intra experiment targets safe_gate=False")
    B, T, H, K, HV = *k.shape, g.shape[2]
    if chunk_size != 64 or K != 128 or H != HV:
        raise NotImplementedError(
            "the first fused-intra experiment requires BT=64, K=128, and H=HV"
        )

    BK = 32
    grid = (triton.cdiv(K, BK), triton.cdiv(T, chunk_size), B * HV)
    chunk_kda_bwd_kernel_intra_chunk_exact[grid](
        q=q,
        k=k,
        g=g,
        beta=beta,
        dAqk=dAqk,
        dAkk=dAkk,
        dq=dq,
        dk=dk,
        dg=dg,
        db=db,
        T=T,
        H=H,
        HV=HV,
        K=K,
        BT=chunk_size,
        BK=BK,
        num_warps=8,
        num_stages=1,
    )
    return dq, dk, db, dg


def chunk_kda_bwd_intra_chunk_blocked(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dAqk: torch.Tensor,
    dAkk: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    db: torch.Tensor,
    dg: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if cu_seqlens is not None or chunk_indices is not None:
        raise NotImplementedError("the first fused-intra experiment is fixed-length only")
    if safe_gate:
        raise NotImplementedError("the first fused-intra experiment targets safe_gate=False")
    B, T, H, K, HV = *k.shape, g.shape[2]
    if chunk_size != 64 or K != 128 or H != HV:
        raise NotImplementedError(
            "the first fused-intra experiment requires BT=64, K=128, and H=HV"
        )

    BC = 16
    BK = 32
    grid = (triton.cdiv(K, BK), triton.cdiv(T, chunk_size), B * HV)
    chunk_kda_bwd_kernel_intra_chunk_blocked[grid](
        q=q,
        k=k,
        g=g,
        beta=beta,
        dAqk=dAqk,
        dAkk=dAkk,
        dq=dq,
        dk=dk,
        dg=dg,
        db=db,
        T=T,
        H=H,
        HV=HV,
        K=K,
        BT=chunk_size,
        BC=BC,
        BK=BK,
        NC=triton.cdiv(chunk_size, BC),
        GUARD_MAX_LOG2_SPAN=232.0,
        GUARD_DOT_PRECISION="tf32x3",
        num_warps=_BLOCKED_NUM_WARPS,
        num_stages=1,
    )
    return dq, dk, db, dg


def chunk_kda_bwd_wy_intra_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dv: torch.Tensor,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    global _PENDING_DAQK
    if _PENDING_DAQK is None:
        raise RuntimeError("the fused WY/intra experiment did not receive dAqk")
    if cu_seqlens is not None or chunk_indices is not None:
        raise NotImplementedError("the first fused WY/intra experiment is fixed-length only")

    B, T, H, K, HV, V = *k.shape, v.shape[2], v.shape[-1]
    if chunk_size != 64 or K != 128 or V != 128 or H != HV:
        raise NotImplementedError(
            "the first fused WY/intra experiment requires BT=64, K=V=128, and H=HV"
        )
    if scale is None:
        scale = K**-0.5

    dAqk = _PENDING_DAQK
    _PENDING_DAQK = None
    dq = g.new_empty(B, T, HV, K, dtype=torch.float)
    dk = g.new_empty(B, T, HV, K, dtype=torch.float)
    dv2 = torch.empty_like(v)
    dg = torch.empty_like(g, dtype=torch.float)
    db = torch.empty_like(beta, dtype=torch.float)
    dAkk = torch.empty_like(A, dtype=torch.float)

    grid = (triton.cdiv(T, chunk_size), B * HV)
    chunk_kda_bwd_kernel_wy_intra_fused[grid](
        q=q,
        k=k,
        v=v,
        v_new=v_new,
        g=g,
        beta=beta,
        A=A,
        h=h,
        do=do,
        dh=dh,
        dq=dq,
        dk=dk,
        dv=dv,
        dv2=dv2,
        dg=dg,
        db=db,
        dAqk=dAqk,
        dAkk=dAkk,
        cu_seqlens=None,
        chunk_indices=None,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=chunk_size,
        BK_WY=64,
        BV=64,
        BK_INTRA=32,
        BC=16,
        NC=4,
        STATE_V_FIRST=state_v_first,
        IS_VARLEN=False,
        GUARD_MAX_LOG2_SPAN=232.0,
        GUARD_DOT_PRECISION="tf32x3",
        num_warps=_FUSED_NUM_WARPS,
        num_stages=2,
    )
    return dq, dk, dv2, db, dg, dAkk


def install_fused_wy_intra_experiment(
    num_warps: int = 4,
) -> tuple[Callable[..., object], ...]:
    """Install the monolithic prototype into FLA for the current process."""
    import fla.ops.kda.chunk_bwd as chunk_bwd

    global _FUSED_NUM_WARPS
    if num_warps not in {4, 8}:
        raise ValueError("the fused candidate supports four or eight warps")
    _FUSED_NUM_WARPS = num_warps
    original_dav = chunk_bwd.chunk_kda_bwd_dAv
    original_wy = chunk_bwd.chunk_kda_bwd_wy_dqkg_fused
    original_intra = chunk_bwd.chunk_kda_bwd_intra

    def capture_daqk(*args, **kwargs):
        global _PENDING_DAQK
        dAqk, dv = original_dav(*args, **kwargs)
        _PENDING_DAQK = dAqk
        return dAqk, dv

    def skip_intra(
        *,
        dq: torch.Tensor,
        dk: torch.Tensor,
        db: torch.Tensor,
        dg: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return dq, dk, db, dg

    chunk_bwd.chunk_kda_bwd_dAv = capture_daqk
    chunk_bwd.chunk_kda_bwd_wy_dqkg_fused = chunk_kda_bwd_wy_intra_fused
    chunk_bwd.chunk_kda_bwd_intra = skip_intra
    return original_dav, original_wy, original_intra


def install_intra_chunk_experiment(
    candidate: str = "blocked",
    num_warps: int = 4,
) -> Callable[..., object]:
    """Install the candidate into FLA's Python orchestrator for this process."""
    import fla.ops.kda.chunk_bwd as chunk_bwd

    global _BLOCKED_NUM_WARPS
    if num_warps not in {2, 4, 8}:
        raise ValueError("num_warps must be one of 2, 4, or 8")
    _BLOCKED_NUM_WARPS = num_warps
    original = chunk_bwd.chunk_kda_bwd_intra
    if candidate == "blocked":
        chunk_bwd.chunk_kda_bwd_intra = chunk_kda_bwd_intra_chunk_blocked
    elif candidate == "exact":
        chunk_bwd.chunk_kda_bwd_intra = chunk_kda_bwd_intra_chunk_exact
    else:
        raise ValueError(f"unknown candidate {candidate!r}")
    return original
