"""Fused KDA preprocessing experiment for the exact OpenKimi SM90 profile.

This is deliberately separated from the supplied monolithic prepare kernel.
The original kernel combines normalization, gate activation, local matrix
construction, inversion, and workspace writes in one very large CuTe program.
Splitting the bandwidth-bound preprocessing makes compilation tractable and
lets us measure fusion independently before coupling it to a tensor-core scan.

The kernel fuses:

* Q and K L2 normalization, including the FP32 reciprocal norms needed by
  backward;
* channel-wise KDA gate activation and a 64-token chunk-local cumulative sum;
* beta sigmoid activation.

Its outputs match the inputs consumed by FLA's existing chunk KDA forward.
"""

from __future__ import annotations

from typing import Final

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cutlass_dsl import Float32, Int32


CHUNK: Final[int] = 64
DIM: Final[int] = 128
THREADS: Final[int] = 256
RCP_LN2: Final[float] = 1.4426950408889634


@cute.jit
def _sigmoid(x: Float32) -> Float32:
    return Float32(1.0) / (
        Float32(1.0) + cute.math.exp(-x, fastmath=True)
    )


@cute.jit
def _softplus(x: Float32) -> Float32:
    ax = x if x >= Float32(0.0) else -x
    positive = x if x >= Float32(0.0) else Float32(0.0)
    return positive + cute.math.log(
        Float32(1.0) + cute.math.exp(-ax, fastmath=True),
        fastmath=True,
    )


@cute.jit
def _warp_sum(value: Float32) -> Float32:
    for shift in cutlass.range_constexpr(5):
        value += cute.arch.shuffle_sync_bfly(value, offset=1 << shift)
    return value


class KdaPreprocessSm90:
    """One-CTA-per-(batch, head, 64-token chunk) preprocessing."""

    @cute.jit
    def __call__(
        self,
        q_ptr: cute.Pointer,
        k_ptr: cute.Pointer,
        raw_gate_ptr: cute.Pointer,
        beta_logits_ptr: cute.Pointer,
        a_log_ptr: cute.Pointer,
        dt_bias_ptr: cute.Pointer,
        q_norm_ptr: cute.Pointer,
        k_norm_ptr: cute.Pointer,
        q_rstd_ptr: cute.Pointer,
        k_rstd_ptr: cute.Pointer,
        gate_cumsum_ptr: cute.Pointer,
        beta_ptr: cute.Pointer,
        B: Int32,
        T: Int32,
        H: Int32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        x_layout = cute.make_layout(
            (B, T, H, DIM),
            stride=(T * H * DIM, H * DIM, DIM, 1),
        )
        token_layout = cute.make_layout(
            (B, T, H),
            stride=(T * H, H, 1),
        )
        parameter_layout = cute.make_layout(
            (H, DIM),
            stride=(DIM, 1),
        )
        mQ = cute.make_tensor(q_ptr, x_layout)
        mK = cute.make_tensor(k_ptr, x_layout)
        mRawGate = cute.make_tensor(raw_gate_ptr, x_layout)
        mBetaLogits = cute.make_tensor(beta_logits_ptr, token_layout)
        mALog = cute.make_tensor(
            a_log_ptr,
            cute.make_layout((H,), stride=(1,)),
        )
        mDtBias = cute.make_tensor(dt_bias_ptr, parameter_layout)
        mQNorm = cute.make_tensor(q_norm_ptr, x_layout)
        mKNorm = cute.make_tensor(k_norm_ptr, x_layout)
        mQRstd = cute.make_tensor(q_rstd_ptr, token_layout)
        mKRstd = cute.make_tensor(k_rstd_ptr, token_layout)
        mGateCumsum = cute.make_tensor(gate_cumsum_ptr, x_layout)
        mBeta = cute.make_tensor(beta_ptr, token_layout)

        self.kernel(
            mQ,
            mK,
            mRawGate,
            mBetaLogits,
            mALog,
            mDtBias,
            mQNorm,
            mKNorm,
            mQRstd,
            mKRstd,
            mGateCumsum,
            mBeta,
            T,
            eps,
        ).launch(
            grid=(cute.ceil_div(T, CHUNK), H, B),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mRawGate: cute.Tensor,
        mBetaLogits: cute.Tensor,
        mALog: cute.Tensor,
        mDtBias: cute.Tensor,
        mQNorm: cute.Tensor,
        mKNorm: cute.Tensor,
        mQRstd: cute.Tensor,
        mKRstd: cute.Tensor,
        mGateCumsum: cute.Tensor,
        mBeta: cute.Tensor,
        T: Int32,
        eps: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        chunk_idx, head_idx, batch_idx = cute.arch.block_idx()
        chunk_start = chunk_idx * CHUNK

        # The lower half of the CTA owns one channel each. Each thread walks
        # time in order, so the channel-wise recurrence never touches shared
        # memory and only writes coalesced 128-channel rows.
        if tidx < DIM:
            channel = tidx
            running = Float32(0.0)
            a = cute.math.exp(
                mALog[head_idx].to(Float32),
                fastmath=True,
            )
            bias = mDtBias[head_idx, channel].to(Float32)
            for token_in_chunk in range(CHUNK):
                token = chunk_start + token_in_chunk
                if token < T:
                    raw = (
                        mRawGate[
                            batch_idx,
                            token,
                            head_idx,
                            channel,
                        ].to(Float32)
                        + bias
                    )
                    running += -a * _softplus(raw) * Float32(RCP_LN2)
                    mGateCumsum[
                        batch_idx,
                        token,
                        head_idx,
                        channel,
                    ] = running

        # Four complete warps normalize 16 rows each while the other half of
        # the CTA owns gate channels. Inputs are read twice:
        # once for the FP32 sum of squares and once for the normalized store.
        # This costs bandwidth but avoids a 64x128 shared-memory tile and keeps
        # the launch occupancy-independent.
        if tidx >= DIM:
            norm_thread = tidx - DIM
            warp = norm_thread // 32
            lane = norm_thread % 32
            for token_iteration in range(CHUNK // 4):
                token_in_chunk = warp + token_iteration * 4
                token = chunk_start + token_in_chunk
                q_sum = Float32(0.0)
                k_sum = Float32(0.0)
                if token < T:
                    for vector in cutlass.range_constexpr(DIM // 32):
                        channel = lane + vector * 32
                        q_value = mQ[
                            batch_idx,
                            token,
                            head_idx,
                            channel,
                        ].to(Float32)
                        k_value = mK[
                            batch_idx,
                            token,
                            head_idx,
                            channel,
                        ].to(Float32)
                        q_sum += q_value * q_value
                        k_sum += k_value * k_value
                q_sum = _warp_sum(q_sum)
                k_sum = _warp_sum(k_sum)
                q_rstd = cute.math.rsqrt(q_sum + eps, fastmath=True)
                k_rstd = cute.math.rsqrt(k_sum + eps, fastmath=True)
                if token < T:
                    for vector in cutlass.range_constexpr(DIM // 32):
                        channel = lane + vector * 32
                        mQNorm[
                            batch_idx,
                            token,
                            head_idx,
                            channel,
                        ] = (
                            mQ[
                                batch_idx,
                                token,
                                head_idx,
                                channel,
                            ].to(Float32)
                            * q_rstd
                        ).to(mQNorm.element_type)
                        mKNorm[
                            batch_idx,
                            token,
                            head_idx,
                            channel,
                        ] = (
                            mK[
                                batch_idx,
                                token,
                                head_idx,
                                channel,
                            ].to(Float32)
                            * k_rstd
                        ).to(mKNorm.element_type)
                    if lane == 0:
                        mQRstd[batch_idx, token, head_idx] = q_rstd
                        mKRstd[batch_idx, token, head_idx] = k_rstd
                        mBeta[batch_idx, token, head_idx] = _sigmoid(
                            mBetaLogits[
                                batch_idx,
                                token,
                                head_idx,
                            ].to(Float32)
                        ).to(mBeta.element_type)
