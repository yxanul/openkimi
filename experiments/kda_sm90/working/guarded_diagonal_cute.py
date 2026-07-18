"""CuTe SM90 midpoint-guarded KDA diagonal backward tile.

Four warps compute the two lower-triangular and two upper-triangular products
concurrently. Tiles whose midpoint factors would approach the FP32 exponent
edge retain the faithful pairwise fallback.
"""

from __future__ import annotations

from typing import Final

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.nvgpu import warp
from cutlass.cutlass_dsl import Float32, Int32

BLOCK: Final[int] = 16
CHANNELS: Final[int] = 32
THREADS: Final[int] = 128


@cute.jit
def _warp_max(value: Float32) -> Float32:
    for shift in cutlass.range_constexpr(5):
        other = cute.arch.shuffle_sync_bfly(value, offset=1 << shift)
        value = value if value >= other else other
    return value


@cute.jit
def _mma_tile(
    sA: cute.Tensor,
    sB: cute.Tensor,
    sC: cute.Tensor,
    lane: Int32,
    tiled_mma: cute.TiledMma,
) -> None:
    thr_mma = tiled_mma.get_slice(lane)
    tCsA = thr_mma.partition_A(sA)
    tCsB = thr_mma.partition_B(sB)
    tCsC = thr_mma.partition_C(sC)
    tCrA = tiled_mma.make_fragment_A(tCsA)
    tCrB = tiled_mma.make_fragment_B(tCsB)
    tCrC = tiled_mma.make_fragment_C(tCsC)
    cute.autovec_copy(tCsA, tCrA)
    cute.autovec_copy(tCsB, tCrB)
    tCrC.fill(0.0)
    for inner_tile in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
        cute.gemm(
            tiled_mma,
            tCrC,
            tCrA[None, None, inner_tile],
            tCrB[None, None, inner_tile],
            tCrC,
        )
    cute.autovec_copy(tCrC, tCsC)


@cute.jit
def _mma_tile_scaled_store(
    sA: cute.Tensor,
    sB: cute.Tensor,
    sScale: cute.Tensor,
    gC: cute.Tensor,
    lane: Int32,
    tiled_mma: cute.TiledMma,
) -> None:
    """Run one MMA, apply its row/channel scale, and store its warp tile."""

    thr_mma = tiled_mma.get_slice(lane)
    tCsA = thr_mma.partition_A(sA)
    tCsB = thr_mma.partition_B(sB)
    tCsScale = thr_mma.partition_C(sScale)
    tCgC = thr_mma.partition_C(gC)
    tCrA = tiled_mma.make_fragment_A(tCsA)
    tCrB = tiled_mma.make_fragment_B(tCsB)
    tCrScale = tiled_mma.make_fragment_C(tCsScale)
    tCrC = tiled_mma.make_fragment_C(tCgC)
    cute.autovec_copy(tCsA, tCrA)
    cute.autovec_copy(tCsB, tCrB)
    cute.autovec_copy(tCsScale, tCrScale)
    tCrC.fill(0.0)
    for inner_tile in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
        cute.gemm(
            tiled_mma,
            tCrC,
            tCrA[None, None, inner_tile],
            tCrB[None, None, inner_tile],
            tCrC,
        )
    tCrC.store(tCrC.load() * tCrScale.load())
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        gC.element_type,
    )
    cute.copy(copy_atom, tCrC, tCgC)


@cute.jit
def _mma_tile_pair(
    sA0: cute.Tensor,
    sA1: cute.Tensor,
    sB: cute.Tensor,
    sC0: cute.Tensor,
    sC1: cute.Tensor,
    lane: Int32,
    tiled_mma: cute.TiledMma,
) -> None:
    """Reuse one 16-channel B fragment across two independent products."""

    thr_mma = tiled_mma.get_slice(lane)
    tCsA0 = thr_mma.partition_A(sA0)
    tCsA1 = thr_mma.partition_A(sA1)
    tCsB = thr_mma.partition_B(sB)
    tCsC0 = thr_mma.partition_C(sC0)
    tCsC1 = thr_mma.partition_C(sC1)
    tCrA0 = tiled_mma.make_fragment_A(tCsA0)
    tCrA1 = tiled_mma.make_fragment_A(tCsA1)
    tCrB = tiled_mma.make_fragment_B(tCsB)
    tCrC0 = tiled_mma.make_fragment_C(tCsC0)
    tCrC1 = tiled_mma.make_fragment_C(tCsC1)
    cute.autovec_copy(tCsA0, tCrA0)
    cute.autovec_copy(tCsA1, tCrA1)
    cute.autovec_copy(tCsB, tCrB)
    tCrC0.fill(0.0)
    tCrC1.fill(0.0)
    for inner_tile in cutlass.range_constexpr(cute.size(tCrA0, mode=[2])):
        cute.gemm(
            tiled_mma,
            tCrC0,
            tCrA0[None, None, inner_tile],
            tCrB[None, None, inner_tile],
            tCrC0,
        )
        cute.gemm(
            tiled_mma,
            tCrC1,
            tCrA1[None, None, inner_tile],
            tCrB[None, None, inner_tile],
            tCrC1,
        )
    cute.autovec_copy(tCrC0, tCsC0)
    cute.autovec_copy(tCrC1, tCsC1)


@cute.jit
def _mma_tile_sum_pair(
    sA0: cute.Tensor,
    sB0: cute.Tensor,
    sA1: cute.Tensor,
    sB1: cute.Tensor,
    sC: cute.Tensor,
    lane: Int32,
    tiled_mma: cute.TiledMma,
) -> None:
    """Compute two 16-channel products and sum their accumulators in registers."""

    thr_mma = tiled_mma.get_slice(lane)
    tCsA0 = thr_mma.partition_A(sA0)
    tCsB0 = thr_mma.partition_B(sB0)
    tCsA1 = thr_mma.partition_A(sA1)
    tCsB1 = thr_mma.partition_B(sB1)
    tCsC = thr_mma.partition_C(sC)
    tCrA0 = tiled_mma.make_fragment_A(tCsA0)
    tCrB0 = tiled_mma.make_fragment_B(tCsB0)
    tCrA1 = tiled_mma.make_fragment_A(tCsA1)
    tCrB1 = tiled_mma.make_fragment_B(tCsB1)
    tCrC0 = tiled_mma.make_fragment_C(tCsC)
    tCrC1 = tiled_mma.make_fragment_C(tCsC)
    cute.autovec_copy(tCsA0, tCrA0)
    cute.autovec_copy(tCsB0, tCrB0)
    cute.autovec_copy(tCsA1, tCrA1)
    cute.autovec_copy(tCsB1, tCrB1)
    tCrC0.fill(0.0)
    tCrC1.fill(0.0)
    for inner_tile in cutlass.range_constexpr(cute.size(tCrA0, mode=[2])):
        cute.gemm(
            tiled_mma,
            tCrC0,
            tCrA0[None, None, inner_tile],
            tCrB0[None, None, inner_tile],
            tCrC0,
        )
        cute.gemm(
            tiled_mma,
            tCrC1,
            tCrA1[None, None, inner_tile],
            tCrB1[None, None, inner_tile],
            tCrC1,
        )
    tCrC0.store(tCrC0.load() + tCrC1.load())
    cute.autovec_copy(tCrC0, tCsC)


class GuardedDiagonalSm90:
    def __init__(
        self,
        b_operand_swizzle_bits: int = 3,
        reciprocal_negative: bool = False,
        cache_gate: bool = True,
        direct_lower_epilogue: bool = False,
        channel_half_warps: bool = False,
        explicit_fallback_diagonal: bool = False,
    ) -> None:
        if b_operand_swizzle_bits not in (0, 1, 2, 3):
            raise ValueError("B-operand swizzle bits must be one of 0, 1, 2, or 3")
        if direct_lower_epilogue and channel_half_warps:
            raise ValueError("direct lower epilogue and channel-half warps are exclusive")
        self.b_operand_swizzle_bits = b_operand_swizzle_bits
        self.reciprocal_negative = reciprocal_negative
        self.cache_gate = cache_gate
        self.direct_lower_epilogue = direct_lower_epilogue
        self.channel_half_warps = channel_half_warps
        self.explicit_fallback_diagonal = explicit_fallback_diagonal

    @cute.jit
    def __call__(
        self,
        gate_ptr: cute.Pointer,
        q_ptr: cute.Pointer,
        k_ptr: cute.Pointer,
        beta_ptr: cute.Pointer,
        da_qk_ptr: cute.Pointer,
        da_kk_ptr: cute.Pointer,
        dq_ptr: cute.Pointer,
        dk_ptr: cute.Pointer,
        dkt_ptr: cute.Pointer,
        blocks: Int32,
        max_log2_span: Float32,
        stream: cuda.CUstream,
    ):
        token_layout = cute.make_layout(
            (blocks, BLOCK, CHANNELS),
            stride=(BLOCK * CHANNELS, CHANNELS, 1),
        )
        beta_layout = cute.make_layout(
            (blocks, BLOCK),
            stride=(BLOCK, 1),
        )
        matrix_layout = cute.make_layout(
            (blocks, BLOCK, BLOCK),
            stride=(BLOCK * BLOCK, BLOCK, 1),
        )
        mGate = cute.make_tensor(gate_ptr, token_layout)
        mQ = cute.make_tensor(q_ptr, token_layout)
        mK = cute.make_tensor(k_ptr, token_layout)
        mBeta = cute.make_tensor(beta_ptr, beta_layout)
        mDaQk = cute.make_tensor(da_qk_ptr, matrix_layout)
        mDaKk = cute.make_tensor(da_kk_ptr, matrix_layout)
        mDq = cute.make_tensor(dq_ptr, token_layout)
        mDk = cute.make_tensor(dk_ptr, token_layout)
        mDkt = cute.make_tensor(dkt_ptr, token_layout)
        tiled_mma = cute.make_tiled_mma(
            warp.MmaTF32Op((16, 8, 8)),
            (1, 1, 1),
            permutation_mnk=(BLOCK, CHANNELS, BLOCK),
        )
        half_tiled_mma = cute.make_tiled_mma(
            warp.MmaTF32Op((16, 8, 8)),
            (1, 1, 1),
            permutation_mnk=(BLOCK, CHANNELS // 2, BLOCK),
        )
        mma_b_layout_base = cute.make_layout(
            (CHANNELS, BLOCK),
            stride=(BLOCK, 1),
        )
        # TF32 is 32 bits, so M=2 preserves each 16-byte group. S=3
        # XORs channel address bits into the bank-select bits. The s128
        # variant (B=3) turns the producer warp's former 16-way stores
        # into at most 2-way conflicts while retaining the logical
        # [channel, row] view consumed by the MMA partition. B=0 is the
        # identity control.
        mma_b_layout = cute.make_composed_layout(
            cute.make_swizzle(
                self.b_operand_swizzle_bits,
                2,
                3,
            ),
            0,
            mma_b_layout_base,
        )
        self.kernel(
            mGate,
            mQ,
            mK,
            mBeta,
            mDaQk,
            mDaKk,
            mDq,
            mDk,
            mDkt,
            max_log2_span,
            tiled_mma,
            half_tiled_mma,
            mma_b_layout,
        ).launch(
            grid=(blocks, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mGate: cute.Tensor,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mBeta: cute.Tensor,
        mDaQk: cute.Tensor,
        mDaKk: cute.Tensor,
        mDq: cute.Tensor,
        mDk: cute.Tensor,
        mDkt: cute.Tensor,
        max_log2_span: Float32,
        tiled_mma: cute.TiledMma,
        half_tiled_mma: cute.TiledMma,
        mma_b_layout: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        warp_index = tidx // 32
        lane = tidx % 32
        smem = utils.SmemAllocator()
        vector_layout = cute.make_layout((CHANNELS,), stride=(1,))
        tile_layout = cute.make_layout(
            (BLOCK, CHANNELS),
            stride=(CHANNELS, 1),
        )
        matrix_layout = cute.make_layout(
            (BLOCK, BLOCK),
            stride=(BLOCK, 1),
        )
        sReference = smem.allocate_tensor(
            cutlass.Float32,
            vector_layout,
            byte_alignment=16,
        )
        if cutlass.const_expr(self.cache_gate):
            sGate = smem.allocate_tensor(
                cutlass.Float32,
                tile_layout,
                byte_alignment=16,
            )
        sGuard = smem.allocate_tensor(
            cutlass.Int32,
            cute.make_layout((1,), stride=(1,)),
            byte_alignment=4,
        )
        sPositive = smem.allocate_tensor(
            cutlass.Float32,
            tile_layout,
            byte_alignment=16,
        )
        sNegative = smem.allocate_tensor(
            cutlass.Float32,
            tile_layout,
            byte_alignment=16,
        )
        sAqkLower = smem.allocate_tensor(
            cutlass.TFloat32,
            matrix_layout,
            byte_alignment=16,
        )
        sAkkLower = smem.allocate_tensor(
            cutlass.TFloat32,
            matrix_layout,
            byte_alignment=16,
        )
        sAqkUpper = smem.allocate_tensor(
            cutlass.TFloat32,
            matrix_layout,
            byte_alignment=16,
        )
        sAkkUpper = smem.allocate_tensor(
            cutlass.TFloat32,
            matrix_layout,
            byte_alignment=16,
        )
        sKNegative = smem.allocate_tensor(
            cutlass.TFloat32,
            mma_b_layout,
            byte_alignment=16,
        )
        sQPositive = smem.allocate_tensor(
            cutlass.TFloat32,
            mma_b_layout,
            byte_alignment=16,
        )
        sKBetaPositive = smem.allocate_tensor(
            cutlass.TFloat32,
            mma_b_layout,
            byte_alignment=16,
        )
        if cutlass.const_expr(not self.direct_lower_epilogue):
            sDq = smem.allocate_tensor(
                cutlass.Float32,
                tile_layout,
                byte_alignment=16,
            )
            sDk = smem.allocate_tensor(
                cutlass.Float32,
                tile_layout,
                byte_alignment=16,
            )
        if cutlass.const_expr(self.channel_half_warps):
            sDkt = smem.allocate_tensor(
                cutlass.Float32,
                tile_layout,
                byte_alignment=16,
            )
        else:
            sDktQ = smem.allocate_tensor(
                cutlass.Float32,
                tile_layout,
                byte_alignment=16,
            )
            sDktK = smem.allocate_tensor(
                cutlass.Float32,
                tile_layout,
                byte_alignment=16,
            )

        if tidx < CHANNELS:
            channel = tidx
            first = mGate[block, 0, channel].to(Float32)
            if cutlass.const_expr(self.cache_gate):
                sGate[0, channel] = first
            previous = first
            valid = first == first
            for row in cutlass.range_constexpr(1, BLOCK):
                value = mGate[block, row, channel].to(Float32)
                if cutlass.const_expr(self.cache_gate):
                    sGate[row, channel] = value
                valid = valid and value == value and value <= previous
                previous = value
            span = first - previous
            span = span if valid else Float32.inf
            sReference[channel] = (first + previous) * Float32(0.5)
            maximum_span = _warp_max(span)
            if tidx == 0:
                sGuard[0] = Int32(1) if maximum_span <= max_log2_span else Int32(0)
        cute.arch.sync_threads()

        if sGuard[0] == 0:
            for iteration in cutlass.range_constexpr(
                (BLOCK * CHANNELS) // THREADS
            ):
                linear = tidx + iteration * THREADS
                row = linear // CHANNELS
                channel = linear % CHANNELS
                if cutlass.const_expr(self.cache_gate):
                    gate_value = sGate[row, channel].to(Float32)
                else:
                    gate_value = mGate[block, row, channel].to(Float32)
                if cutlass.const_expr(self.explicit_fallback_diagonal):
                    own_k = mK[block, row, channel].to(Float32)
                    own_qk = mDaQk[block, row, row].to(Float32)
                    own_kk = mDaKk[block, row, row].to(Float32)
                    dq = own_qk * own_k
                    dk = own_kk * own_k
                    dkt = (
                        own_qk * mQ[block, row, channel].to(Float32)
                        + own_kk
                        * own_k
                        * mBeta[block, row].to(Float32)
                    )
                else:
                    dq = Float32(0.0)
                    dk = Float32(0.0)
                    dkt = Float32(0.0)
                for partner in cutlass.range_constexpr(BLOCK):
                    if cutlass.const_expr(self.cache_gate):
                        partner_gate = sGate[partner, channel].to(Float32)
                    else:
                        partner_gate = mGate[
                            block,
                            partner,
                            channel,
                        ].to(Float32)
                    if cutlass.const_expr(self.explicit_fallback_diagonal):
                        lower = row > partner
                    else:
                        lower = row >= partner
                    if lower:
                        decay = cute.math.exp2(
                            gate_value - partner_gate,
                            fastmath=True,
                        )
                        partner_k = mK[block, partner, channel].to(Float32)
                        dq += (
                            mDaQk[block, row, partner].to(Float32)
                            * partner_k
                            * decay
                        )
                        dk += (
                            mDaKk[block, row, partner].to(Float32)
                            * partner_k
                            * decay
                        )
                    if cutlass.const_expr(self.explicit_fallback_diagonal):
                        upper = row < partner
                    else:
                        upper = row <= partner
                    if upper:
                        decay = cute.math.exp2(
                            partner_gate - gate_value,
                            fastmath=True,
                        )
                        dkt += (
                            mDaQk[block, row, partner].to(Float32)
                            * mQ[block, partner, channel].to(Float32)
                            * decay
                        )
                        dkt += (
                            mDaKk[block, row, partner].to(Float32)
                            * mK[block, partner, channel].to(Float32)
                            * mBeta[block, partner].to(Float32)
                            * decay
                        )
                mDq[block, row, channel] = dq
                mDk[block, row, channel] = dk
                mDkt[block, row, channel] = dkt
        else:
            for iteration in cutlass.range_constexpr(
                (BLOCK * BLOCK) // THREADS
            ):
                linear = tidx + iteration * THREADS
                row = linear // BLOCK
                partner = linear % BLOCK
                qk = mDaQk[block, row, partner].to(Float32)
                kk = mDaKk[block, row, partner].to(Float32)
                sAqkLower[row, partner] = (
                    qk if row >= partner else Float32(0.0)
                ).to(cutlass.TFloat32)
                sAkkLower[row, partner] = (
                    kk if row >= partner else Float32(0.0)
                ).to(cutlass.TFloat32)
                sAqkUpper[row, partner] = (
                    qk if row <= partner else Float32(0.0)
                ).to(cutlass.TFloat32)
                sAkkUpper[row, partner] = (
                    kk if row <= partner else Float32(0.0)
                ).to(cutlass.TFloat32)
            for iteration in cutlass.range_constexpr(
                (BLOCK * CHANNELS) // THREADS
            ):
                linear = tidx + iteration * THREADS
                row = linear // CHANNELS
                channel = linear % CHANNELS
                reference = sReference[channel].to(Float32)
                if cutlass.const_expr(self.cache_gate):
                    gate_value = sGate[row, channel].to(Float32)
                else:
                    gate_value = mGate[block, row, channel].to(Float32)
                positive = cute.math.exp2(
                    gate_value - reference,
                    fastmath=True,
                )
                if cutlass.const_expr(self.reciprocal_negative):
                    negative = Float32(1.0) / positive
                else:
                    negative = cute.math.exp2(
                        reference - gate_value,
                        fastmath=True,
                    )
                sPositive[row, channel] = positive
                sNegative[row, channel] = negative
                key = mK[block, row, channel].to(Float32)
                sKNegative[channel, row] = (
                    key * negative
                ).to(cutlass.TFloat32)
                sQPositive[channel, row] = (
                    mQ[block, row, channel].to(Float32) * positive
                ).to(cutlass.TFloat32)
                sKBetaPositive[channel, row] = (
                    key * mBeta[block, row].to(Float32) * positive
                ).to(cutlass.TFloat32)
            cute.arch.sync_threads()

            if cutlass.const_expr(self.channel_half_warps):
                if warp_index < 2:
                    channel_half = warp_index
                    _mma_tile_pair(
                        sAqkLower,
                        sAkkLower,
                        cute.local_tile(
                            sKNegative,
                            (CHANNELS // 2, BLOCK),
                            (channel_half, 0),
                        ),
                        cute.local_tile(
                            sDq,
                            (BLOCK, CHANNELS // 2),
                            (0, channel_half),
                        ),
                        cute.local_tile(
                            sDk,
                            (BLOCK, CHANNELS // 2),
                            (0, channel_half),
                        ),
                        lane,
                        half_tiled_mma,
                    )
                else:
                    channel_half = warp_index - 2
                    _mma_tile_sum_pair(
                        sAqkUpper,
                        cute.local_tile(
                            sQPositive,
                            (CHANNELS // 2, BLOCK),
                            (channel_half, 0),
                        ),
                        sAkkUpper,
                        cute.local_tile(
                            sKBetaPositive,
                            (CHANNELS // 2, BLOCK),
                            (channel_half, 0),
                        ),
                        cute.local_tile(
                            sDkt,
                            (BLOCK, CHANNELS // 2),
                            (0, channel_half),
                        ),
                        lane,
                        half_tiled_mma,
                    )
            else:
                if warp_index == 0:
                    if cutlass.const_expr(self.direct_lower_epilogue):
                        _mma_tile_scaled_store(
                            sAqkLower,
                            sKNegative,
                            sPositive,
                            mDq[block, None, None],
                            lane,
                            tiled_mma,
                        )
                    else:
                        _mma_tile(sAqkLower, sKNegative, sDq, lane, tiled_mma)
                if warp_index == 1:
                    if cutlass.const_expr(self.direct_lower_epilogue):
                        _mma_tile_scaled_store(
                            sAkkLower,
                            sKNegative,
                            sPositive,
                            mDk[block, None, None],
                            lane,
                            tiled_mma,
                        )
                    else:
                        _mma_tile(sAkkLower, sKNegative, sDk, lane, tiled_mma)
                if warp_index == 2:
                    _mma_tile(sAqkUpper, sQPositive, sDktQ, lane, tiled_mma)
                if warp_index == 3:
                    _mma_tile(sAkkUpper, sKBetaPositive, sDktK, lane, tiled_mma)
            cute.arch.sync_threads()

            for iteration in cutlass.range_constexpr(
                (BLOCK * CHANNELS) // THREADS
            ):
                linear = tidx + iteration * THREADS
                row = linear // CHANNELS
                channel = linear % CHANNELS
                positive = sPositive[row, channel].to(Float32)
                negative = sNegative[row, channel].to(Float32)
                if cutlass.const_expr(not self.direct_lower_epilogue):
                    mDq[block, row, channel] = (
                        sDq[row, channel].to(Float32) * positive
                    )
                    mDk[block, row, channel] = (
                        sDk[row, channel].to(Float32) * positive
                    )
                if cutlass.const_expr(self.channel_half_warps):
                    mDkt[block, row, channel] = (
                        sDkt[row, channel].to(Float32) * negative
                    )
                else:
                    mDkt[block, row, channel] = (
                        sDktQ[row, channel].to(Float32)
                        + sDktK[row, channel].to(Float32)
                    ) * negative
