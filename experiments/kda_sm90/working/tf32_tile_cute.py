"""CuTe TF32 warp-MMA building block for guarded KDA diagonal products.

This is an isolated 16x16 @ 16x32 kernel, not a training backend. It validates
the exact warp-MMA layout and accumulator epilogue needed by the complete
guarded KDA backward tile.
"""

from __future__ import annotations

from typing import Final

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.nvgpu import warp
from cutlass.cutlass_dsl import Float32, Int32

M: Final[int] = 16
N: Final[int] = 32
K: Final[int] = 16
THREADS: Final[int] = 32


class Tf32TileSm90:
    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        scale_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        blocks: Int32,
        stream: cuda.CUstream,
    ):
        mA = cute.make_tensor(
            a_ptr,
            cute.make_layout((blocks, M, K), stride=(M * K, K, 1)),
        )
        mB = cute.make_tensor(
            b_ptr,
            cute.make_layout((blocks, K, N), stride=(K * N, N, 1)),
        )
        mScale = cute.make_tensor(
            scale_ptr,
            cute.make_layout((blocks, M, N), stride=(M * N, N, 1)),
        )
        mOutput = cute.make_tensor(
            output_ptr,
            cute.make_layout((blocks, M, N), stride=(M * N, N, 1)),
        )
        tiled_mma = cute.make_tiled_mma(
            warp.MmaTF32Op((16, 8, 8)),
            (1, 1, 1),
            permutation_mnk=(M, N, K),
        )
        self.kernel(mA, mB, mScale, mOutput, tiled_mma).launch(
            grid=(blocks, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mScale: cute.Tensor,
        mOutput: cute.Tensor,
        tiled_mma: cute.TiledMma,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        smem = utils.SmemAllocator()
        sA = smem.allocate_tensor(
            cutlass.TFloat32,
            cute.make_layout((M, K), stride=(K, 1)),
            byte_alignment=16,
        )
        # MMA B is logically [N,K] even though the input is conventional [K,N].
        sB = smem.allocate_tensor(
            cutlass.TFloat32,
            cute.make_layout((N, K), stride=(K, 1)),
            byte_alignment=16,
        )
        sC = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((M, N), stride=(N, 1)),
            byte_alignment=16,
        )
        for iteration in cutlass.range_constexpr((M * K) // THREADS):
            linear = tidx + iteration * THREADS
            row = linear // K
            inner = linear % K
            sA[row, inner] = mA[block, row, inner].to(cutlass.TFloat32)
        for iteration in cutlass.range_constexpr((K * N) // THREADS):
            linear = tidx + iteration * THREADS
            inner = linear // N
            column = linear % N
            sB[column, inner] = mB[block, inner, column].to(cutlass.TFloat32)
        cute.arch.sync_threads()

        thr_mma = tiled_mma.get_slice(tidx)
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
        cute.arch.sync_threads()

        for iteration in cutlass.range_constexpr((M * N) // THREADS):
            linear = tidx + iteration * THREADS
            row = linear // N
            column = linear % N
            mOutput[block, row, column] = (
                sC[row, column].to(Float32)
                * mScale[block, row, column].to(Float32)
            )
