"""
Backward stages for the SM90 CuTe DSL KDA lowering.

The implementation is deliberately explicit and uses stable pairwise decay
differences in the local backward.  Major matrix regions are written as FP32
loops so the file is a correctness-first CuTe lowering; replace them with the
TF32/BF16 warp-MMA helpers described in SM90_LOWERING.md after on-H100
validation.

ABI:
  replay_segment  : materialize at most R chunk-start states from one boundary
  scan_segment_bwd: reverse those chunks and emit chunk-level adjoints
  local_bwd       : token-parallel Q/K/gate/beta gradients
  reduce_params   : deterministic sum of per-(B,H) parameter partials
"""

from __future__ import annotations

try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass.cutlass_dsl import Float32, Int32
except Exception:
    cuda = None
    cutlass = None
    cute = None
    utils = None
    Float32 = None
    Int32 = None

from sm90_cute_dsl import CHUNK, DIM, KdaSm90Config

if cute is not None:
    from sm90_cute_dsl import _sigmoid, _softplus, _exp


if cute is not None:

    class ReplaySegmentSm90:
        """Recompute chunk-start states for one checkpoint segment."""

        def __init__(self, config: KdaSm90Config = KdaSm90Config()):
            config.validate()
            self.cfg = config

        @cute.jit
        def __call__(
            self,
            mV: cute.Tensor,
            mKd: cute.Tensor,
            mKc: cute.Tensor,
            mE: cute.Tensor,
            mA: cute.Tensor,
            mBeta: cute.Tensor,
            mBoundaries: cute.Tensor,
            mReplay: cute.Tensor,
            segment_idx: Int32,
            T: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                mV,
                mKd,
                mKc,
                mE,
                mA,
                mBeta,
                mBoundaries,
                mReplay,
                segment_idx,
                T,
            ).launch(
                grid=(mV.shape[0], mV.shape[2], 1),
                block=(256, 1, 1),
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mV: cute.Tensor,
            mKd: cute.Tensor,
            mKc: cute.Tensor,
            mE: cute.Tensor,
            mA: cute.Tensor,
            mBeta: cute.Tensor,
            mBoundaries: cute.Tensor,
            mReplay: cute.Tensor,
            segment_idx: Int32,
            T: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            b, h, _ = cute.arch.block_idx()
            NC = cute.ceil_div(T, CHUNK)
            first_chunk = segment_idx * self.cfg.checkpoint_chunks

            smem = utils.SmemAllocator()
            state_layout = cute.make_layout((DIM, DIM), stride=(DIM, 1))
            tile_layout = cute.make_layout((CHUNK, DIM), stride=(DIM, 1))
            mat_layout = cute.make_layout((CHUNK, CHUNK), stride=(CHUNK, 1))
            sS = smem.allocate_tensor(
                cutlass.Float32, state_layout, byte_alignment=128
            )
            sKd = smem.allocate_tensor(
                cutlass.BFloat16, tile_layout, byte_alignment=128
            )
            sKc = smem.allocate_tensor(
                cutlass.BFloat16, tile_layout, byte_alignment=128
            )
            sV = smem.allocate_tensor(
                cutlass.BFloat16, tile_layout, byte_alignment=128
            )
            sP = smem.allocate_tensor(
                cutlass.Float32, tile_layout, byte_alignment=128
            )
            sU = smem.allocate_tensor(
                cutlass.Float32, tile_layout, byte_alignment=128
            )
            sA = smem.allocate_tensor(
                cutlass.Float32, mat_layout, byte_alignment=16
            )
            sE = smem.allocate_tensor(
                cutlass.Float32,
                cute.make_layout((DIM,), stride=(1,)),
                byte_alignment=16,
            )
            sBeta = smem.allocate_tensor(
                cutlass.Float32,
                cute.make_layout((CHUNK,), stride=(1,)),
                byte_alignment=16,
            )

            # Load the segment boundary.
            for it in cutlass.range_constexpr((DIM * DIM) // 256):
                idx = tidx + 256 * it
                d = idx // DIM
                vv = idx % DIM
                sS[d, vv] = mBoundaries[b, h, segment_idx, d, vv]
            cute.arch.sync_threads()

            for slot in cutlass.range_constexpr(self.cfg.checkpoint_chunks):
                chunk = first_chunk + slot
                if chunk < NC:
                    # Save S before this chunk.
                    for it in cutlass.range_constexpr((DIM * DIM) // 256):
                        idx = tidx + 256 * it
                        d = idx // DIM
                        vv = idx % DIM
                        mReplay[b, h, slot, d, vv] = sS[d, vv]

                    for it in cutlass.range_constexpr(
                        (CHUNK * DIM) // 256
                    ):
                        idx = tidx + 256 * it
                        i = idx // DIM
                        d = idx % DIM
                        tg = chunk * CHUNK + i
                        sKd[i, d] = mKd[b, h, chunk, i, d]
                        sKc[i, d] = mKc[b, h, chunk, i, d]
                        sV[i, d] = (
                            mV[b, tg, h, d]
                            if tg < T
                            else cutlass.BFloat16(0.0)
                        )
                    if tidx < CHUNK * CHUNK:
                        i = tidx // CHUNK
                        j = tidx % CHUNK
                        sA[i, j] = mA[b, h, chunk, i, j].to(Float32)
                    if tidx < CHUNK:
                        sBeta[tidx] = mBeta[b, h, chunk, tidx].to(Float32)
                    if tidx < DIM:
                        sE[tidx] = mE[b, h, chunk, tidx].to(Float32)
                    cute.arch.sync_threads()

                    # P=Kd@S.
                    for it in cutlass.range_constexpr(
                        (CHUNK * DIM) // 256
                    ):
                        idx = tidx + 256 * it
                        i = idx // DIM
                        vv = idx % DIM
                        p = Float32(0.0)
                        for d in cutlass.range_constexpr(DIM):
                            p += sKd[i, d].to(Float32) * sS[d, vv]
                        sP[i, vv] = p
                    cute.arch.sync_threads()

                    # U=A@(beta*(V-P)).
                    for it in cutlass.range_constexpr(
                        (CHUNK * DIM) // 256
                    ):
                        idx = tidx + 256 * it
                        i = idx // DIM
                        vv = idx % DIM
                        u = Float32(0.0)
                        for j in cutlass.range_constexpr(CHUNK):
                            u += sA[i, j] * sBeta[j] * (
                                sV[j, vv].to(Float32) - sP[j, vv]
                            )
                        sU[i, vv] = u
                    cute.arch.sync_threads()

                    # S=E*S+Kc.T@U.
                    for it in cutlass.range_constexpr(
                        (DIM * DIM) // 256
                    ):
                        idx = tidx + 256 * it
                        d = idx // DIM
                        vv = idx % DIM
                        update = Float32(0.0)
                        for i in cutlass.range_constexpr(CHUNK):
                            update += sKc[i, d].to(Float32) * sU[i, vv]
                        sS[d, vv] = sE[d] * sS[d, vv] + update
                    cute.arch.sync_threads()


    class ScanSegmentBwdSm90:
        """Reverse one replayed checkpoint segment and emit chunk adjoints."""

        def __init__(self, config: KdaSm90Config = KdaSm90Config()):
            config.validate()
            self.cfg = config

        @cute.jit
        def __call__(
            self,
            mV: cute.Tensor,
            mGradOut: cute.Tensor,
            mQd: cute.Tensor,
            mKd: cute.Tensor,
            mKc: cute.Tensor,
            mE: cute.Tensor,
            mA: cute.Tensor,
            mM: cute.Tensor,
            mL: cute.Tensor,
            mBeta: cute.Tensor,
            mReplay: cute.Tensor,
            mDState: cute.Tensor,
            mDQd: cute.Tensor,
            mDKd: cute.Tensor,
            mDKc: cute.Tensor,
            mDE: cute.Tensor,
            mDL: cute.Tensor,
            mDM: cute.Tensor,
            mDBeta: cute.Tensor,
            mDV: cute.Tensor,
            segment_idx: Int32,
            T: Int32,
            scale: Float32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                mV,
                mGradOut,
                mQd,
                mKd,
                mKc,
                mE,
                mA,
                mM,
                mL,
                mBeta,
                mReplay,
                mDState,
                mDQd,
                mDKd,
                mDKc,
                mDE,
                mDL,
                mDM,
                mDBeta,
                mDV,
                segment_idx,
                T,
                scale,
            ).launch(
                grid=(mV.shape[0], mV.shape[2], 1),
                block=(256, 1, 1),
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mV: cute.Tensor,
            mGradOut: cute.Tensor,
            mQd: cute.Tensor,
            mKd: cute.Tensor,
            mKc: cute.Tensor,
            mE: cute.Tensor,
            mA: cute.Tensor,
            mM: cute.Tensor,
            mL: cute.Tensor,
            mBeta: cute.Tensor,
            mReplay: cute.Tensor,
            mDState: cute.Tensor,
            mDQd: cute.Tensor,
            mDKd: cute.Tensor,
            mDKc: cute.Tensor,
            mDE: cute.Tensor,
            mDL: cute.Tensor,
            mDM: cute.Tensor,
            mDBeta: cute.Tensor,
            mDV: cute.Tensor,
            segment_idx: Int32,
            T: Int32,
            scale: Float32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            b, h, _ = cute.arch.block_idx()
            NC = cute.ceil_div(T, CHUNK)
            first_chunk = segment_idx * self.cfg.checkpoint_chunks
            remaining = NC - first_chunk
            count = (
                self.cfg.checkpoint_chunks
                if remaining >= self.cfg.checkpoint_chunks
                else remaining
            )

            smem = utils.SmemAllocator()
            state_layout = cute.make_layout((DIM, DIM), stride=(DIM, 1))
            tile_layout = cute.make_layout((CHUNK, DIM), stride=(DIM, 1))
            mat_layout = cute.make_layout((CHUNK, CHUNK), stride=(CHUNK, 1))

            sS = smem.allocate_tensor(
                cutlass.Float32, state_layout, byte_alignment=128
            )
            sDS = smem.allocate_tensor(
                cutlass.Float32, state_layout, byte_alignment=128
            )
            sQd = smem.allocate_tensor(
                cutlass.BFloat16, tile_layout, byte_alignment=128
            )
            sKd = smem.allocate_tensor(
                cutlass.BFloat16, tile_layout, byte_alignment=128
            )
            sKc = smem.allocate_tensor(
                cutlass.BFloat16, tile_layout, byte_alignment=128
            )
            sV = smem.allocate_tensor(
                cutlass.BFloat16, tile_layout, byte_alignment=128
            )
            sP = smem.allocate_tensor(
                cutlass.Float32, tile_layout, byte_alignment=128
            )
            sRhs = smem.allocate_tensor(
                cutlass.Float32, tile_layout, byte_alignment=128
            )
            sU = smem.allocate_tensor(
                cutlass.Float32, tile_layout, byte_alignment=128
            )
            sDQd = smem.allocate_tensor(
                cutlass.Float32, tile_layout, byte_alignment=128
            )
            sDKd = smem.allocate_tensor(
                cutlass.Float32, tile_layout, byte_alignment=128
            )
            sDKc = smem.allocate_tensor(
                cutlass.Float32, tile_layout, byte_alignment=128
            )
            sA = smem.allocate_tensor(
                cutlass.Float32, mat_layout, byte_alignment=16
            )
            sM = smem.allocate_tensor(
                cutlass.Float32, mat_layout, byte_alignment=16
            )
            sL = smem.allocate_tensor(
                cutlass.Float32, mat_layout, byte_alignment=16
            )
            sDA = smem.allocate_tensor(
                cutlass.Float32, mat_layout, byte_alignment=16
            )
            sTmp = smem.allocate_tensor(
                cutlass.Float32, mat_layout, byte_alignment=16
            )
            sDM = smem.allocate_tensor(
                cutlass.Float32, mat_layout, byte_alignment=16
            )
            sE = smem.allocate_tensor(
                cutlass.Float32,
                cute.make_layout((DIM,), stride=(1,)),
                byte_alignment=16,
            )
            sDE = smem.allocate_tensor(
                cutlass.Float32,
                cute.make_layout((DIM,), stride=(1,)),
                byte_alignment=16,
            )
            sBeta = smem.allocate_tensor(
                cutlass.Float32,
                cute.make_layout((CHUNK,), stride=(1,)),
                byte_alignment=16,
            )
            sDBeta = smem.allocate_tensor(
                cutlass.Float32,
                cute.make_layout((CHUNK,), stride=(1,)),
                byte_alignment=16,
            )

            # dS entering the segment is carried in global memory.
            for it in cutlass.range_constexpr((DIM * DIM) // 256):
                idx = tidx + 256 * it
                d = idx // DIM
                vv = idx % DIM
                sDS[d, vv] = mDState[b, h, d, vv]
            cute.arch.sync_threads()

            for rev in cutlass.range_constexpr(self.cfg.checkpoint_chunks):
                if rev < count:
                    slot = count - 1 - rev
                    chunk = first_chunk + slot

                    for it in cutlass.range_constexpr(
                        (DIM * DIM) // 256
                    ):
                        idx = tidx + 256 * it
                        d = idx // DIM
                        vv = idx % DIM
                        sS[d, vv] = mReplay[b, h, slot, d, vv]

                    for it in cutlass.range_constexpr(
                        (CHUNK * DIM) // 256
                    ):
                        idx = tidx + 256 * it
                        i = idx // DIM
                        d = idx % DIM
                        tg = chunk * CHUNK + i
                        sQd[i, d] = mQd[b, h, chunk, i, d]
                        sKd[i, d] = mKd[b, h, chunk, i, d]
                        sKc[i, d] = mKc[b, h, chunk, i, d]
                        sV[i, d] = (
                            mV[b, tg, h, d]
                            if tg < T
                            else cutlass.BFloat16(0.0)
                        )
                    if tidx < CHUNK * CHUNK:
                        i = tidx // CHUNK
                        j = tidx % CHUNK
                        sA[i, j] = mA[b, h, chunk, i, j].to(Float32)
                        sM[i, j] = mM[b, h, chunk, i, j].to(Float32)
                        sL[i, j] = mL[b, h, chunk, i, j].to(Float32)
                    if tidx < CHUNK:
                        sBeta[tidx] = mBeta[b, h, chunk, tidx].to(Float32)
                    if tidx < DIM:
                        sE[tidx] = mE[b, h, chunk, tidx].to(Float32)
                    cute.arch.sync_threads()

                    # Recompute P, rhs, U.
                    for it in cutlass.range_constexpr(
                        (CHUNK * DIM) // 256
                    ):
                        idx = tidx + 256 * it
                        i = idx // DIM
                        vv = idx % DIM
                        p = Float32(0.0)
                        for d in cutlass.range_constexpr(DIM):
                            p += sKd[i, d].to(Float32) * sS[d, vv]
                        sP[i, vv] = p
                        sRhs[i, vv] = sBeta[i] * (
                            sV[i, vv].to(Float32) - p
                        )
                    cute.arch.sync_threads()
                    for it in cutlass.range_constexpr(
                        (CHUNK * DIM) // 256
                    ):
                        idx = tidx + 256 * it
                        i = idx // DIM
                        vv = idx % DIM
                        u = Float32(0.0)
                        for j in cutlass.range_constexpr(CHUNK):
                            u += sA[i, j] * sRhs[j, vv]
                        sU[i, vv] = u
                    cute.arch.sync_threads()

                    # dQd = dZ @ S.T.
                    for it in cutlass.range_constexpr(
                        (CHUNK * DIM) // 256
                    ):
                        idx = tidx + 256 * it
                        i = idx // DIM
                        d = idx % DIM
                        acc = Float32(0.0)
                        tg = chunk * CHUNK + i
                        if tg < T:
                            for vv in cutlass.range_constexpr(DIM):
                                acc += (
                                    mGradOut[b, tg, h, vv].to(Float32)
                                    * scale
                                    * sS[d, vv]
                                )
                        sDQd[i, d] = acc

                    # dM = dZ @ U.T.
                    if tidx < CHUNK * CHUNK:
                        i = tidx // CHUNK
                        j = tidx % CHUNK
                        acc = Float32(0.0)
                        tg = chunk * CHUNK + i
                        if tg < T:
                            for vv in cutlass.range_constexpr(DIM):
                                acc += (
                                    mGradOut[b, tg, h, vv].to(Float32)
                                    * scale
                                    * sU[j, vv]
                                )
                        sDM[i, j] = acc
                        mDM[b, h, chunk, i, j] = acc

                    # dU = M.T @ dZ + Kc @ dS.
                    for it in cutlass.range_constexpr(
                        (CHUNK * DIM) // 256
                    ):
                        idx = tidx + 256 * it
                        i = idx // DIM
                        vv = idx % DIM
                        acc = Float32(0.0)
                        for row in cutlass.range_constexpr(CHUNK):
                            tg = chunk * CHUNK + row
                            if tg < T:
                                acc += (
                                    sM[row, i]
                                    * mGradOut[b, tg, h, vv].to(Float32)
                                    * scale
                                )
                        for d in cutlass.range_constexpr(DIM):
                            acc += sKc[i, d].to(Float32) * sDS[d, vv]
                        sU[i, vv] = acc  # U buffer now holds dU.
                    cute.arch.sync_threads()

                    # dE, dKc.
                    if tidx < DIM:
                        d = tidx
                        acc = Float32(0.0)
                        for vv in cutlass.range_constexpr(DIM):
                            acc += sDS[d, vv] * sS[d, vv]
                        sDE[d] = acc
                    for it in cutlass.range_constexpr(
                        (CHUNK * DIM) // 256
                    ):
                        idx = tidx + 256 * it
                        i = idx // DIM
                        d = idx % DIM
                        acc = Float32(0.0)
                        for vv in cutlass.range_constexpr(DIM):
                            # Recompute forward U_i,v from A@rhs.
                            uf = Float32(0.0)
                            for j in cutlass.range_constexpr(CHUNK):
                                uf += sA[i, j] * sRhs[j, vv]
                            acc += uf * sDS[d, vv]
                        sDKc[i, d] = acc
                    cute.arch.sync_threads()

                    # dS = E*dS + Qd.T@dZ, in-place after all consumers of
                    # incoming dS above have finished.
                    for it in cutlass.range_constexpr(
                        (DIM * DIM) // 256
                    ):
                        idx = tidx + 256 * it
                        d = idx // DIM
                        vv = idx % DIM
                        acc = sE[d] * sDS[d, vv]
                        for i in cutlass.range_constexpr(CHUNK):
                            tg = chunk * CHUNK + i
                            if tg < T:
                                acc += (
                                    sQd[i, d].to(Float32)
                                    * mGradOut[b, tg, h, vv].to(Float32)
                                    * scale
                                )
                        sDS[d, vv] = acc
                    cute.arch.sync_threads()

                    # dA=dU@rhs.T.
                    if tidx < CHUNK * CHUNK:
                        i = tidx // CHUNK
                        j = tidx % CHUNK
                        acc = Float32(0.0)
                        for vv in cutlass.range_constexpr(DIM):
                            acc += sU[i, vv] * sRhs[j, vv]
                        sDA[i, j] = acc
                    cute.arch.sync_threads()

                    # drhs=A.T@dU, overwrite rhs.
                    for it in cutlass.range_constexpr(
                        (CHUNK * DIM) // 256
                    ):
                        idx = tidx + 256 * it
                        i = idx // DIM
                        vv = idx % DIM
                        acc = Float32(0.0)
                        for j in cutlass.range_constexpr(CHUNK):
                            acc += sA[j, i] * sU[j, vv]
                        sRhs[i, vv] = acc
                    cute.arch.sync_threads()

                    # dX=-A.T@dA@A.T.  sTmp=A.T@dA, sDA becomes dX.
                    if tidx < CHUNK * CHUNK:
                        i = tidx // CHUNK
                        j = tidx % CHUNK
                        acc = Float32(0.0)
                        for kk in cutlass.range_constexpr(CHUNK):
                            acc += sA[kk, i] * sDA[kk, j]
                        sTmp[i, j] = acc
                    cute.arch.sync_threads()
                    if tidx < CHUNK * CHUNK:
                        i = tidx // CHUNK
                        j = tidx % CHUNK
                        acc = Float32(0.0)
                        for kk in cutlass.range_constexpr(CHUNK):
                            acc += sTmp[i, kk] * sA[j, kk]
                        sDA[i, j] = -acc
                    cute.arch.sync_threads()

                    # dbeta and dL.
                    if tidx < CHUNK:
                        i = tidx
                        db = Float32(0.0)
                        for j in cutlass.range_constexpr(CHUNK):
                            db += sDA[i, j] * sL[i, j]
                        tg = chunk * CHUNK + i
                        if tg < T:
                            for vv in cutlass.range_constexpr(DIM):
                                db += sRhs[i, vv] * (
                                    sV[i, vv].to(Float32) - sP[i, vv]
                                )
                        sDBeta[i] = db
                        mDBeta[b, h, chunk, i] = db
                    if tidx < CHUNK * CHUNK:
                        i = tidx // CHUNK
                        j = tidx % CHUNK
                        dl = (
                            sBeta[i] * sDA[i, j]
                            if i > j
                            else Float32(0.0)
                        )
                        mDL[b, h, chunk, i, j] = dl
                    cute.arch.sync_threads()

                    # dV, dP, dKd.
                    for it in cutlass.range_constexpr(
                        (CHUNK * DIM) // 256
                    ):
                        idx = tidx + 256 * it
                        i = idx // DIM
                        vv = idx % DIM
                        tg = chunk * CHUNK + i
                        dv = sBeta[i] * sRhs[i, vv]
                        if tg < T:
                            mDV[b, tg, h, vv] = dv
                        sP[i, vv] = -dv  # dP
                    cute.arch.sync_threads()

                    for it in cutlass.range_constexpr(
                        (CHUNK * DIM) // 256
                    ):
                        idx = tidx + 256 * it
                        i = idx // DIM
                        d = idx % DIM
                        acc = Float32(0.0)
                        for vv in cutlass.range_constexpr(DIM):
                            acc += sP[i, vv] * sS[d, vv]
                        sDKd[i, d] = acc
                    cute.arch.sync_threads()

                    # dS += Kd.T@dP.
                    for it in cutlass.range_constexpr(
                        (DIM * DIM) // 256
                    ):
                        idx = tidx + 256 * it
                        d = idx // DIM
                        vv = idx % DIM
                        acc = sDS[d, vv]
                        for i in cutlass.range_constexpr(CHUNK):
                            acc += sKd[i, d].to(Float32) * sP[i, vv]
                        sDS[d, vv] = acc
                    cute.arch.sync_threads()

                    # Emit tile adjoints.
                    for it in cutlass.range_constexpr(
                        (CHUNK * DIM) // 256
                    ):
                        idx = tidx + 256 * it
                        i = idx // DIM
                        d = idx % DIM
                        mDQd[b, h, chunk, i, d] = sDQd[i, d]
                        mDKd[b, h, chunk, i, d] = sDKd[i, d]
                        mDKc[b, h, chunk, i, d] = sDKc[i, d]
                    if tidx < DIM:
                        mDE[b, h, chunk, tidx] = sDE[tidx]
                    cute.arch.sync_threads()

            # Carry dS to the preceding segment.
            for it in cutlass.range_constexpr((DIM * DIM) // 256):
                idx = tidx + 256 * it
                d = idx // DIM
                vv = idx % DIM
                mDState[b, h, d, vv] = sDS[d, vv]


    class LocalBwdSm90:
        """Token-parallel stable local backward for one 16-token chunk."""

        def __init__(self, config: KdaSm90Config = KdaSm90Config()):
            config.validate()
            self.cfg = config

        @cute.jit
        def __call__(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mRawDecay: cute.Tensor,
            mBetaLogits: cute.Tensor,
            mALog: cute.Tensor,
            mDtBias: cute.Tensor,
            mQd: cute.Tensor,
            mKd: cute.Tensor,
            mKc: cute.Tensor,
            mE: cute.Tensor,
            mDQd: cute.Tensor,
            mDKd: cute.Tensor,
            mDKc: cute.Tensor,
            mDE: cute.Tensor,
            mDL: cute.Tensor,
            mDM: cute.Tensor,
            mDBeta: cute.Tensor,
            mDQ: cute.Tensor,
            mDK: cute.Tensor,
            mDRawDecay: cute.Tensor,
            mDBetaLogits: cute.Tensor,
            mPartialALog: cute.Tensor,
            mPartialDtBias: cute.Tensor,
            T: Int32,
            stream: cuda.CUstream,
        ):
            NC = cute.ceil_div(T, CHUNK)
            self.kernel(
                mQ,
                mK,
                mRawDecay,
                mBetaLogits,
                mALog,
                mDtBias,
                mQd,
                mKd,
                mKc,
                mE,
                mDQd,
                mDKd,
                mDKc,
                mDE,
                mDL,
                mDM,
                mDBeta,
                mDQ,
                mDK,
                mDRawDecay,
                mDBetaLogits,
                mPartialALog,
                mPartialDtBias,
                T,
            ).launch(
                grid=(NC, mQ.shape[2], mQ.shape[0]),
                block=(256, 1, 1),
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mRawDecay: cute.Tensor,
            mBetaLogits: cute.Tensor,
            mALog: cute.Tensor,
            mDtBias: cute.Tensor,
            mQd: cute.Tensor,
            mKd: cute.Tensor,
            mKc: cute.Tensor,
            mE: cute.Tensor,
            mDQd: cute.Tensor,
            mDKd: cute.Tensor,
            mDKc: cute.Tensor,
            mDE: cute.Tensor,
            mDL: cute.Tensor,
            mDM: cute.Tensor,
            mDBeta: cute.Tensor,
            mDQ: cute.Tensor,
            mDK: cute.Tensor,
            mDRawDecay: cute.Tensor,
            mDBetaLogits: cute.Tensor,
            mPartialALog: cute.Tensor,
            mPartialDtBias: cute.Tensor,
            T: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            chunk, h, b = cute.arch.block_idx()
            token = tidx // 16
            lane = tidx % 16
            tg = chunk * CHUNK + token
            valid = tg < T
            remaining_tokens = T - chunk * CHUNK
            valid_count = (
                CHUNK if remaining_tokens >= CHUNK else remaining_tokens
            )
            last_valid = valid_count - 1

            smem = utils.SmemAllocator()
            row_layout = cute.make_layout((CHUNK, DIM), stride=(DIM, 1))
            mat_layout = cute.make_layout((CHUNK, CHUNK), stride=(CHUNK, 1))
            sQ = smem.allocate_tensor(
                cutlass.Float32, row_layout, byte_alignment=16
            )
            sK = smem.allocate_tensor(
                cutlass.Float32, row_layout, byte_alignment=16
            )
            sG = smem.allocate_tensor(
                cutlass.Float32, row_layout, byte_alignment=16
            )
            sDQn = smem.allocate_tensor(
                cutlass.Float32, row_layout, byte_alignment=16
            )
            sDKn = smem.allocate_tensor(
                cutlass.Float32, row_layout, byte_alignment=16
            )
            sDg = smem.allocate_tensor(
                cutlass.Float32, row_layout, byte_alignment=16
            )
            sReduceQ = smem.allocate_tensor(
                cutlass.Float32,
                cute.make_layout((CHUNK, 16), stride=(16, 1)),
                byte_alignment=16,
            )
            sReduceK = smem.allocate_tensor(
                cutlass.Float32,
                cute.make_layout((CHUNK, 16), stride=(16, 1)),
                byte_alignment=16,
            )
            sDL = smem.allocate_tensor(
                cutlass.Float32, mat_layout, byte_alignment=16
            )
            sDM = smem.allocate_tensor(
                cutlass.Float32, mat_layout, byte_alignment=16
            )

            qss = Float32(0.0)
            kss = Float32(0.0)
            for vec in cutlass.range_constexpr(DIM // 16):
                d = lane + 16 * vec
                qv = mQ[b, tg, h, d].to(Float32) if valid else Float32(0.0)
                kv = mK[b, tg, h, d].to(Float32) if valid else Float32(0.0)
                sQ[token, d] = qv
                sK[token, d] = kv
                qss += qv * qv
                kss += kv * kv
            sReduceQ[token, lane] = qss
            sReduceK[token, lane] = kss
            cute.arch.sync_threads()
            if lane == 0:
                sq = Float32(0.0)
                sk = Float32(0.0)
                for i in cutlass.range_constexpr(16):
                    sq += sReduceQ[token, i]
                    sk += sReduceK[token, i]
                # Keep both the norm and reciprocal.
                nq = cute.math.sqrt(sq, fastmath=True)
                nk = cute.math.sqrt(sk, fastmath=True)
                sReduceQ[token, 0] = nq
                sReduceQ[token, 1] = Float32(1.0) / max(
                    nq, Float32(self.cfg.norm_eps)
                )
                sReduceK[token, 0] = nk
                sReduceK[token, 1] = Float32(1.0) / max(
                    nk, Float32(self.cfg.norm_eps)
                )
            cute.arch.sync_threads()
            for vec in cutlass.range_constexpr(DIM // 16):
                d = lane + 16 * vec
                sQ[token, d] *= sReduceQ[token, 1]
                sK[token, d] *= sReduceK[token, 1]
            cute.arch.sync_threads()

            # Cumulative exact gate.
            if tidx < DIM:
                d = tidx
                running = Float32(0.0)
                a = _exp(mALog[h].to(Float32))
                for i in cutlass.range_constexpr(CHUNK):
                    ti = chunk * CHUNK + i
                    z = (
                        mRawDecay[b, ti, h, d].to(Float32)
                        + mDtBias[h, d].to(Float32)
                        if ti < T
                        else Float32(0.0)
                    )
                    running += -a * _softplus(z) if ti < T else Float32(0.0)
                    sG[i, d] = running
            if tidx < CHUNK * CHUNK:
                i = tidx // CHUNK
                j = tidx % CHUNK
                sDL[i, j] = mDL[b, h, chunk, i, j].to(Float32)
                sDM[i, j] = mDM[b, h, chunk, i, j].to(Float32)
            cute.arch.sync_threads()

            # Direct stable pairwise derivative.  Each (token,channel) thread
            # loops over 16 partners; all exponents are g_later-g_earlier <= 0.
            for it in cutlass.range_constexpr((CHUNK * DIM) // 256):
                idx = tidx + 256 * it
                i = idx // DIM
                d = idx % DIM
                ti = chunk * CHUNK + i
                if ti < T:
                    G = _exp(sG[i, d])
                    qn = sQ[i, d]
                    kn = sK[i, d]
                    dqn = mDQd[b, h, chunk, i, d].to(Float32) * G
                    dkn = mDKd[b, h, chunk, i, d].to(Float32) * G
                    dg = (
                        mDQd[b, h, chunk, i, d].to(Float32)
                        * mQd[b, h, chunk, i, d].to(Float32)
                        + mDKd[b, h, chunk, i, d].to(Float32)
                        * mKd[b, h, chunk, i, d].to(Float32)
                    )

                    # Kc_i = k_i * exp(g_last-g_i).
                    hdec = _exp(sG[CHUNK - 1, d] - sG[i, d])
                    dkc = mDKc[b, h, chunk, i, d].to(Float32)
                    dkn += dkc * hdec
                    kc_dot = dkc * mKc[b, h, chunk, i, d].to(Float32)
                    dg -= kc_dot

                    # Row-i pairwise terms.
                    for j in cutlass.range_constexpr(CHUNK):
                        tj = chunk * CHUNK + j
                        if j <= i and tj < T:
                            decay = _exp(sG[i, d] - sG[j, d])
                            dm = sDM[i, j]
                            term_m = dm * qn * sK[j, d] * decay
                            dqn += dm * sK[j, d] * decay
                            dg += term_m
                            if j < i:
                                dl = sDL[i, j]
                                term_l = dl * kn * sK[j, d] * decay
                                dkn += dl * sK[j, d] * decay
                                dg += term_l

                    # Column-i pairwise terms.
                    for row in cutlass.range_constexpr(CHUNK):
                        tr = chunk * CHUNK + row
                        if row >= i and tr < T:
                            decay = _exp(sG[row, d] - sG[i, d])
                            dm = sDM[row, i]
                            term_m = dm * sQ[row, d] * kn * decay
                            dkn += dm * sQ[row, d] * decay
                            dg -= term_m
                            if row > i:
                                dl = sDL[row, i]
                                term_l = dl * sK[row, d] * kn * decay
                                dkn += dl * sK[row, d] * decay
                                dg -= term_l

                    # Kc contributes + to g_last.
                    if i == last_valid:
                        extra = mDE[b, h, chunk, d].to(Float32) * mE[
                            b, h, chunk, d
                        ].to(Float32)
                        for row in cutlass.range_constexpr(CHUNK):
                            extra += (
                                mDKc[b, h, chunk, row, d].to(Float32)
                                * mKc[b, h, chunk, row, d].to(Float32)
                            )
                        dg += extra

                    sDQn[i, d] = dqn
                    sDKn[i, d] = dkn
                    sDg[i, d] = dg
                else:
                    sDQn[i, d] = Float32(0.0)
                    sDKn[i, d] = Float32(0.0)
                    sDg[i, d] = Float32(0.0)
            cute.arch.sync_threads()

            # Reverse cumulative sum and exact gate chain; one thread/channel.
            if tidx < DIM:
                d = tidx
                suffix = Float32(0.0)
                partial_a = Float32(0.0)
                partial_dt = Float32(0.0)
                a = _exp(mALog[h].to(Float32))
                for rev in cutlass.range_constexpr(CHUNK):
                    i = CHUNK - 1 - rev
                    ti = chunk * CHUNK + i
                    suffix += sDg[i, d]
                    if ti < T:
                        z = (
                            mRawDecay[b, ti, h, d].to(Float32)
                            + mDtBias[h, d].to(Float32)
                        )
                        log_alpha = -a * _softplus(z)
                        dz = suffix * (-a) * _sigmoid(z)
                        mDRawDecay[b, ti, h, d] = dz.to(
                            mDRawDecay.element_type
                        )
                        partial_dt += dz
                        partial_a += suffix * log_alpha
                # Atomics are avoided: each chunk writes a partial.  The final
                # reduction kernel sums chunk partials.
                mPartialDtBias[b, h, chunk, d] = partial_dt
            cute.arch.sync_threads()
            if tidx == 0:
                total_a = Float32(0.0)
                a = _exp(mALog[h].to(Float32))
                for dd in cutlass.range_constexpr(DIM):
                    # Recompute the per-channel A_log contribution to avoid a
                    # second shared buffer.
                    suffix2 = Float32(0.0)
                    for rev in cutlass.range_constexpr(CHUNK):
                        ii = CHUNK - 1 - rev
                        suffix2 += sDg[ii, dd]
                        tii = chunk * CHUNK + ii
                        if tii < T:
                            zz = (
                                mRawDecay[b, tii, h, dd].to(Float32)
                                + mDtBias[h, dd].to(Float32)
                            )
                            total_a += suffix2 * (-a * _softplus(zz))
                mPartialALog[b, h, chunk] = total_a
            cute.arch.sync_threads()

            # L2 normalization backward and beta sigmoid chain.
            for it in cutlass.range_constexpr((CHUNK * DIM) // 256):
                idx = tidx + 256 * it
                i = idx // DIM
                d = idx % DIM
                ti = chunk * CHUNK + i
                if ti < T:
                    qproj = Float32(0.0)
                    kproj = Float32(0.0)
                    for dd in cutlass.range_constexpr(DIM):
                        qproj += sDQn[i, dd] * sQ[i, dd]
                        kproj += sDKn[i, dd] * sK[i, dd]
                    qnorm = sReduceQ[i, 0]
                    knorm = sReduceK[i, 0]
                    dq = (
                        (sDQn[i, d] - sQ[i, d] * qproj)
                        / max(qnorm, Float32(self.cfg.norm_eps))
                        if qnorm > Float32(self.cfg.norm_eps)
                        else sDQn[i, d] / Float32(self.cfg.norm_eps)
                    )
                    dk = (
                        (sDKn[i, d] - sK[i, d] * kproj)
                        / max(knorm, Float32(self.cfg.norm_eps))
                        if knorm > Float32(self.cfg.norm_eps)
                        else sDKn[i, d] / Float32(self.cfg.norm_eps)
                    )
                    mDQ[b, ti, h, d] = dq.to(mDQ.element_type)
                    mDK[b, ti, h, d] = dk.to(mDK.element_type)

            if tidx < CHUNK:
                i = tidx
                ti = chunk * CHUNK + i
                if ti < T:
                    beta = _sigmoid(mBetaLogits[b, ti, h].to(Float32))
                    db = mDBeta[b, h, chunk, i].to(Float32)
                    mDBetaLogits[b, ti, h] = (
                        db * beta * (Float32(1.0) - beta)
                    ).to(mDBetaLogits.element_type)


    class ReduceParamsSm90:
        """Reduce chunk partials to dA_log[H] and ddt_bias[H,D]."""

        @cute.jit
        def __call__(
            self,
            mPartialALog: cute.Tensor,
            mPartialDtBias: cute.Tensor,
            mDALog: cute.Tensor,
            mDDtBias: cute.Tensor,
            stream: cuda.CUstream,
        ):
            self.kernel(
                mPartialALog,
                mPartialDtBias,
                mDALog,
                mDDtBias,
            ).launch(
                grid=(mPartialALog.shape[1], 1, 1),
                block=(256, 1, 1),
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mPartialALog: cute.Tensor,
            mPartialDtBias: cute.Tensor,
            mDALog: cute.Tensor,
            mDDtBias: cute.Tensor,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            h, _, _ = cute.arch.block_idx()
            B = mPartialALog.shape[0]
            NC = mPartialALog.shape[2]

            if tidx < DIM:
                d = tidx
                acc_dt = Float32(0.0)
                for b in range(B):
                    for c in range(NC):
                        acc_dt += mPartialDtBias[b, h, c, d].to(Float32)
                mDDtBias[h, d] = acc_dt.to(mDDtBias.element_type)

            if tidx == 0:
                acc_a = Float32(0.0)
                for b in range(B):
                    for c in range(NC):
                        acc_a += mPartialALog[b, h, c].to(Float32)
                mDALog[h] = acc_a.to(mDALog.element_type)


else:

    class ReplaySegmentSm90:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("CUTLASS CuTe DSL is not installed")

    class ScanSegmentBwdSm90(ReplaySegmentSm90):
        pass

    class LocalBwdSm90(ReplaySegmentSm90):
        pass

    class ReduceParamsSm90(ReplaySegmentSm90):
        pass
