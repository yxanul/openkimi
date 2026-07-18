"""
SM90 CuTe DSL lowering draft for KDA training.

Target: CUTLASS 4.5-era CuTe DSL, SM90a, BF16 inputs, D=V=128,
C=16, FP32 state and gradients.

This file intentionally keeps the mathematical stages explicit.  The direct
FP32 loops are the strict numerical path.  The marked GEMM regions are the
places to substitute the warp-MMA/TF32 helper from the CUTLASS tensorop GEMM
example after compiling on H100.

The file is Python-syntax checked in the delivered environment.  It has not
been JIT-compiled here because CUTLASS CuTe DSL and an SM90 GPU are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass.cutlass_dsl import Float32, Int32
except Exception:  # Importing the design on a non-CUDA host is allowed.
    cuda = None
    cutlass = None
    cute = None
    utils = None
    Float32 = None
    Int32 = None


CHUNK: Final[int] = 16
DIM: Final[int] = 128
THREADS_PREPARE: Final[int] = 256
THREADS_SCAN: Final[int] = 256


@dataclass(frozen=True)
class KdaSm90Config:
    chunk_size: int = CHUNK
    dim: int = DIM
    checkpoint_chunks: int = 8
    norm_eps: float = 1e-6
    state_math: str = "strict_fp32"  # or "tf32_tensorcore"

    def validate(self) -> None:
        if self.chunk_size != CHUNK:
            raise ValueError("the first specialization requires chunk_size=16")
        if self.dim != DIM:
            raise ValueError("the first specialization requires D=V=128")
        if self.checkpoint_chunks <= 0:
            raise ValueError("checkpoint_chunks must be positive")
        if self.state_math not in {"strict_fp32", "tf32_tensorcore"}:
            raise ValueError("unknown state_math")


@dataclass(frozen=True)
class WorkspaceShapes:
    B: int
    T: int
    H: int
    checkpoint_chunks: int = 8

    @property
    def NC(self) -> int:
        return (self.T + CHUNK - 1) // CHUNK

    @property
    def NS(self) -> int:
        return (self.NC + self.checkpoint_chunks - 1) // self.checkpoint_chunks

    def as_dict(self) -> dict[str, tuple[int, ...]]:
        # Layout order is chosen so a (chunk,head,batch) CTA sees a contiguous
        # token x channel tile.
        return {
            "qd": (self.B, self.H, self.NC, CHUNK, DIM),
            "kd": (self.B, self.H, self.NC, CHUNK, DIM),
            "kc": (self.B, self.H, self.NC, CHUNK, DIM),
            "e": (self.B, self.H, self.NC, DIM),
            "ainv": (self.B, self.H, self.NC, CHUNK, CHUNK),
            "mqk": (self.B, self.H, self.NC, CHUNK, CHUNK),
            "l": (self.B, self.H, self.NC, CHUNK, CHUNK),
            "beta": (self.B, self.H, self.NC, CHUNK),
            # Boundary 0 is the zero initial state.
            "boundaries": (self.B, self.H, self.NS + 1, DIM, DIM),
            # Reused by the reverse persistent CTA; independent of T.
            "replay": (
                self.B,
                self.H,
                self.checkpoint_chunks,
                DIM,
                DIM,
            ),
        }


if cute is not None:

    @cute.jit
    def _sigmoid(x: Float32) -> Float32:
        # Stable enough for the activation path and easy to replace with
        # tanh.approx.f32 after SASS inspection.
        return Float32(1.0) / (Float32(1.0) + cute.math.exp(-x, fastmath=True))


    @cute.jit
    def _softplus(x: Float32) -> Float32:
        # max(x,0) + log(1 + exp(-abs(x))).
        ax = x if x >= Float32(0.0) else -x
        m = x if x >= Float32(0.0) else Float32(0.0)
        return m + cute.math.log(
            Float32(1.0) + cute.math.exp(-ax, fastmath=True),
            fastmath=True,
        )


    @cute.jit
    def _exp(x: Float32) -> Float32:
        return cute.math.exp(x, fastmath=True)


    class KdaPrepareFwdSm90:
        """Token-parallel preprocessing and 16x16 local algebra."""

        def __init__(self, config: KdaSm90Config = KdaSm90Config()):
            config.validate()
            self.cfg = config

        @cute.jit
        def __call__(
            self,
            q_ptr: cute.Pointer,
            k_ptr: cute.Pointer,
            raw_decay_ptr: cute.Pointer,
            beta_logits_ptr: cute.Pointer,
            a_log_ptr: cute.Pointer,
            dt_bias_ptr: cute.Pointer,
            qd_ptr: cute.Pointer,
            kd_ptr: cute.Pointer,
            kc_ptr: cute.Pointer,
            e_ptr: cute.Pointer,
            ainv_ptr: cute.Pointer,
            mqk_ptr: cute.Pointer,
            l_ptr: cute.Pointer,
            beta_ptr: cute.Pointer,
            B: Int32,
            T: Int32,
            H: Int32,
            stream: cuda.CUstream,
        ):
            bf16 = cutlass.BFloat16
            f32 = cutlass.Float32
            NC = cute.ceil_div(T, CHUNK)

            mQ = cute.make_tensor(
                q_ptr,
                cute.make_layout(
                    (B, T, H, DIM),
                    stride=(T * H * DIM, H * DIM, DIM, 1),
                ),
            )
            mK = cute.make_tensor(k_ptr, mQ.layout)
            mG = cute.make_tensor(raw_decay_ptr, mQ.layout)
            mBetaLogits = cute.make_tensor(
                beta_logits_ptr,
                cute.make_layout(
                    (B, T, H), stride=(T * H, H, 1)
                ),
            )
            mALog = cute.make_tensor(
                a_log_ptr, cute.make_layout((H,), stride=(1,))
            )
            mDt = cute.make_tensor(
                dt_bias_ptr,
                cute.make_layout((H, DIM), stride=(DIM, 1)),
            )

            tile5 = cute.make_layout(
                (B, H, NC, CHUNK, DIM),
                stride=(
                    H * NC * CHUNK * DIM,
                    NC * CHUNK * DIM,
                    CHUNK * DIM,
                    DIM,
                    1,
                ),
            )
            mQd = cute.make_tensor(qd_ptr, tile5)
            mKd = cute.make_tensor(kd_ptr, tile5)
            mKc = cute.make_tensor(kc_ptr, tile5)
            mE = cute.make_tensor(
                e_ptr,
                cute.make_layout(
                    (B, H, NC, DIM),
                    stride=(H * NC * DIM, NC * DIM, DIM, 1),
                ),
            )
            mat = cute.make_layout(
                (B, H, NC, CHUNK, CHUNK),
                stride=(
                    H * NC * CHUNK * CHUNK,
                    NC * CHUNK * CHUNK,
                    CHUNK * CHUNK,
                    CHUNK,
                    1,
                ),
            )
            mAinv = cute.make_tensor(ainv_ptr, mat)
            mMqk = cute.make_tensor(mqk_ptr, mat)
            mLout = cute.make_tensor(l_ptr, mat)
            mBeta = cute.make_tensor(
                beta_ptr,
                cute.make_layout(
                    (B, H, NC, CHUNK),
                    stride=(H * NC * CHUNK, NC * CHUNK, CHUNK, 1),
                ),
            )

            self.kernel(
                mQ,
                mK,
                mG,
                mBetaLogits,
                mALog,
                mDt,
                mQd,
                mKd,
                mKc,
                mE,
                mAinv,
                mMqk,
                mLout,
                mBeta,
                T,
            ).launch(
                grid=(NC, H, B),
                block=(THREADS_PREPARE, 1, 1),
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
            mAinv: cute.Tensor,
            mMqk: cute.Tensor,
            mLout: cute.Tensor,
            mBeta: cute.Tensor,
            T: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            chunk_idx, head_idx, batch_idx = cute.arch.block_idx()
            token = tidx // 16
            lane = tidx % 16
            token_global = chunk_idx * CHUNK + token
            token_valid = token_global < T

            smem = utils.SmemAllocator()
            row_layout = cute.make_layout((CHUNK, DIM), stride=(DIM, 1))
            mat_layout = cute.make_layout((CHUNK, CHUNK), stride=(CHUNK, 1))

            sQ = smem.allocate_tensor(
                cutlass.Float32, row_layout, byte_alignment=16
            )
            sK = smem.allocate_tensor(
                cutlass.Float32, row_layout, byte_alignment=16
            )
            sCumG = smem.allocate_tensor(
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
            sBeta = smem.allocate_tensor(
                cutlass.Float32,
                cute.make_layout((CHUNK,), stride=(1,)),
                byte_alignment=16,
            )
            sL = smem.allocate_tensor(
                cutlass.Float32, mat_layout, byte_alignment=16
            )
            sM = smem.allocate_tensor(
                cutlass.Float32, mat_layout, byte_alignment=16
            )
            sA = smem.allocate_tensor(
                cutlass.Float32, mat_layout, byte_alignment=16
            )

            # Sixteen 16-thread groups load and normalize Q/K.
            qss = Float32(0.0)
            kss = Float32(0.0)
            for vec in cutlass.range_constexpr(DIM // 16):
                d = lane + 16 * vec
                qv = (
                    mQ[batch_idx, token_global, head_idx, d].to(Float32)
                    if token_valid
                    else Float32(0.0)
                )
                kv = (
                    mK[batch_idx, token_global, head_idx, d].to(Float32)
                    if token_valid
                    else Float32(0.0)
                )
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
                sReduceQ[token, 0] = cute.math.rsqrt(
                    max(sq, Float32(self.cfg.norm_eps**2)),
                    fastmath=True,
                )
                sReduceK[token, 0] = cute.math.rsqrt(
                    max(sk, Float32(self.cfg.norm_eps**2)),
                    fastmath=True,
                )
                sBeta[token] = (
                    _sigmoid(
                        mBetaLogits[
                            batch_idx, token_global, head_idx
                        ].to(Float32)
                    )
                    if token_valid
                    else Float32(0.0)
                )
            cute.arch.sync_threads()

            qinv = sReduceQ[token, 0]
            kinv = sReduceK[token, 0]
            for vec in cutlass.range_constexpr(DIM // 16):
                d = lane + 16 * vec
                sQ[token, d] = sQ[token, d] * qinv
                sK[token, d] = sK[token, d] * kinv
            cute.arch.sync_threads()

            # One thread per channel computes exact gate activation and the
            # 16-token cumulative log decay in FP32.
            if tidx < DIM:
                d = tidx
                running = Float32(0.0)
                a = _exp(mALog[head_idx].to(Float32))
                for i in cutlass.range_constexpr(CHUNK):
                    tg = chunk_idx * CHUNK + i
                    z = (
                        mRawDecay[batch_idx, tg, head_idx, d].to(Float32)
                        + mDtBias[head_idx, d].to(Float32)
                        if tg < T
                        else Float32(0.0)
                    )
                    log_alpha = -a * _softplus(z) if tg < T else Float32(0.0)
                    running += log_alpha
                    sCumG[i, d] = running
                mE[batch_idx, head_idx, chunk_idx, d] = _exp(running)
            cute.arch.sync_threads()

            # Qd/Kd/Kc and activated beta workspace.
            linear = tidx
            for it in cutlass.range_constexpr((CHUNK * DIM) // THREADS_PREPARE):
                idx = linear + it * THREADS_PREPARE
                i = idx // DIM
                d = idx % DIM
                tg = chunk_idx * CHUNK + i
                if tg < T:
                    G = _exp(sCumG[i, d])
                    E = _exp(sCumG[CHUNK - 1, d])
                    mQd[batch_idx, head_idx, chunk_idx, i, d] = (
                        sQ[i, d] * G
                    ).to(mQd.element_type)
                    mKd[batch_idx, head_idx, chunk_idx, i, d] = (
                        sK[i, d] * G
                    ).to(mKd.element_type)
                    mKc[batch_idx, head_idx, chunk_idx, i, d] = (
                        sK[i, d] * _exp(
                            sCumG[CHUNK - 1, d] - sCumG[i, d]
                        )
                    ).to(mKc.element_type)
                else:
                    mQd[batch_idx, head_idx, chunk_idx, i, d] = 0
                    mKd[batch_idx, head_idx, chunk_idx, i, d] = 0
                    mKc[batch_idx, head_idx, chunk_idx, i, d] = 0
            if tidx < CHUNK:
                mBeta[batch_idx, head_idx, chunk_idx, tidx] = sBeta[
                    tidx
                ].to(mBeta.element_type)
            cute.arch.sync_threads()

            # Exact lower-triangular pairwise interactions.  The exponent
            # sCumG[i]-sCumG[j] is non-positive for i>=j, avoiding overflow.
            i = tidx // CHUNK
            j = tidx % CHUNK
            tg_i = chunk_idx * CHUNK + i
            tg_j = chunk_idx * CHUNK + j
            lval = Float32(0.0)
            mval = Float32(0.0)
            if i >= j and tg_i < T and tg_j < T:
                for d in cutlass.range_constexpr(DIM):
                    decay = _exp(sCumG[i, d] - sCumG[j, d])
                    mval += sQ[i, d] * sK[j, d] * decay
                    if i > j:
                        lval += sK[i, d] * sK[j, d] * decay
            sL[i, j] = lval
            sM[i, j] = mval
            sA[i, j] = Float32(1.0) if i == j else Float32(0.0)
            cute.arch.sync_threads()

            # Invert X=I+diag(beta)L by unit-lower-triangular forward
            # substitution.  Sixteen lanes compute one inverse row.
            inv_lane = tidx % CHUNK
            inv_warp = tidx // CHUNK
            if inv_warp == 0:
                for row in cutlass.range_constexpr(1, CHUNK):
                    if inv_lane < row:
                        acc = Float32(0.0)
                        for kk in cutlass.range_constexpr(CHUNK):
                            if kk >= inv_lane and kk < row:
                                acc += (
                                    sBeta[row]
                                    * sL[row, kk]
                                    * sA[kk, inv_lane]
                                )
                        sA[row, inv_lane] = -acc
                    cute.arch.sync_threads()
            else:
                # All threads must execute matching barriers.
                for _ in cutlass.range_constexpr(1, CHUNK):
                    cute.arch.sync_threads()

            if tidx < CHUNK * CHUNK:
                ii = tidx // CHUNK
                jj = tidx % CHUNK
                mAinv[
                    batch_idx, head_idx, chunk_idx, ii, jj
                ] = sA[ii, jj].to(mAinv.element_type)
                mMqk[
                    batch_idx, head_idx, chunk_idx, ii, jj
                ] = sM[ii, jj].to(mMqk.element_type)
                mLout[
                    batch_idx, head_idx, chunk_idx, ii, jj
                ] = sL[ii, jj].to(mLout.element_type)


    class KdaScanFwdSm90:
        """Persistent per-head FP32 recurrence.

        This strict path uses explicit FP32 FMA loops.  The tensor-core lowering
        replaces the three marked GEMM regions with TF32 mma.sync while keeping
        the state and accumulators FP32.
        """

        def __init__(self, config: KdaSm90Config = KdaSm90Config()):
            config.validate()
            self.cfg = config

        @cute.jit
        def __call__(
            self,
            v_ptr: cute.Pointer,
            qd_ptr: cute.Pointer,
            kd_ptr: cute.Pointer,
            kc_ptr: cute.Pointer,
            e_ptr: cute.Pointer,
            ainv_ptr: cute.Pointer,
            mqk_ptr: cute.Pointer,
            l_ptr: cute.Pointer,
            beta_ptr: cute.Pointer,
            out_ptr: cute.Pointer,
            boundaries_ptr: cute.Pointer,
            B: Int32,
            T: Int32,
            H: Int32,
            scale: Float32,
            stream: cuda.CUstream,
        ):
            NC = cute.ceil_div(T, CHUNK)
            NS = cute.ceil_div(NC, self.cfg.checkpoint_chunks)
            in_layout = cute.make_layout(
                (B, T, H, DIM),
                stride=(T * H * DIM, H * DIM, DIM, 1),
            )
            mV = cute.make_tensor(v_ptr, in_layout)
            mO = cute.make_tensor(out_ptr, in_layout)

            tile5 = cute.make_layout(
                (B, H, NC, CHUNK, DIM),
                stride=(
                    H * NC * CHUNK * DIM,
                    NC * CHUNK * DIM,
                    CHUNK * DIM,
                    DIM,
                    1,
                ),
            )
            mQd = cute.make_tensor(qd_ptr, tile5)
            mKd = cute.make_tensor(kd_ptr, tile5)
            mKc = cute.make_tensor(kc_ptr, tile5)
            mE = cute.make_tensor(
                e_ptr,
                cute.make_layout(
                    (B, H, NC, DIM),
                    stride=(H * NC * DIM, NC * DIM, DIM, 1),
                ),
            )
            mat = cute.make_layout(
                (B, H, NC, CHUNK, CHUNK),
                stride=(
                    H * NC * CHUNK * CHUNK,
                    NC * CHUNK * CHUNK,
                    CHUNK * CHUNK,
                    CHUNK,
                    1,
                ),
            )
            mA = cute.make_tensor(ainv_ptr, mat)
            mM = cute.make_tensor(mqk_ptr, mat)
            mBeta = cute.make_tensor(
                beta_ptr,
                cute.make_layout(
                    (B, H, NC, CHUNK),
                    stride=(H * NC * CHUNK, NC * CHUNK, CHUNK, 1),
                ),
            )
            mBoundaries = cute.make_tensor(
                boundaries_ptr,
                cute.make_layout(
                    (B, H, NS + 1, DIM, DIM),
                    stride=(
                        H * (NS + 1) * DIM * DIM,
                        (NS + 1) * DIM * DIM,
                        DIM * DIM,
                        DIM,
                        1,
                    ),
                ),
            )

            self.kernel(
                mV,
                mQd,
                mKd,
                mKc,
                mE,
                mA,
                mM,
                mBeta,
                mO,
                mBoundaries,
                T,
                scale,
            ).launch(
                grid=(B, H, 1),
                block=(THREADS_SCAN, 1, 1),
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mV: cute.Tensor,
            mQd: cute.Tensor,
            mKd: cute.Tensor,
            mKc: cute.Tensor,
            mE: cute.Tensor,
            mA: cute.Tensor,
            mM: cute.Tensor,
            mBeta: cute.Tensor,
            mO: cute.Tensor,
            mBoundaries: cute.Tensor,
            T: Int32,
            scale: Float32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            batch_idx, head_idx, _ = cute.arch.block_idx()
            NC = cute.ceil_div(T, CHUNK)

            smem = utils.SmemAllocator()
            state_layout = cute.make_layout((DIM, DIM), stride=(DIM, 1))
            tile_layout = cute.make_layout((CHUNK, DIM), stride=(DIM, 1))
            local_mat = cute.make_layout((CHUNK, CHUNK), stride=(CHUNK, 1))

            sState = smem.allocate_tensor(
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
            sQ0 = smem.allocate_tensor(
                cutlass.Float32, tile_layout, byte_alignment=128
            )
            sU = smem.allocate_tensor(
                cutlass.Float32, tile_layout, byte_alignment=128
            )
            sA = smem.allocate_tensor(
                cutlass.Float32, local_mat, byte_alignment=16
            )
            sM = smem.allocate_tensor(
                cutlass.Float32, local_mat, byte_alignment=16
            )
            sBeta = smem.allocate_tensor(
                cutlass.Float32,
                cute.make_layout((CHUNK,), stride=(1,)),
                byte_alignment=16,
            )
            sE = smem.allocate_tensor(
                cutlass.Float32,
                cute.make_layout((DIM,), stride=(1,)),
                byte_alignment=16,
            )

            # Zero initial state and boundary zero.
            for it in cutlass.range_constexpr((DIM * DIM) // THREADS_SCAN):
                idx = tidx + it * THREADS_SCAN
                d = idx // DIM
                vv = idx % DIM
                sState[d, vv] = Float32(0.0)
                mBoundaries[batch_idx, head_idx, 0, d, vv] = Float32(0.0)
            cute.arch.sync_threads()

            for chunk_idx in range(NC):
                # Coalesced stage load.  Replace with PipelineTmaAsync in the
                # tuned path; the shared layouts are already separated.
                for it in cutlass.range_constexpr(
                    (CHUNK * DIM) // THREADS_SCAN
                ):
                    idx = tidx + it * THREADS_SCAN
                    i = idx // DIM
                    d = idx % DIM
                    tg = chunk_idx * CHUNK + i
                    sQd[i, d] = mQd[
                        batch_idx, head_idx, chunk_idx, i, d
                    ]
                    sKd[i, d] = mKd[
                        batch_idx, head_idx, chunk_idx, i, d
                    ]
                    sKc[i, d] = mKc[
                        batch_idx, head_idx, chunk_idx, i, d
                    ]
                    sV[i, d] = (
                        mV[batch_idx, tg, head_idx, d]
                        if tg < T
                        else cutlass.BFloat16(0.0)
                    )
                if tidx < CHUNK * CHUNK:
                    i = tidx // CHUNK
                    j = tidx % CHUNK
                    sA[i, j] = mA[
                        batch_idx, head_idx, chunk_idx, i, j
                    ].to(Float32)
                    sM[i, j] = mM[
                        batch_idx, head_idx, chunk_idx, i, j
                    ].to(Float32)
                if tidx < CHUNK:
                    sBeta[tidx] = mBeta[
                        batch_idx, head_idx, chunk_idx, tidx
                    ].to(Float32)
                if tidx < DIM:
                    sE[tidx] = mE[
                        batch_idx, head_idx, chunk_idx, tidx
                    ].to(Float32)
                cute.arch.sync_threads()

                # GEMM region 1: P=Kd@S and Q0=Qd@S.
                # Tensor-core path: TF32 mma.sync.m16n8k8, FP32 accum.
                for out_it in cutlass.range_constexpr(
                    (CHUNK * DIM) // THREADS_SCAN
                ):
                    out_idx = tidx + out_it * THREADS_SCAN
                    i = out_idx // DIM
                    vv = out_idx % DIM
                    p = Float32(0.0)
                    q0 = Float32(0.0)
                    for d in cutlass.range_constexpr(DIM):
                        sv = sState[d, vv]
                        p += sKd[i, d].to(Float32) * sv
                        q0 += sQd[i, d].to(Float32) * sv
                    sP[i, vv] = p
                    sQ0[i, vv] = q0
                cute.arch.sync_threads()

                # U=A @ (beta*(V-P)).
                for out_it in cutlass.range_constexpr(
                    (CHUNK * DIM) // THREADS_SCAN
                ):
                    out_idx = tidx + out_it * THREADS_SCAN
                    i = out_idx // DIM
                    vv = out_idx % DIM
                    u = Float32(0.0)
                    for j in cutlass.range_constexpr(CHUNK):
                        rhs = sBeta[j] * (
                            sV[j, vv].to(Float32) - sP[j, vv]
                        )
                        u += sA[i, j] * rhs
                    sU[i, vv] = u
                cute.arch.sync_threads()

                # O=scale*(Q0+M@U).
                for out_it in cutlass.range_constexpr(
                    (CHUNK * DIM) // THREADS_SCAN
                ):
                    out_idx = tidx + out_it * THREADS_SCAN
                    i = out_idx // DIM
                    vv = out_idx % DIM
                    corr = Float32(0.0)
                    for j in cutlass.range_constexpr(CHUNK):
                        corr += sM[i, j] * sU[j, vv]
                    tg = chunk_idx * CHUNK + i
                    if tg < T:
                        mO[batch_idx, tg, head_idx, vv] = (
                            scale * (sQ0[i, vv] + corr)
                        ).to(mO.element_type)
                cute.arch.sync_threads()

                # GEMM region 2 and row scaling:
                # S=diag(E)S+Kc.T@U.
                for out_it in cutlass.range_constexpr(
                    (DIM * DIM) // THREADS_SCAN
                ):
                    out_idx = tidx + out_it * THREADS_SCAN
                    d = out_idx // DIM
                    vv = out_idx % DIM
                    update = Float32(0.0)
                    for i in cutlass.range_constexpr(CHUNK):
                        update += sKc[i, d].to(Float32) * sU[i, vv]
                    sState[d, vv] = sE[d] * sState[d, vv] + update
                cute.arch.sync_threads()

                is_boundary = (
                    (chunk_idx + 1) % self.cfg.checkpoint_chunks == 0
                    or chunk_idx + 1 == NC
                )
                if is_boundary:
                    boundary_idx = (
                        (chunk_idx + 1 + self.cfg.checkpoint_chunks - 1)
                        // self.cfg.checkpoint_chunks
                    )
                    for it in cutlass.range_constexpr(
                        (DIM * DIM) // THREADS_SCAN
                    ):
                        idx = tidx + it * THREADS_SCAN
                        d = idx // DIM
                        vv = idx % DIM
                        mBoundaries[
                            batch_idx, head_idx, boundary_idx, d, vv
                        ] = sState[d, vv]
                cute.arch.sync_threads()


    class KdaBackwardSm90:
        """Reverse persistent scan + local analytic backward lowering.

        The exact executable equations are in ``kda_chunk_oracle.py``.  The
        production lowering is intentionally split into scan/local kernels so
        chunk-local matrix work regains [B,H,NC] parallelism.  This class records
        the device ABI and launch sequence used by the PyTorch binding.

        Implementations should lower:
          1. segment replay from FP32 boundaries into bounded replay scratch;
          2. reverse chunk equations from SM90_LOWERING.md;
          3. chunk-adjoint stores;
          4. token-parallel inverse/pairwise/norm/gate backward;
          5. FP32 parameter partial reduction.

        Keeping this ABI stable lets the recurrence GEMMs be swapped between
        strict FP32 SIMT and TF32 tensor-core implementations without changing
        autograd.
        """

        def __init__(self, config: KdaSm90Config = KdaSm90Config()):
            config.validate()
            self.cfg = config

        @cute.jit
        def __call__(self, *args, **kwargs):
            raise RuntimeError(
                "The backward CuTe lowering needs on-SM90 compilation and "
                "register/smem iteration. Use kda_chunk_oracle.py as the exact "
                "training implementation and SM90_LOWERING.md as the complete "
                "reverse-mode contract."
            )


    class KdaSm90Operator:
        """Forward launch graph and stable workspace ABI."""

        def __init__(self, config: KdaSm90Config = KdaSm90Config()):
            config.validate()
            self.cfg = config
            self.prepare = KdaPrepareFwdSm90(config)
            self.scan = KdaScanFwdSm90(config)
            self.backward = KdaBackwardSm90(config)

        @staticmethod
        def workspace_shapes(
            B: int, T: int, H: int, checkpoint_chunks: int = 8
        ) -> dict[str, tuple[int, ...]]:
            return WorkspaceShapes(
                B=B, T=T, H=H, checkpoint_chunks=checkpoint_chunks
            ).as_dict()


else:

    class KdaSm90Operator:  # pragma: no cover - host without CUTLASS
        def __init__(self, config: KdaSm90Config = KdaSm90Config()):
            config.validate()
            raise RuntimeError(
                "CUTLASS CuTe DSL is not installed. Install a CUTLASS 4.5-era "
                "CuTe DSL build with CUDA 12.9+ and compile for sm_90a."
            )

        @staticmethod
        def workspace_shapes(
            B: int, T: int, H: int, checkpoint_chunks: int = 8
        ) -> dict[str, tuple[int, ...]]:
            return WorkspaceShapes(
                B=B, T=T, H=H, checkpoint_chunks=checkpoint_chunks
            ).as_dict()
