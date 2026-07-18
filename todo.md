# H100 kernel evaluation — 2026-07-19

This is the runbook for choosing the next OpenKimi training backends. Preserve the
model mathematics first; throughput wins only count after forward, backward, and
optimizer-step parity pass.

## Next: guarded KDA diagonal MMA and backward fusion

The saved-intermediate exact FLA configuration is the control baseline. On the
2026-07-19 H100 it measures 16.947 ms for standalone KDA forward+backward and
143,068–143,194 tok/s for the complete update. Directly identifiable KDA chunk
kernels account for 23.6% of GPU kernel time (approximately 27% with Q/K
normalization and gate transforms), so KDA backward remains the first
optimization target.

### 1. Guarded diagonal MMA — full integration packaged and verified

- [x] Confirm the pinned FLA `safe_gate=False` path: both diagonal phases in
  `chunk_kda_bwd_kernel_intra` execute 16 scalar partner loops, recompute a full
  `[16,32]` `exp2` tile per partner, and do not use tensor cores.
- [x] Add a device-independent oracle for the guarded factorization:
  `experiments/kda_sm90/guarded_diagonal_oracle.py`.
- [x] Add CPU tests covering factorized parity, exact fallback for large gate
  spans, rejection of non-monotonic blocks, and guard validation.
- [x] Add the isolated H100 Triton benchmark
  `experiments/kda_sm90/benchmark_guarded_diagonal.py`. It compares the exact
  scalar FLA algebra against the runtime-guarded TF32 `16x16 @ 16x32` path for
  both diagonal phases.
- [x] Measure the actual distribution of `g_max - g_min` over every
  `[BC=16,BK=32]` tile from all 12 KDA layers at the primary initialized-model
  shape. The 2,359,296 tiles have p50/p90/p99/p99.9/max spans of
  `143.673/233.876/248.796/254.614/265.768`.
- [x] Reject the first-row threshold plan. Threshold 30 accepts only 2.778% of
  actual initialized-model tiles. Sweep the midpoint-reference policy instead;
  threshold 248 accepts 98.742% while bounding each factor near `2**±124`.
- [x] Record guard-hit rate, latency distribution, maximum error, relative
  error, and cold compile time for the isolated Triton schedule:

  ```bash
  uv run --extra cuda python experiments/kda_sm90/benchmark_guarded_diagonal.py \
    --blocks 196608 --mean-log2-decay 8 \
    --reference-policy midpoint --thresholds 128,160,192,224,240,248,250 \
    --warmup 10 --repetitions 30
  ```

- [x] Require the exact pairwise FMA fallback whenever the tile is
  non-monotonic, non-finite, or exceeds the selected span. The first-row
  and centered reference policies are both covered by the CPU oracle and H100
  benchmark.
- [x] Lower the isolated four-product tile to native CuTe DSL 4.6 with TF32 MMA
  and FP32 accumulation.
- [x] Remove the CuTe B-operand shared-memory bank conflict. The plain
  `[channel,row] = [32,16]` stride-16 layout has predicted 16-way producer-store
  conflicts. The s32/s64/s128 sweep reduces this to 8/4/2-way and measures
  1.295/1.136/1.121 ms versus 1.597 ms plain. The selected s128 tile is 29.8%
  faster than plain, 1.948x faster than exact, and within 2.2% of the Triton
  prototype, with unchanged <=7.64e-4 relative error. Nsight Compute confirms
  66.7% fewer shared-load and 63.2% fewer shared-store bank conflicts across
  128 CTAs.
- [x] Test replacing the negative factor's `exp2` with a reciprocal of the
  positive factor. It regresses from 1.125 to 1.241 ms on the s128 layout, so
  retain two `exp2` evaluations.
- [x] Cache the gate tile during validation and simplify the reference to the
  first/last monotonic values. This reduces the controlled median from 1.118
  to 1.037 ms (7.23%), cuts global-load L1 sectors by 25.1%, reaches 2.10x
  versus exact, and beats the Triton prototype by 5.44%. Select gate caching
  even though it increases allocated registers from 48 to 64.
- [x] Test channel-half warp ownership and register-local upper summation.
  Reversed-order timing measures 1.048 ms versus 1.033 ms for product ownership
  (1.43% slower), so retain one warp per product.
- [x] Test direct Dq/Dk accumulator stores. Removing 4 KiB of shared epilogue
  scratch regresses from 1.042 to 1.074 ms (3.05%), so retain the coalescing
  shared-memory epilogue.
- [x] Test an explicit exact-fallback diagonal. It improves an all-fallback
  workload by 10.8%, but enlarging the mixed kernel regresses the realistic
  98.39%-fast workload by 1.28%. Retain the compact loop fallback; do not add a
  17 KiB decay table to the common kernel.
- [x] Verify that FLA's existing unguarded factorized path is not usable as a
  shortcut. With the faithful gate equation precomputed, `safe_gate=True`
  produces NaN output and gradients at `[1,256,6,128]`.
- [x] Integrate the provisional midpoint-248 guard into FLA's complete
  `chunk_kda_bwd_kernel_intra` and compare output plus all seven differentiable
  inputs at `[32,4096,6,128]`. Plain TF32 is rejected: raw-decay, `A_log`, and
  `dt_bias` gradient errors are 2.62%, 4.67%, and 13.81%. Triton `tf32x3`
  passes this synthetic production-shape target, but threshold 248 is not the
  final selection because the controlled real-model replay below is stricter.
- [x] Extend the TF32x3 parity gate to near-threshold, high-decay,
  cancellation-heavy, 4,103-token partial-chunk, non-monotonic/fallback, and
  real-model gradient fixtures. High-decay and non-monotonic fallbacks are
  bit-identical. For a deliberately near-zero cancellation reference, accept
  only if either relative L2 is below `5e-3` or maximum absolute error is below
  `5e-5`; all other tensors retain the relative target and no non-finite values
  are allowed.
- [x] Reject thresholds 248 and 240 after replaying actual activations and
  downstream gradients captured from one exact 4,096-token full-model
  backward. Threshold 248 fails two of 204 KDA mixer tensors and threshold 240
  fails one. Threshold **232** is the highest passing value: zero failures,
  worst relative L2 `2.409e-3`, and an 88.615% initialized-model guard-hit rate.
  A naive two-full-model comparison is not a valid gate because even two exact
  runs differ by up to 18.9% across nondeterministic downstream reductions.
- [x] Lower the winning schedule as an ABI-identical Triton integration for
  `chunk_kda_bwd_intra`, retaining FLA's `BT=64`, `BC=16`, saved-intermediate
  ABI, FP32 accumulators, and exact fallback. The reversible experiment is
  `experiments/kda_sm90/patch_fla_guarded_intra.py`. It verifies the pinned FLA
  source hash before modifying the isolated site-package and preserves an
  exact-source restore.
- [x] Productionize the winning integration as the exact-pinned fork
  `yxanul/flash-linear-attention@ee8369bb735bcc91aefc967ea911cc75248a1b79`,
  based on upstream `ccb0ff944cbff035fa59ac47a4cc8fd2e079bb17`.
  `pyproject.toml` and `uv.lock` install it directly; bootstrap and H100 backend
  resolution fail closed unless patch version 1, TF32x3, and span `<=232` are
  present. The training path no longer modifies `site-packages`.
- [x] Verify a clean `uv sync --locked --extra cuda` on H100 resolves the fork
  and exact commit through `direct_url.json`, then rerun the four backend tests,
  20 H100 parity/platform tests, five adversarial fixtures, and the controlled
  12-mixer/204-tensor replay. All pass. The packaged no-regression benchmark is
  884.806 ms per 131,072-token microstep (148,136 tok/s), equivalent to
  1,769.611 ms for the configured two-microstep 262,144-token update.
- [ ] Lower the selected native CuTe schedule into the full ABI only if it can
  beat the now-faster TF32x3 Triton integration: midpoint guard 232, s128 B
  layout, cached gate, product-owned warps, shared output epilogue, two
  exponentials, and compact fallback. Do not extend the old per-16-chunk
  workspace ABI.
- [ ] Revisit storing two full A matrices only after full-ABI profiling. The
  cached kernel is register-limited at eight CTAs/SM, so adding register masks
  is unlikely to improve isolated-tile residency.
- [x] Compare the full integration against exact FLA. Saved-intermediate KDA
  at the selected threshold 232 falls from 16.947 to 16.054 ms (`-5.27%` total,
  `-6.76%` backward). Exact FLA controls measure 1,832.31/1,830.69 ms per
  update; guarded TF32x3-232 measures 1,787.93 ms and 146,619 tok/s:
  `+2.39–2.48%` throughput at the same 58.66 GiB peak allocation. The earlier
  147,440 tok/s threshold-248 result is rejected on real-model gradient
  accuracy. Final results are in
  `profiles/h100-sm90-kda-fla-guarded-validation-2026-07-19.json`.

### 2. Fuse `wy_dqkg_fused` and `intra` after the drop-in wins

- [x] Prototype the fused CTA before committing to a full CuTe rewrite. A
  process-local Triton candidate executes the existing WY body followed by one
  coarsened, blocked intra phase per `(chunk,b,hv)`. It preserves numerical
  parity (`4.31e-4` worst relative L2), but the four-warp kernel regresses from
  1.862 to 2.191 ms (`+17.6%` latency, `-15.0%` speed). It retains global `dA`
  scratch, so it is a rejection probe rather than a production implementation.
  The eight-warp version cannot lower the current SM90 MMA layout.
- [x] Budget SMEM/registers before deeper implementation. The existing WY
  kernel already consumes 255 registers/thread and 49,664 B dynamic SMEM. The
  fused candidate spills 1,152 B of local stack per thread. This invalidates
  mechanical Triton inlining; a future attempt needs a dedicated CuTe schedule
  with explicit phase lifetimes and shared/register ownership.
- [x] Re-measure the target against a fresh control. Nsight now reports
  111.909 ms/update for WY and 100.100 ms/update for intra, or 212.009 ms
  combined rather than the older 255.28 ms estimate. The speculative
  100–140 ms target was not achieved.
- [x] Retain the independently useful coarsened intra only as an experiment.
  Four warps reduce the intra region by 3.93% and the combined WY+intra region
  by 1.85%. Full-step A/B/A improves throughput by only 0.32–0.37%
  (146,541 to 147,006–147,087 tok/s). A controlled replay of all 12 real KDA
  mixers passes all 204 tensors with `2.431e-3` worst relative L2. This is not
  enough to replace the pinned production provider.

### 3. Backward epilogue and persistent scan follow-ups

- [x] Do not fold reverse chunk-local `dg` cumsum, KDA gate backward, Q/K L2
  backward, or beta-sigmoid backward into the rejected CTA. This experiment
  was explicitly conditional on fused chunk CTA speedup; that prerequisite
  failed.
- [x] Do not test the persistent `(b,hv)` reverse scan yet. Its fused-CTA and
  epilogue prerequisites both failed, so a 192-CTA persistent grid on the
  132-SM H100 would add a second independent scheduling risk without a winning
  producer to integrate.
- [x] Leave forward specialization alone because backward was not proven. The
  measured
  standalone split is 3.77 ms forward versus 12.94 ms backward. Any forward
  work should fuse normalization/gate/beta into the existing intra prologue;
  the standalone CuTe preprocessing launch already lost to FLA
  (0.602 ms versus 0.490 ms).

## Most likely wins

1. [x] **Remove unnecessary outer checkpoint replay.** The current profile has
   65 AttnRes forward launches and 33 backward launches. The architecture has
   33 original AttnRes reads (two per layer plus the final read), so the other
   32 forwards are consistent with PyTorch checkpoint replay. Those duplicated
   AttnRes forwards cost approximately
   `86.447 ms * 32 / 65 = 42.6 ms`, or 1.9% of the update, before counting the
   replayed KDA/MLA, FFN, router, expert, and FP8 quantization work.
2. [x] **Tune the FP8 LM-head token chunk.** With the selected microbatch-32,
   accumulation-2 policy, the automatic heuristic selects 1,024 rows and
   produces 256 LM-head forward GEMMs, CE launches, and backward GEMMs per
   262,144-token update. Test 2K, 4K, 8K, 16K, and 32K while the machine has memory
   headroom. This can improve the entire LM-head region, not just the CE
   reduction.
3. [x] **Replace Liger CE with QuACK only after chunk tuning.** Compare both at
   the best common chunk size so a chunking improvement is not misattributed to
   the CE implementation.
4. [x] **Replace routed-expert permutation with SonicMoE's GPU-resident
   bitmatrix path.** This is the most promising structural MoE optimization but
   needs an external-routing adapter and broader gradient testing.
5. [ ] **Specialize AttnRes only after profiling by source count and replay
   status.** The existing FLA kernel already fuses online softmax, residual
   accumulation, and output RMSNorm; a replacement must demonstrate a
   data-movement or exact-shape advantage.
6. [ ] **Treat cuLA as an inference-prefill experiment for now.** Its Hopper
   kernel is forward-only and `safe_gate=True`, so it cannot replace faithful
   training KDA.

## Baselines and test profiles

- [ ] Record the exact OpenKimi commit, dependency revisions, and any uncommitted
  changes before testing.
- [x] Reproduce the current measured baseline before changing a backend:
  `115,548 tok/s`, `2,268.7 ms/update`, `35.14 GiB peak allocated`, one H100,
  `64 x 4,096 = 262,144` tokens/update, no gradient accumulation, eager execution,
  activation checkpointing enabled, and TE FP8 Current Scaling where configured.
- [ ] Keep two model profiles separate in every result:
  - **Measured profile:** latent width 192, routed hidden 512, shared hidden 512,
    top-4, no MTP.
  - **Candidate profile:** latent width 256, routed hidden 768, shared hidden 1,024,
    top-2 and top-4 variants, MTP depth 3 and 4.
- [ ] Use standalone kernel benchmarks for candidate shapes until the candidate
  model configuration and MTP implementation are committed. Do not label a
  standalone estimate as full-model throughput.
- [ ] Do not change sequence length, token batch, precision policy, checkpointing,
  or model dimensions during an apples-to-apples backend comparison.

## Test order

Run in this order if GPU time is limited:

1. [x] Environment inventory and current-baseline reproduction.
2. [x] Outer-checkpoint and AttnRes checkpoint-level matrix.
3. [x] FP8 LM-head chunk-size sweep using the current Liger CE.
4. [x] QuACK cross-entropy parity and timing at the tuned chunk size.
5. [x] SonicMoE routed-expert parity and timing.
6. [x] Combined winning checkpoint, LM-head, CE, and MoE settings.
7. [ ] cuLA H100 fused-forward smoke test and model-shape benchmark.
8. [ ] `torch.compile` and CUDA Graph experiments only for eager backends that
   already win.
9. [ ] Candidate top-2/top-4 and MTP-depth sweeps after their implementation is
   ready.

## Reproducible H100 setup

- [x] Inventory the machine before installing anything:
  - GPU name, compute capability, VRAM, power limit, clocks, and ECC state.
  - NVIDIA driver, `nvcc`, system CUDA toolkit, and glibc versions.
  - Python, PyTorch, PyTorch CUDA, Triton, cuDNN, NCCL, FLA, MegaBlocks,
    grouped-GEMM, Liger, Transformer Engine, QuACK, SonicMoE, and cuLA versions.
- [x] Confirm the machine is an H100/SM90. Do not extrapolate H200 bandwidth
  results directly to H100.
- [x] Use `uv` and isolated environments for experimental stacks. Keep the
  current locked environment intact because SonicMoE, QuACK, and cuLA currently
  have newer and mutually sensitive Python/PyTorch/CUDA requirements.
- [x] Pin and record exact source commits:
  - OpenKimi parent: `0b7ac3b12eee`.
  - SonicMoE inspected revision: `0349404`.
  - QuACK inspected revision: `3a1c687` / release `v0.6.1`.
  - cuLA inspected revision: `9ff1edb1a027`; record the actually tested revision
    rather than using an unrecorded `main`.
- [ ] Verify PyTorch CUDA and the system CUDA toolkit match before building cuLA.
  Its current documented stack is Python 3.12+, CUDA/NVCC 12.9+, and
  PyTorch 2.9.1+.
- [x] Use fixed seeds and synthetic static-shape inputs for kernel comparisons;
  dataset/tokenizer throughput must not contaminate GPU kernel timing.
- [x] Warm all compilation, autotuning, and allocator paths before measuring.
  Record cold-start time separately.
- [ ] Use at least 5 warmups and 10 measured iterations when time permits. Report
  median, minimum, p10, and p90 rather than one favorable iteration.
- [x] Reset peak-memory statistics between providers and synchronize only outside
  the measured region.
- [x] Save compact JSON results under `profiles/`. Keep large Nsight traces
  outside Git and record their paths plus exact capture commands.

## Baseline reproduction and profiling

- [x] Run all existing CPU tests before using the GPU:

  ```bash
  uv run pytest
  uv run ruff check .
  ```

- [x] On H100, run the CUDA parity tests:

  ```bash
  uv run --extra cuda pytest -m gpu tests/test_h100_parity.py -v
  ```

- [x] Reproduce the eager FP8 Current Scaling step:

  ```bash
  uv run --extra cuda python scripts/profile_h100_step.py \
    --config configs/h100-fp8-current.json \
    --warmup 5
  ```

- [x] Accept the baseline only if median throughput is within 3% of the prior
  `115,548 tok/s` result and memory is within 1 GiB. Investigate clocks,
  dependency drift, or a changed config otherwise.
- [x] Capture one warmed optimizer update with Nsight Systems and the existing
  NVTX ranges. Confirm the selected providers in the emitted backend metadata.
- [x] Record:
  - Total update time and tokens/s.
  - Forward, backward, AdamW, gradient clipping, and zeroing time.
  - Peak allocated and reserved memory.
  - Kernel-launch count, GPU-busy percentage, CPU gaps, and any D2H copies.
  - KDA forward/backward, AttnRes, routed experts, shared experts, LM head, and
    cross-entropy contributions.

## Checkpoint recomputation policy

This is the first optimization experiment. It requires no new kernel dependency,
does not alter the architecture, and trades currently unused memory capacity for
less work.

### Result — 2026-07-18

- [x] Reproduced the clean baseline at 116,135 tok/s; the repeated policy-A
  median was 116,197 tok/s at 35.13 GiB.
- [x] Batch 64 OOMed when either outer checkpoint family was removed, at about
  77.2-77.5 GiB allocated.
- [x] The winning equal-token policy is microbatch 32, accumulation 2, attention
  checkpointing on, FFN checkpointing off, and AttnRes level 1:
  121,687 tok/s at 60.84 GiB, a 4.73% throughput improvement.
- [x] Attention-only checkpoint removal at microbatch 32 reached 119,594 tok/s;
  removing both checkpoint families at microbatch 16 reached 120,405 tok/s.
- [x] AttnRes level 0 was only 0.16% faster than level 1 on the winning outer
  policy and used another 3.18 GiB, so level 1 remains selected.
- [x] Nsight confirmed 98 AttnRes forwards and 66 backwards across two
  microsteps: 66 original reads plus 32 attention replays, with the 32 FFN
  replays removed.
- [x] All nine H100 GPU tests passed, and ten consecutive measured optimizer
  updates were stable at 121,680 tok/s.
- [x] Results are recorded in
  `profiles/h100-sm90-checkpoint-policy-2026-07-18.json`; the candidate config is
  `configs/h100-fp8-current-checkpoint-optimized.json`.

### Instrumentation

- [x] Make outer attention checkpointing, outer FFN checkpointing, and FLA
  AttnRes `checkpoint_level` separately configurable. Keep the existing single
  `activation_checkpointing` setting as a compatibility/default shorthand.
- [ ] Add NVTX ranges containing layer, sublayer, and source count:
  - `attnres.attention.layer_00.sources_1`
  - `attnres.ffn.layer_00.sources_2`
  - Continue through all 16 layers and the final read.
- [ ] Mark or otherwise distinguish original forward execution from checkpoint
  replay. If explicit replay ranges are awkward for Transformer Engine's
  checkpoint wrapper, use per-layer launch counts to distinguish one execution
  from two.
- [ ] Add ranges around the complete attention and FFN bodies so the measurement
  includes replayed KDA/MLA, router, expert, shared expert, and FP8 quantization
  work, not only `attnres_fwd_kernel`.
- [x] Confirm the current policy produces 65 AttnRes forward and 33 backward
  launches before evaluating a change.

### Policy matrix

- [ ] At fixed batch 64, sequence 4,096, and 262,144 tokens/update, test:

  | Case | Outer attention | Outer FFN | AttnRes level | Purpose |
  |---|---:|---:|---:|---|
  | A | on | on | 1 | Current baseline |
  | B | off | on | 1 | Isolate attention/mixer replay |
  | C | on | off | 1 | Isolate FFN/MoE replay |
  | D | off | off | 1 | Remove all outer sublayer replay |
  | E | best A-D | same | 0 | Save AttnRes pre-norm mixture |

- [ ] If time permits, run the complete 2 x 2 outer-policy matrix with AttnRes
  levels 0 and 1. Do not assume level 0 helps when an enclosing checkpoint
  discards its original saved tensors.
- [ ] FLA AttnRes level 1 saves logits/statistics but recomputes the pre-norm
  weighted residual mixture from the sources during backward. Level 0 saves one
  BF16 `o_pre` tensor per read instead.
- [ ] At the exact `64 x 4096 x 768` shape, each saved BF16 `o_pre` is about
  0.375 GiB; retaining it for 33 reads has an approximate 12.4 GiB upper-bound
  cost. Measure actual allocator peaks rather than relying on this estimate.
- [ ] Do not change FLA KDA's internal recompute policy during this matrix. The
  immediate question is whether the outer checkpoint is replaying whole
  sublayers unnecessarily; KDA can remain a separately measured baseline.
- [ ] Record for every case:
  - Full update latency and tokens/s.
  - Peak allocated/reserved memory and remaining H100 headroom.
  - AttnRes, KDA/MLA, routed/shared FFN, and FP8 quantization launch counts/time.
  - Forward and backward numerical parity with case A.
- [x] If case D OOMs, test the best partial policy first. Then reduce microbatch
  only as a secondary experiment and restore the 262,144-token global batch with
  gradient accumulation for a fair throughput comparison.
- [x] Leave at least 5-8 GiB of practical memory margin for allocator variance,
  DDP/NCCL buffers, and real training inputs; do not select a synthetic profile
  that only barely fits.

### AttnRes follow-up

- [ ] Aggregate AttnRes time by layer and exact source count. Determine whether
  cost is concentrated in later many-source reads, uniform across depth, replay,
  or backward rereading.
- [x] Measure `checkpoint_level=0` versus 1 for complete AttnRes forward+backward,
  not only the forward kernel.
- [ ] Use the measured source distribution for any exact `D=768` specialization.
  Test small-source and late-layer many-source cases separately.
- [ ] Quantify the theoretical region-level opportunity correctly:
  - Current principal AttnRes kernels: about 213.6 ms or 9.4% of GPU time.
  - A 1.5x complete-region speedup saves about 71 ms, roughly 3.1% of the update.
  - A 2x complete-region speedup saves about 107 ms, roughly 4.7%.
- [ ] Profile auxiliary backward reductions and BF16 residual vector additions
  before considering fused adjacent accumulation.
- [ ] Do not change `attnres_block_size` in a kernel comparison. Source-count
  changes are an architecture ablation and need a separate quality run.

### Checkpoint decision gate

- [x] Select the fastest equal-token policy that passes parity, fits with a
  5-8 GiB operational margin, and improves median full-step throughput by at
  least 1%.
- [x] Run at least 10 optimizer updates with the selected policy and require
  finite loss, gradients, parameters, and optimizer state before combining it
  with another backend.

## FP8 LM-head chunk-size tuning

This is the second optimization experiment and must precede the QuACK comparison.
The current chunk is derived from the Liger fused-linear memory heuristic rather
than tuned for H100 Transformer Engine Current Scaling.

### Result — 2026-07-18

- [x] The automatic control resolved to 1,024 rows and 256 chunks/update:
  121,634 tok/s at 60.84 GiB allocated.
- [x] Throughput rose monotonically through 2K, 4K, 8K, and 16K. The 16K winner
  reached 127,858 tok/s at 68.21 GiB allocated and 71.14 GiB reserved, a 5.12%
  improvement with 8.04 GiB of reserved-memory headroom.
- [x] The 32K candidate OOMed in the FP8 LM-head backward after reaching
  72.09 GiB allocated and requesting another 3.91 GiB.
- [x] At the actual `D=768`, logical-vocabulary 128,001, physical-vocabulary
  128,016 shape, every non-OOM chunk matched the BF16 reference with about
  2.67% hidden-gradient and 2.65% tied-weight-gradient relative error. All 15
  physical dummy rows had exactly zero gradient.
- [x] Ten consecutive 16K updates were stable at 127,747 tok/s. Nsight recorded
  16 Liger CE launches and 201.52 ms of CE kernels, down from 256 launches and
  about 216.22 ms for the checkpoint-only profile.
- [x] The tuned config is
  `configs/h100-fp8-current-lm-head-optimized.json`; complete results are in
  `profiles/h100-sm90-fp8-lm-head-chunk-2026-07-18.json`.

### Chunk sweep

- [x] Make `fp8_lm_head_chunk_size` an explicit configuration/benchmark override while
  preserving the current automatic mode.
- [ ] Add an NVTX range for the full LM-head loss region and optional per-chunk
  ranges covering:
  - TE input quantization.
  - FP8 LM-head forward GEMM.
  - In-place CE.
  - FP8/BF16 gradient quantization.
  - LM-head input-gradient and tied-weight-gradient GEMMs.
- [x] At 262,144 tokens, sweep:

  | Chunk rows | Chunks / CE launches | Approx. BF16 logits per chunk |
  |---:|---:|---:|
  | 2,048 | 128 | 0.49 GiB |
  | 4,096 | 64 | 0.98 GiB |
  | 8,192 | 32 | 1.95 GiB |
  | 16,384 | 16 | 3.91 GiB |
  | 32,768 | 8 | 7.81 GiB |

- [x] Stop increasing the chunk when peak memory loses the operational margin,
  GEMM/CE latency regresses, or allocator behavior becomes unstable.
- [x] For every chunk, compare loss, hidden gradient, and tied embedding/LM-head
  gradient against the 2,048-row baseline. Confirm all 15 physical dummy
  vocabulary rows still have exactly zero gradient.
- [ ] Record:
  - Full LM-head region forward/backward time.
  - CE-only time and launch count.
  - FP8 forward, input-gradient, and weight-gradient GEMM time/efficiency.
  - Quantization time, temporary allocations, peak model memory, and full-step
    tokens/s.
- [ ] Use Nsight Compute on the best two sizes to inspect tensor-core utilization,
  TMA behavior, occupancy, DRAM traffic, and whether the larger GEMM tiles are
  actually more efficient.
- [x] Do not credit reduced Python/operator dispatch unless it changes measured
  wall time; the current profile is already about 99.1% GPU busy.

### Chunk decision gate

- [x] Use the fastest parity-clean chunk with 5-8 GiB memory margin as the new
  Liger baseline.
- [x] Require at least 1% full-step improvement to change the default. Otherwise
  retain automatic 2,048-row chunking and record the sweep.
- [x] Compare QuACK against the tuned Liger baseline at the same chunk first.
  Then allow a small QuACK-specific chunk sweep so each backend also gets its
  best valid configuration.

## QuACK cross-entropy

Reference: [Dao-AILab/quack](https://github.com/Dao-AILab/quack).

### Integration needed before the benchmark

- [x] Add QuACK as an optional loss provider without changing the default lock or
  macOS resolution.
- [x] Extend `scripts/benchmark_loss_backends.py` with a QuACK provider and reuse
  the configurable logits chunk size from the preceding sweep.
- [x] Keep TE FP8 Current Scaling for the LM-head linear projection and apply
  QuACK to each BF16 logits chunk first. This isolates the CE replacement without
  giving up the existing FP8 GEMM.
- [x] Use QuACK's output/in-place forward where possible so the BF16 logits buffer
  is overwritten by `dlogits` instead of allocating both.
- [x] Implement exact logical-vocabulary handling:
  - Physical padded vocabulary: 128,016 if the GEMM requires it.
  - Logical vocabulary: 128,001.
  - Dummy logits must behave as negative infinity in the softmax.
  - Dummy-column gradients must be exactly zero.
  - Do not accept an approximation that includes padding columns in the
    denominator.
- [x] Preserve tied input-embedding/LM-head gradient accumulation.
- [x] Do not switch first to QuACK's chunked-linear CE as if it were equivalent:
  that path currently uses ordinary `torch.mm` for parts of the computation and
  does not preserve our TE Current Scaling projection.

### Correctness matrix

- [x] Compare FP32 PyTorch reference, current Liger fused linear CE, and QuACK for:
  - Token chunks of 2,048, 4,096, and 8,192.
  - Hidden width 768.
  - Logical vocabulary 128,001 and physical vocabulary 128,016.
  - BF16 logits and FP32 loss accumulation.
  - Mean reduction and `ignore_index=-100`.
- [x] Check loss, `dlogits`, hidden-state gradient, and tied LM-head/embedding
  gradient.
- [x] Include random labels, ignored labels, repeated labels, boundary IDs
  `0`/`128000`, extreme positive/negative logits, and all-ignored input.
- [x] Require finite outputs and approximately `5e-3` BF16 relative error against
  the reference, with stricter checks where FP32 accumulation permits.
- [x] Assert exact zero gradients for every physical padding column.

### Performance matrix

- [x] Measure isolated forward+backward for Liger and QuACK first at the tuned
  Liger chunk, then at neighboring powers of two.
- [x] Compare TE FP8 LM head + Liger CE against TE FP8 LM head + QuACK CE.
- [ ] Benchmark BF16 QuACK chunked-linear CE as a separate ablation only if the
  selected TE FP8 composition needs another loss-level control.
- [x] Record CE wall time, kernel time, peak memory, temporary allocations, launch
  count, and full optimizer-step tokens/s.
- [x] Verify that no full `[262144, 128001]` logits tensor is materialized.
- [x] Treat published QuACK-vs-Liger charts cautiously: the repository's
  [Liger comparison issue](https://github.com/Dao-AILab/quack/issues/9) remains
  open. Our decision must use the same GEMM, chunking, inputs, and measurement
  method for both providers.

### QuACK decision gate

- [x] Adopt only if all parity checks pass and it improves isolated loss time by
  at least 10% **and** full-step throughput by at least 2%, without increasing
  peak memory by more than 1 GiB.
- [x] If the only blocker is logical-vocabulary support, keep the patch small and
  upstreamable; do not fork unrelated QuACK code.

## SonicMoE GPU-resident bitmatrix path

Reference: [Dao-AILab/sonic-moe](https://github.com/Dao-AILab/sonic-moe).

The target is to replace the current:

```text
stable sort -> histogram/offsets -> gather -> grouped FC1 -> weighted SwiGLU
            -> grouped FC2 -> collision-heavy scatter/index_add
```

with SonicMoE's GPU-resident bitmatrix metadata, fused expert execution, and fused
aggregation, while keeping routing probabilities and expert mathematics unchanged.

### Adapter work

- [x] Add SonicMoE as an optional routed-expert backend; keep MegaBlocks as the
  reference and fallback.
- [x] Start from SonicMoE's general/external-routing interface because OpenKimi
  routes from full-width 768-dimensional tokens and executes experts on
  compressed latent tokens. Do not use a high-level API that assumes the router
  and expert input tensors are the same.
- [x] Prefer a fixed-top-k adapter that consumes the router's existing selected
  indices and normalized weights. Avoid variable-routing binary search and token
  rounding in the faithful path.
- [x] Preserve the current combined FC1 layout:
  - OpenKimi `gate_up_weight`: `[E, 2I, H]`.
  - Sonic view: `[2I, H, E]` with concatenated gate/up layout.
  - OpenKimi `down_weight`: `[E, I, H]`.
  - Sonic view: `[H, I, E]`.
- [ ] Reuse Sonic's device-resident expert counts for the router auxiliary loss
  and diagnostics where possible. Do not run a duplicate histogram.
- [x] Audit the hot path for `.cpu()`, `.item()`, `.tolist()`, host-visible
  counts, or implicit synchronization. Confirm their absence in Nsight Systems.
- [x] Keep diagnostics such as entropy, dead-expert streak, and maximum-load
  violation on the configured logging interval rather than every layer/step.
- [x] Keep Sonic routed experts in BF16 initially. Do not use its experimental
  FP8/MXFP8 branches for the primary test.
- [x] Do not add expert parallelism; the current target remains ordinary DDP with
  complete experts on every rank.

### Correctness matrix

- [ ] Compare MegaBlocks and Sonic outputs and gradients for:
  - Latent width `H`: 192 and 256.
  - Expert hidden `I`: 512 and 768.
  - Experts `E`: 64.
  - Top-k `K`: 2 and 4.
  - Source tokens `T`: 4,096; 16,384; 65,536; and 262,144 as memory allows.
- [x] Compare gradients for latent inputs, selected router weights,
  `gate_up_weight`, and `down_weight`.
- [ ] Test balanced, random, strongly skewed, one-hot-to-one-expert, empty-expert,
  and maximum-load routing.
- [ ] Test non-contiguous weight views used by the zero-copy layout adapter.
- [ ] Test activation checkpointing both off and on; backward recomputation must
  remain deterministic enough for the chosen tolerance.
- [x] Require approximately `5e-3` BF16 relative error and no NaN/Inf values.

### Performance matrix

- [x] Measure cold compile/autotune time separately from warmed execution.
- [ ] Measure router/metadata construction, permutation/materialization, FC1,
  weighted SwiGLU, FC2, aggregation/unpermutation, and backward individually.
- [x] Compare allocated bytes and peak memory. Specifically verify that Sonic does
  not materialize the current `O(T*K*H)` gathered input/output tensors.
- [x] Capture Nsight Systems for one full optimizer step. Compare launch counts,
  GPU gaps, D2H copies, and CPU launch overhead.
- [ ] Capture a separate targeted Nsight Systems trace for one routed layer if
  finer phase attribution is needed beyond the isolated CUDA-event benchmark.
- [ ] Capture targeted Nsight Compute metrics only for the dominant Sonic and
  MegaBlocks kernels: achieved occupancy, tensor-core utilization, DRAM
  throughput, L2 hit rate, and register/shared-memory pressure.
- [x] Run a full model step with all 15 routed FFNs changed together; isolated
  layer wins are not sufficient.
- [x] Repeat top-2 and top-4 candidate backend tests without changing routing
  weights or active routed computation.
- [ ] Compare top-2 and top-4 model quality separately; do not present the
  top-k architecture ablation as a pure backend speedup.

### Result — 2026-07-18

- [x] Pinned SonicMoE `0349404` and QuACK 0.6.1 in a combined isolated target;
  the locked Torch/FLA/MegaBlocks environment remains unchanged.
- [x] The fixed-top-k adapter preserves the existing router, selected
  probabilities, and `[E,2I,H]`/`[E,I,H]` parameters. It replaces only expert
  metadata, grouped GEMMs, weighted SwiGLU, and aggregation.
- [x] All 19 H100 tests passed. Sonic matched MegaBlocks output and latent,
  router-weight, FC1, and FC2 gradients for random and heavily skewed/empty-expert
  routing at 192/512/top-4 and 256/768/top-2/top-4.
- [x] At 131,072 tokens, current 192/512/top-4 expert latency fell from 7.402 to
  5.607 ms (`-24.3%`) and steady-state peak allocation fell from 2.909 to
  2.780 GiB.
- [x] The candidate 256/768 experts improved from 6.048 to 4.309 ms (`-28.7%`)
  at top-2 and 10.391 to 7.654 ms (`-26.3%`) at top-4.
- [x] The complete FP8 Current Scaling + QuACK update improved from 1,900.6 to
  1,844.1 ms and from 137,926 to 142,152 tokens/s (`+3.06%`). Peak allocation
  fell from 68.22 to 58.66 GiB.
- [x] Nsight reports 10,515 kernel launches and 1,831.9 ms GPU kernel time,
  versus 11,205 and 1,887.6 ms for MegaBlocks. Only two one-byte D2H copies
  remain per update; there are no 15-layer expert-count synchronizations.
- [x] Ten measured optimizer updates completed at a 1,846.6 ms median. Results
  are in `profiles/h100-sm90-sonic-moe-2026-07-18.json`.

### SonicMoE decision gate

- [x] Adopt if parity passes and either:
  - Routed-layer forward+backward improves by at least 10% and the full step by
    at least 2%; or
  - Peak activation memory falls by at least 10% with no meaningful throughput
    regression, enabling a materially larger microbatch.
- [x] Reject any apparent win caused by capacity truncation, dropped tokens,
  rounded routing, changed top-k weights, or skipped gradients.

## cuLA KDA evaluation

References:
[cuLA H200 results](https://github.com/inclusionAI/cuLA/blob/main/BENCHMARK_H200.md)
and [inclusionAI/cuLA](https://github.com/inclusionAI/cuLA).

### What the published number does and does not show

- [ ] Record this before testing: the H200 table measures the **fully fused SM90
  KDA forward-prefill kernel**, BF16, `D=128`, `H=64`, `safe_gate=True`, against
  FLA v0.5.0. It reports an average 1.58x speedup across fixed and variable
  lengths, with speedups ranging from about 1.02x to 2.51x.
- [ ] Do not treat that table as a training benchmark:
  - The Hopper fused kernel's backward raises `NotImplementedError`.
  - It asserts `safe_gate=True`.
  - cuLA's trainable modular KDA forward currently asserts Blackwell/SM10X.
  - The roadmap still lists backward-pass optimization as unfinished.
- [ ] The faithful primary OpenKimi training run uses `safe_gate=False`. Enabling
  `safe_gate=True` with a `-5` lower bound changes the gate activation and is an
  explicit ablation, not a drop-in faithful replacement.

### H100 tests worth running

- [ ] Install cuLA in a separate CUDA 12.9 / PyTorch 2.9.1 environment first;
  do not disturb the working CUDA 12.6 / PyTorch 2.7 environment.
- [ ] Reproduce a small subset of the published benchmark on H100:
  - Published-like `H=64`, `D=128`, `B={1,2}`,
    `T={512,1024,4096,8192,16384}`.
  - Compare against both the cuLA-pinned FLA v0.5.0 baseline and OpenKimi's
    currently pinned FLA revision.
- [ ] Benchmark OpenKimi-relevant shapes:
  - `H=6`, `D=128`, `B={1,2,64}`, `T=4096`, as memory permits.
  - Packed total-token variants that match 262,144 tokens.
  - Fixed-length mode is the priority because training emits padding-free
    4,096-token samples.
- [ ] Measure the complete KDA forward module, not only the recurrent core:
  short convolution, Q/K L2 normalization, channel-wise gate construction,
  beta, KDA, output RMSNorm, sigmoid output gate, and projection.
- [ ] Check forward output and final-state parity against FLA and the FP32
  recurrence. Include causality, multiple batches, and deterministic repeats.
- [ ] Measure with and without an initial state and with/without final-state
  output if the inference use case needs both.
- [ ] Record kernel-only time, complete-layer time, peak workspace, launch count,
  and small-head occupancy. cuLA's own roadmap calls out small-B/H/S
  optimization, and OpenKimi uses only six heads rather than the published 64.
- [ ] Capture one Nsight profile to determine whether the H100 result is
  compute-, bandwidth-, occupancy-, or launch-limited.

### cuLA decision gate

- [ ] Do **not** integrate the current SM90 fused path into training because it has
  no backward and does not support faithful `safe_gate=False`.
- [ ] Keep it as an optional future prefill/inference backend only if the
  OpenKimi-shape forward benchmark and parity pass.
- [ ] Reconsider it for training only when SM90 supports forward and backward with
  `safe_gate=False`; then compare all gradients (`q`, `k`, `v`, gate, beta,
  `A_log`, `dt_bias`, and initial state) and require a full-step improvement.
- [ ] Do not spend tomorrow writing a custom backward around a forward-only fused
  kernel. Current profiles show KDA backward is already a major cost, so a
  forward-only substitution cannot solve the training bottleneck.

## Existing backend checks to retain

- [ ] Confirm FLA `chunk_kda` and fused AttnRes are actually selected in the full
  model and no reference fallback is silently active.
- [ ] Re-run FLA KDA and fused AttnRes forward/gradient parity at the exact
  OpenKimi `D=128`, `H=6`, `T=4096` shapes after dependency changes.
- [ ] Keep fused KDA output RMSNorm + sigmoid gate and fused weighted SwiGLU
  enabled; verify their kernels remain selected after backend changes.
- [ ] Keep TE FP8 Current Scaling only on supported dense/latent/shared/LM-head
  projections, with KDA, MLA attention, routers, reductions, and routed experts
  in BF16/FP32 for the first combined test.
- [ ] Recheck FP8-vs-BF16 loss and gradient parity after replacing CE or routed
  experts.
- [ ] Retain eager execution as the baseline. The prior whole-model
  `torch.compile` result was neutral/slightly negative.

## Combined winner and launch-overhead experiments

- [x] Combine the winning checkpoint policy, LM-head chunk, QuACK, and SonicMoE
  only after each passes independently. Preserve per-feature toggles so any
  interaction or regression can be bisected.
- [ ] Re-run the full optimizer-step profile at `64 x 4096`, checkpointing on,
  FP8 Current Scaling, no accumulation, and one H100.
- [ ] Compare full-step tokens/s, peak memory, GPU-busy time, launch count, CPU
  gaps, and numerical outputs against the untouched baseline.
- [ ] Run at least 10 optimizer updates and require finite loss, gradients, AdamW
  state, and parameters.
- [ ] Run a small fixed-batch overfit check and verify loss decreases.
- [ ] Log router auxiliary loss, z-loss, expert loads, entropy, dead experts, and
  maximum-load violation at logging intervals. Check that the backend does not
  change routing statistics.
- [ ] Test the largest microbatch that fits after any memory reduction. Compare
  equal-global-token throughput, not only a larger amount of work per update.
- [ ] Try `torch.compile` only around stable boundaries or the whole model after
  the eager combined path wins. Record graph breaks and compilation time.
- [ ] Attempt CUDA Graph capture only after Nsight confirms no host-visible expert
  metadata or other capture blockers. Use fixed sequence, batch, top-k, and
  checkpointing shapes.
- [ ] If a graph captures, compare eager vs graph replay over many updates and
  verify optimizer and RNG behavior, not just a forward pass.

## Candidate architecture sweeps

These are separate from backend parity and require the candidate configuration to
be implemented first.

- [ ] Add explicit configs for latent 256, routed hidden 768, shared hidden 1,024,
  and top-2/top-4; keep the measured profile available.
- [ ] Add MTP depth 3 and depth 4 configs with loss weights and inference
  acceptance diagnostics.
- [ ] Measure top-2 vs top-4:
  - Total and active parameters.
  - Routed/shared FLOPs and memory.
  - Full-step tokens/s.
  - Router balance, dead experts, and validation loss on an equal-token smoke run.
- [ ] Measure MTP depth 3 vs 4:
  - Added parameters, training time, and peak memory.
  - Main-token loss and per-depth MTP loss.
  - Short stability/overfit behavior.
  - Draft-token acceptance by depth during a small inference evaluation.
- [ ] Do not mix MuonClip into the kernel comparison. Keep AdamW fixed; evaluate
  MuonClip later as an optimizer experiment with its own stability and
  convergence baseline.

## Acceptance summary and artifacts

- [ ] Produce one result table with:
  provider, commit, shape, precision, forward ms, backward ms, full-step ms,
  tokens/s, peak GiB, launch count, numerical error, and decision.
- [ ] Save the machine/software inventory and benchmark results to
  `profiles/h100-kernel-evaluation-2026-07-19.json`.
- [ ] Record exact commands and the location of Nsight Systems/Compute traces.
- [ ] Mark each candidate as one of:
  `adopt`, `keep experimental`, `inference only`, or `reject`.
- [ ] Update the backend documentation and dependency pins only for candidates
  that meet the decision gates.
- [ ] Run the full test suite after integration and commit the results separately
  from implementation changes.
