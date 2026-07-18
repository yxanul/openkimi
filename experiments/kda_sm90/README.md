# SM90 KDA experiment

`original/` contains the two CuTe DSL drafts supplied for this experiment,
copied byte-for-byte before any H100 work:

- `sm90_cute_dsl.py`
  (`sha256: 90f51f9cbee008d1e75299c029f83eda31e9dce1a2f5e55efcf421078a6f325b`)
- `sm90_cute_bwd.py`
  (`sha256: 51c81f872fa93399e4bd01527908a63f036e217bb9c30d5849981c58d3b7b484`)

The originals are retained as design inputs rather than imported by training.
They describe a 16-token, 128-channel KDA decomposition with FP32 state, but
they are not yet a PyTorch operator:

- the forward scan's large matrix products are explicit SIMT FP32 loops;
- TMA and tensor-core substitutions are comments rather than implementations;
- the forward file's `KdaBackwardSm90` deliberately raises;
- the separate backward stages have no launch orchestrator, workspace owner,
  PyTorch autograd wrapper, or DLPack bridge;
- their chunk size and checkpoint ABI differ from FLA's current 64-token path;
- they were written against a CUTLASS 4.5-era API, while the test stack has
  NVIDIA CUTLASS DSL 4.6.0.

Candidate kernels derived from these files must retain the exact OpenKimi
semantics: BF16 Q/K/V/raw gate/beta inputs, FP32 `A_log` and `dt_bias`, in-kernel
Q/K L2 normalization, channel-wise
`-exp(A_log) * softplus(raw_gate + dt_bias)`, sigmoid beta, `safe_gate=False`,
FP32 recurrent state, scale `128**-0.5`, and gradients for every input.

The acceptance baseline is the faster of FLA's recompute and saved-intermediate
policies at `[B,T,H,D] = [32,4096,6,128]`, followed by the complete
262,144-token optimizer update. Use `scripts/benchmark_kda_backends.py` for the
standalone comparison.

## H100 results (2026-07-18)

The exact-shape standalone comparison used BF16 inputs, FP32 `A_log` and
`dt_bias`, `safe_gate=False`, seven measured repetitions, and gradients for all
seven inputs.

| FLA policy | Forward | Backward | Total | Peak allocated |
|---|---:|---:|---:|---:|
| Recompute backward intermediates | 3.639 ms | 14.503 ms | 18.144 ms | 6.590 GiB |
| Save backward intermediates | 3.773 ms | 12.942 ms | 16.718 ms | 7.530 GiB |

The two policies produced bit-identical outputs and gradients in this test.
Saving intermediates reduced standalone KDA forward+backward time by 7.86%.
In the complete FP8-current + QuACK + SonicMoE optimizer update, enabling
`kda_disable_recompute` changed the seven-run median from 1,845.016 ms
(142,082 tok/s) to 1,821.177 ms (143,942 tok/s), a 1.31% throughput gain. The
saved-intermediate profile is therefore the baseline for subsequent custom
kernel work.

The saved-intermediate Nsight trace attributed 427.5 ms of 1,808.5 ms total GPU
kernel time (23.6%) to the directly identifiable KDA chunk forward/backward
path. Including Q/K L2 normalization and gate transforms raises the region to
approximately 27%. The two largest individual KDA kernels were:

- `chunk_kda_bwd_kernel_intra`: 144.36 ms (7.98%);
- `chunk_kda_bwd_kernel_wy_dqkg_fused`: 110.92 ms (6.13%).

## CuTe draft status

The original monolithic prepare kernel did not finish compilation within 180
seconds on the H100. Replacing its 128-term compile-time expansions with device
loops reduced frontend expansion but did not make the complete program
practical. Its recurrent forward products still need BF16/TF32 MMA tiles, and
the separate backward file still needs an orchestrator, workspace ownership,
autograd binding, and mathematical parity tests.

`working/preprocess_cute.py` is the first compiling extraction from that design.
It fuses Q/K L2 normalization, reciprocal-norm saves, channel-wise gate
activation/cumsum, and beta sigmoid into one CuTe launch. At the target shape:

| Preprocessing path | Median |
|---|---:|
| Existing FLA launches | 0.490 ms |
| Fused CuTe experiment | 0.602 ms |

Q/K and beta BF16 results were effectively identical; reciprocal norms differed
by at most `2.24e-8`, and the gate cumulative sum by at most `3.05e-4`.
Nevertheless, the CuTe path was 22.9% slower than FLA, so it is retained only
as an experiment and is **not selected by training**.

No supplied/custom CuTe KDA backend has therefore met the swap bar yet. Current
training uses FLA with saved intermediates. The next kernel effort should start
with the two measured backward kernels and use real tensor-core MMA tiles
rather than extending the scalar draft.

## Guarded diagonal experiment

Inspection of the pinned FLA source found that both diagonal phases of
`chunk_kda_bwd_kernel_intra` use 16 scalar partner iterations when
`safe_gate=False`. `guarded_diagonal_oracle.py` records the stable
first-row-reference factorization and mandatory exact fallback.
`benchmark_guarded_diagonal.py` contains an isolated SM90 Triton prototype that
uses TF32 matrix products only when the runtime within-block log2 gate span
passes a configurable guard.

The isolated prototype is not imported by training.
`patch_fla_guarded_intra.py` generated and verified the full backward-intra
change against the exact upstream source hash during development. The selected
source now lives in the pinned FLA fork used directly by training.

### H100 result (2026-07-18)

`scripts/profile_kda_gate_spans.py` captured the exact log2 cumulative gates
entering all 12 KDA layers of the initialized primary model at
`[B,T,H,D] = [32,4096,6,128]`. Across 2,359,296 `[BC=16,BK=32]` tiles, the
within-tile span distribution was:

| percentile | log2 span |
|---|---:|
| p50 | 143.673 |
| p90 | 233.876 |
| p99 | 248.796 |
| p99.9 | 254.614 |
| maximum | 265.768 |

The original first-row reference is therefore rejected: a threshold of 30
accepts only 2.778% of real tiles. Centering each channel at
`(minimum + maximum) / 2` minimizes the largest factor exponent. A provisional
span guard of 248 accepts 98.742% of initialized-model tiles while keeping the
two FP32 factors at approximately `2**[-124,124]`.

At 196,608 synthetic tiles with mean log2 decay 8 and the same 248 guard:

| implementation | median | speedup vs exact | relative error |
|---|---:|---:|---:|
| Exact pairwise Triton | 2.183 ms | 1.00x | — |
| Guarded midpoint Triton | 1.097 ms | 1.99x | <= 7.64e-4 |
| Guarded midpoint CuTe, plain B SMEM | 1.597 ms | 1.37x | <= 7.64e-4 |
| Guarded midpoint CuTe, s128 B SMEM | 1.121 ms | 1.95x | <= 7.64e-4 |
| Guarded midpoint CuTe, s128 + gate cache | **1.037 ms** | **2.10x** | <= 7.64e-4 |

The CuTe kernel uses four warps for the two lower- and two upper-triangular
products and takes the exact pairwise fallback for rejected, non-monotonic, or
non-finite tiles. It compiles against CUTLASS DSL 4.6 on the clean H100 host.
The original `[channel,row] = [32,16]` TF32 B operands used physical stride 16.
A producer warp writing one fixed row therefore mapped its 32 lanes onto two
shared-memory banks. CuTe s32/s64/s128 XOR swizzles reduced the predicted
maximum conflict from 16-way to 8/4/2-way and reduced the measured median from
1.597 to 1.295/1.136/1.121 ms, respectively. The selected s128 isolated tile is
29.8% faster than the plain CuTe layout and only 2.2% slower than the Triton
prototype. Nsight Compute independently measured 66.7% fewer shared-load bank
conflicts and 63.2% fewer shared-store bank conflicts for s128 than for the
plain layout over 128 CTAs.

Replacing the negative `exp2` with `1 / positive` was also measured on the
s128 layout. It regressed from 1.125 to 1.241 ms despite effectively unchanged
error, so the two-exp2 formulation remains selected.

Caching the gate tile while the first warp validates monotonicity is selected.
It removes the second fast-path global load and lets the exact fallback reuse
the shared copy. The controlled median falls from 1.118 to 1.037 ms, another
7.23% reduction. Nsight Compute records 25.1% fewer global-load L1 sectors.
This wins despite increasing allocated registers from 48 to 64 and reducing
the resident-block limit from nine to eight, so the saved load stream—not
occupancy—is the relevant mechanism. The cached CuTe tile is 5.44% faster than
the earlier Triton prototype.

The other proposed schedules were tested independently:

- Channel-half warp ownership is numerically identical but consistently slower.
  With reversed provider order and 20 warmups, product ownership measures
  1.033 ms versus 1.048 ms for channel-half, a 1.43% regression. Reusing the
  lower B fragment and fusing the upper sum does not offset the less favorable
  half-width schedule.
- Direct Dq/Dk accumulator stores remove 4 KiB of output scratch but regress
  from 1.042 to 1.074 ms (3.05%). The generated global-store pattern loses
  more than the shared epilogue costs.
- Handling the fallback diagonal outside the partner loop improves an
  all-fallback workload from 1.911 to 1.704 ms (10.8%). In the real mixed
  workload, where only 1.61% of synthetic tiles and 1.26% of initialized-model
  tiles fall back, the larger mixed kernel regresses from 1.041 to 1.054 ms.
  Keep the compact loop fallback.

Storing only two unmasked A matrices is deferred. Applying triangular masks in
registers adds instructions and likely register pressure to an already
register-limited cached kernel, without independently improving residency.
Likewise, a 17 KiB fallback decay table would penalize the 98% fast path unless
it lived in a separately dispatched kernel. These are not attractive before
the full FLA ABI integration.

Neither complete path is imported by training yet.

Reproduce the CuTe result after `scripts/bootstrap_h100.sh` with:

```bash
scripts/run_with_quack.sh .venv/bin/python \
  experiments/kda_sm90/benchmark_guarded_diagonal_cute.py \
  --blocks 196608 --mean-log2-decay 8 --threshold 248 \
  --warmup 10 --repetitions 30 --swizzle-bits 3 \
  --gate-cache-modes uncached,cached
```

As a control, the faithful unbounded gate equation was precomputed and passed
to FLA's existing `safe_gate=True` midpoint path. It produced NaN outputs and
all-NaN gradients even at `[1,256,6,128]`. This confirms that the production
kernel cannot merely flip `safe_gate`; it needs the runtime span check and
exact fallback in both diagonal backward phases.

### Full FLA integration result (2026-07-19)

The first complete integration keeps FLA's full `chunk_kda_bwd_intra` ABI and
changes only the two `safe_gate=False` diagonal branches. Each tile validates
monotonicity, finiteness, and a configurable maximum log2 span. Passing tiles use the
midpoint factorization; rejected tiles execute the original pairwise scalar
loops.

Plain TF32 MMA is fast but fails the all-gradient gate. At production shape its
relative L2 errors reach 2.62% for raw decay, 4.67% for `A_log`, and 13.81% for
`dt_bias`; those aggregate decay gradients amplify small tensor-core errors.
Triton's three-pass `tf32x3` mode is selected instead. The provisional
threshold-248 path passes a synthetic production-shape comparison: at
`[32,4096,6,128]`, output and value gradient are bit-identical and the relative
errors for Q, K, raw decay, beta logits, `A_log`, and `dt_bias` are
`2.31e-5`, `2.35e-5`, `7.47e-4`, `1.12e-7`, `3.15e-3`, and `3.55e-3`.
It is nevertheless rejected by the controlled real-model gate below.

The final H100 gate adds near-threshold, high-decay, cancellation-heavy,
4,103-token partial-chunk, non-monotonic fallback, and actual initialized-model
fixtures. A negative runtime threshold forces the same patched kernel through
FLA's original scalar branch, so exact and guarded paths share source and
autotuner choices. High-decay and non-monotonic fallbacks are bit-identical.
The cancellation fixture creates an `A_log` reference gradient with L2 norm
`2.48e-8`; its 52.3% relative error is only `9.22e-9` maximum absolute error.
The gate therefore uses relative L2 `<=5e-3`, with a `5e-5` maximum-absolute
escape only for near-zero references, and rejects every non-finite result.

Comparing two entire model backward passes is invalid here: even exact versus
exact differs by as much as 18.9% for 69 of 180 KDA parameter tensors because
of nondeterministic downstream GPU reductions. Instead, one exact full-model
backward captures the actual activation and downstream gradient for every KDA
mixer. Each of the 12 initialized mixers is then replayed exact and guarded
with identical weights, activation, and upstream gradient, comparing 204
output, input-gradient, and parameter-gradient tensors.

Thresholds 248 and 240 fail two and one of those tensors, respectively.
Threshold **232** is the highest passing value: zero failures and a worst
relative L2 error of `2.409e-3`. It accepts 88.615% of the 2,359,296
initialized-model tiles, compared with 98.742% at the rejected threshold 248.

At production shape, selected threshold 232 changes saved-intermediate FLA
from:

| path | forward | backward | total |
|---|---:|---:|---:|
| Exact pairwise FLA | 3.798 ms | 13.155 ms | 16.947 ms |
| Guarded midpoint TF32x3-232 | 3.786 ms | 12.265 ms | 16.054 ms |

That is a 6.76% backward reduction and a 5.27% total KDA reduction. Exact FLA
controls take 1,832.31 and 1,830.69 ms for the complete 262,144-token optimizer
update; guarded TF32x3-232 takes 1,787.93 ms. Throughput rises from
143,068–143,194 to 146,619 tok/s (`+2.39–2.48%`) with unchanged 58.66 GiB peak
allocation. The faster 147,440 tok/s result at threshold 248 is retained only
as a rejected historical measurement.

The selected integration is productionized as
`yxanul/flash-linear-attention@ee8369bb735bcc91aefc967ea911cc75248a1b79`,
one commit on top of upstream
`ccb0ff944cbff035fa59ac47a4cc8fd2e079bb17`. `pyproject.toml` and `uv.lock`
install that exact source. OpenKimi and `scripts/bootstrap_h100.sh` fail closed
unless patch version 1, TF32x3, and span `<=232` are active; no runtime
`site-packages` mutation remains. The reversible patch script stays only as
experiment provenance and a source-transform verification tool. Native CuTe
full-ABI work now has to beat the selected TF32x3-232 result, not the old scalar
baseline.

A clean locked H100 reinstall resolved that exact commit without invoking the
development patcher. The installed source reports patch version 1, TF32x3, and
span 232, and its SHA-256 is
`d036a681ebc1f6ea81b23209d3e89cdab3dd04d83b0d53d29d36616bdb0c0f3d`.
Four backend tests, 20 H100 parity/platform tests, all five adversarial fixtures,
and the 12-mixer/204-tensor controlled replay pass from the installed package.
The post-package no-regression run measures 884.806 ms for one
131,072-token microstep, or 148,136 tok/s; two configured accumulation
microsteps correspond to 1,769.611 ms for 262,144 tokens. This confirms the
packaging did not erase the selected kernel win; the A/B/A result above remains
the controlled comparison against exact upstream FLA.

Raw measurements are recorded in:

- `profiles/h100-sm90-kda-gate-spans-2026-07-18.json`
- `profiles/h100-sm90-kda-guarded-midpoint-2026-07-18.json`
- `profiles/h100-sm90-kda-guarded-diagonal-cute-2026-07-18.json`
- `profiles/h100-sm90-kda-guarded-diagonal-cute-swizzle-2026-07-18.json`
- `profiles/h100-sm90-kda-guarded-diagonal-cute-exp2-2026-07-18.json`
- `profiles/h100-sm90-kda-guarded-diagonal-cute-bank-conflicts-2026-07-18.json`
- `profiles/h100-sm90-kda-guarded-diagonal-cute-gate-cache-2026-07-19.json`
- `profiles/h100-sm90-kda-guarded-diagonal-cute-gate-cache-ncu-2026-07-19.json`
- `profiles/h100-sm90-kda-guarded-diagonal-cute-direct-epilogue-2026-07-19.json`
- `profiles/h100-sm90-kda-guarded-diagonal-cute-channel-half-2026-07-19.json`
- `profiles/h100-sm90-kda-guarded-diagonal-cute-channel-half-reverse-2026-07-19.json`
- `profiles/h100-sm90-kda-guarded-diagonal-cute-fallback-diagonal-2026-07-19.json`
- `profiles/h100-sm90-kda-guarded-diagonal-cute-fallback-mixed-2026-07-19.json`
- `profiles/h100-sm90-kda-fla-safe-control-2026-07-18.json`
- `profiles/h100-sm90-kda-before-guarded-integration-2026-07-19.json`
- `profiles/h100-sm90-kda-guarded-fla-tf32x3-2026-07-19.json`
- `profiles/h100-sm90-kda-fla-guarded-integration-2026-07-19.json`
- `profiles/h100-sm90-kda-guarded-adversarial-232-2026-07-19.json`
- `profiles/h100-sm90-kda-gate-spans-232-2026-07-19.json`
- `profiles/h100-sm90-kda-guarded-real-model-threshold-sweep-2026-07-19.json`
- `profiles/h100-sm90-kda-guarded-fla-tf32x3-threshold232-2026-07-19.json`
- `profiles/h100-sm90-kda-fla-guarded-validation-2026-07-19.json`
- `profiles/h100-sm90-kda-pinned-fork-validation-2026-07-19.json`
- `profiles/h100-sm90-kda-pinned-fork-benchmark-2026-07-19.json`
- `profiles/h100-sm90-kda-pinned-fork-2026-07-19.json`

## WY/intra backward fusion result (2026-07-19)

A fresh Nsight control supersedes the older 255.28 ms estimate. Across the
configured 262,144-token update, `wy_dqkg_fused` takes 111.909 ms and `intra`
takes 100.100 ms, for 212.009 ms combined. WY already uses 255
registers/thread and 49,664 B of dynamic shared memory; intra uses 128
registers/thread and 16,512 B.

`fused_wy_intra_triton.py` contains two process-local experiments and does not
alter the pinned FLA package:

- A coarsened blocked intra maps one CTA to a complete 64-token chunk for each
  32-channel slice. Four warps reduce the Nsight intra region to 96.165 ms
  (`-3.93%`) and WY+intra to 208.086 ms (`-1.85%`). Full-step A/B/A improves
  from 146,541 to 147,006–147,087 tok/s, only 0.32–0.37%. A controlled replay
  with identical real-model activations and upstream gradients passes all 204
  tensors from 12 KDA mixers; the worst relative L2 error is `2.431e-3`.
- A monolithic four-warp kernel executes the existing WY body and coarsened
  intra body in one CTA. It remains accurate (`4.31e-4` worst relative L2) but
  regresses from 1.862 to 2.191 ms (`+17.6%` latency, `-15.0%` speed). The
  generated kernel uses 255 registers/thread and spills 1,152 B/thread of local
  stack. An eight-warp attempt fails Triton's SM90 MMA lowering because the
  current tile layout requires each chunk to be filled by one warp.

The monolithic candidate is rejected, and the small coarsened-intra win remains
experimental. A viable fusion would require a dedicated CuTe schedule with
explicit phase lifetimes and shared/register ownership, not mechanical Triton
inlining. Because the fused chunk CTA did not win, the conditional fused
backward epilogue and persistent `(b,hv)` reverse scan were deliberately not
implemented. The production pinned FLA backend is unchanged.

Reproduce the isolated comparison with:

```bash
scripts/run_with_sonic.sh .venv/bin/python \
  experiments/kda_sm90/benchmark_fused_backward.py \
  --batch 2 --sequence-length 4096 --candidate blocked \
  --candidate-num-warps 4
```

Run the controlled real-model replay with:

```bash
scripts/run_with_sonic.sh .venv/bin/python \
  experiments/kda_sm90/validate_fused_backward.py
```

The summary and raw measurements are:

- `profiles/h100-sm90-kda-backward-fusion-2026-07-19.json`
- `profiles/h100-sm90-kda-intra-chunk-blocked-w2-b2-2026-07-19.json`
- `profiles/h100-sm90-kda-intra-chunk-blocked-w4-b2-2026-07-19.json`
- `profiles/h100-sm90-kda-intra-chunk-blocked-full-step-2026-07-19.json`
- `profiles/h100-sm90-kda-intra-chunk-control-full-step-2026-07-19.json`
- `profiles/h100-sm90-kda-intra-chunk-blocked-full-step-after-2026-07-19.json`
- `profiles/h100-sm90-kda-intra-chunk-blocked-real-model-2026-07-19.json`
- `profiles/h100-sm90-kda-wy-intra-fused-b2-2026-07-19.json`
