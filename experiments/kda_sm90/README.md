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
