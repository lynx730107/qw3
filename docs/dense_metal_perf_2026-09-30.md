# Dense Metal optimization measurements (2026-09-30)

Hardware: fanless Apple M5, 24 GB. Model:
`Qwen3.8-27B-UD-Q3_K_XL.gguf`, full 64-layer offload, F16 GQA KV.
Baseline source: `6b9cf46`. Measurements run sequentially, with untimed warmup
and `caffeinate -disu`; no concurrent inference benchmarks. These are short
observations, not sustained-speed guarantees.

## Accepted changes

- Keep the two-column DeltaNet prefill state in FP32 registers for the whole
  chunk, then write the final state once. The algorithm, per-token core output,
  and state layout are unchanged; no quantization or tolerance changes.
- Expand IQ3_S magnitudes into an exact FP32 workgroup table for gate/up pairs.
  Keep the smaller packed table in single matvec: expanding it there did not
  help. At this revision, the original signed/scaled accumulation order was
  retained; the later vector decode pass below reassociates those sums.
- Generalize grouped FlashAttention decode to the dense 6:1 Q/KV geometry.
  Automatically use it from 8192 tokens; retain the old short-context path.
  `QW3_METAL_GQA_FLASH_DECODE_GROUPED=0` disables it, `=1` forces eligibility
  from 1024 tokens. The existing 8:1 MoE threshold stays unchanged.

## Isolated comparisons

| Measurement | Baseline | Candidate |
| --- | ---: | ---: |
| `pp128`, 5 repeats | 144.55 +/- 0.81 tok/s | 152.84 +/- 1.15 tok/s |
| Warm IQ3_S gate/up pair, 100 calls | 0.9682 ms | 0.8448 ms |
| Warm IQ3_S single, 100 calls | 0.5343 ms | 0.5331 ms |
| GQA attention + output at 16385 tokens, 64 calls | 1.8105 ms | 1.5833 ms |

The prefill comparison is about +5.7% throughput. The pair kernel time falls
about 13%. Neither isolated kernel improvement is an end-to-end decode claim.
The GQA timing used the initial constant-input probe; the final regression now
varies input embeddings every 64 positions so constant V cannot conceal wrong
head routing, attention weights, or split/tail reduction.

## Whole-model comparison and variability

Reference: local `llama.cpp` build `4fea119de` (11041), the same dense GGUF,
full GPU offload, FlashAttention on, F16 K/V, batch and microbatch 128. An
initial sequence gave qw3 prefill 149.01 +/- 0.60 versus llama 146.37 +/- 0.56
tok/s, but alternating runs did **not** confirm that apparent overtake:

| Sequential run | `pp128`, 5 repeats |
| --- | ---: |
| llama | 160.29 +/- 1.03 tok/s |
| qw3 candidate | 151.56 +/- 0.99 tok/s |
| llama | 159.54 +/- 0.87 tok/s |
| qw3 candidate | 150.26 +/- 0.91 tok/s |

The recent prefill gap is still roughly 5-6%; parity is not demonstrated.
For generation at depth 128, a `tg32` sequence gave qw3 7.16 +/- 0.09 versus
llama 8.38 +/- 0.31 tok/s. A later baseline/candidate `tg32` sequence gave
7.12 +/- 0.11 / 6.78 +/- 0.28, so the isolated pair speedup does not establish
a stable whole-model gain. Reverse-order `tg16`, 3-repeat runs yielded:

| Sequential run | tok/s |
| --- | ---: |
| qw3 candidate | 7.38 +/- 0.03 |
| qw3 old kernels | 7.20 +/- 0.08 |
| qw3 candidate | 7.17 +/- 0.05 |
| qw3 old kernels | 7.00 +/- 0.03 |

These short pairs suggest a 2-3% advantage, but also show session drift.
Generation remains below llama; no sustained decode speedup is claimed.
The old-kernel comparisons use the same current host binary and a saved
baseline Metal source directory through `QW3_METAL_KERNEL_DIR`.

## Rejected experiments

- Dense tiled single-token DeltaNet, after correcting its gate launch to cover
  all 48 value heads: alternating `tg16` measurements were 7.35 / 7.28 /
  7.26 / 7.22 tok/s (serial / tiled / serial / tiled). Do not enable it in the
  dense decoder. Its CPU logit and 8-step greedy comparisons passed.
- Byte-expanded IQ3_S tables: no demonstrated whole-model benefit.
- FP32-expanded tables for single IQ3_S matvec: retain the packed-table path.
- Grouped dense attention below 8K: no consistent advantage in the 1K/4K probe;
  leave the earlier decode kernel selected there by default.

## Further decode pass (baseline `423e673`)

This follow-up changes IQ3_XXS and IQ2_XXS/XS/S single-token projections to
32-value work units per SIMD lane, with two output rows per SIMD group and
vector-loaded activations reused across them. At 5120 inputs, the former
whole-block mapping had only 20 blocks for 32 lanes. IQ3_S paired gate/up also
uses vector dot products. No prefill matrix kernel is changed in this pass;
mixed-prefill fallback matvecs do benefit.

The lookup magnitudes and signs are exact. IQ2 caches the small integer
magnitudes in half to bound workgroup memory (4/8/16 KiB for XXS/XS/S), then
converts them to FP32. Scaled weights, activations, products and accumulators
stay FP32. Dot-product sums and scale application are reassociated, so this is
not a bit-identical arithmetic change. Existing differential tolerances are
unchanged.

Sequential warmed probes, 100 calls each:

| Projection | `423e673` kernels | New kernels |
| --- | ---: | ---: |
| IQ2_XXS, 5120 -> 17408 | 0.7660 ms | 0.4737 ms |
| IQ2_XS, 5120 -> 17408 | 0.8527 ms | 0.5204 ms |
| IQ2_S, 5120 -> 17408 | 0.8775 ms | 0.6710 ms |
| IQ3_XXS, 5120 -> 17408 | 1.0815 ms | 0.4959 ms |
| IQ3_XXS, 17408 -> 5120 | 0.9601 ms | 0.4918 ms |
| IQ3_S gate/up pair, 5120 -> 17408 each | 0.8523 ms | 0.8016 ms |

These are isolated observations. Unchanged kernels also show timing drift;
do not interpret every probe difference as a source-code speedup.

Whole-model `tg32`, depth 128, context allocation 256, F16 KV. The following
runs occurred in order, using five timed repetitions and untimed warmup:

| Run | tok/s |
| --- | ---: |
| New IQ2/IQ3 kernels | 8.50 +/- 0.11 |
| New IQ3, old IQ2 (`QW3_METAL_DENSE_IQ2_VECTOR=0`) | 8.13 +/- 0.12 |
| All `423e673` kernels | 7.18 +/- 0.14 |
| New IQ2/IQ3 kernels | 8.29 +/- 0.13 |
| llama.cpp `4fea119de`, same GGUF | 8.69 +/- 0.15 |

The adjacent old/new comparison is +15.5% decode throughput. The last new
run is about 4.6% below the llama reference. This narrows the gap, but does not
establish parity or sustained performance on the fanless machine.

A subsequent depth-1024 comparison (`tg32`, ctx 1152, three repetitions)
gave old kernels **7.39 +/- 0.08**, new kernels **8.63 +/- 0.20**, then
llama.cpp **8.66 +/- 0.31** tok/s. The measured decode gain is +16.8%; new qw3
and llama are essentially tied in this short sample, not proof of a sustained
or across-context overtake.

Earlier calibration was noisier: old kernels initially measured 5.20 +/- 0.14
tok/s; a subsequent new/old/new sequence, before the IQ2 changes, gave
8.24 +/- 0.02 / 6.75 +/- 0.07 / 8.18 +/- 0.08 (three repetitions).
Do not use the anomalously slow first sample to advertise a 60% speedup.

For all-old-kernel comparisons, the current host executable uses a saved Metal
directory from `423e673`, with `QW3_METAL_DENSE_IQ2_VECTOR=0` so its dispatch
matches the old IQ2 kernels. This flag alone reverts IQ2, not IQ3.

Validation passed: full dense bring-up (all 13 input and 8 inner quantization
formats, output projection, two-token CPU logits, unchanged eight greedy IDs,
F16/F32 KV, 128-token prefill/continuation and exact session roundtrip), plus
the MoE logit/greedy/prefill suite. The dense diagnostic now compares both
IQ3_S pair outputs directly and checks 60 cases: input sizes 256/768/1280,
output sizes 1/3/7/13, across IQ2_XXS/XS/S, IQ3_XXS and paired IQ3_S. The test
script independently rejects non-finite or excessive pair/tail error metrics;
stdout/stderr are separated for the large diagnostic to prevent log lines
being interleaved mid-record. Runtime-fingerprint and agent parser/verification
self-tests also passed. The real dense agent git-tool smoke test passed with
cold, warm and disabled prefix cache, without empty-response retries; the
dense agent generation/checkpoint/prefix-cache smoke test passed as well.
CPU vectors, model profiles and the quant-format comparison against local
ggml also passed.

## Reproduction and validation

```sh
./qw3-bench --metal -m ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf \
  --ngl 64 --llama-style -p 128 -n 0 -r 5 --ctx-alloc 256
./qw3-bench --metal -m ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf \
  --ngl 64 --llama-style -p 0 -n 32 -d 128 -r 5 --ctx-alloc 256
./qw3-bench --metal -m ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf \
  --ngl 64 --llama-style -p 0 -n 32 -d 1024 -r 3 --ctx-alloc 1152
QW3_METAL_DENSE_PROBE_REPS=100 ./qw3-test --metal \
  -m ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf --ctx 32 \
  --metal-session-quant-types-test 66
make test-dense-metal-bringup test-dense-gqa-decode test-metal-logits \
  test-metal-gqa-decode test-dense-agent test-agent-python-smoke \
  test-agent-git-smoke test-agent-verify test-runtime-fingerprint
../../llama.cpp/build/bin/llama-bench \
  -m ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf \
  -ngl 99 -p 128 -n 0 -r 5 -ctk f16 -ctv f16 -fa on -b 128 -ub 128 -o json
../../llama.cpp/build/bin/llama-bench \
  -m ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf \
  -ngl 99 -p 0 -n 32 -d 128 -r 5 -ctk f16 -ctv f16 -fa on -b 128 -ub 128 -o json
../../llama.cpp/build/bin/llama-bench \
  -m ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf \
  -ngl 99 -p 0 -n 32 -d 1024 -r 3 -ctk f16 -ctv f16 -fa on -b 128 -ub 128 -o json
```

The dense GQA target covers 1023, 1024, 1025, 4097 and 16385 tokens against
CPU, including forced grouped dispatch and uneven tails. Dense bring-up covers
all supported quantizations, CPU logits, 8 greedy tokens, 128-token prefill
and continuation, F32 KV, and exact session export/import. No test thresholds
were loosened.
