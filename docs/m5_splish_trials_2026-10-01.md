# M5 single-token and staging trials after inspecting Splish

Production baseline: `a3f7670`, base Apple M5, 24 GB, macOS 26.6.2.
Target: `../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf`, full 64-layer offload,
F16 GQA KV. All GPU runs below were sequential under `caffeinate -disu`.
No MTP, DFlash, model conversion, sampler change or agent change is included.
**No candidate from this investigation was promoted to production.**

## What transfers from Splish

Inspected [Splish](https://github.com/publicExcess/splish) at
`c1b4aef460a0cd5680a196b3d7bb5d2815155b06`. Its published 178–191 tok/s
27B results use a 40-core M5 Max and DFlash drafting, not our ordinary
one-token decoder. The package quantization and device differ too.
[Method and results](https://github.com/publicExcess/splish/blob/c1b4aef460a0cd5680a196b3d7bb5d2815155b06/RESULTS.md).

Its adopted GGUF G4a loop uses a single weight stage and input prefetch;
its tuned policies depend on GPU and matrix shape. Those are hypotheses
to test locally, not performance numbers to import.
[GGUF stage](https://github.com/publicExcess/splish/blob/c1b4aef460a0cd5680a196b3d7bb5d2815155b06/runtime/metal/kernels/common/gguf_staged_tile.h).
We also use paired whole-operation measurements, following the warning that
isolated winners need not improve the full step, and Student-t intervals
for small repetition counts.
[M5 Pro measurements](https://github.com/publicExcess/splish/issues/1).

## Fresh local reference (production code unchanged)

Five repetitions with warmup. These are sequential observations, not an
interleaved cross-engine confidence interval or a sustained-speed guarantee.
Both engines use the same GGUF, full GPU offload and F16 K/V. llama.cpp is
the local `build/bin/llama-bench`, batch/microbatch 128, FlashAttention on.
qw3 allocates context 256. `ngl=99` in llama means full available offload,
not a different number of model layers than qw3's `ngl=64`.

| Workload | qw3 tok/s (sample SD) | llama tok/s (sample SD) |
| --- | ---: | ---: |
| pp128, depth 0 | 154.97 (0.29) | 160.060 (1.264) |
| tg32, depth 0 | 9.16 (0.09) | 9.165 (0.050) |
| tg32, depth 128 | 8.78 (0.05) | 9.134 (0.066) |

pp128 and depth-128 generation remain below the reference. Empty-context
generation being close does not prove the complete objective.

Logs: `/tmp/qw3-m5-stream-baseline.log`, `/tmp/qw3-m5-stream-llama.log`,
`/tmp/qw3-m5-stream-baseline-d128.log`, `/tmp/qw3-m5-stream-llama-d128.log`.

## Decode experiments

`tests/iq3_decode_candidates.metal` is compiled only by the optional
`qw3-iq3-stream-bench`. No production shader or dispatch switch is added.

- IQ3_XXS: exact magnitudes in a `uchar4` table rather than `float4`
  (1 KiB instead of 4 KiB).
- Read packed quants/signs through scalar 16-bit loads, respecting GGUF's
  two-byte alignment. No unsafe four-byte-aligned weight reads.
- Prefetch the next packed payload into registers before the current dot.
- Float-table variants isolate the payload/prefetch effects.
- IQ3_S paired gate/up: exact byte magnitudes (2 KiB instead of 8 KiB).

Products and accumulators remain FP32, with the same dot-product grouping.
All six candidates passed six shapes, including 256->1, 4352->33,
5120->4097/4111/17408 and 17408->5120. Alternate shapes reinterpret a
bounded real-weight tensor prefix; they are not all actual model shapes.
Both gate and up outputs are checked. Buffer offsets and guard regions
are nonzero. Full old/new GPU vectors were identical; an independent CPU
double dot on first/middle/last rows agreed within 1e-4. IQ3_S has a private
scalar dequantization reference because `tensor_read_dense_row` does not
support that format. The shell test checks 36 finite metric records.

Whole-model tests use independent sampled sessions, seed 42, temperature
0.6, top-k 20, top-p 0.95, repetition penalty 1.06, alternating AB/BA
forwards, eight untimed warmup forwards and 120 timed forwards. Sampling
is also measured. All vocabulary logits are checked after every one of
the 128 forwards, not just the sampled token IDs.

| Candidate | Paired baseline / candidate tok/s | Forward throughput change |
| --- | ---: | ---: |
| IQ3_XXS bytes + word loads | 9.113 / 9.129 | +0.19% |
| IQ3_XXS bytes + word loads + prefetch | 9.069 / 9.050 | -0.19% |
| IQ3_S paired byte table | 8.963 / 8.977 | +0.18% |

All three had **exactly identical logits** at every step and identical
128-token hash `6e27f5726999f2da`. The results do not establish a useful
speedup. Some isolated projections looked faster, but that did not carry
over to the model. Two-session rates are not normal single-session rates.

Logs: `/tmp/qw3-iq3-stream-full-words.log`,
`/tmp/qw3-iq3-stream-full-prefetch.log`,
`/tmp/qw3-iq3-stream-full-sbytes.log`,
`/tmp/qw3-iq3-stream-final-quant.log`.

## Prefill experiments

Eight paired pp128 trials, one untimed paired warmup, identical synthetic
seed-1 token prefix, independent ctx-256 sessions. Compare the entire
vocabulary and history after prefill and after **eight greedy continuation
tokens per trial**. Every comparison was bit-identical. Timing excludes
logit checks and continuation. The 95% paired interval uses Student-t
with seven degrees of freedom, not the large-sample 1.96 multiplier.

| Candidate | Baseline / candidate tok/s | Change | Saved ms, paired 95% interval |
| --- | ---: | ---: | ---: |
| 4 KiB dynamic weight stage instead of 8 KiB | 151.858 / 152.197 | +0.22% | 1.875 +/- 5.068 |
| Explicit half4 stage writes (IQ3_S and IQ4_XS) | 150.501 / 150.777 | +0.18% | 1.555 +/- 8.590 |
| Vector IQ3_S dequantization + half4 stage writes | 149.621 / 149.778 | +0.11% | 0.900 +/- 8.513 |

**Every interval contains zero; none was retained.** The temporary 4 KiB
host allocation switch was removed completely. The optional
`qw3-prefill-staging-bench` reproduces the other two trials by compiling
modified source in memory and privately selecting the two pipelines.
It does not edit source files or change the normal executable. Source
replacement counts are asserted so a changed shader fails explicitly.

A final rebuilt vector-dequantization replication, with the explicit F16
KV guard, measured 150.004 / 149.879 tok/s (-0.08%), saved
-0.713 +/- 9.614 ms, again with exact logits and continuations. This
reinforces rejection rather than a positive speed claim.
Log: `/tmp/qw3-prefill-staging-final.log`.

Logs: `/tmp/qw3-prefill-memory-paired.log`,
`/tmp/qw3-prefill-vector-store-paired.log`,
`/tmp/qw3-prefill-vector-dequant-paired.log`.

## Reproduce

```sh
make qw3-iq3-stream-bench qw3-prefill-staging-bench
caffeinate -disu make test-iq3-stream-quant
caffeinate -disu ./qw3-iq3-stream-bench \
  ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf --full iq3_words
caffeinate -disu ./qw3-iq3-stream-bench \
  ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf --full iq3_prefetch
caffeinate -disu ./qw3-iq3-stream-bench \
  ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf --full iq3s_bytes
caffeinate -disu ./qw3-prefill-staging-bench \
  ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf vector_store
caffeinate -disu ./qw3-prefill-staging-bench \
  ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf vector_dequant
./qw3-bench --metal -m ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf \
  --ngl 64 --llama-style -p 128 -n 32 -r 5 --ctx-alloc 256
./qw3-bench --metal -m ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf \
  --ngl 64 --llama-style -p 0 -n 32 -d 128 -r 5 --ctx-alloc 256
../../llama.cpp/build/bin/llama-bench \
  -m ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf -ngl 99 \
  -p 128 -n 32 -r 5 -ctk f16 -ctv f16 -fa on -b 128 -ub 128 -o json
../../llama.cpp/build/bin/llama-bench \
  -m ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf -ngl 99 \
  -p 0 -n 32 -d 128 -r 5 -ctk f16 -ctv f16 -fa on -b 128 -ub 128 -o json
```

Do not run inference benchmarks concurrently. No speed threshold in the
new quant test is used as a correctness gate. The immediate remaining
performance targets are roughly 3.3% for pp128 and 4% for depth-128 tg32,
with another interleaved cross-engine confirmation required before claiming
parity. Speculative speedups remain a different measurement.

Validation after removing the temporary host allocation change:
`make test-iq3-stream-quant` (36 cases), `make test-runtime-fingerprint`
(two tests), and `make test-dense-metal-bringup test-dense-agent
test-agent-verify` passed. Bring-up includes the existing CPU projection/
logit, greedy, batch-continuation and state-roundtrip checks; the agent smoke
includes generation, checkpoint and prefix-cache reuse. Those production
tests exercise the unchanged default, not a promoted candidate.
Log: `/tmp/qw3-m5-stream-production-regressions.log`.
The existing sequential Metal-vs-llama full-logit and greedy differential
also passed via `make test-dense-metal-logits` on the unchanged default.
Log: `/tmp/qw3-m5-stream-llama-correctness.log`.
