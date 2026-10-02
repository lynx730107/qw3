# M5 IQ4 layout and prefill activation staging trials

Starting production revision: `69d12f1`. Base Apple M5, 24 GB,
Qwen3.8-27B UD-Q3_K_XL GGUF, full 64-layer offload, F16 GQA KV.
GPU workloads ran sequentially under `caffeinate -disu`.
**No candidate in this report is adopted in production.**

This continues the [earlier Splish inspection](m5_splish_trials_2026-10-01.md).
Splish's current `m5` branch still resolves to
`c1b4aef460a0cd5680a196b3d7bb5d2815155b06`. Its
[prepared GGUF layout](https://github.com/publicExcess/splish/blob/c1b4aef460a0cd5680a196b3d7bb5d2815155b06/runtime/metal/abi/QuantFormat.h)
and [single-stage/input-prefetch loop](https://github.com/publicExcess/splish/blob/c1b4aef460a0cd5680a196b3d7bb5d2815155b06/runtime/metal/kernels/common/gguf_staged_tile.h)
provide local hypotheses. Its DFlash, affine-4-bit and M5 Max results do not
predict ordinary one-token decode on this base M5.

## Native IQ4_XS payload layout

`tests/iq4_layout_probe.inc` privately prepares four layouts:

- `blocks4`, `blocks16`: transpose complete 136-byte GGUF blocks across
  tiles of four or sixteen output rows.
- `planes4`, `planes16`: separate the eight-byte metadata headers and
  transpose eight-byte quant payloads across the row tile.

All 136 native bytes per block are preserved. Final row tiles are compact;
there is no row padding, tensor expansion or dequantized weight cache.
An additional 32-byte guard at each end is used only by the experiment.
Every prepared byte is compared against its native source. Kernel address
calculations change, while FP32 dot grouping, per-lane accumulation and
SIMD reduction order remain the same.

All four alternatives passed single, paired gate/up and residual-add
projections on six shapes (72 cases): 256->1, 4352->33,
5120->4097/4111/17408, 17408->5120. Alternate shapes reinterpret a bounded
real tensor prefix, rather than claiming that all are actual model tensors.
Complete GPU vectors match exactly and remain finite. Independent double
CPU dots on first/middle/last rows pass max 1e-4; nonzero buffer offsets,
negative residual scale, nonzero residuals and guards are checked.
Micro timings use four warmups, 40 alternating AB/BA repetitions and GPU
command-buffer durations. Complete-block layouts were generally slower;
the plane layouts were mixed across operations.

The `planes16` full-model probe prepares all 69 IQ4_XS dense FFN tensors,
3115.78 MiB, and retains their native mapped weights. Therefore this
prototype adds about **3.04 GiB of live weight buffers**, despite its
unchanged payload size. Preparation and complete byte checking took 0.775 s
in the first run. No loading/memory policy is enabled in the normal engine.

The whole-model method is the
[sampled two-session decode comparison](m5_decode_trials_2026-10-02.md):
same 26-token chat prefix, seed 42, temperature 0.6, top-k 20, top-p 0.95,
repetition penalty 1.06; 128 generated tokens, eight untimed warmups,
120 alternating timed forwards, and separately timed sampling included in
reported tok/s. All vocabulary logits and histories are checked after
every forward; live K/V, full DeltaNet and convolution state are compared
after 154 committed tokens. Unused KV and derived FlashAttention scratch
are excluded from canonical-state comparison.

| Decode trial | Baseline / candidate tok/s | Forward change | Saved ms/token |
| --- | ---: | ---: | ---: |
| Planes16 | 9.104 / 9.164 | +0.69% | 0.751883 |
| Planes16 replication | 9.034 / 9.068 | +0.40% | 0.442475 |
| Native-copy allocation control | 8.858 / 8.853 | -0.03% | -0.039175 |

The `native_copy` control keeps the original contiguous GGUF layout and
original pipelines, copies the same 69 tensors into allocated buffers, and
uses the same private call wrappers. Its effect was essentially neutral.
This addresses the mapped-file versus allocated-buffer confound, but one
noisy control does not establish a precise causal estimate for layout.
It had a descriptive token-level 95% interval of +/-0.633414 ms, assuming
independent tokens. Tokens are correlated, so that interval is not a
promotion criterion. Do not compare absolute rates from separate runs.

Each full layout/control run verified exactly 55 redirected FFN dispatches
per generated token (7040 total), identical hash `6e27f5726999f2da`, exact
full logits, exact canonical state and intact weight guards. The modest
gain does not justify the prototype's extra allocation and new weight path
on a 24 GB machine. `blocks4`, `blocks16` and `planes4` were measured only
as projection microbenchmarks in this investigation.

## IQ4_XS payload prefetch on the original layout

`tests/iq4_prefetch_probe.inc` loads the next native block's metadata and
two quant words into private payload variables before the current block's
single/paired/residual dots. The next block is still accumulated only in
its original iteration. No prepared weights or additional model allocation
is used.

All 18 projection shape/operation cases passed exact finite GPU vectors,
CPU sampled-row dots and guards. Actual large projection timings were mixed.

| Decode trial | Baseline / candidate tok/s | Forward change | Saved ms/token |
| --- | ---: | ---: | ---: |
| Payload prefetch | 9.087 / 9.099 | +0.16% | 0.176183 |
| Payload prefetch replication | 8.608 / 8.616 | +0.11% | 0.128808 |

Descriptive token-level intervals are +/-0.181739 and +/-0.476660 ms;
both contain zero. Both runs pass exact full logits, histories and canonical
state, with the same 128-token hash. No dependable speedup is established.

## Prefill geometry and input staging

The existing private pp128 probe retains 64 output columns, K=32 and
128-token weight reuse. `sg8` distributes the chunk as sixteen token rows
per SIMD group over eight groups/256 threads, instead of thirty-two rows
over four groups/128 threads. Only the original 128 threads dequantize
weights; the extra threads still participate in every barrier and matmul.
The private thread-count hook defaults to constant 128 in normal builds.

`input_stage` first copies the current 128x32 FP32 input tile into a static
16 KiB threadgroup array. `input_prefetch` additionally loads the next tile
into 32 FP32 values per thread before the current matmul. Weight decoding,
FP32 accumulation, output stores and K-loop order remain as before.

Eight alternating paired pp128 trials follow one untimed paired warmup,
using independent ctx-256 sessions and the same seed-1 synthetic prefix.
GPU row copies are off on both sides to isolate the matrix candidates.
Every trial compares the full vocabulary after prefill and after eight
greedy continuations. Final canonical-state checks are at 136 tokens.
Checks/readbacks are outside timing. Intervals use Student-t df=7.

| Prefill candidate | Baseline / candidate tok/s | Change | Saved ms +/- paired 95% interval |
| --- | ---: | ---: | ---: |
| Eight SIMD groups, 16 token rows each | 150.001 / 139.938 | -6.71% | -61.363 +/- 5.648 |
| FP32 input stage | 151.415 / 125.466 | -17.14% | -174.834 +/- 6.544 |
| FP32 input stage + next-tile prefetch | 150.570 / 49.918 | -66.85% | -1714.119 +/- 16.701 |

All three passed bit-identical logits, greedy continuations and canonical
states, but all were substantially slower. The added live values and shared
memory are possible costs; no register-spill or hardware occupancy measurement
was taken, so the cause should not be presented as proven.

An `input_half` experiment halves the input stage to 8 KiB. It fails the
exact prefill check at vocabulary ID 0: 0.721617818 versus 0.721832395.
It is rejected before timed trial results; thresholds were not relaxed.
The analogous `input_half_prefetch` mode is available privately but was
not run because the input conversion has already failed the exactness gate.

## Reproduce and logs

```sh
make qw3-decode-trials-bench qw3-prefill-staging-bench
caffeinate -disu ./qw3-decode-trials-bench \
  ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf --micro planes16
caffeinate -disu ./qw3-decode-trials-bench \
  ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf --full planes16
caffeinate -disu ./qw3-decode-trials-bench \
  ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf --full native_copy
caffeinate -disu ./qw3-decode-trials-bench \
  ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf --micro iq4_prefetch
caffeinate -disu ./qw3-decode-trials-bench \
  ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf --full iq4_prefetch
caffeinate -disu ./qw3-prefill-staging-bench \
  ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf sg8
caffeinate -disu ./qw3-prefill-staging-bench \
  ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf input_stage
caffeinate -disu ./qw3-prefill-staging-bench \
  ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf input_prefetch
```

Source extraction/replacement counts are asserted. Optional probes do not
add a normal CLI/agent dependency or a production runtime selector.

Logs: `/tmp/qw3-layout-{blocks4,blocks16,planes4,planes16}-micro.log`,
`/tmp/qw3-layout-planes16-full.log`,
`/tmp/qw3-layout-planes16-full-repeat.log`,
`/tmp/qw3-iq4-native-copy-full.log`,
`/tmp/qw3-iq4-prefetch-micro.log`,
`/tmp/qw3-iq4-prefetch-full.log`,
`/tmp/qw3-iq4-prefetch-full-repeat.log`,
`/tmp/qw3-prefill-sg8.log`, `/tmp/qw3-prefill-input-stage.log`,
`/tmp/qw3-prefill-input-prefetch.log`, `/tmp/qw3-prefill-input-half.log`.

Production validation passed `test-runtime-fingerprint` (two tests),
`test-dense-metal-bringup`, `test-dense-agent`, `test-agent-verify`,
`test-metal-logits` (MoE), and the existing 42-case
`test-decode-trials-quant`. Bring-up covers the normal 128-token prefill,
continuation, CPU projection/logit checks, greedy IDs, F16/F32 sessions and
state roundtrip. The agent smoke covers generation, checkpoints and prefix
reuse. Normal CLI/agent/bench binaries were rebuilt successfully.
Log: `/tmp/qw3-layout-staging-production-regressions.log`.

These trials do not prove parity or an overtake of llama.cpp. The production
prefill-copy improvement remains the latest adopted speed change. No MTP,
DFlash, sampler, agent prompt or tool behavior is changed here.
