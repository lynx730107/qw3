# M5 prefill geometry and bit-preserving staging copies

Base Apple M5, 24 GB, same dense Qwen3.8-27B UD-Q3_K_XL GGUF,
full 64-layer GPU offload and F16 KV. Starting revision: `e0ec3ea`.
All GPU workloads were sequential under `caffeinate -disu`.
No MTP, model conversion, precision reduction, sampling or agent change.

## Reject tile changes; reduce staging command count instead

The private staging harness changes only IQ3_S/IQ4_XS staged128 kernels
in memory. Smaller column tiles guard unused dequantization threads while
preserving all workgroup barriers; the 128-column variant gives each
thread two weight rows. K=64 stages two 16-element pieces per thread.
The default production tile remains 64 columns, K=32, 128 threads.
The compile-time geometry hook is constant 64 in normal builds; there
is no production geometry setting from these failed experiments.

Eight AB/BA paired pp128 measurements, after one paired warmup; independent
ctx-256 sessions with the same synthetic seed-1 128-token prefix. Every
trial checks all vocabulary logits after prefill and after eight greedy
continuation steps. No arithmetic reassociation or threshold relaxation.

| Candidate | Baseline / candidate tok/s | Change | Saved ms +/- paired 95% CI |
| --- | ---: | ---: | ---: |
| 32 output columns | 150.764 / 126.368 | -16.18% | -163.904 +/- 12.251 |
| 128 output columns | 148.384 / 57.787 | -61.06% | -1352.403 +/- 14.297 |
| K=64, 64 output columns | 145.612 / 138.660 | -4.77% | -44.077 +/- 10.811 |

All full-vector comparisons and continuations were bit-identical, but all
three were substantially slower. None enters production. Tile16 is available
in the private probe but was not measured in this investigation.

The next experiment targets host/GPU staging, not matmul arithmetic:
GQA staging previously emitted three blit copies per token, DeltaNet
staging two, and output compaction one. The new strided copy kernel replaces
each projection/compaction loop with one compute dispatch. It copies raw
32-bit words, preserving negative zero and NaN payloads as well as FP32
values. It does not change weights, activations, accumulation or state layout.

| Copy experiment | Baseline / candidate tok/s | Change | Saved ms +/- paired 95% CI |
| --- | ---: | ---: | ---: |
| Private prototype | 151.051 / 152.181 | +0.75% | 6.290 +/- 3.614 |
| Rebuilt prototype + raw copy tests | 149.454 / 150.649 | +0.80% | 6.792 +/- 8.113 |
| Production implementation | 149.674 / 150.540 | +0.58% | 4.920 +/- 5.513 |
| Final implementation + state checks | 145.626 / 146.289 | +0.46% | 3.982 +/- 7.585 |

The four series agree in direction, but three individual intervals include
zero. Pooling the 32 measured differences gives **5.496 +/- 2.678 ms**
(Student-t, df=31). This is exploratory evidence for a small benefit, not
a large acceleration or a guarantee for sustained fanless-M5 operation.
Each eight-pair interval uses df=7 and multiplier 2.365. Raw pair data is
logged; pooling does not turn separate series into a randomized trial.

Production enables the copies automatically only on the measured exact
`Apple M5` device and chunks of at least 128 tokens. Smaller chunks and
other devices retain blits. Missing optional copy pipeline also falls back.
`QW3_METAL_DENSE_GPU_COPY=0` restores blits; `=1` explicitly opts in for
chunks of at least 16 tokens. The runtime fingerprint includes this setting.
There is no one-token decode change here.

## Correctness and reproducibility

`test-prefill-copy` checks both the private and production copy pipeline
on five shapes: rows/columns 1/1, 3/7, 31/33, 128/4097 and 129/17, with
nonzero source/destination offsets, padded strides and destination guards.
Inputs include arbitrary word patterns, negative zero and a NaN payload.
Every output and padding word is compared exactly, including tail rows.
The full-model test additionally compares all live K/V bytes in 16 layers,
the entire DeltaNet state and convolution state after 128+8 tokens. These
checks are outside timing. Unused KV capacity and derived FlashAttention
scratch are not compared. Speed is not a correctness test threshold.

```sh
make qw3-prefill-staging-bench qw3-bench qw3-agent
caffeinate -disu make test-prefill-copy
caffeinate -disu ./qw3-prefill-staging-bench \
  ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf tile32
caffeinate -disu ./qw3-prefill-staging-bench \
  ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf tile128
caffeinate -disu ./qw3-prefill-staging-bench \
  ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf k64
```

Logs: `/tmp/qw3-prefill-tile32-paired.log`,
`/tmp/qw3-prefill-tile128-paired.log`, `/tmp/qw3-prefill-k64-paired.log`,
`/tmp/qw3-prefill-copies-paired.log`, `/tmp/qw3-prefill-copies-repeat.log`,
`/tmp/qw3-prefill-production-copy-paired.log`, `/tmp/qw3-prefill-copy-final.log`.

Production validation passed `test-runtime-fingerprint` (two tests),
`test-dense-metal-bringup`, `test-dense-agent`, `test-agent-verify`, and
`test-metal-logits` (MoE). Bring-up covers CPU projections/logits, greedy,
default 128-token prefill/continuation, F16/F32 decode and state roundtrip;
the dense-agent smoke covers generation and checkpoint/prefix-cache reuse.
Log: `/tmp/qw3-copy-production-regressions.log`.

`test-dense-metal-logits` also passed the sequential local llama.cpp
full-vocabulary reference: four-token prefix RMSE 0.0000474260, maximum
0.000237703; eight-token prefix RMSE 0.0000686683, maximum 0.000369251.
Both top-eight and eight greedy IDs match. These short prefixes exercise
ordinary decode, not the new 128-row copy path (covered by the exact paired
test and default pp128 CPU differential above).
`test-agent-git-smoke` passed cold, warm-prefix and cache-disabled runs:
the dense model chose the real `bash` tool, retrieved three commits from an
isolated temporary repository, reported their IDs/messages, and completed
without an empty-response retry or loop recovery. No tool call was injected.
Log: `/tmp/qw3-copy-reference-agent.log`.

## Fresh sequential reference, with substantial session drift

Five repetitions and warmup per workload, same GGUF, full GPU offload,
F16 K/V, batch/microbatch 128 for llama.cpp (`4fea119de`, build 11041).
qw3 context allocation is 256. Each row below ran in sequence, not with
interleaved cross-engine pairs. These observations are not causal estimates.

| Engine/configuration | pp128 depth 0 tok/s (SD) | tg32 depth 0 tok/s (SD) | tg32 depth 128 tok/s (SD) |
| --- | ---: | ---: | ---: |
| qw3, copies off | 129.33 (2.73) | 7.57 (0.03) | not measured |
| qw3, new default | 133.94 (0.59) | 7.30 (0.22) | 6.24 (0.07) |
| llama.cpp | 127.088 (2.530) | 6.373 (0.202) | 5.833 (0.123) |

Absolute speeds decreased strongly in both engines versus the previous
session, and later decode runs were slower. The copy-off pp128 control also
fell to 129.33, so this is not an isolated new-copy regression. AC power was
connected, free-memory percentage was about 30%, and `pmset -g therm`
reported no recorded warning. That does **not** prove stable clocks or
identify the cause of drift. We did not stop unrelated user applications.
The apparent default/llama lead cannot establish a sustained overtake or
goal completion under these conditions; the small in-process copy effect
above is the attributable result. No one-token decode kernel was changed.

Logs: `/tmp/qw3-copy-off-bench.log`, `/tmp/qw3-copy-default-bench.log`,
`/tmp/qw3-copy-default-d128.log`, `/tmp/qw3-copy-llama-bench.json`,
`/tmp/qw3-copy-llama-d128.json` (the llama JSON includes every timed sample).

The overall goal remains open: the preceding matched reference had roughly
a 3.3% pp128 gap and 4% depth-128 tg32 gap. A new alternating cross-engine
confirmation under sufficiently stable conditions is still required. This
small prefill-only change does not improve ordinary one-token decoding.
