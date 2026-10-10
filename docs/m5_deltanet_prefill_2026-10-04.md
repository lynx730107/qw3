# GatedDeltaNet prefill candidates — base Apple M5, 2026-10-04

Branch: `codex/m5-deltanet-prefill`. This experiment starts from archival
commit `4f033f6`, retaining the ordinary dense engine optimizations and MTP
removal. The abandoned MoE/dense supervisor, expert flags/tool, live hybrid
tests and pilot are removed on this branch. The repeat guard, Unicode handling,
TUI and ordinary tool/session path remain. `main` itself is unchanged.

## Acceptance and comparison

Use the existing Qwen3.8-27B-UD-Q3_K_XL GGUF, all 64 trunk layers on Metal,
F16 KV, context allocation 512, four helper threads, seed-1 fixed IDs and
excluded warmup. Measure pp128 separately from 32 ordinary decode forwards
at depth128, including logits. No speculative decoding or sampling.

Require at least 5% repeatable end-to-end pp128 gain, no repeatable tg32
regression above 2%, and OFF latency spread no higher than 5% in either
phase. Each candidate uses OFF/candidate/OFF/candidate/OFF with equal
40-second cooldowns. The combined follow-up averages three repetitions
per phase. The initial packed4 series has a tg32 OFF spread slightly above
5%; it is inconclusive as a joint comparison and is repeated with three
repetitions, 60-second cooldowns and OFF/candidate/OFF. Both sides use the same diagnostic executable and
compile all available candidate pipelines before the excluded warmup. Dispatch
counters confirm that OFF never uses a candidate and candidate runs do.

Commands, binary/source hashes, counts, raw stdout/stderr, machine-wide
swap before/after each job and nearby-control comparisons are recorded by
`tools/bench_qw3_candidates.py`. Swap observations describe the entire
machine and cannot identify a process-specific memory leak.

The ordinary engine, bridge, benchmark and production Metal shaders are
byte-identical to the archived baseline. Candidates are compiled only in
optional diagnostic binaries, with no production dispatch or setting change.
Intrusive per-stage profiling is disabled. These runs establish effects
within their own series, not comparisons to older dates or llama versions.

## Mechanisms

The existing recurrence keeps two value columns per 32-lane SIMD group.
Each lane owns four key coordinates of each column, with `dot(float4)`
and full-SIMD reduction for each prediction/output.

`packed4` assigns four lanes to each value column: eight columns per SIMD,
32 columns per 128-thread workgroup, eight float4 state blocks per lane.
`packed8` assigns eight lanes per column: four columns per SIMD,
16 columns per workgroup, four float4 state blocks per lane. Both keep
the existing key-major state storage, head mapping, FP32 state, gates and
sequential token recurrence. Local partial trees reproduce the native
ascending butterfly; remaining reductions use xor shuffles within each
column's lanes. `dot(float4)` and state-update arithmetic keep native
fast-math. Only the explicit local tree disables floating-point reassociation.

This is inspired by the [MLX packed GatedDeltaNet implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gated_delta.py).
It is a new lane/state mapping, beyond the earlier wider-column grouping
and gate-precomputation experiments. It does not copy MLX's different
state layout, gate arithmetic or grouped-head mapping.

`pipeline` keeps the original two-column recurrence/reductions and reads
next-token q/k/v and gates before completing the current token's dependent
state work. Its motivation is overlapping independent input loads with the
recurrence, as also explored by [oMLX prefill kernels](https://github.com/jundot/omlx/blob/main/omlx/custom_kernels/qwen35_prefill/gdn.py).
The last token has no speculative out-of-range read. All candidates
use the native fallback for head dimensions other than 128 and chunks
shorter than 16; single-token decode is unchanged.

The initial packed8 and pipeline series both showed small positive pp128
changes. This justified one follow-up combination, `packed8_pipeline`,
with the packed8 state/reduction geometry and next-token input preloading.
It passed the same exact quality suite before its own measurement series.
It is compared only to controls from its own newly built executable,
which compiles four pipelines on both sides.

## Correctness

[Quality manifest](measurements/gdn_prefill_quality_2026-10-04/results.json)
and logs show that the first three final candidates pass exact comparisons of
all vocabulary logits, greedy IDs and canonical live F16 K/V,
convolution and DeltaNet state. Prefix lengths 16, 31 and 128 activate
the candidates; 1 and 15 check fallback behavior. The 128-token case
extends by 31 tokens to test a nonzero initial recurrent state. Every
case saves/loads the candidate's disk payload before 32 greedy continuation
forwards. Candidate dispatch count is 192 per complete quality run.
Unused KV capacity and derived FlashAttention scratch are excluded.

The first packed attempt failed numerical equality. Disabling reassociation
for the whole packed kernel still failed numerical equality.
Restricting it to the local reduction helper preserves the explicit tree
without changing the native dot/update arithmetic. Failed logs are retained;
they are not accepted correctness results.

Ordinary agent checks passed for status, context ledger, tool parsing,
verification guard, repeat guard, bash/edit tools, colors and trace.
The real 35B repeat-format regression passed after a read tool, preserving
a Unicode diagram/ASCII separator and resuming KV on the next turn.
The 27B agent smoke also passed prompt/offload/KV validation, generation,
checkpoint save/resume and prefix-cache creation/resume. CPU agent and Metal
CLI build successfully; the unchanged CPU engine reports its existing
unused-function warnings. These checks cover the tested ordinary path.

## Measurements

No candidate is adopted: none reaches the predeclared 5% pp128 gate.
The initial packed4 series is inconclusive as a joint pp/tg comparison:
tg32 OFF spread is 5.047%, exceeding 5%. Its two pp128 effects disagree
(+0.49% and -2.93%). It is retained, not pooled with the repeat.

The repeat uses three repetitions per phase, 60-second cooldowns and one
candidate block bracketed by two OFF blocks. Its controls are stable, and
pp128 improves 2.83%; it is a bounded follow-up, not two independently
repeated accepted candidate blocks. That small gain does not supersede
the initial inconsistent series or meet the adoption gate.

| Candidate / series | pp128 OFF / candidate tok/s | pp change | tg32 OFF / candidate tok/s | tg change | OFF spread pp / tg |
| --- | ---: | ---: | ---: | ---: | ---: |
| [packed4 repeat](measurements/gdn_prefill_packed4_repeat_2026-10-04/results.json) | 152.86 / 157.19 | +2.83% | 8.766 / 8.664 | -1.16% | 0.51% / 0.04% |
| [packed8](measurements/gdn_prefill_packed8_2026-10-04/results.json) | 151.85 / 156.46 | +3.04% | 8.673 / 8.666 | -0.08% | 2.83% / 4.73% |
| [pipeline](measurements/gdn_prefill_pipeline_2026-10-04/results.json) | 153.57 / 155.01 | +0.93% | 8.868 / 8.850 | -0.20% | 1.90% / 2.87% |
| [packed8 + pipeline](measurements/gdn_prefill_combined_2026-10-04/results.json) | 154.10 / 154.80 | +0.45% | 8.776 / 8.813 | +0.42% | 0.79% / 4.55% |

Effects are throughput from pooled mean phase latency, within each series.
Do not compare absolute rates across different runs/executable hashes.
Packed8's two nearby-control pp128 changes are +2.50% and +3.08%.
Pipeline gives +1.66% and +0.53%; the combination gives +0.46% and +0.62%.
The combination does not add their gains. No repeatable tg32 regression
above 2% is established in the stable series, but that cannot rescue a
failed pp128 gate. Negative results apply to these exact implementations
and settings, not to every possible optimization on the M5.

All candidate quality suites passed exact full logits, greedy continuations,
canonical states and payload continuation. The ordinary inference sources
are unchanged. Failed early numeric prototypes remain only in the quality
logs; corrected prototypes remain optional diagnostics. No speculative
path, expert mode, model conversion or changed KV/offload is introduced.

These are short warmed workloads. They do not prove a sustained thermal
ceiling, a long-context advantage, or parity/superiority over current llama.cpp.
The previous llama comparisons remain historical; this experiment does not
reinterpret them as a fresh controlled comparison.

## Reproduce

Run from the repository root, with a new output directory:

```sh
make qw3-gdn-prefill-test qw3-gdn-prefill-bench
QW3_METAL_KV_F16=1 QW3_METAL_KV_Q8_0=0 caffeinate -disu \
  ./qw3-gdn-prefill-test ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf packed4
python3 tools/bench_qw3_candidates.py --output /tmp/qw3-gdn-packed4 \
  --binary ./qw3-gdn-prefill-bench --order off,candidate,off,candidate,off \
  --candidate-env QW3_PRIVATE_GDN_LAYOUT=packed4 --cooldown 40
```

Substitute `packed8`, `pipeline` or `packed8_pipeline` for the other candidates.
For the combined series add `--repetitions 3`; for the packed4 repeat use
`--repetitions 3 --cooldown 60 --order off,candidate,off`. The private
setting is not a supported client/agent option. This short ctx512 test
does not establish sustained fanless throughput or long-context gains.
