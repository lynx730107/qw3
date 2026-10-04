# Ordinary qw3 optimization — Apple M5, 2026-10-03

The target is a measurable local improvement, not a requirement to beat
llama.cpp. Keep Qwen3.8-27B-UD-Q3_K_XL unchanged, full 64-layer Metal
offload, F16 KV, ctx512, four helper threads and fixed seed-1 input IDs.
Measure pp128 separately from 32 ordinary decode forwards at depth128.
This pure qw3-bench generation convention differs from llama-server's
31 timed forwards for 32 generated outputs.

Acceptance rules set before candidate measurement:

- OFF control latency spread must be at most 5% in each phase. Otherwise
  the comparison is inconclusive.
- Require at least 5% repeatable end-to-end improvement in the target phase.
- Reject a repeatable regression above 2% in the other phase.
- Preserve existing logit tolerances, greedy IDs, recurrent/KV continuation,
  session persistence, client and agent behavior for any adopted change.
- Negative candidate results are a valid outcome; do not reintroduce MTP
  or implement DFlash2 in this work.

## Baseline

[Raw baseline](measurements/qw3_ordinary_baseline_2026-10-03/results.json):
four independent OFF blocks, 40 seconds idle before each block. Each
phase runs in a new process, with the standard excluded warmup and one
measured repetition. GPU jobs are sequential. No intrusive stage profiler
is enabled. Throughput below is token count divided by mean phase latency.

| Phase | Throughput | OFF latency spread |
| --- | ---: | ---: |
| pp128 | 145.91 tokens/s | 0.71% |
| tg32, depth128 | 8.33 tokens/s | 0.54% |

The spread is max/min latency minus one, which also bounds the throughput
range. The baseline satisfies the 5% gate. It is a control for this series,
not a sustained hardware limit or a comparison with earlier measurements.

## Candidates

Both prototypes are private to optional diagnostic binaries. The engine,
bridge and Metal shader files used by normal client/agent builds have no
new candidate setting or altered dispatch policy.

**FFN IQ4_XS groups:** use eight SIMD groups/256 threads instead of two
groups/64 threads for the paired gate/up and down-residual projections.
This changes how many independent output rows share a workgroup; FP32 dot
grouping, per-lane accumulation and SIMD reduction stay unchanged. The
target uses 14 such FFN pairs and 32 such down projections per forward.
The experiment tests workgroup overhead/occupancy, not another lookup-table
location, payload layout, prefetch or previously rejected SwiGLU fusion.

**DeltaNet gate precomputation:** compute beta sigmoid and alpha softplus/
decay once per token/value head, then have the existing two-column register
recurrence read those values. Baseline recurrence repeats this gate math
for every SIMD group of state columns. FP32 state/update/reduction order
stays unchanged. There is one added bounded gate dispatch per DeltaNet
prefill layer; it reuses the alpha/beta scratch slots and allocates no new
model or state buffers. Single-token decode keeps the ordinary path.

The [private adapter](../tests/dense_candidates.inc) compiles modified
pipelines in memory and duplicates only two IQ4 dispatch wrappers, with
the original validation/bindings and wider launch geometry. Native wrapper
implementations are called for OFF controls. Both sets of pipelines are
compiled even for OFF, avoiding asymmetric initialization. Compilation and
standard warmup are outside phase timing.

## Candidate correctness

[Quality logs](measurements/qw3_candidates_quality_2026-10-03/ffn.log):
FFN passed exact finite output comparisons and output guards for both
operations at 256→1, 4352→33, 5120→4097/4111/17408 and 17408→5120.
Those are bounded real tensor-prefix reinterpretations, including actual
projection dimensions and partial output workgroups. Full-vocabulary logits
and 32 greedy continuation forwards after a 128-token prefix matched
exactly. Final live F16 K/V, convolution and DeltaNet state were identical.

[DeltaNet quality log](measurements/qw3_candidates_quality_2026-10-03/gdn.log):
16-, 31- and 128-token prefixes passed exact full logits and canonical
state comparison. The 128-token case was extended by 31 tokens to test a
nonzero initial recurrent state. Each case also passed 32 exact greedy
continuation/logit comparisons and final canonical state comparison.
The candidate gate kernel was dispatched 192 times, confirming the tested
path was active. Unused KV capacity and derived FlashAttention scratch are
excluded from canonical-state equality.

These exact private comparisons are stronger than a greedy-only check.
They do not replace client/agent/session tests if a candidate is adopted.

## Candidate measurement

The candidate series use OFF/candidate/OFF/candidate/OFF, equal 40-second
cooldowns, the same private executable for both sides, and the same phase
parameters as the baseline. Compare each candidate to its nearest OFF
controls; do not infer gains from different executable runs or dates.
Raw commands, environment, binary hash, results, stderr and stdout are
retained by [the measurement tool](../tools/bench_qw3_candidates.py).
Dispatch counters verify that OFF controls do not use a candidate and that
candidate jobs actually execute the relevant kernels.

FFN [raw measurements](measurements/qw3_ffn_sg8_2026-10-03/results.json)
and [nearby-control analysis](measurements/qw3_ffn_sg8_2026-10-03/analysis.json)
are complete. OFF spread is 0.46% for pp128 and 0.09% for tg32, satisfying
the baseline gate in both phases. Pooled tg32 is 8.810 OFF versus 8.795
candidate tokens/s, -0.17%. Nearby-control effects are -0.175% and -0.141%.
Pooled pp128 changes by -0.14%. The candidate passes correctness but does
not meet the 5% improvement requirement, and is **rejected**.

The private OFF rate differs from the earlier ordinary-baseline rate.
Those series ran at different times and with different executable hashes;
the difference is not attributed to this candidate. All candidate effects
above are measured within their own interleaved series.

DeltaNet [raw measurements](measurements/qw3_gdn_gates_2026-10-03/results.json)
are also complete. OFF spread is 0.15% for pp128 and 0.21% for tg32,
satisfying the baseline gate. Pooled pp128 is 153.278 OFF versus 153.137
candidate tokens/s, -0.09%. Nearby-control effects are -0.099% and -0.080%.
The incidental tg32 change is +0.13%, with the gate kernel used only during
untimed depth setup. Neither number establishes a useful acceleration.
This candidate also fails the 5% target-phase requirement and is **rejected**.

| Candidate | Target phase | OFF / candidate tokens/s | Target-phase change | OFF spread, pp / tg |
| --- | --- | ---: | ---: | ---: |
| FFN IQ4_XS groups | tg32 | 8.810 / 8.795 | -0.17% | 0.46% / 0.09% |
| DeltaNet gates | pp128 | 153.278 / 153.137 | -0.09% | 0.15% / 0.21% |

## Outcome and completion evidence

No candidate is adopted. Hash comparisons confirm the ordinary benchmark
executable and `qw3.c`, `qw3_metal.m`, `qw3_bench.c` are unchanged from the
initial baseline. Production Metal shaders are unchanged as well. The new
targets are optional and are not dependencies of the client, agent or
normal benchmark. Rejected prototypes are retained for reproducibility.

| Goal requirement | Evidence | Outcome |
| --- | --- | --- |
| Separate repeatable pp128/tg32, same model/offload/KV/input | Baseline JSON, raw commands and phase counts, 0.71%/0.54% spread | Complete |
| Interleaved OFF controls with spread ≤5% | Both candidate JSONs and nearby-control effects above | Complete |
| One FFN and one DeltaNet prefill candidate, with new motivation | Private adapter, compile log, active dispatch counters and prior-trial review | Complete |
| Adopt only repeatable ≥5% gains, other phase regression ≤2% | Both target effects below threshold; no candidate adopted | Complete, negative result |
| Preserve logits/greedy/recurrent and KV state for candidate evaluation | Exact finite quant/tail checks and complete vocabulary/canonical-state/continuation logs | Complete |
| Preserve client, agent and session behavior for adopted changes | No inference change adopted; ordinary source and binary hashes preserved | Complete by retaining baseline |
| No MTP or DFlash2 implementation | Optional diagnostics only; ordinary engine unchanged | Complete |

These are short warmed pp128/tg32 observations at ctx512, not a claim about
long-context inference or sustained heavy-load throughput. They show that
these two mechanisms are not productive candidates on the tested M5 path.
The work establishes a stable control method and rejects two ideas without
adding unsuccessful complexity to production. It demonstrates no new
end-to-end performance improvement.

## Reproduce

Run from the repository root; output directories must be new:

```sh
make qw3-dense-candidates-bench qw3-dense-candidates-test
caffeinate -disu ./qw3-dense-candidates-test \
  ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf ffn
caffeinate -disu ./qw3-dense-candidates-test \
  ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf gdn
python3 tools/bench_qw3_candidates.py --output /tmp/qw3-baseline --cooldown 40
python3 tools/bench_qw3_candidates.py --output /tmp/qw3-ffn \
  --binary ./qw3-dense-candidates-bench \
  --order off,candidate,off,candidate,off \
  --candidate-env QW3_PRIVATE_FFN_SG8=1 --cooldown 40
python3 tools/bench_qw3_candidates.py --output /tmp/qw3-gdn \
  --binary ./qw3-dense-candidates-bench \
  --order off,candidate,off,candidate,off \
  --candidate-env QW3_PRIVATE_GDN_GATES=1 --cooldown 40
```

The `QW3_PRIVATE_*` switches exist only in these optional prototypes.
They are not supported settings of the ordinary client or agent. The
FFN result's nearby-control analysis is a separate `analysis.json`; the
DeltaNet result contains the same analysis in `bracketed_comparisons`.
