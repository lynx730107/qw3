# Ordinary qw3 versus llama.cpp on Apple M5, 2026-10-11

The fresh direct comparison places prefill very close to llama.cpp and
ordinary decode about 2.40% below its point estimate. Both phases pass the
≤5% spread gate for each engine's run means. Neither paired interval confirms
a lead; matching both phases remains an unachieved objective.

| Phase | qw3 token/s | llama token/s | Rate difference | qw3 / llama spread |
| --- | ---: | ---: | ---: | ---: |
| pp128 | 158.07973 | 157.96486 | +0.073% | 0.680% / 0.376% |
| tg32, depth 128 | 8.73875 | 8.95342 | −2.398% | 3.021% / 4.149% |

The paired saved time is +0.589 ±4.747 ms per 128-token prefill and
−87.797 ±113.343 ms per 32-token decode. These are two-sided 95% Student-t
intervals over five paired run means (four degrees of freedom). Both include
zero. The prefill point estimate is close, but this is not a formal
equivalence test. The decode interval is wide, so the 2.40% point difference
must not be reported as a precisely established stable gap.

The paired differences also associate with run order. Prefill saves a mean
+3.373 ms when qw3 runs first and −3.588 ms when it runs second; decode saves
−24.927 ms first and −182.103 ms second. This describes the observed ordering,
not its cause. Only the beginning of each pair has a cooldown in this protocol,
and five pairs contain three AB and two BA orders. A balanced follow-up with
the same idle interval before every engine run is needed to separate this
ordering effect from an engine difference.

Five alternating AB/BA pairs per phase, five repetitions per engine run,
60 seconds idle before each pair: 20 runs and all 100 samples are retained.
The ordinary production `qw3-bench` executable is used; the rejected aligned
loads and row-reuse candidates are not enabled. No sampling, MTP or acceptance
rate is involved. Prefill is outside the timed decode region. Each engine
uses its native synthetic token generator, so the token streams differ.

Both use the same resolved `Qwen3.8-27B-UD-Q3_K_XL.gguf` (13,146,393,504 file
bytes), Apple M5, F16 KV, ctx256 and full trunk/head GPU offload. Requested
thread count is four. qw3's engine option does not establish the CPU helper
thread count; this workload runs fully on Metal. llama uses batch/ubatch 128,
Flash Attention enabled, and 999 requested GPU layers; logs confirm ctx256,
all layers and the output layer on the GPU. The llama executable reports
build 11041, commit `4fea119de`; its checkout revision is recorded separately
and must not be substituted for the executable build identity.

QW3/GGML/Metal environment overrides are cleared by the launcher. The runner
sets qw3 F16 KV, disables Q8 KV and enables raw sample printing. Production
FAST/K64/GDN cache64 defaults are retained. No inference code is modified by
this campaign. The previous full logits/state/checkpoint evidence remains in
the [K64](m5_kstage_prefill_2026-10-10.md),
[GDN cache64](m5_gdn_cache_2026-10-10.md), and
[Flash Attention mask guard](m5_hazard_fences_2026-10-10.md) reports; this timing
campaign does not repeat those numerical or checkpoint tests.

All source snapshot hashes and executable hashes match the completed run.
Model file size and modification time remain unchanged. Recorded swap usage
is 831.12 MiB before and after every run; this observation does not establish
constant temperature or clock frequency. Absolute rates from different
campaigns are not optimization effects.

The [raw results and full commands](measurements/native_reference_2026-10-11/results.json),
[validation and paired analysis](measurements/native_reference_2026-10-11/analysis.json),
launcher metadata, all stdout/stderr streams and immutable source snapshot
are retained in the same directory.

```sh
make qw3-bench
# Clear QW3/GGML/Metal overrides before launch; use a new output directory.
python3 tools/bench_dense_reference.py \
  --output /tmp/qw3-native-reference \
  --jobs pp128,tg32_d128 --pairs 5 --repetitions 5 --cooldown 60
```

The runner now also accepts `--cooldown-scope run`, inserting the same pause
before each engine instead of only before the pair. The default remains
`pair` for legacy reproduction. Scope and both idle intervals are recorded;
even pair counts balance AB and BA. The runner clears QW3/GGML/MTL overrides
itself and reports saved-time means separately by order. Three
`tests/test_bench_dense_reference.py` fixture tests verify legacy pacing,
equal per-engine pacing and preservation of failed stdout/stderr, without
running inference.

The next decode replication uses six balanced pairs:

```sh
python3 tools/bench_dense_reference.py \
  --output /tmp/qw3-run-cooled-reference \
  --jobs tg32_d128 --pairs 6 --repetitions 5 \
  --cooldown 60 --cooldown-scope run
```
