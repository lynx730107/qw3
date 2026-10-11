# Balanced, per-run-cooled decode comparison on Apple M5, 2026-10-11

Ordinary qw3 decode remains below llama.cpp after balancing the run order
and applying the same idle interval before every engine run. qw3 measures
8.86247 token/s and llama.cpp 9.13410: a −2.974% rate difference. Reaching
llama's measured rate requires about 3.065% more qw3 throughput. The objective
of matching both prefill and decode remains unachieved.

| Phase | qw3 token/s | llama token/s | qw3 / llama run-mean spread |
| --- | ---: | ---: | ---: |
| tg32, depth 128 | 8.86247 | 9.13410 | 0.600% / 0.606% |

The paired saved time for qw3 is −107.375 ±6.764 ms per 32-token decode.
The two-sided 95% Student-t interval is [−114.140, −100.611] ms over six
paired run means (five degrees of freedom). It confirms a llama lead in
this series; both engines pass the ≤5% spread gate. All 60 samples from
12 runs are retained, with no discarded samples.

Six AB/BA pairs balance three runs with each engine first. Each engine run
contains five repetitions, with 60 seconds idle before every run. Saved
time averages −103.011 ms with qw3 first and −111.740 ms with qw3 second.
Both order groups favor llama. The much larger order association in the
[earlier pair-cooled campaign](m5_native_reference_2026-10-11.md) does not
persist at that scale; this comparison does not identify its cause.

The ordinary production `qw3-bench` is used, with the adopted packed8,
K64 prefill and GDN cache64 defaults. Both engines use the same resolved
`Qwen3.8-27B-UD-Q3_K_XL.gguf` (13,146,393,504 file bytes), Apple M5,
F16 KV, ctx256 and full trunk/head GPU offload. The requested thread count
is four; qw3's engine option does not establish its CPU helper thread count.
llama uses batch/ubatch 128 and Flash Attention. Its executable reports
build 11041, commit `4fea119de`; the checkout revision is recorded separately.

Prefill is outside the timed decode region. Each engine uses its native
synthetic token generator, so the token streams differ. No sampling, MTP,
acceptance-rate adjustment or private optimization candidate is involved.
The launcher clears QW3/GGML/MTL overrides before setting the recorded
configuration. This campaign times decode only; the earlier pp128 result
remains close to llama without a confirmed lead or formal equivalence test.

Every recorded source snapshot and executable hash matches the completed
run. Model size and modification time remain unchanged. Swap usage is
807.06 MiB before and after every run; this does not establish constant
temperature or clock frequency. Numerical and checkpoint evidence for
the unchanged production inference paths is linked in the earlier report;
this timing series does not repeat those checks.

The [raw results and full commands](measurements/native_reference_run_cooldown_2026-10-11/results.json),
[validation and paired analysis](measurements/native_reference_run_cooldown_2026-10-11/analysis.json),
all stdout/stderr streams and immutable source snapshots are retained.
Absolute rates from different campaigns must not be interpreted as an
optimization effect.

```sh
python3 tools/bench_dense_reference.py \
  --output /tmp/qw3-run-cooled-reference \
  --jobs tg32_d128 --pairs 6 --repetitions 5 \
  --cooldown 60 --cooldown-scope run
```
