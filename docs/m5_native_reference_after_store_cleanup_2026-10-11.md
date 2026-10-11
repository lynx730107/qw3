# Ordinary decode comparison after store cleanup, 2026-10-11

The unchanged production qw3 binary remains behind llama.cpp in ordinary
decode: 9.07176 versus 9.34049 token/s for tg32 at depth 128. qw3's rate
is 2.877% lower; reaching this measured llama rate requires 2.962% more
qw3 throughput. Matching both prefill and decode remains the objective.

| Phase | qw3 token/s | llama token/s | qw3 / llama run-mean spread |
| --- | ---: | ---: | ---: |
| tg32, depth 128 | 9.07176 | 9.34049 | 0.942% / 0.584% |

All six paired comparisons favor llama. The paired saved time for qw3 is
−101.488 ±6.155 ms per 32-token decode, with a two-sided 95% Student-t
interval of [−107.643, −95.333] ms. The interval uses six paired engine-run
means, five degrees of freedom and critical value 2.571; the 60 individual
repetitions are not treated as independent experimental units. Both
engines pass the pre-existing ≤5% max/min run-mean spread gate. All samples
are retained, including the final pair.

Six AB/BA pairs balance three runs with each engine first. Each process
has five repetitions and a 60-second idle interval before launch. Saved
time averages −102.026 ms with qw3 first and −100.950 ms with qw3 second;
both order groups favor llama. Only one model process runs at a time.

Both engines use the same resolved `Qwen3.8-27B-UD-Q3_K_XL.gguf`, with
13,146,393,504 file bytes, Apple M5, F16 KV, ctx256 and full trunk/head
GPU offload. llama uses batch/ubatch 128 and Flash Attention. Its executable
reports build 11041, commit `4fea119de`; checkout revision is recorded
separately. The requested thread count is four; qw3's engine option does
not establish its CPU helper thread count. Prefill is outside the timed
decode region. The engines use different native synthetic token streams,
without sampling or MTP.

The production executables and all 20 recorded source hashes match the
[previous balanced comparison](m5_native_reference_run_cooldown_2026-10-11.md).
The launcher clears QW3/GGML/MTL overrides, records the absence of DYLD
overrides, and verifies eight directly referenced local llama libraries
before and after the campaign. The immutable source snapshots are reused
from the indexed previous archive. Model size and modification time remain
unchanged. No private optimization candidate, copied weight layout or new
session data buffer is enabled. Every qw3 run reports 176,761,680 session
bytes. Numerical and checkpoint checks for the unchanged inference paths
are documented in the earlier comparison; this timing campaign does not
repeat them.

Swap is constant within every process run. The first eleven runs have
1,173.06 MiB used; before the final qw3 run it drops to 1,165.06 MiB and
remains there. Thus swap is not constant over the whole campaign, and no
increase is observed at the recorded boundaries. These observations do
not establish constant temperature, clock frequency or peak memory use.
An initial metadata verifier incorrectly required identical swap readings
across the entire campaign. The verifier now records the cross-run change
and checks within-run equality; no timing sample was removed or rerun.

The [authorized store cleanup](agent_store_2026-10-11.md) removed
1,115,649,836 logical bytes of hybrid artifacts, prefix caches and spills.
The store remains history-only at 30,109 bytes, with its hash unchanged
through this comparison. Its cleanup is a disk-space result; neither the
higher absolute rates than the earlier campaign nor the small change in
relative gap can be attributed to it. Production inference code is
unchanged, and this campaign contains no intervention/control comparison
for the cleanup. Standalone diagnostic rates must also not be compared
against llama rates from a different campaign.

This campaign times decode only. The earlier pp128 measurement remains
close to llama without a confirmed lead or formal equivalence test.
The full goal remains unachieved.

[Raw results and commands](measurements/native_reference_after_store_cleanup_2026-10-11/results.json),
[launch and snapshot references](measurements/native_reference_after_store_cleanup_2026-10-11/launch.json),
[paired analysis](measurements/native_reference_after_store_cleanup_2026-10-11/analysis.json),
all stdout/stderr streams and the
[verification script](measurements/native_reference_after_store_cleanup_2026-10-11/verify_campaign.py)
are retained.

```sh
python3 tools/bench_dense_reference.py \
  --output /tmp/qw3-after-store-cleanup-reference \
  --jobs tg32_d128 --pairs 6 --repetitions 5 \
  --cooldown 60 --cooldown-scope run
```
