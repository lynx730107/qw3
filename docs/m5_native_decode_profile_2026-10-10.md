# Native decode pipeline attribution on Apple M5

The diagnostic profile points to IQ4_XS and IQ3_S projections as the next
optimization targets. Across four sampled decode steps, the three IQ4_XS
pipelines sum to 40.577 ms per step and the two IQ3_S pipelines to 30.643 ms.
The existing cache64 DeltaNet pipeline sums to 2.484 ms over 48 dispatches.
These are means of encoder timestamp durations under instrumentation, not
native throughput measurements or a comparison with llama.cpp.

The harness runs the Qwen3.8-27B-UD-Q3_K_XL GGUF with all 64 layers on Metal,
F16 KV, ctx512, native K64 prefill and native GDN cache64 enabled. Two sessions
receive a fixed 128-token prefix (LCG seed 1) and 32 fixed decode tokens
(seed 2). Only steps 0, 8, 16 and 24 of the second session use separate compute
encoders with stage timestamp attachments. The first session always retains
native encoder boundaries. M5 does not support dispatch-boundary sampling.

Each sampled step has 978 passes and no pass containing multiple pipeline
names. The harness checks the entire finite vocabulary bit for bit after
every decode forward, and canonical history, KV and recurrent state before
and after the sequence. All checks passed with Metal API validation enabled.
This harness does not test checkpoint roundtrips. Production source and the
profiling binary remained unchanged during the run.

The [manifest](measurements/native_decode_profile_2026-10-10/manifest.json),
[raw stdout](measurements/native_decode_profile_2026-10-10/profile.stdout.log),
[raw stderr](measurements/native_decode_profile_2026-10-10/profile.stderr.log),
source snapshot and [derived ranking](measurements/native_decode_profile_2026-10-10/results.json)
retain the complete run. The Q5_K entry includes all Q5_K projections; it
must not be interpreted as the language-model head alone.

Reproduce the diagnostic with:

```sh
make qw3-decode-profile-test
QW3_METAL_KV_F16=1 QW3_METAL_NGL=64 QW3_METAL_DENSE_PREFILL_FAST=1 \
QW3_METAL_DENSE_PREFILL_K64=1 QW3_METAL_DENSE_GDN_CACHE=1 MTL_DEBUG_LAYER=1 \
caffeinate -disu ./qw3-decode-profile-test ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf
```

Run with other QW3/GGML/Metal overrides cleared. The engine receives the
`n_threads=4` option; the workload is fully on the GPU and this option does
not establish the actual CPU helper thread count.
