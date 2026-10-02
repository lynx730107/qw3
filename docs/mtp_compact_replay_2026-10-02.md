# Compact accepted-prefix replay (2026-10-02)

Experimental dense MTP decoder only, base Apple M5, the existing
Qwen3.8-27B UD-Q3_K_XL GGUF, full offload, F16 KV and ctx512. Inspired by
Backburner's compact GDN input replay, not a copied implementation:
[source reviewed at f8c76832](https://github.com/StayLameBro/backburner-llama.cpp/blob/f8c76832b9bb7710db52143d9c89adcda5402718/src/models/qwen35.cpp#L732).
No iPhone, new quantization, model download or ANE use is involved.

## Implementation and boundary

`qw3-mtp-decode` has a new explicit `serial-replay` mode. Its scalar verifier
records projected QKV, Z, alpha and beta before each layer's convolution.
A compile-time hook in `qw3.c` defaults to constant 1: production builds
allocate no replay log and dispatch no recording work. Only the private
diagnostic overrides it. Recording checks complete per-token layer coverage.

After each verified token, the diagnostic also records that position's F16
K/V rows for every full-attention layer, and the final raw hidden, normalized
hidden and logit rows. Inputs are GPU-to-GPU copies; no new weight layout or
full dequantized model is stored.

On rejection:

1. Restore the cycle's original GPU snapshot.
2. For each linear layer, replay only the accepted logged projections through
   the **same** scalar convolution, Q/K normalization and fused DeltaNet kernel
   that ordinary decode used. No embedding, projection, attention output,
   FFN or LM head is rerun.
3. Install only accepted KV rows. Rebuild their F16 FlashAttention ring entries
   from those same half values, including position-32 wraparound. Other slots
   retain their pre-cycle contents.
4. Install the saved output buffers for the last accepted token and truncate
   position/history to the accepted boundary. CPU logits become invalid.

Each layer's replay depends only on its own logged input sequence and initial
recurrent state, so replay can be grouped by layer without recomputing the
hidden propagated through other layers. Arithmetic inside a layer is unchanged.
The implementation reuses the existing gate/output work in the fused recurrence
even though only its state result is needed; a state-only kernel is future work.

The current full 190.61 MiB ctx512 snapshot is still used. The log is bounded
by block width (up to eight tokens), but the snapshot still scales with context.
This is not yet a long-context production rollback format, nor the deferred
lazy-state scheme of Backburner. The `batch` verifier remains disallowed for
this recorder and retains its previously measured numerical failures.

## Checks and timing interpretation

The independent oracle still restores the original snapshot and runs the
complete scalar target on the accepted prefix. Every persistent buffer must
then equal the compact result byte for byte: DeltaNet, convolution, complete
KV (including unused rows), Flash tails, raw/normalized hidden and logits.
All proposed rows, emitted IDs, final state against ordinary generation and
eight subsequent full-logit continuation steps are checked too.

`restore_replay_ms` includes restoring the snapshot and installing compact
state. `scalar_restore_reference_ms` times the independent full-model restore
and replay. The oracle runs after the compact path; these are correctness-
instrumented component observations, not an alternating promotion benchmark.
The emitted `experimental_tps` also counts the logging cost during verification
and keeps the previous `not_promotion_benchmark=1` marker.

## Measured results

The six-case compact regression passed: 88 emitted tokens matched ordinary
greedy generation, all 179 proposed hidden/logit rows had zero max/RMS error,
and all 21 rejection-state comparisons were byte-identical. Each case also
passed the final full-state comparison and eight subsequent continuation
steps. The last forced rejection crosses the 32-token Flash tail boundary.
Additional forced-rejection runs at block widths 2 and 3 also passed with
zero row error, exact rollback/final state and eight continuation steps each.
Their replay logs occupy 8.13 and 12.20 MiB respectively, including saved
output rows; the separate full snapshot remains 190.61 MiB.

Times below are sums over each case's rejected cycles, not per-token latency.

| Case (block 8) | Rejections | Full scalar reference (ms) | Compact restore/replay (ms) | Component ratio |
| --- | ---: | ---: | ---: | ---: |
| Code, 32 tokens | 8 | 1706.677 | 95.690 | 17.8x |
| Math, 16 tokens | 4 | 1579.728 | 71.795 | 22.0x |
| Story, 16 tokens | 5 | 1607.132 | 78.353 | 20.5x |
| Forced first rejection | 2 | 469.925 | 27.631 | 17.0x |
| Forced middle rejection | 1 | 469.932 | 26.644 | 17.6x |
| Forced last rejection | 1 | 843.042 | 36.055 | 23.4x |

**This is not an overall decode speedup.** In the code case the instrumented
compact decoder measured 3.358 tok/s versus 9.114 tok/s for its ordinary
scalar baseline. Serial verification alone took 8569.125 ms, dwarfing the
95.690 ms replay cost. Fast, numerically correct short-block verification is
still required. These are diagnostic observations, not a paired comparison
against llama.cpp or a production throughput benchmark.

The unchanged production path passed `test-runtime-fingerprint`,
`test-dense-metal-bringup`, `test-dense-agent` and `test-agent-verify`.
An eight-token `serial` control also passed all row/state/continuation checks
and reported `replay_log_mib=0.00`, `compact_replay=0`.

## Reproduce

```sh
make qw3-mtp-decode
caffeinate -disu make test-mtp-compact-replay
caffeinate -disu ./qw3-mtp-decode \
  ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf 32 8 serial-replay
# Width/layout checks, including an odd block width:
caffeinate -disu ./qw3-mtp-decode \
  ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf 8 2 serial-replay \
  'Spiega dettagliatamente come implementare una tabella hash in C.' 1
caffeinate -disu ./qw3-mtp-decode \
  ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf 8 3 serial-replay \
  'Spiega dettagliatamente come implementare una tabella hash in C.' 2
# Original full-model replay remains available as a control:
caffeinate -disu make test-mtp-decode
```

The test covers autonomous code/math/story prompts and forced rejection at
the first, middle and last proposal. Fault injection is not acceptance-rate
evidence. Correct, fast short-block verification remains the next dependency;
this work alone does not make MTP faster than ordinary generation or enable
it in the CLI/agent.
