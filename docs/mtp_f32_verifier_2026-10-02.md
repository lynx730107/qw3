# All-row FP32 MTP verifier (2026-10-02)

Base Apple M5, dense Qwen3.8-27B UD-Q3_K_XL, ctx512, full offload and
F16 KV. This is a private verifier experiment, not agent/CLI integration or
an accelerated production generation setting. Default decoding, prefill and
the existing `serial`, `serial-replay` and `batch` probe policies are unchanged.
Work started on October 2; the final regression and paired checks completed
on October 3.

## Precision investigation

The earlier `batch` verifier failed its first hidden row (max 0.00333309,
RMS 0.000534521). Reverting only to the legacy batch DeltaNet kernel did
not fix it: max 0.00434725, RMS 0.000555875. Private compile-time hooks now
record every layer's post-attention and post-FFN residual rows, with complete
layer/site/row coverage required before reporting the first-cycle trace.
Production hooks reduce to constants and allocate/copy nothing.

The opt-in `batch-f32` probe avoids the mapped F16 matrix/pair tiles for
short IQ3_S and IQ4_XS projections. It also retains the scalar-ordered fused
batch DeltaNet recurrence instead of the tiled dot-reduction path. Using
FP32 projections alone with the tiled recurrence still failed the extended
code test (cycle 2, row 2, hidden max 0.00102996826). No tolerance was widened.

Two private kernels share each quantized weight row among up to four tokens.
They retain FP32 activations/products, GGUF magnitudes and float scales;
loading the model's stored half scale does not create a rounded half weight
tile. IQ3_S caches its exact 512-entry packed ushort grid in 1 KiB and
expands signs/magnitudes before FP32 vector dots. IQ4_XS uses the exact
16-entry table in threadgroup storage. Offsets, padded strides, odd output
counts and partial token groups are explicitly supported.

`QW3_MTP_SHORT_VECTOR=0/1/2/3` is a private ablation mask: scalar projection
control, IQ3_S only, IQ4_XS only, or both. The probe's default is both.
`batch-f32-scalar` is an explicit projection control, not a claim that its
different reduction grouping passes the full all-row corpus.

## Checks and adoption boundary

The direct kernel test has 84 cases across IQ3_S/IQ4_XS, input sizes
256/4352/5120/17408, output sizes 1/33/4097/4111/5120/17408 and token counts
1/2/3/4/5/8/15. Inputs and outputs have nonzero offsets and odd padded
strides. Complete outputs are compared with the ordinary scalar kernels;
boundary rows also use independent llama.cpp `ggml_get_type_traits(...)->to_float`
dequantization and double-precision CPU dots. Guard gaps must stay intact.
Limits remain max 0.0001, RMS 0.00001, CPU boundary max 0.0001. Reinterpreted
shapes use only a bounded prefix of real quantized tensor data.

The MTP regression requires all proposed vocabulary and normalized hidden
rows to remain finite, with the original logit max/RMS limits 0.005/0.001,
hidden limits 0.001/0.0001 and matching greedy IDs. Rejected prefixes still
use full scalar replay and byte-identical independent rollback-state checks.
Final logits and eight continuation steps are checked against untouched
ordinary generation; continuation max/RMS are now reported and shell-checked.

**Fully accepted batched state is not asserted byte-identical to scalar
generation.** The probe reports `serial_state_exact=0` for batched modes.
The original serial/compact regressions still require exact state and zero
row/continuation error. Compact rollback is not combined with this verifier:
its stored scalar inputs and byte-exact invariants cannot simply be assumed
valid for a numerically different batch transition.

## Rejected performance candidates

Three alternating paired trials, after a discarded paired warmup, compared
short FP32 vector projections against scalar FP32 projections, with identical
initial states and the same precise recurrence on both sides. These cost
probes compute only the last head; they do not substitute for all-row tests.

| Candidate | Scalar FP32 control | Candidate | Decision |
| --- | ---: | ---: | --- |
| Initial two-kernel layout | 902.331 ms | 965.999 ms | Slower |
| IQ3_S only, expanded grid | 903.446 ms | 951.646 ms | Slower |
| IQ4_XS only, shared table | 899.736 ms | 869.525 ms | Faster cost, fails all-row correctness |

The IQ4_XS-only candidate failed the extended code test at cycle 2, row 2:
hidden max 0.00243186951, RMS 0.000189609457. Its component gain is not an
adoptable optimization. This motivated the compact IQ3_S grid layout.

A disabled `QW3_MTP_GDN_CACHED=1` trial retains 128-column state in a private
kernel. It passed a 32-token all-row code check but showed no material gain
in instrumented verifier totals (9705.6 versus 9713.9 ms in separate runs).
That is not an alternating speed benchmark; the trial is not enabled by default.

## Final compact-grid results

The default two-kernel compact-grid configuration passed the complete six-case
regression: 88 emitted tokens matched ordinary generation, all 179 proposed
rows passed the unchanged numerical thresholds, all 21 rejection restores
were byte-identical to independent full scalar replay, and all six eight-token
continuations passed. The forced-last case crosses the 32-token F16 tail ring.

Worst proposed logit max/RMS were 0.000515938/0.0000679806; hidden max/RMS
were 0.000505447/0.0000510053. Worst continuation max/RMS were
0.000352383/0.0000539738. This is tolerance-checked equivalence, **not** zero
error or bit-identical accepted batched state. The separate 84-case kernel
test also passed (worst scalar GPU max/RMS 0.00000572205/0.000000556939,
worst CPU boundary max 0.000000604061).

Five alternating paired trials plus a discarded warmup gave 912.943 ms for
the scalar-FP32 projection control and 904.378 ms for the compact-grid vector
candidate (about 0.9% throughput difference). This small component result
does not establish a useful overall speedup or a gain in ordinary generation.
The cost probe compares two precision-preserving variants, not the original
faster-but-failing F16 batch path.

The full code MTP run still measured 2.729 instrumented tok/s versus its
ordinary baseline of 8.729 tok/s; verification consumed 9054.661 ms for
76 rows and full accepted-prefix replay 1775.965 ms. Separate sessions ran
serially on a fanless M5; ordinary baseline rates varied across the corpus.
Clocks/temperatures were not measured, so the cause of that variation is not
established. These are diagnostic costs, not an interleaved overall promotion benchmark.
Further verification/head optimization and compatible rollback remain needed.

Production regression passed `test-runtime-fingerprint`,
`test-dense-metal-bringup`, `test-dense-agent` and `test-agent-verify`.
Fresh eight-token `serial` and `serial-replay` controls retained zero row
and continuation error, byte-identical final state and matching token history.
The original `batch` negative control still exits nonzero at cycle 0, row 0
with hidden max 0.00333309174/RMS 0.000534521453, confirming that this mode
was not silently replaced by the new verifier.

## Reproduce

```sh
make qw3-mtp-decode qw3-mtp-short-bench
caffeinate -disu make test-mtp-short-f32-quant test-mtp-f32
# Alternating precise batch control/candidate cost (last head only):
QW3_BLOCK_BASELINE_BATCH=1 caffeinate -disu ./qw3-mtp-short-bench \
  ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf 3 8
# First-cycle all-row internal trace; not timing evidence:
QW3_MTP_LAYER_TRACE=1 caffeinate -disu ./qw3-mtp-decode \
  ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf 32 8 batch-f32
```

The diagnostic still uses greedy proposals, fixed output counts, conservative
full snapshots and scalar heads. It lacks production sampling/EOS/cancellation
and public MTP session integration. Correctness progress must not be presented
as a speedup over ordinary decoding or llama.cpp.

The subsequent optional shared Q5_K head and all-row paired head-cost probe
are documented in [the October 3 experiment](mtp_shared_head_2026-10-03.md).
This report's original scalar-head timings remain historical controls.
