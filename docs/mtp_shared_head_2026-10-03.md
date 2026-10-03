# Shared FP32 MTP head (October 3, 2026)

Base Apple M5, dense Qwen3.8-27B UD-Q3_K_XL, context 512, full Metal
offload, F16 KV. These private diagnostics do not change ordinary decoding,
prefill or the agent. The shared head is opt-in even in `batch-f32`.

## Implementation

The existing precise verifier performs a separate Q5_K vocabulary projection
for every proposed token. The new private kernel decodes each Q5_K weight
block once for up to four token rows, retaining FP32 activations/products
and the scalar kernel's per-token reduction structure. GGUF half scales are
read as stored; no additional half weight or activation tiles are introduced.
Only the private diagnostic compiles or dispatches this kernel.

Final normalization uses the ordinary scalar RMSNorm pipeline on every
batch residual row. Normalized rows and complete vocabulary rows are written
directly to the diagnostic's existing shared buffers, avoiding intermediate
per-row copies. The last residual, normalized hidden row and vocabulary row
are installed in the target session, preserving ordinary continuation and
the existing full-snapshot/scalar rejection replay. All proposed rows,
including those after a rejected draft, are still checked.

The first implementation interleaved accumulation for all tokens inside
the element loop. It was numerically correct but slower. Reordering the
work to finish each token's local accumulations before the next token
improved the component cost. Register occupancy/spilling was not measured;
it is a possible explanation, not an established hardware diagnosis.

`QW3_MTP_BATCH_HEAD=1` enables the private head in `batch-f32` only.
`QW3_MTP_HEAD_GROUP=1/2/4` selects its compiled token grouping; default 4.
Invalid settings are rejected. No public session API, CLI or agent flag is
added. Compact rollback, sampling, EOS stopping and cancellation remain
outside this experiment.

## Numerical checks

The direct test has 42 cases: six shapes, input sizes 256/4352/5120,
output sizes 1/33/4097/4111/248319/248320, and token counts
1/2/3/4/5/8/15. Inputs and outputs have nonzero offsets and odd padded
strides; output guard gaps must remain intact. Shapes use bounded prefixes
of the actual output tensor. Every output is compared with ordinary GPU
Q5_K GEMV. Boundary rows also use independent llama.cpp CPU dequantization
and double-precision dots. No numerical limit was loosened.

All 42 checks passed for both final two/four-token layouts. Worst GPU
max/RMS differences were 0.000000953674/0.000000112652; independent CPU
boundary max was 0.000000464295. These are not bit-identical logits.

The full four-token-head MTP suite passed all six cases: 88 emitted token
IDs matched ordinary decode, all 179 proposed rows passed original
thresholds, 21 rejection restores were byte-identical to independent full
scalar replay, and all six eight-token continuations passed. Worst proposed
logit max/RMS were 0.000515938/0.0000679808; hidden max/RMS were
0.000505447/0.0000510053. Worst continuation max/RMS were
0.000352383/0.0000539738. Accepted batched state is still reported as
`serial_state_exact=0`; unchanged tolerance checks must not be described
as scalar byte equivalence.

Final rebuild regression also passed the existing 84 IQ3_S/IQ4_XS kernel
checks, runtime fingerprint tests, dense Metal bring-up, dense agent smoke
test and executor verification-guard self-test. No production arithmetic
or tool behavior was changed.
Fresh eight-token `serial` and `serial-replay` controls also retained zero
row/continuation error, byte-identical final state and the same token IDs.

## Performance scope

Direct full-vocabulary head probes used alternating order, two discarded
paired warmups and ten measured pairs. Each side emitted all eight rows
with the same input buffer. Costs include command submission/waiting, not
prefix preparation or CPU-reference checks.

| Layout | Eight scalar heads | Shared heads | Result |
| --- | ---: | ---: | --- |
| Initial interleaved four-token | 53.282 ms | 141.648 ms | Rejected layout |
| Initial interleaved two-token | 56.967 ms | 95.764 ms | Rejected layout |
| Initial one-token control | 53.647 ms | 54.446 ms | No useful gain |
| Reordered two-token | 56.904 ms | 45.257 ms | Lower component cost |
| Reordered four-token | 55.398 ms | 42.392 ms | About 23.5% less head time |

The four-token ratio is 1.307 (about 30.7% greater **head-only** throughput),
not 30.7% faster generation. The head is a small fraction of total verifier
cost, and all these are separate head-only trials.
The final rebuilt four-token variant repeated the 42 numerical checks and
ten alternating pairs: 53.812 versus 41.704 ms, ratio 1.290. This confirms
the component result's approximate size, not an end-to-end speedup.

A new all-row alternating verifier probe holds the backbone, proposals and
initial GPU checkpoint identical on both sides. A discarded warmup and
five measured pairs gave 896.615 ms with serial heads and 889.611 ms with
shared heads, a ratio of 1.008. Only three of five pairs were faster, so
the small approximately 0.8% difference is noisy, not a robust overall
promotion result. Every vocabulary row matched greedy IDs and strict
max/RMS limits (0.00000214577/0.000000309999). Hidden rows and a subsequent
ordinary-decode continuation were byte-identical between the two head
variants. These A/B checks are outside timing.

The full instrumented code MTP run emitted 32 tokens at 2.927 tok/s versus
its ordinary baseline's 9.319 tok/s. Verification took 8419.067 ms for 76
rows, full rejection replay 1667.003 ms and draft generation 686.183 ms.
This separate diagnostic run is not an alternating end-to-end benchmark;
its higher rate than earlier runs must not be attributed entirely to the
head change. It remains far slower than ordinary decoding.

The outcome narrows the next target: most verifier cost remains in the
backbone, not the vocabulary head. Shared heads alone cannot make MTP
competitive. Compatible compact rollback and better proposal acceptance
are also still required. No result here establishes a gain over llama.cpp.

## Reproduce

```sh
make qw3-mtp-decode qw3-mtp-head-bench
caffeinate -disu make test-mtp-head-quant test-mtp-batch-head
# Same precise backbone; every output row, not just the last head:
caffeinate -disu ./qw3-mtp-head-bench \
  ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf 5 8
# Group-size control:
QW3_MTP_HEAD_GROUP=2 caffeinate -disu make test-mtp-head-quant
```
