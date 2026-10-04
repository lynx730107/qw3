# Text IMRoPE correction (2026-10-01)

Apple M5, 24 GB, Qwen3.8-27B UD-Q3_K_XL. Local llama.cpp reference build
`4fea119de` (11041). All inference runs are sequential. The separate MTP
experiment that uncovered this correction was removed on 2026-10-03.

## Correctness issue uncovered by the MTP differential

The initial native MTP graph matched four greedy IDs, but its full logit
vector differed from the reference (RMSE 0.08876, max absolute error 0.5518).
Intermediate traces matched embedding/hidden normalization, the combined
projection and Q/K/V projections. Divergence began in attention. Both the
CPU helper and Metal kernels rotated adjacent coordinates; Qwen text
IMRoPE pairs coordinates in the two halves of the rotated prefix instead.
The multimodal section interleaving chooses positions, not adjacent pairs.
For the supported text-only engine, section positions are identical.

Fixing only RoPE in the diagnostic reduced MTP RMSE to 0.0003276 and max
error to 0.002892, inside the existing dense differential limits. The same
correction was then applied to CPU, Metal single-token, batch and fused
normalization/RoPE paths, without changing weights or precision.

A direct reproduction uses the same engine binary, sequential llama.cpp
reference and F16 KV, switching only to an archived pre-fix Metal source:

| Main-model prefix | Before RMSE / max | After RMSE / max |
| --- | ---: | ---: |
| `66,30057,14578,1789` | 0.015219 / 0.10585 | 0.00004743 / 0.0002377 |
| `66,561,1118,321,2018,41131,314,279` | not measured | 0.00006867 / 0.0003693 |

Top-eight ranks and eight subsequent greedy IDs matched the reference even
before the fix in the first case; testing only IDs had missed the drift.
The strict limits remain RMSE <= 0.001, max <= 0.005. The reference now
evaluates prefix tokens individually, matching qw3 GEMV and cache precision.
The old multi-row llama prefill can select different GEMM arithmetic and
is available explicitly with `QW3_DENSE_LLAMA_BATCH=1` for diagnostics.

Validation passed the no-model CPU/Metal analytic RoPE checks (16 cases,
positions 0/1/28/511, head dimensions 128/256, rotated prefix 64), both dense
llama differentials, a four-token MoE llama differential, full dense bring-up,
MoE logit/greedy/prefill regression, dense agent smoke, executor verification
self-test and runtime fingerprint tests. No thresholds were relaxed.
This correction is about numerical fidelity, not a throughput claim.
The sampled 128-token output changed from the old rotation, as expected;
an IQ2 geometry A/B with corrected rotation still matched every token and
retained +0.61% forward throughput. Whole-model sampled decode in a subsequent
warm single-session run was 8.503 tok/s. Generation is still the bottleneck.
