# M5 short-batch vector kernels (2026-10-01)

Base Apple M5, fanless, 24 GB; Qwen3.8-27B UD-Q3_K_XL, full Metal offload,
F16 KV. Baseline source `9fb45aa`. Inference runs are sequential under
`caffeinate -disu`. This implements short-batch projection kernels, not MTP
generation, acceptance, rollback or a CLI/agent MTP flag.

## What changed

The previous IQ3_XXS/IQ2_S four-token fallback assigned whole 256-weight
quantization blocks to SIMD lanes. A 5120-input projection has only 20 such
blocks, leaving lanes idle and doing long scalar dequantization/dot loops.
The new kernels distribute 32-weight units instead, using FP32 vector dots
and sharing the decoded weights among up to four token rows. Both table
magnitudes and input activations retain their original precision; no half
weight tile is introduced. The dot grouping follows the existing vector
GEMV, but results are not claimed bit-identical to the old scalar batch path.

IQ3_XXS caches a 256-entry FP32 magnitude table plus sign bytes; IQ2_S caches
exact byte magnitudes and converts them to FP32 when used. Every workgroup
initializes its table before the barrier, including partial output groups.
Partial token groups compute only real rows, with explicit input/output
offsets and strides. Kernels still use 128 threads, four output rows and up
to four token rows per workgroup.

Default dispatch changes only on the exact device name `Apple M5`, for
2..8 token rows and projections with at least 4096 inputs and outputs.
Other formats, smaller shapes, other GPUs, one-token decode and >=9-row
prefill retain their prior paths. Optional pipeline creation failure or an
insufficient thread limit falls back to the previous kernel.
`QW3_METAL_DENSE_SHORT_VECTOR=0` disables the policy; `=1` opts in on another
GPU while retaining the shape/token guards. Kernel/source and environment
fingerprints include the change; old saved states are not silently reused.

## Measured short-block gain

The benchmark uses two independent 512-token sessions and a true 33-token
greedy suffix, as in the earlier [cost probe](decode_block_cost_2026-10-01.md).
Every trial resets both sessions and prefills with the **old** policy outside
timing, requiring identical initial logits and token history. A/B order
alternates, with one paired warmup omitted. Both sides compute just the last
LM head; full last-row logits and a serial continuation are checked outside
timing. GPU work is completed before the timed call returns.

With `QW3_BLOCK_BASELINE_BATCH=1`, the comparison is old batch versus new
batch. The production-policy A/B, five timed pairs, measured:

| Size | Old batch | New batch | Throughput gain / time saved |
| --- | ---: | ---: | ---: |
| 8 | 669.235 ms | 590.429 ms | +13.3% / 11.8% |

An earlier private-prototype five-pair run gave 670.644 / 592.768 ms (+13.1%).
The final guarded-kernel replica gave 686.863 / 599.134 ms (+14.6%), with
all five timed pairs favoring the new path and identical comparison errors.
These are paired short-block cost measurements, **not** gains in normal
single-token generation or end-to-end speculative decoding.

Without that option, the baseline is serial backbone evaluation, computing
only the last LM head. The combined-kernel prototype versus serial gave:

| Prompt, block 8 | Serial / new batch | Worst max / RMS including continuation |
| --- | ---: | ---: |
| Hash table in C, 3 timed pairs | 833.510 / 595.485 ms | 0.003992 / 0.0007269 |
| Equal groups of pencils, 2 pairs | 837.565 / 602.151 ms | 0.003405 / 0.0006575 |
| Library story, 2 pairs | 845.680 / 602.609 ms | 0.004560 / 0.0008298 |

The final production-policy corpus reproduced these errors and passed all
five checks (code blocks 2/4/8 plus math/story block 8). Old/new batch full
vectors also passed (max 0.004222, RMS 0.0006167 for code8). Every tested
final and continuation argmax matched. Limits remain max <= 0.005 and
RMS <= 0.001, all values finite; none were relaxed.

For size 2, the new batch was still slower than serial (360.966 / 214.312 ms);
size 4 was approximately tied (435.721 / 431.126 ms). Thus this does not
make a short two-token MTP rollout automatically beneficial. Large-block
numerical differences found in the prior probe are not fixed by this change.

The IQ3_XXS-only prototype was not retained alone: despite a cost improvement,
its code8 serial comparison exceeded the maximum error limit (0.005298).
The combined IQ3_XXS/IQ2_S variant passed the same unchanged threshold on all
tested corpus cases. This is why matching greedy IDs alone was not used as
the acceptance condition.

## Validation and reproduction

`tests/bench_short_batch.m` checks both production kernels directly against
the previous batch kernel and independent CPU dequantization/double dots at
boundary output rows. There are 84 cases: input sizes 256/4352/5120/17408,
output sizes 1/33/4097/4111/5120/17408 and token counts 1/2/3/4/5/8/15.
All cases use nonzero offsets, odd padded strides and guarded output gaps.
Full GPU vector limits are max <= 0.0001, RMS <= 0.00001; independent CPU
boundary-dot error must also be <= 0.0001. All cases passed. The policy test
checks opt-out, opt-in, other-GPU default and shape/token exclusions.
Alternate shapes reinterpret a bounded prefix of actual quantized tensor
blocks on both sides; they are not asserted to be actual layer dimensions.

The optional shell corpus test independently checks finite numerical metrics
and thresholds, without prescribing speed. Production validation passed:
full dense Metal bring-up (CPU projections, logits, greedy sequence, tails,
F16/F32 KV, prefill/state continuation and roundtrip), MoE Metal logits,
dense agent smoke, runtime fingerprint tests, and CPU/Metal analytic RoPE.

A subsequent ordinary single-session 128-token sampled run gave 8.889 tok/s
(112.125 ms forward, 0.377 ms sampling), with the unchanged token hash
`6e27f5726999f2da`. This is not an interleaved speed comparison; variation
from earlier single-session runs must not be attributed to the short-batch
kernels, which do not execute in this benchmark's one-token forwards.

```sh
make qw3-cli qw3-agent qw3-short-batch-bench
caffeinate -disu make test-short-batch
QW3_BLOCK_BASELINE_BATCH=1 caffeinate -disu ./qw3-short-batch-bench \
  ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf 5 8
# Quant/tail/policy checks without the model's full forward passes:
make test-short-batch-quant
```

`QW3_BLOCK_PROMPT` selects another prompt; the block probe still returns only
the last logit row, so it cannot establish all-row MTP verification. One-token
generation remains on the existing GEMV kernels. The next requirements are
further short-block cost reduction, all-row target logits/hidden states and
correct state handling before any claim of accelerated generation.
