# Decode head and short-block cost (2026-10-01)

Base Apple M5, fanless, 24 GB; Qwen3.8-27B UD-Q3_K_XL, full Metal
offload, F16 KV, 512-token sessions. Corrected text IMRoPE is included.
Production source baseline is `8576897`.
Inference runs are sequential under `caffeinate -disu`; no CLI or agent
integration, tool/prompt changes, new quantization or relaxed tolerances.
The started CLI MTP scaffolding was removed at the user's request.

## Q5_K LM head: no credible geometry gain

The 5120 -> 248320 output projection reads 874,086,400 bytes of Q5_K
weights. A standalone shader varies rows per SIMD and SIMD groups per
workgroup while retaining FP32 row calculations and sharing activation
loads. The diagnostic includes a private Metal bridge and overrides only
the final head, not other model projections. Production kernels are unchanged.

Each microbenchmark alternates A/B and B/A calls, omits eight warmup pairs
and times 72 pairs with completed GPU work. Six full-vector checks cover
the actual head, a partial final workgroup, odd output counts 4097/4111,
and 4352/5120 inputs. Limits are max <= 0.0001, RMS <= 0.00001, finite
values required. All tested configurations passed these checks; multi-row
candidates differ by small FP32 compiler/reduction effects, not bit-for-bit.

| Rows per SIMD / groups | Baseline / candidate ms | Micro gain |
| --- | ---: | ---: |
| 1 / 2, baseline clone | 7.4419 / 7.4182 | +0.32% |
| 1 / 4 | 6.9514 / 6.9468 | +0.07% |
| 1 / 8 | 7.0554 / 7.0352 | +0.29% |
| 2 / 2 | 7.0559 / 7.0500 | +0.08% |
| 2 / 4 | 7.1536 / 7.1454 | +0.12% |
| 2 / 8 | 7.0336 / 7.0508 | -0.24% |
| 4 / 2 | 7.1254 / 7.1467 | -0.30% |

These are within-run comparisons, not comparisons between separate rows.
The candidate gains do not exceed the clone/noise control. A subsequent
2/2 micro run was slower by 1.07%. Its full-model paired run independently
sampled 128 tokens with seed 42, temperature 0.6, top-k 20, top-p 0.95,
repetition penalty 1.06. Every token matched; hash `6e27f5726999f2da`.
The first four generated full logit vectors passed the same strict checks.
Timing discards eight forwards and alternates A/B order at every token.

Full-model rates including sampling were 8.587 / 8.618 tok/s; forward-only
gain was +0.31%, with individual blocks ranging from -1.76% to +2.17%.
This is not a credible sustained improvement. Neither candidate nor switch
was installed in production. Absolute paired-session rates are not a
single-session throughput claim.

After building the final optional target, another 128-token paired run passed
all ID/vector checks with the same hash and gave 8.286 / 8.295 tok/s, +0.10%
forward throughput. Its micro-only gain was +1.20%, illustrating again that
an isolated measurement does not establish a model-wide benefit.

## Existing short-block path: favorable cost probe, not MTP verification

`tests/bench_verify_blocks.c` generates a 33-token greedy suffix first,
then compares two independent sessions consuming the same true tokens.
Both sessions are reset and synchronized to the identical 26-token prefix
outside timing for every trial. The final harness also checks initial prefix
logits bit-for-bit. Order alternates A/B and B/A; one warm pair per shape
is excluded, followed by three timed pairs.

The serial side computes the backbone for each token but skips all LM heads
except the last; the batch side also computes only the last LM head. Timers
include completed GPU work, excluding logit readback. Full last-row vectors
are compared, then one identical serial continuation token checks recurrent
and cache continuation outside timing. Token history and position are checked
as well. Limits remain max <= 0.005, RMS <= 0.001, all finite, matching
argmax. The suffix hash is `edf04b19a85a0c46`.

| Block size | Serial / batch ms | Backbone ratio | Worst max / RMS | Pass |
| --- | ---: | ---: | ---: | --- |
| 2 | 215.148 / 398.964 | 0.539x | 0.003328 / 0.0005214 | yes |
| 4 | 429.488 / 461.893 | 0.930x | 0.003698 / 0.0006799 | yes |
| 8 | 859.334 / 682.332 | 1.259x | 0.004657 / 0.0006621 | yes |
| 16 | 1714.476 / 478.966 | 3.580x | 0.011205 / 0.002231 | no |
| 32 | 3496.959 / 511.575 | 6.836x | 0.021278 / 0.002791 | no |

All final and continuation greedy IDs matched, including the failing vector
comparisons. The diagnostic deliberately exits 2 when equivalence limits
are exceeded rather than treating matching greedy IDs as sufficient.
Batch GEMM and serial GEMV use different arithmetic paths; this finding
does not, by itself, demonstrate a reasoning regression or a new production
bug. It prevents claiming equivalence for speculative verification under the
chosen strict comparison. Larger blocks still need a proper reference and
all-row comparison, not looser thresholds to make this probe pass.

A final rebuilt all-size run (one timed pair per size plus warmup), including
the initial-prefix bit-equality check, reproduced identical numerical errors
and pass/fail outcomes. Its backbone ratios were 0.547/0.934/1.262/3.606/6.851;
the all-size run correctly exited 2. CPU/Metal analytic RoPE and runtime
fingerprint regression checks also passed. No production source was changed.

Forcing the existing Q3_K and GQA batch options for block 8 passed the same
quality checks but measured 883.102 / 875.408 ms (1.009x). GQA alone gave
858.882 / 685.450 ms (1.253x), essentially the original ratio. These are
separate diagnostic runs, not interleaved evidence of a new kernel gain.
Neither option was made default.

These ratios are **not generation speedups**. Real verification needs all
logit/hidden rows, draft generation, acceptance/resampling, and accepted-prefix
state handling, including rejected proposals. A useful batch-8 optimization
must reduce the present backbone cost substantially before that overhead
can be assessed. The large jump at 16 rows confirms that the larger matrix
path does not automatically solve the short-block case.

## Reproduce

Run from the repository root. Both targets are optional; neither is a normal
CLI/agent build dependency. Finite checks compile with `-fno-finite-math-only`.

```sh
make qw3-q5-head-bench qw3-block-cost-bench
caffeinate -disu ./qw3-q5-head-bench \
  ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf 2 2 --full
caffeinate -disu ./qw3-block-cost-bench \
  ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf 3
# Select just one block size:
caffeinate -disu ./qw3-block-cost-bench \
  ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf 3 8
```

The Q5 probe supports 1/2/4 rows and 2/4/8 groups, except the unimplemented
4-row/8-group combination. Tests do not change normal model execution.
The block probe accepts repeat counts 1..20 and sizes 2/4/8/16/32. Its
default all-size run is expected to report the measured vector failures;
do not interpret its cost ratios as a successful MTP implementation.
