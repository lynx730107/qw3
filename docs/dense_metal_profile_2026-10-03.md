# Ordinary dense Metal cost profile — 2026-10-03

Apple M5, existing Qwen3.8-27B-UD-Q3_K_XL GGUF, 64 target trunk
layers on Metal, F16 KV, context allocation 512 and four helper threads.
MTP is removed; no drafter is loaded. This profile was collected after the
sequential [DFlash2 evaluation](dflash2_evaluation_2026-10-03.md), with no
concurrent inference process. It is a cost diagnosis, not a comparison
against llama.cpp or evidence of a new throughput improvement.

The reproducible [profiling script](../tools/profile_dense_metal.py)
stores exact commands, environment, hardware/model identity and current
source hashes alongside [raw measurements](measurements/dense_metal_profile_2026-10-03/results.json).
The existing native profiling hooks were used without changing engine or
kernel code. One complete warmup is excluded from each phase. Assertions
check 64 measured prefill layers and 32 measured decode positions 128–159.

## Prefill pp128

Synthetic 128-token input, one measured repetition after a warmup. These
are sums over the 64 instrumented layers; embedding and final output norm/
logits are outside this layer breakdown.

| Instrumented stage | Sum ms | Share of layer time |
| --- | ---: | ---: |
| Attention, all layers | 566.71 | 52.1% |
| — DeltaNet, 48 layers | 454.80 | 41.8% |
| — GQA, 16 layers | 111.91 | 10.3% |
| Batched FFN | 520.49 | 47.9% |
| Total layers | 1087.20 | 100% |

DeltaNet attention averages 9.48 ms per layer; GQA averages 6.99 ms.
They include their projections, recurrent/cache operations and submission
waits. They do not isolate the recurrent update kernel from matrix work.
The six largest FFN observations were layers 63, 56, 43, 42, 41 and 12;
a single observation does not establish a stable per-layer ranking.

## Decode tg32 at depth 128

Synthetic fixed token IDs, 32 measured forwards after a 128-token prefill.
Each forward computes output logits. No sampling or speculative acceptance
is involved. Values are averages across the 32 measured forwards.

| Instrumented stage | Mean ms/forward | Share of instrumented total |
| --- | ---: | ---: |
| FFN gate/up | 60.67 | 31.9% |
| FFN down | 37.32 | 19.6% |
| DeltaNet, all 48 layers | 41.55 | 21.8% |
| GQA, all 16 layers | 16.25 | 8.5% |
| Attention normalization | 13.68 | 7.2% |
| FFN normalization | 13.31 | 7.0% |
| Output norm/logits/submission | 7.42 | 3.9% |
| Embedding | 0.16 | 0.1% |
| Total | 190.38 | 100% |

FFN projections together account for 51.5% of this instrumented total.
Per-forward totals range from 180.29 to 212.76 ms.

## Interpretation and next optimization targets

Decode profiling synchronizes after every stage in every layer and starts
a new command buffer. Prefill profiling disables the normal combined layer
command batching and separates attention from FFN submissions. The timers
mix CPU submission, GPU execution and waits. In particular, normalization
times and the 190 ms total must not be interpreted as normal production
costs or converted into a production throughput claim. No normal-baseline
speedup can be inferred from these fractions.

The useful next candidates are the mixed-quantization FFN projections for
decode and batched DeltaNet attention/projections for pp128. Before
changing a default, isolate the candidate without these intrusive fences,
compare it to the existing kernel in alternating runs, and confirm an
end-to-end gain with nearby ordinary baseline controls. The existing
head-geometry trials already found no credible full-model gain; this
profile supplies no new reason to repeat that search first.

Any production change must retain full-logit tolerances, greedy IDs,
recurrent/KV continuation, session roundtrips and agent verification. The
CPU/Metal build, RoPE, dense reference logits, bring-up/session and agent
checks passed after MTP removal; this evaluation changed only measurement
tools and documentation, so no new inference behavior is being claimed.

## Reproduce

```sh
make qw3-bench
python3 tools/profile_dense_metal.py --output /tmp/qw3-dense-profile
```

Use a new output directory. The script runs pp128 and tg32 sequentially,
retains unfiltered stderr/stdout, and validates warmup exclusion and sample
positions. Its output is explicitly unsuitable for comparing ordinary
throughput against the llama-server timing convention used by DFlash2.
