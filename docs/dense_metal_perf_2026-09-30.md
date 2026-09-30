# Dense Metal optimization measurements (2026-09-30)

Hardware: fanless Apple M5, 24 GB. Model:
`Qwen3.8-27B-UD-Q3_K_XL.gguf`, full 64-layer offload, F16 GQA KV.
Baseline source: `6b9cf46`. Measurements run sequentially, with untimed warmup
and `caffeinate -disu`; no concurrent inference benchmarks. These are short
observations, not sustained-speed guarantees.

## Accepted changes

- Keep the two-column DeltaNet prefill state in FP32 registers for the whole
  chunk, then write the final state once. The algorithm, per-token core output,
  and state layout are unchanged; no quantization or tolerance changes.
- Expand IQ3_S magnitudes into an exact FP32 workgroup table for gate/up pairs.
  Keep the smaller packed table in single matvec: expanding it there did not
  help. The original signed/scaled accumulation order is retained.
- Generalize grouped FlashAttention decode to the dense 6:1 Q/KV geometry.
  Automatically use it from 8192 tokens; retain the old short-context path.
  `QW3_METAL_GQA_FLASH_DECODE_GROUPED=0` disables it, `=1` forces eligibility
  from 1024 tokens. The existing 8:1 MoE threshold stays unchanged.

## Isolated comparisons

| Measurement | Baseline | Candidate |
| --- | ---: | ---: |
| `pp128`, 5 repeats | 144.55 +/- 0.81 tok/s | 152.84 +/- 1.15 tok/s |
| Warm IQ3_S gate/up pair, 100 calls | 0.9682 ms | 0.8448 ms |
| Warm IQ3_S single, 100 calls | 0.5343 ms | 0.5331 ms |
| GQA attention + output at 16385 tokens, 64 calls | 1.8105 ms | 1.5833 ms |

The prefill comparison is about +5.7% throughput. The pair kernel time falls
about 13%. Neither isolated kernel improvement is an end-to-end decode claim.
The GQA timing used the initial constant-input probe; the final regression now
varies input embeddings every 64 positions so constant V cannot conceal wrong
head routing, attention weights, or split/tail reduction.

## Whole-model comparison and variability

Reference: local `llama.cpp` build `4fea119de` (11041), the same dense GGUF,
full GPU offload, FlashAttention on, F16 K/V, batch and microbatch 128. An
initial sequence gave qw3 prefill 149.01 +/- 0.60 versus llama 146.37 +/- 0.56
tok/s, but alternating runs did **not** confirm that apparent overtake:

| Sequential run | `pp128`, 5 repeats |
| --- | ---: |
| llama | 160.29 +/- 1.03 tok/s |
| qw3 candidate | 151.56 +/- 0.99 tok/s |
| llama | 159.54 +/- 0.87 tok/s |
| qw3 candidate | 150.26 +/- 0.91 tok/s |

The recent prefill gap is still roughly 5-6%; parity is not demonstrated.
For generation at depth 128, a `tg32` sequence gave qw3 7.16 +/- 0.09 versus
llama 8.38 +/- 0.31 tok/s. A later baseline/candidate `tg32` sequence gave
7.12 +/- 0.11 / 6.78 +/- 0.28, so the isolated pair speedup does not establish
a stable whole-model gain. Reverse-order `tg16`, 3-repeat runs yielded:

| Sequential run | tok/s |
| --- | ---: |
| qw3 candidate | 7.38 +/- 0.03 |
| qw3 old kernels | 7.20 +/- 0.08 |
| qw3 candidate | 7.17 +/- 0.05 |
| qw3 old kernels | 7.00 +/- 0.03 |

These short pairs suggest a 2-3% advantage, but also show session drift.
Generation remains below llama; no sustained decode speedup is claimed.
The old-kernel comparisons use the same current host binary and a saved
baseline Metal source directory through `QW3_METAL_KERNEL_DIR`.

## Rejected experiments

- Dense tiled single-token DeltaNet, after correcting its gate launch to cover
  all 48 value heads: alternating `tg16` measurements were 7.35 / 7.28 /
  7.26 / 7.22 tok/s (serial / tiled / serial / tiled). Do not enable it in the
  dense decoder. Its CPU logit and 8-step greedy comparisons passed.
- Byte-expanded IQ3_S tables: no demonstrated whole-model benefit.
- FP32-expanded tables for single IQ3_S matvec: retain the packed-table path.
- Grouped dense attention below 8K: no consistent advantage in the 1K/4K probe;
  leave the earlier decode kernel selected there by default.

## Reproduction and validation

```sh
./qw3-bench --metal -m ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf \
  --ngl 64 --llama-style -p 128 -n 0 -r 5 --ctx-alloc 256
./qw3-bench --metal -m ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf \
  --ngl 64 --llama-style -p 0 -n 32 -d 128 -r 5 --ctx-alloc 256
QW3_METAL_DENSE_PROBE_REPS=100 ./qw3-test --metal \
  -m ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf --ctx 32 \
  --metal-session-quant-types-test 66
make test-dense-metal-bringup test-dense-gqa-decode test-metal-logits \
  test-metal-gqa-decode test-dense-agent test-agent-python-smoke \
  test-agent-verify test-runtime-fingerprint
../../llama.cpp/build/bin/llama-bench \
  -m ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf \
  -ngl 99 -p 128 -n 0 -r 5 -ctk f16 -ctv f16 -fa on -b 128 -ub 128 -o json
../../llama.cpp/build/bin/llama-bench \
  -m ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf \
  -ngl 99 -p 0 -n 32 -d 128 -r 5 -ctk f16 -ctv f16 -fa on -b 128 -ub 128 -o json
```

The dense GQA target covers 1023, 1024, 1025, 4097 and 16385 tokens against
CPU, including forced grouped dispatch and uneven tails. Dense bring-up covers
all supported quantizations, CPU logits, 8 greedy tokens, 128-token prefill
and continuation, F32 KV, and exact session export/import. No test thresholds
were loosened.
