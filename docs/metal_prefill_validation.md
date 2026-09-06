# Metal Prefill Validation

This project treats logits as the regression boundary for Metal prefill changes.
Performance-only checks are not enough: every prefill optimization must preserve
CPU/Metal final logits and greedy top-1 choices before it can be enabled by
default.

## 2026-06-11 MoE Fast-Layout Probe Notes

Two DS4/llama.cpp-inspired MoE prefill probes were evaluated and not retained:

- Pair-token side map: `qw3_moe_topk_expert_map` was temporarily extended to
  write a `pair -> token` buffer, and `gate_up_pair_mpp` used it instead of
  recomputing `pid / n_active` in the hot loop. It was logits-safe, but
  `pp4096` dropped to 437.81 tok/s over 3 reps and the MoE profile still showed
  `gate_up_pair_mpp` around 60-62 ms/layer. This confirms the bottleneck is not
  the integer division/modulo bookkeeping.
- Naive `NR0=128` gate/up pair tile: rejected before compile. The existing
  128-thread tile loader only covers 64 A rows, and the B loader layout would
  write past the current `NR1 x NK` RHS tile if thread count were simply raised.
  A real 128-row tile needs a redesigned A/B cooperative load, not a constant
  change.
- Opt-in RHS packed prototype: a DS4-style compact RHS f16 buffer was tested
  behind `QW3_METAL_MOE_PACK_RHS=1`. The first version exposed why agent tool
  smoke tests are mandatory: the pack capacity missed per-expert block padding
  and the pack kernel only wrote 256 of 2048 tile elements. After fixing both,
  `make test-metal-logits` and an agent `write` tool smoke passed, but
  corrected `pp4096` was 432.72 tok/s over 3 reps. The apparent earlier 550
  tok/s was from the incomplete/invalid pack. Do not keep or promote this
  version.

Validation gates used after rollback:
- `make test-metal-logits`: passed; Metal logits, greedy, and prefill batch
  regression all OK.
- Agent tool smoke: `qw3-agent --ctx 1600 --nothink -p ...` successfully called
  the native `write` tool and created a temporary file with exact content.

Next serious direction remains a real DS4-style fast-layout RHS path, but it
must avoid repacking the whole 32x64 RHS tile for every layer in the naive way.
Smaller bookkeeping changes are below the noise floor.

Default safety policy:
- `QW3_METAL_KV_Q8_0` is not part of the default validation path.
- `QW3_METAL_KV_F16=1` enables an experimental f16 GQA KV cache. It is logits
  safe under the regression tests and useful for memory pressure, but it is not
  a default prefill speed path because it did not improve the long-prompt
  benchmark on M5.
- The llama.cpp-style concurrent Metal encoder is enabled by default for
  prefill frontiers. Set `QW3_METAL_PREFILL_CONCURRENT_DISABLE=1`,
  `QW3_METAL_PREFILL_CONCURRENT=0`, or `GGML_METAL_CONCURRENCY_DISABLE=1`
  for the legacy serial encoder.
- `QW3_METAL_PREFILL_BATCH` defaults to 4096, the current Metal batch cap.
- DeltaNet batch GDN uses the two-column tiled recurrent core by default.
  `QW3_METAL_BATCH_GDN_TILED4=1` enables the experimental four-column core for
  profiling; keep it opt-in until repeated end-to-end benches show a stable win.
- DS4-style Metal4 direct-RHS Q8_0 prefill matmul is enabled by default for
  aligned projection batches when the Metal4 tensor API probe succeeds. Set
  `QW3_METAL_Q8_NAX_DISABLE=1` for the legacy Q8 MM path.
  `QW3_METAL_Q8_NAX_TILE=32|64|128` can force the token tile for profiling.
- Expert-major MoE gate/up is enabled inside batch prefill with at least 32
  tokens; set `QW3_METAL_MOE_MAP_GATEUP_DISABLE=1` for legacy comparisons.
- Expert-major MoE down is enabled inside IQ4_XS batch prefill with at least 32
  tokens; set `QW3_METAL_MOE_MAP_DOWN_DISABLE=1` for legacy comparisons.
- Metal4 TensorOps MoE kernels are enabled automatically on M5/M6/A19/A20
  devices after a successful compile probe. Set `QW3_METAL_DISABLE_METAL4=1`
  to disable the feature probe, or `QW3_METAL_MOE_MPP_DISABLE=1` for legacy
  MoE prefill comparisons. Gate/up and down can also be disabled separately
  with `QW3_METAL_MOE_MPP_GATEUP_DISABLE=1` and
  `QW3_METAL_MOE_MPP_DOWN_DISABLE=1`.
- Q6_K expert-down prefill uses a Metal4 TensorOps mapped MPP kernel when
  available. Set `QW3_METAL_MOE_Q6_MPP_DISABLE=1` for the legacy mapped Q6_K
  comparison path.
- IQ4_XS MoE pair-MPP preweights the activated mid buffer with router weights
  before expert-down when the mapped f32 down MPP path is active. This keeps
  the algebra equivalent while moving the scale out of the larger `n_embd`
  down-output store.
- GQA batch prefill fuses RMSNorm, Q gate copy, and RoPE by default. Set
  `QW3_METAL_GQA_NORM_ROPE_SPLIT=1` for the legacy split-kernel comparison.
- GQA cached prefill attention uses the llama/ds4 FlashAttention kernel by
  default when the shared Metal source is available. Set
  `QW3_METAL_GQA_FLASH_ATTN=0` or `QW3_METAL_GQA_FLASH_ATTN_DISABLE=1` to use
  the native `block4` fallback. In fallback mode, set
  `QW3_METAL_GQA_ATTEND_BLOCK2=1` for the two-query comparison path, or
  `QW3_METAL_GQA_ATTEND_BLOCK1=1` for the legacy one-query kernel.
- FlashAttention prefill uses the CPU dense-mask fill plus block-scan path by
  default. Set `QW3_METAL_GQA_FLASH_GPU_MASK=1` to test the experimental
  GPU-generated causal block map that only writes boundary mask blocks.
- Metal session reset does not zero the GQA KV buffers by default. The valid
  KV range is controlled by the session position and every prefill/decode step
  writes the entries it can later read. Set `QW3_METAL_FORCE_KV_CLEAR=1` only
  when bisecting memory bugs against the legacy full-clear behavior.
- Metal session reset also skips prefill work-buffer clears by default. The
  batched prefill pipeline overwrites `prefillX0`, `prefillX1`, and scratch
  ranges before reading them. Set `QW3_METAL_FORCE_PREFILL_CLEAR=1` to restore
  the legacy eager clear while debugging scratch lifetime issues.
- `QW3_METAL_PROFILE_PREFILL_GQA_SYNC=1` is a diagnostic-only sync profiler
  for GQA prefill stages: attention norm, qkv projection, norm/RoPE, cache
  write, attend, output projection, residual norm.
- `QW3_METAL_PROFILE_PREFILL_LINEAR_SYNC=1` is a diagnostic-only sync profiler
  for linear-attention prefill stages: attention norm, qkv/gate/alpha/beta
  projection, conv1d, q/k l2norm, DeltaNet GDN, output projection, residual
  norm.
- `QW3_METAL_PROFILE_PREFILL_LINEAR_PROJ_SYNC=1` is a more intrusive
  diagnostic split for the linear-attention projection group. It serializes and
  measures qkv, gate, and alpha/beta projections separately, so use it only to
  choose the next optimization target.
- `QW3_METAL_PROFILE_PREFILL_MOE_SYNC=1` is a diagnostic-only sync profiler
  for the batched routed MoE stages: map, gate, up, activation, down, reduce.
- `QW3_METAL_MOE_MAP_GATEUP_PAIR=1` enables the experimental fused mapped
  gate/up/SwiGLU kernel. It is not default because the current version is
  correct but slower on the validation prompt.
- `QW3_METAL_MOE_MID_F16=1` enables the DS4-style F16 routed-MoE intermediate.
  It is correct under logits regression, but remains opt-in until it shows a
  repeatable speed win on long prompts.
- The routed-MoE IQ4_XS down path uses a compact F32 SwiGLU intermediate by
  default. This keeps the previous F32 precision while avoiding the larger
  token-stride scratch layout for the down projection. Set
  `QW3_METAL_MOE_MID_F32_DISABLE=1` for legacy comparisons.
- Linear-attention batch DeltaNet uses the two-column tiled recurrent core plus
  a separate gated RMSNorm node by default. Set
  `QW3_METAL_BATCH_GDN_TILED2_DISABLE=1` for the previous one-column tiled
  comparison path, or `QW3_METAL_BATCH_GDN_LEGACY=1` for the old scalar fused
  GDN kernel.
- Metal command buffers use unretained references by default, matching
  llama.cpp's graph compute path. Set `QW3_METAL_RETAINED_COMMAND_BUFFERS=1`
  for legacy comparisons.

Required checks after each Metal prefill change:
1. Build `qw3-metal`.
2. Run `make test-metal-logits`.
3. If the change touches a layer-local primitive, run the matching
   `--metal-session-...-test` diagnostic as well.
4. For batched prefill work, also run the logits tests with the target
   `QW3_METAL_PREFILL_BATCH` value before enabling it by default.
5. Only then benchmark with `--metal-run` or `qw3-agent`.

Useful commands:

```sh
make qw3-metal
make test-metal-logits
make test-metal-logits-concurrent
./qw3-metal -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 1024 \
  --metal-session-decode-test -p "ciao"
./qw3-metal -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 1024 \
  --metal-greedy-test 4 -p "ciao"
env QW3_METAL_PREFILL_TEST_TOKENS=64 ./qw3-metal \
  -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 1024 \
  --metal-session-prefill-q8-batch-test 66
```

## 2026-06-11 MoE Mid Preweight

The IQ4_XS MoE pair-MPP path now applies `router_weights[pair]` while writing
the activated f32 mid buffer, and the mapped f32 down MPP path skips the same
scale on its `down_slots` output. This is equivalent because the expert-down
projection is linear, but it moves the multiply from `n_embd` outputs per
token/expert pair to `n_ff` mid values.

Validation:
- `make qw3-bench-metal`
- `make test-metal-logits`
- `./qw3-bench-metal -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx-alloc 16000 --llama-style -p 4096 -n 0 -r 3 --no-warmup`

Observed result on Apple M5:
- `pp4096`: 441.89 tok/s over 3 repetitions, stdev 5.85.

Rejected probe:
- Direct atomic accumulation from down MPP into `x0` passed logits, but removed
  the reduce kernel without a stable end-to-end win (`pp4096` stayed around
  444-446 tok/s). Do not pursue that atomic path as the next optimization.

## 2026-06-06 GQA FlashAttention Prefill Default

GQA full-attention prefill now uses the shared llama/ds4 FlashAttention Metal
kernel by default. The QW3 wrapper builds the causal mask block map and pads
non-64-aligned K/V tails into a temporary interleaved layout compatible with
the imported kernel, so real prompts such as `prompt_perf.txt` no longer fall
back to the older scalar/block4 attention path.

Validation after the change:
- `make qw3-metal`
- `make test-metal-logits`
- `./qw3-metal -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 16000 --nothink --prompt-file ./prompt_perf.txt -n 128`
- `make qw3-bench-metal`
- `./qw3-bench-metal -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --llama-style -p 4096 -n 0 -r 3`
- `./qw3-bench-metal -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --llama-style -p 4095 -n 0 -r 1`

Benchmark notes on Apple M5:
- `prompt_perf.txt`: 6399 prompt tokens, 21063.1 ms prefill, 303.80 tok/s;
  generated text was coherent on the previous garbage-regression prompt.
- `pp4096`: 315.88 tok/s average across 3 runs.
- `pp4095`: 315.15 tok/s in a single non-aligned run.
- Previous default `pp4096` baseline was about 212 tok/s; explicit
  FlashAttention before padded tails/block maps was about 240 tok/s.

## 2026-06-06 DeltaNet Two-Column Tiled GDN

The linear-attention prefill GDN recurrence now computes two DeltaNet state
columns per simdgroup. This keeps the recurrent token loop and F32 state update
unchanged, while reducing repeated Q/K loads and threadgroup scheduling in the
dominant linear-attention stage.

Validation after the change:
- `make qw3-metal`
- `make test-metal-logits`
- `./qw3-metal -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 16000 --nothink --prompt-file ./prompt_perf.txt -n 128`
- `make qw3-bench-metal`
- `./qw3-bench-metal -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --llama-style -p 4096 -n 0 -r 1`

Benchmark notes on Apple M5:
- Opt-in validation before promotion: `pp4096` 350.65 tok/s in one run.
- `prompt_perf.txt` default TILED2: 6399 prompt tokens, 19554.3 ms prefill,
  327.24 tok/s; generated text was coherent.
- `QW3_METAL_BATCH_GDN_TILED2_DISABLE=1` on the same prompt: 6399 prompt
  tokens, 24125.4 ms prefill, 265.24 tok/s.
- Linear profiler with the new kernel shows `deltanet_gdn` mostly around
  40-52 ms per linear layer, down from roughly 75-90 ms.

## 2026-06-06 Concurrent Prefill And Q8 NAX Probe

The prefill command encoder now defaults to the concurrent Metal dispatch mode,
matching the graph-frontier orchestration used by llama.cpp/ds4 more closely.
This does not change the math or buffer layout; it only allows independent
dispatches in the same prefill frontier to be scheduled concurrently. The
legacy serial encoder is available with the opt-outs listed above.

The DS4-style Metal4 Q8_0 NAX direct-RHS matmul was also ported as an
experimental opt-in for aligned Q8_0 projection batches. It compiled and was
logits-safe, but initially stayed disabled because the current QW3 scratch
layout did not show a repeatable win. It was revalidated and promoted on
2026-06-07 after the projection profiler showed a repeatable win on M5.

Validation after the change:
- `make qw3-metal`
- `make test-metal-logits-concurrent`
- `env QW3_METAL_Q8_NAX=1 make test-metal-logits`
- `env QW3_METAL_PREFILL_CONCURRENT=1 ./qw3-metal -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 16000 --nothink --prompt-file ./prompt_perf.txt -n 128`

Benchmark notes on Apple M5:
- Serial default before promotion: `pp4096` 347.72 tok/s in a nearby run.
- Explicit concurrent prefill: `pp4096` 354.20 tok/s.
- Default after promotion under later thermal conditions: `pp4096` 331.47
  tok/s, while `QW3_METAL_PREFILL_CONCURRENT_DISABLE=1` in the same window was
  293.74 tok/s.
- Real `prompt_perf.txt` after promotion: 6399 prompt tokens, 19182.1 ms
  prefill, 333.59 tok/s; generated text was coherent.
- `QW3_METAL_Q8_NAX=1`: `pp4096` 352.54 tok/s; tile 64 and tile 32 were worse
  at 309.43 and 291.00 tok/s.
- Historical opt-in probe before the 2026-06-07 promotion:
  `QW3_METAL_Q8_NAX=1 QW3_METAL_PREFILL_CONCURRENT=1` measured `pp4096`
  341.78 tok/s in that older thermal/code window. The later revalidation below
  supersedes this result.

## 2026-06-07 Q8 NAX Default For Prefill Projections

The Q8_0 NAX direct-RHS matmul is now the default aligned Q8 projection path on
Metal4-capable devices. The linear projection profiler showed QKV projection
around 7.5 ms/layer and gate projection around 3.9 ms/layer with NAX, compared
with about 20 ms and 10 ms on the legacy Q8 MM path at `pp2048`. The path still
falls back automatically when the tensor API is unavailable or the shape is not
aligned.

Validation after the change:
- `make qw3-metal`
- `make qw3-bench-metal`
- `make test-metal-logits`
- `make test-metal-logits-concurrent`
- `./qw3-metal -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 16000 --nothink --prompt-file ./prompt_perf.txt -n 128`

Benchmark notes on Apple M5:
- `pp2048` with detailed projection profiling and NAX: 434.82 tok/s.
- `pp4096` default after NAX promotion: 431.19 tok/s; nearby default before
  NAX promotion was 351.11 tok/s.
- `prompt_perf.txt` default after NAX promotion: 6399 prompt tokens,
  17043.3 ms prefill, 375.46 tok/s; generated text was coherent on the previous
  garbage-regression prompt.
- `QW3_METAL_Q8_NAX_DISABLE=1` restores the legacy Q8 MM path for comparisons.

## 2026-06-07 Q6_K MoE Down TensorOps MPP

The Q6_K routed-MoE down projection now has the same Metal4 TensorOps mapped
MPP treatment used by the IQ4_XS down path. This keeps the existing Q6_K
dequantization math and mapped expert/token layout, but feeds the tile through
`matmul2d` instead of the older scalar mapped down kernel. The path is enabled
only when the Metal4 tensor API probe succeeds.

Validation after the change:
- `make qw3-bench-metal`
- `make test-metal-logits`
- `make test-metal-logits-concurrent`
- `./qw3-metal -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 16000 --nothink --prompt-file ./prompt_perf.txt -n 128`
- `env QW3_METAL_MOE_Q6_MPP_DISABLE=1 ./qw3-bench-metal -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --llama-style -p 4096 -n 0 -r 1`

Benchmark notes on Apple M5:
- `pp4096` default: 351.11 tok/s in one run.
- `pp4096` with `QW3_METAL_MOE_Q6_MPP_DISABLE=1`: 346.20 tok/s in the matched
  opt-out run.
- MoE sync profile at `pp2048` confirms Q6_K layers 34, 38, and 39 now report
  `stage=down_mpp`; their down stage was about 34-36 ms in the validation run,
  versus about 42-44 ms on the previous mapped scalar path.
- `prompt_perf.txt`: 6399 prompt tokens, 19383.1 ms prefill, 330.13 tok/s;
  generated text was coherent on the previous garbage-regression prompt.

## 2026-06-01 GQA Prefill Softmax Check

The GQA prefill attention kernels compute the online softmax max/denominator
once per grouped-query head in threadgroup memory. This removes redundant
per-dimension `exp()` work while preserving the same logits boundary.

Validation after the change:
- `make qw3-metal`
- `make test-metal-logits`
- `make test-metal-smoke`
- `./qw3-metal -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 1024 --nothink -p ciao -n 32`
- `make test-metal-smoke`
- `./qw3-metal -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 2048 --nothink -p ciao -n 32`

Benchmark notes on Apple M5:
- `docs/metal_prefill_validation.md`: 733 prompt tokens, 4220.0 ms prefill.
- `/private/tmp/qw3_prefill_3k.md`: 3413 prompt tokens, 19221.7 ms prefill
  (about 177.6 tok/s). Previous default-batch result was about 163 tok/s.

## 2026-06-01 Q6_K MoE Down Mapping

Q6_K expert-down prefill now uses the same expert-mapped tiled path as the
IQ4_XS down projection. This follows the llama.cpp direction for quantized
prefill: use `mul_mm`/`mul_mm_id`-style tiled work instead of a row-wise
reduce path for sparse expert matmuls.

Validation after the change:
- `make qw3-metal`
- `make test-metal-logits`
- `make test-metal-smoke`
- `./qw3-metal -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 2048 --nothink -p ciao -n 32`

Benchmark notes on Apple M5 with `/private/tmp/qw3_prefill_3k.md`:
- No-profile run: 3413 prompt tokens, 18892.8 ms prefill.
- Profile run: Q6_K layers 34, 38, and 39 dropped from about 287 ms sparse MoE
  to about 179 ms, matching the IQ4_XS mapped layer range.

## 2026-06-01 GQA Norm/RoPE Fusion

GQA batch prefill now fuses Q RMSNorm, Q gate copy, and RoPE into one kernel,
and fuses K RMSNorm plus RoPE into one kernel. The split path remains available
through `QW3_METAL_GQA_NORM_ROPE_SPLIT=1`. This mirrors the llama.cpp graph
direction of reducing small intermediate kernels around attention setup, while
leaving the logits boundary unchanged.

Validation after the change:
- `make qw3-metal`
- `make test-metal-logits`
- `make test-metal-smoke`
- `./qw3-metal -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 1024 --nothink -p ciao -n 32`

Benchmark notes on Apple M5 with `/private/tmp/qw3_prefill_3k.md`:
- Fused path: 3413 prompt tokens, 18797.2 ms prefill.
- Legacy split path: 3413 prompt tokens, 18900.8 ms prefill.

## 2026-06-01 IQ3_S Gate/Up Prefill Dequant

The mapped IQ3_S routed-MoE gate/up kernel now dequantizes each 16-value
sub-block from precomputed block pointers, scales, qh bits, signs, and expanded
grid entries. This follows llama.cpp's `dequantize_iq3_s` shape more closely
than the old per-element `k` decoder, while keeping the same mapped
`mul_mm_id`-style tiling.

Validation after the change:
- `make qw3-metal`
- `make test-metal-logits`
- `make test-metal-smoke`
- `./qw3-metal -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 1024 --nothink -p ciao -n 32`

Benchmark notes on Apple M5 with `/private/tmp/qw3_prefill_3k.md`:
- Before this change: 3413 prompt tokens, 18797.2 ms prefill.
- After this change: 3413 prompt tokens, 17839.4 ms prefill.
- `QW3_METAL_PROFILE_PREFILL_MOE_SYNC=1` shows mapped IQ3_S gate/up dropping
  from about 46-47 ms each per layer to about 35-36 ms each per layer.

## 2026-06-01 GQA Cached Attention Blocking

The cached GQA prefill attention path now groups up to four causal query
positions per threadgroup by default. This is a conservative step toward the
llama.cpp flash-attention direction: K/V are reused across adjacent queries and
the online softmax remains per query/head, so the logits boundary stays
unchanged. The two-query path remains available with
`QW3_METAL_GQA_ATTEND_BLOCK2=1`; the old one-query path remains available with
`QW3_METAL_GQA_ATTEND_BLOCK1=1`.

`QW3_METAL_PROFILE_PREFILL_GQA_SYNC=1` was added to split the full-attention
stage into graph-like nodes. On the 3413-token validation prompt it showed that
`attend` dominated GQA, at roughly 520-545 ms per full-attention layer before
this change.

Validation after the change:
- `make qw3-metal`
- `make test-metal-logits`

Benchmark notes on Apple M5 with `/private/tmp/qw3_prefill_3k.md`:
- New `block4` default: 3413 prompt tokens, 16352.2-16820.1 ms prefill
  across two no-profile runs; sync-profile run was 16340.0 ms.
- `QW3_METAL_GQA_ATTEND_BLOCK2=1`: 3413 prompt tokens, 17358.6 ms prefill.
- Legacy `QW3_METAL_GQA_ATTEND_BLOCK1=1`: 3413 prompt tokens, 18002.7 ms
  prefill.
- GQA profiler with `block4` shows the `attend` substage around 362-386 ms per
  full-attention layer.

## 2026-06-01 Linear-Attention Prefill Profiler

`QW3_METAL_PROFILE_PREFILL_LINEAR_SYNC=1` splits the linear-attention prefill
stage into graph-like nodes, matching the same profiling style used for GQA and
MoE. It is diagnostic-only and does not alter the default execution path.

Profile notes on Apple M5 with `/private/tmp/qw3_prefill_3k.md`:
- Typical linear layer total: about 165-170 ms.
- DeltaNet GDN dominates: about 89-93 ms per linear layer.
- Q8 qkv/gate plus F32 alpha/beta projections: about 52-54 ms per linear
  layer, with layer 0 warmup and layer 32 outliers.
- Output projection: about 17 ms. Conv1d and q/k l2norm are each around 2 ms.

## 2026-06-01 DeltaNet Batch Tiled Core

The batch DeltaNet GDN path now uses a tiled recurrent core derived from the
single-token simdgroup/float4 kernel, followed by a separate batch gated RMSNorm
node. This mirrors the llama.cpp graph-node direction better than the previous
single scalar fused kernel: the recurrent state update is vectorized per state
row, while normalization remains a separate row-reduction node.

Validation after the change:
- `make qw3-metal`
- `env QW3_METAL_BATCH_GDN_TILED=1 make test-metal-logits` while opt-in
- `make test-metal-logits` after promoting tiled GDN to default
- `make test-metal-smoke`
- `./qw3-metal -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 1024 --nothink -p ciao -n 32`

Benchmark notes on Apple M5 with `/private/tmp/qw3_prefill_3k.md`:
- Tiled path, opt-in before promotion: 3413 prompt tokens, 15535.6 ms prefill.
- Tiled path, default after promotion: 3413 prompt tokens, 15270.0 ms prefill.
- Previous default after GQA block4: 3413 prompt tokens, 16352.2-16820.1 ms
  prefill.

## 2026-06-01 IQ4_XS Down Prefill Dequant

The mapped routed-MoE IQ4_XS down kernels now dequantize 16 contiguous values
from one pre-decoded block scale and q-byte pointer instead of rebuilding the
IQ4_XS block metadata for each scalar `k`. This mirrors the earlier IQ3_S
gate/up improvement and applies to the legacy scratch, compact F32, and F16-mid
mapped down variants.

Validation after the change:
- `make qw3-metal`
- `make test-metal-logits`
- `make test-metal-smoke`

Benchmark notes on Apple M5 with `/private/tmp/qw3_prefill_3k.md`:
- Compact F32 mid default before this change: 3413 prompt tokens, 15312.8 ms
  prefill.
- After this change: 3413 prompt tokens, 14964.5-14972.1 ms prefill across two
  no-profile runs.
- `QW3_METAL_PROFILE_PREFILL_MOE_SYNC=1` shows routed-MoE down dropping from
  about 68.3 ms/layer to about 56.7 ms/layer.

## 2026-06-02 Metal4 MoE TensorOps Prefill

QW3 now probes the Metal4 Tensor API before compiling the main Metal shader
library. On M5/M6/A19/A20 devices with a successful probe it defines
`QW3_METAL_HAS_TENSOR` and enables TensorOps/MPP kernels by default for mapped
IQ3_S gate/up and compact F32 IQ4_XS down prefill. The legacy kernels remain
available through the documented opt-outs above.

Validation after the change:
- `make qw3-metal`
- `make test-metal-logits`
- `make test-metal-smoke`
- `env QW3_METAL_MOE_MPP_GATEUP=1 make test-metal-logits`
- `env QW3_METAL_MOE_MPP_DOWN=1 make test-metal-logits`
- `env QW3_METAL_MOE_MPP_GATEUP=1 QW3_METAL_MOE_MPP_DOWN=1 make test-metal-logits`

Benchmark notes on Apple M5 with `/private/tmp/qw3_prefill_3k.md`:
- Legacy current default before MPP promotion: 3413 prompt tokens, 15090.4 ms
  prefill.
- Down MPP only: 3413 prompt tokens, 14262.6 ms prefill.
- Gate/up plus down MPP opt-in: 3413 prompt tokens, 13762.6 ms prefill.
- Default after promotion: 3413 prompt tokens, 13830.8 ms prefill.
- `QW3_METAL_PROFILE_PREFILL_MOE_SYNC=1` shows mapped IQ3_S gate/up dropping
  from about 36 ms/layer to about 27 ms/layer, and compact IQ4_XS down dropping
  from about 56.7 ms/layer to about 41-43 ms/layer.

## 2026-06-02 Experimental F16 GQA KV Cache

`QW3_METAL_KV_F16=1` stores the GQA K/V cache as f16 instead of f32, without
using q8 quantization. It updates both batched prefill cache writes and
single-token decode cache writes, and the cached GQA attention kernels select
f32 or f16 reads through an explicit `kv_type` argument. The default remains
f32 because this is a memory feature, not a measured prefill speed win.
The same mode is available from `qw3-metal` and `qw3-agent` with `--kv-f16`
or with the llama-style spelling `-ctk f16 -ctv f16`. For large contexts on
24 GB unified memory, this avoids the f32 GQA KV pressure that can make decode
collapse at `--ctx 32000` and should be the first option to use for
`--ctx 64000`.

Validation after the change:
- `make qw3-metal`
- `make test-metal-logits`
- `env QW3_METAL_KV_F16=1 make test-metal-logits`

Benchmark notes on Apple M5 with `/private/tmp/qw3_prefill_3k.md`:
- Default f32 KV after Metal4 MoE MPP promotion: 3413 prompt tokens, 13830.8 ms
  prefill.
- `./qw3-agent ... --ctx 32000 --kv-f16 --nothink -p ciao -n 16` keeps the
  GQA KV estimate at 625.0 MiB and decodes the short greeting at about
  31 tok/s.
- `./qw3-metal ... --ctx 64000 --kv-f16 --nothink -p ciao -n 16` starts and
  generates the expected greeting, but the 1.25 GiB f16 GQA KV cache still
  causes decode to drop to about 4 tok/s on the 24 GB test machine. This is the
  same pressure point as f32 KV at `--ctx 32000`; making 64k fast will need a
  further memory strategy, such as safe q8 KV, paged/growable KV, or CPU layer
  offload.
- `QW3_METAL_KV_F16=1`: 3413 prompt tokens, 13928.9 ms prefill in a no-profile
  run.
- `QW3_METAL_KV_F16=1 QW3_METAL_PROFILE_PREFILL_GQA_SYNC=1` shows GQA `attend`
  still around 367-382 ms per full-attention layer, so the remaining GQA
  bottleneck is kernel shape/FlashAttention-style tiling rather than cache
  bandwidth alone.

## 2026-06-07 Routed MoE Pair TensorOps Prefill

The routed MoE prefill path now defaults to a Metal 4 TensorOps gate/up pair
kernel when available. The paired kernel writes the SwiGLU intermediate in the
compact `pid * n_ff` layout, so the existing IQ4_XS `mid_f32_mpp` down path can
remain active. If the paired TensorOps kernel is unavailable or disabled, the
host falls back to the previous separated gate/up MPP path; the old legacy pair
path remains reachable only when explicitly requested and TensorOps pair is
disabled.

Validation on Apple M5 with `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`:
- `make qw3-metal`
- `make qw3-bench-metal`
- `make test-metal-logits`
- `make test-metal-logits-concurrent`
- `./qw3-bench-metal --llama-style ... --ctx-alloc 16000 -p 4096 -n 0 -r 1`
  now reports `pp4096 = 451.49 tok/s` without environment overrides. The
  same shape was about `430 tok/s` before this pair-MPP promotion.
- `./qw3-metal ... --ctx 16000 --nothink --prompt-file ./prompt_perf.txt -n 128`
  generated coherent Italian text and reported `6399` prompt tokens at
  `394.01 tok/s`, with generation at `32.85 tok/s`.

Profiling notes:
- `QW3_METAL_PROFILE_PREFILL_MOE_SYNC=1` shows the default IQ4_XS layers using
  `stage=gate_up_pair_mpp` followed by `stage=down_mpp` with `mid=f32c_mpp`.
- The Q6_K MPP down tile now dequantizes each 16-value chunk with shared block,
  scale, and segment metadata instead of recomputing them per scalar. This
  reduced Q6 down MPP stage time in profile, but Q6 layers are few enough that
  the end-to-end `pp4096` effect is small.
- A f16 RHS MPP down path for IQ4_XS was added and passes logits under
  `QW3_METAL_MOE_MID_F16=1`, but it was not promoted because `pp4096` did not
  improve over the f32 compact path in the no-profile runs.

## 2026-06-07 Lazy Session Clears

Metal session reset now skips full GQA KV zero-fill and prefill work-buffer
zero-fill by default. The cached attention kernels only read positions below
the current logical context, and prefill/decode writes those positions before
they are visible to attention. The batched prefill pipeline likewise overwrites
its token rows before reading them. Avoiding the eager clear prevents large
private Metal buffers from polluting GPU memory residency immediately before a
long prompt run, especially with `--ctx 32000` and larger.

Debug fallback:
- `QW3_METAL_FORCE_KV_CLEAR=1` restores the old full K/V blit clear.
- `QW3_METAL_FORCE_PREFILL_CLEAR=1` restores the old prefill X/scratch clear.

Validation on Apple M5, Qwen3.6 35B A3B IQ4_XS:
- `make qw3-metal`
- `make test-metal-logits`
- `make test-metal-logits-concurrent`
- `./qw3-bench-metal --llama-style ... --ctx-alloc 16000 -p 4096 -n 0 -r 1`:
  `434.61 tok/s`
- `./qw3-bench-metal --llama-style ... --ctx-alloc 32000 -p 4096 -n 0 -r 1`:
  `423.73 tok/s`
- `./qw3-metal ... --ctx 16000 --nothink --prompt-file ./prompt_perf.txt -n 64`
  produced coherent Italian output with `prefill=6399` at `335.11 tok/s` and
  generation at `26.49 tok/s`.

## 2026-06-07 Experimental GPU Causal Flash Mask

The QW3 FlashAttention prefill path now builds its causal block map on the GPU
when `QW3_METAL_GQA_FLASH_GPU_MASK=1` is set. The default path fills a dense
`n_tokens * n_keys` half mask on the CPU, synchronizes the Metal batch, then
scans the dense mask with `kernel_flash_attn_ext_blk`. The experimental
`qw3_gqa_flash_causal_mask_block` kernel computes the same block states
directly and only writes element masks for boundary blocks. Fully unmasked and
fully masked causal blocks no longer need a dense mask write.

This mostly helps long prompt chunks where `pos0` is already large:
- `./qw3-bench-metal --llama-style ... --ctx-alloc 16000 -d 4096 -p 2303 -n 0 -r 1`
  improved from `269.18 tok/s` to `316.19 tok/s`.
- `./qw3-bench-metal --llama-style ... --ctx-alloc 16000 -p 4096 -n 0 -r 1`
  remained strong at `437.44 tok/s`.
- `env QW3_METAL_GQA_FLASH_GPU_MASK=1 ./qw3-metal ... --ctx 16000 --nothink --prompt-file ./prompt_perf.txt -n 64`
  produced coherent Italian output with `prefill=6399` at `372.36 tok/s` and
  generation at `31.38 tok/s`.

Guardrail:
- The GPU mask path remains opt-in because it changed real agent/tool behavior
  in one-shot tests. Keep default runs on the CPU dense-mask path until logits
  parity is proven for long prompts and agent system prompts.

## 2026-06-08 DeltaNet Four-Column Tiled GDN Probe

Added opt-in `QW3_METAL_BATCH_GDN_TILED4=1`, a four-column DeltaNet recurrent
core that keeps the F32 state recurrence unchanged while reusing each loaded
Q/K vector across four state columns per simdgroup. The default remains the
two-column tiled core.

Validation on Apple M5:
- `make qw3-bench-metal`
- `env QW3_METAL_BATCH_GDN_TILED4=1 make test-metal-logits`
- `env QW3_METAL_BATCH_GDN_TILED4=1 ./qw3-bench-metal -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --llama-style -p 4096 -n 0 -r 3 --ctx-alloc 16000 --no-warmup`

Observed results:
- Linear profiler with `QW3_METAL_BATCH_GDN_TILED4=1` shows `deltanet_gdn`
  mostly around 30-32 ms per linear layer, down from the default tiled2
  40-50 ms band.
- `pp4096` with tiled4: 435.70 tok/s over 3 repetitions, stdev 22.93.
- Default `pp4096` in the same session: 431.50 tok/s over 3 repetitions,
  stdev 2.49.

Conclusion: tiled4 is logits-safe and improves the isolated GDN stage, but the
end-to-end pp4096 gain is still too small/noisy to promote. Keep it as an
opt-in probe while larger prefill wins are pursued in MoE and projection/GDN
orchestration.

## 2026-06-12 IQ3_S 4x4 Dequant For MoE Gate/Up MPP

The default Metal MoE gate/up MPP path now dequantizes each IQ3_S half-block
as a 4x4 group, mirroring the shape used by llama.cpp `MUL_MAT_ID`, instead
of rebuilding every scalar element independently. The dispatch topology stays
the same: expert/token mapping, compact MoE blocks, router preweighting, and
the downstream IQ4_XS/Q6_K paths are unchanged.

Validation:
- `make qw3-bench-metal`
- `make test-metal-logits`
- No-garbage prompt: `./qw3-metal ... --ctx 16000 --nothink --prompt-file ./prompt_perf.txt -n 96`
  generated coherent English diff analysis; no repeated garbage pattern.
- Agent tool smoke: `./qw3-agent ... --ctx 1600 --nothink -p 'Crea il file /tmp/qw3_agent_tool_smoke.txt ...'`
  called `[tool] write` and wrote `tool-ok`.

Observed results:
- `QW3_METAL_PROFILE_PREFILL_MOE_SYNC=1` at `pp512`: `gate_up_pair_mpp`
  dropped to 255.022 ms total over 40 layers, 6.376 ms/layer. The previous
  same-path measurement was about 12.8 ms/layer.
- `down_mpp` stayed essentially unchanged at 10.575 ms/layer, as expected.
- `pp4096`, `ctx=16000`, 3 repetitions, no warmup: 483.20 tok/s, stdev
  18.94, 8485.73 ms average.
- `prompt_perf.txt` prefill: 6399 tokens in 15261.1 ms, 419.30 tok/s.

Conclusion: keep this patch. The next MoE target is the down projection, then
larger llama.cpp-style orchestration around `MUL_MAT_ID`-like blocks.

## 2026-06-12 IQ4_XS 4x4 Dequant For MoE Down MPP

The default IQ4_XS down-projection MPP path now dequantizes each selected
half-block as a 4x4 group before loading it into the tensor matmul tile. This
matches the successful gate/up cleanup shape and keeps the existing compact
expert/token map and f32 mid buffer.

Validation:
- `make qw3-bench-metal`
- `make test-metal-logits`
- No-garbage prompt: `./qw3-metal ... --ctx 16000 --nothink --prompt-file ./prompt_perf.txt -n 96`
  generated coherent English diff analysis.
- Agent tool smoke: `./qw3-agent ... --ctx 1600 --nothink -p 'Crea il file /tmp/qw3_agent_tool_smoke.txt ...'`
  called `[tool] write` and wrote `tool-ok`.

Observed results:
- `QW3_METAL_PROFILE_PREFILL_MOE_SYNC=1` at `pp512`: `down_mpp`
  improved to 391.633 ms total over 40 layers, 9.791 ms/layer. The previous
  post-IQ3-gate/up value was about 10.575 ms/layer.
- `gate_up_pair_mpp` stayed stable at 6.314 ms/layer.
- `pp4096`, `ctx=16000`, 3 repetitions, no warmup: 515.13 tok/s, stdev
  8.62, 7952.93 ms average.
- `prompt_perf.txt` prefill: 6399 tokens in 14660.1 ms, 436.49 tok/s.

Conclusion: keep this patch. It is a smaller win than the IQ3_S gate/up 4x4
dequant, but it moves the current MoE bottleneck in the right direction.

## 2026-06-12 Constant IQ4_NL Lookup Table

QW3 previously implemented IQ4_NL value decoding as a `switch` in
`qw3_iq4nl_val()`. llama.cpp keeps the same 16 values in a constant lookup
table (`kvalues_iq4nl_f`). Moving QW3 to the same table-shaped lookup removes
heavy branchy scalar decoding from IQ4_XS down projection and all other IQ4_NL
call sites.

Validation:
- `make qw3-bench-metal`
- `make test-metal-logits`
- No-garbage prompt: `./qw3-metal ... --ctx 16000 --nothink --prompt-file ./prompt_perf.txt -n 96`
  generated coherent English diff analysis.
- Agent tool smoke: `./qw3-agent ... --ctx 1600 --nothink -p 'Crea il file /tmp/qw3_agent_tool_smoke.txt ...'`
  called `[tool] write` and wrote `tool-ok`.

Observed results:
- `QW3_METAL_PROFILE_PREFILL_MOE_SYNC=1` at `pp512`: `down_mpp`
  improved to 155.567 ms total over 40 layers, 3.889 ms/layer.
- `gate_up_pair_mpp` stayed stable at 6.357 ms/layer.
- `pp512` profile run rose to 414.70 tok/s despite stage synchronization.
- `pp4096`, `ctx=16000`, 3 repetitions, no warmup: 574.49 tok/s, stdev
  18.43, 7134.73 ms average.
- `prompt_perf.txt` prefill: 6399 tokens in 12715.9 ms, 503.23 tok/s.

Conclusion: keep this patch. It directly confirms a block-level llama.cpp
advantage in IQ4_XS down projection: constant table lookup beats QW3's old
switch-based scalar value decode by a wide margin.

## 2026-06-12 Unrolled IQ3_S 4x4 Dequant Assignments

The IQ3_S 4x4 helper used by MoE gate/up now writes the 16 decoded values with
explicit assignments instead of a small dynamic loop. This follows the same
general shape as llama.cpp's quantized matmul helpers: fixed-lane constants
are exposed directly to the Metal compiler.

Validation:
- `make qw3-bench-metal`
- `make test-metal-logits`
- No-garbage prompt: `./qw3-metal ... --ctx 16000 --nothink --prompt-file ./prompt_perf.txt -n 96`
  generated coherent English diff analysis.
- Agent tool smoke: `./qw3-agent ... --ctx 1600 --nothink -p 'Crea il file /tmp/qw3_agent_tool_smoke.txt ...'`
  called `[tool] write` and wrote `tool-ok`.

Observed results:
- `QW3_METAL_PROFILE_PREFILL_MOE_SYNC=1` at `pp512`: `gate_up_pair_mpp`
  stayed slightly improved/stable at 252.453 ms total over 40 layers,
  6.311 ms/layer.
- `down_mpp` stayed stable at 156.278 ms total, 3.907 ms/layer.
- `pp4096`, `ctx=16000`, 3 repetitions, no warmup: 584.33 tok/s, stdev
  21.41, 7016.17 ms average.
- `prompt_perf.txt` prefill remained coherent at 500.29 tok/s.

Conclusion: keep this patch as a small compiler-friendly cleanup. The next
larger target is no longer scalar IQ4/IQ3 decode, but broader MoE/projection
orchestration and the remaining linear-layer GDN/projection cost.

## 2026-06-13 GDN Tiled2 Cleanup And Rejected NAX Branch Removal

The default DeltaNet GDN tiled2 kernel was tightened without changing the
algorithm:
- Hoist per-head constants `a[hv]`, `dt_bias[hv]`, and
  `rsqrt(head_dim)` out of the token loop.
- Remove the `j1 < head_dim` tail branch from tiled2 only when the host
  selects tiled2 with `head_dim % 8 == 0`; otherwise the generic tiled path
  remains available.
- Keep the earlier pair-MPP MoE tile-load cleanup that removes redundant
  hot-path guards for the QW3 fixed dimensions.

Validation:
- `make qw3-metal && make qw3-agent && make test-metal-logits`: passed.
- `pp4096`, `ctx=4097`, 3 repetitions, no warmup:
  `603.61 tok/s`, stdev `7.67`, average `6786.56 ms`.
- `prompt_perf.txt`, `ctx=16000`, `n=128`: coherent output,
  `prefill=6399` at `530.15 tok/s`, generation at `31.93 tok/s`.
- Agent interactive tool smoke:
  `bash date` was called through `[tool] bash` and returned the correct
  2026-06-13 CEST timestamp.

Rejected in the same session:
- Removing the tail guards from `qw3_matmul_q8_0_nax_direct_rhs` looked
  safe for the active QW3 dimensions, but the matched `pp4096` result was
  `599.45 tok/s`, below the nearby `602-604 tok/s` runs. The experiment was
  reverted and should not be retried unless a deeper NAX rewrite changes the
  surrounding kernel shape.

## 2026-06-13 MoE Down MPP Branch Cleanup

The IQ4_XS and Q6_K mapped TensorOps expert-down kernels now drop redundant
hot-path tail guards when loading their `A`/`B` tiles. This is valid for the
active QW3 prefill path because the host already requires `n_ff % 256 == 0`
and each compact block has `r1u < count` with `lr1` clamped inside the block.

Validation:
- `make qw3-metal && make test-metal-logits`: passed.
- MoE profile at `pp4096`: Q6_K `down_mpp` improved from about
  `51.7 ms/layer` to about `37.4 ms/layer` on the three Q6_K layers.
- `pp4096`, 3 repetitions, no warmup: `608.92 tok/s`, stdev `3.27`,
  average `6726.81 ms`.
- `prompt_perf.txt`, `ctx=16000`, `n=128`: coherent output,
  `prefill=6399` at `539.75 tok/s`, generation at `32.04 tok/s`.
- Agent interactive tool smoke still called `[tool] bash` successfully.

## 2026-06-05 Llama-Style Bench Guardrail

`qw3-bench` now has `--llama-style`, a synthetic benchmark mode shaped like
`llama-bench`: `-p/--n-prompt` and `-n/--n-gen` are measured as separate `pp`
and `tg` rows when both are non-zero, and `-d/--depth` prefills a context
outside the timed token-generation loop. This makes the QW3 numbers easier to
compare with llama.cpp without mixing prompt processing and decode in one row.

Useful commands:
- `./qw3-bench-metal -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --llama-style -p 4096 -n 128 -r 3`
- `./qw3-bench-metal -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --llama-style -p 0 -n 128 -d 4096 -r 2`
- `env QW3_METAL_GQA_FLASH_ATTN=1 ./qw3-bench-metal -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --llama-style -p 4096 -n 0 -r 3`

Apple M5 notes with the Qwen3.6 35B A3B IQ4_XS model:
- Default `pp4096`: 212.21 tok/s over 3 repetitions.
- Default `tg128` at depth 0: 40.46 tok/s over 3 repetitions.
- Default `tg128` at depth 4096: 26.57 tok/s over 2 repetitions.
- `QW3_METAL_GQA_FLASH_ATTN=1` `pp4096`: 240.02 tok/s over 3 repetitions.

Profiler notes from one `pp4096` no-warmup run:
- `QW3_METAL_PROFILE_PREFILL_GQA_SYNC=1` shows full-attention `attend` at
  about 760-820 ms per full-attention layer.
- `QW3_METAL_PROFILE_PREFILL_LINEAR_SYNC=1` shows linear-layer DeltaNet GDN at
  about 73-90 ms/layer and qkv/gate/alpha/beta projection at about
  63-72 ms/layer after warmup.
- `QW3_METAL_PROFILE_PREFILL_MOE_SYNC=1` shows mapped MoE gate and up around
  31-33 ms each, IQ4_XS down MPP around 48-52 ms, and Q6_K down around
  75 ms.

Optimization priority after this profile: first reduce the full-attention
`attend` cost with a more llama.cpp-like tiled/FlashAttention path, then revisit
linear DeltaNet/projection batching and MoE down.

## 2026-06-21 Partial Offload Diagnostics

`qw3-bench` now accepts `--ngl N`, matching the client option, and
`make test-prefill-bench` runs a conservative `pp4096` throughput guard. The
guard is intentionally not part of the default regression target because it is
performance-sensitive, but it should be run during prefill/Metal optimization.

For llama.cpp-style split offload, `--ngl 35` keeps the last 35 layers on Metal
and evaluates the first 6 layers on CPU. A small decode profile:

```sh
QW3_METAL_PROFILE_CPU_SPAN=1 QW3_METAL_PROFILE=1 \
./qw3-bench -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
  --llama-style --ngl 35 -p 0 -n 4 -d 16 -r 1 --no-warmup
```

showed the CPU prefix at about `33-36 ms/token`, with total decode around
`55 ms/token` after the short setup phase. Stage profiling with
`QW3_METAL_PROFILE_CPU_STAGE=1` showed routed MoE as the main CPU-prefix cost:
roughly `3 ms/layer` for `moe_routed`, while linear projection is about
`1.2-1.7 ms/layer` and the short-context CPU GQA attend part is negligible.

The first small optimization fused CPU IQ3_S expert `gate`+`up` evaluation for
the routed MoE path into one parallel row pass. Validation:

- `make test-regression`: passed.
- `make test-prefill-bench`: passed, `pp4096` above `600 tok/s`.
- `./qw3-cli ... --ctx 4096 --ngl 35 --nothink -p ciao -n 32`: coherent
  greeting, no garbage.
- `./qw3-bench ... --llama-style --ngl 35 -p 0 -n 8 -d 16 -r 3 --no-warmup`:
  `tg` improved to about `19.7 tok/s` in the short-depth diagnostic case.

Remaining concern: if a CPU-prefix full-attention layer remains on CPU at very
large contexts, its scalar attention path will scale with context length. That
needs a separate long-context measurement and likely a dedicated policy or
optimized CPU attention path.

An opt-in policy was added for this:
`QW3_METAL_SPLIT_AVOID_CPU_FULL_ATTN=1`. When enabled with the llama-style
split, the CPU prefix is shortened so that the first full-attention layer is
kept on Metal. For `--ngl 35` this means layers `0..2` run on CPU and
layers `3..39` run on Metal. This is not the default because it changes the
effective number of Metal layers and increases Metal KV/state residency, but it
is the path to test for long-context usability on machines with enough memory.

## 2026-06-02 Partial Metal Layer Offload

`QW3_METAL_NGL=N`, exposed as `--ngl N` on `qw3-metal` and `qw3-agent`, keeps
only `N` transformer layers active on Metal and evaluates the
remaining layers on the CPU reference path. With the current default
llama.cpp-style split, those are the final `N` layers, so lower-numbered
prefix layers run on CPU. This mirrors the operational role
of llama.cpp `--ngl`: it reduces Metal KV/state residency for very large
contexts, especially `--ctx 64000`, where f16 KV alone is still too much
pressure on the 24 GB test machine.

Current scope:
- Valid range is `0..40`; default is `40` and preserves the full Metal path.
- Partial offload disables the batched Metal prefill path for now, so it is a
  correctness and residency feature first, not a prefill-speed feature.
- The model tensor mmap is still global; the immediate win is lower active
  Metal cache/state allocation and fewer layer weights touched by Metal.
- The Metal prefix command buffer must be synchronized before reading the
  boundary activation into the CPU tail; otherwise the CPU sees stale zeroed
  `x0` and logits collapse to token `0`.

Suggested 64k smoke shape:
- `./qw3-metal -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 64000 --kv-f16 --ngl 35 --nothink -p ciao -n 16`

Validation after the first partial-offload implementation:
- `make qw3-metal`
- `make qw3-agent`
- `make test-metal-logits`
- `./qw3-metal ... --ctx 1024 --ngl 35 --metal-session-decode-test -p ciao`
  passes with matching top logits.
- `./qw3-metal ... --ctx 1024 --ngl 35 --nothink -p ciao -n 16` generates
  the expected greeting, but at about 5.5 tok/s because layers 35..39 run on
  CPU.
- `./qw3-metal ... --ctx 64000 --kv-f16 --ngl 35 --nothink -p ciao -n 4`
  starts successfully with a Metal memory estimate of 1057.6 MiB
  (`gqa_kv=1000.0 MiB`, `deltanet=54.0 MiB`) and generates the expected prefix.

## 2026-07-10 Prefill Headroom Profile

The warmed `pp4096` baseline on the M5 test machine was `632.19 tok/s` over
three repetitions (standard deviation `3.89`). This leaves roughly an 8%
gap from the locally observed llama.cpp result near `687 tok/s`.

Synchronized stage profiles attribute the main costs as follows. The numbers
are diagnostic totals and include profiler barriers, so they describe the
relative optimization targets rather than normal end-to-end timing:

- Linear-attention layers: about `2.86 s`, including `1.40 s` in DeltaNet GDN
  and `0.89 s` in QKV/gate/alpha-beta projections.
- Routed MoE: about `1.78 s`, including `0.95 s` in IQ3_S gate/up and `0.64 s`
  in expert down projection.
- Full-attention layers: about `0.89 s`, including `0.58 s` in FlashAttention.

A batch version of the decode-time DeltaNet gate precomputation was tested and
fully reverted. It was logits-safe, but reduced warmed `pp4096` from `638.54`
to `623.76 tok/s`; the profiled GDN total increased from roughly `1.40 s` to
`1.55 s`. Do not retry a separate alpha/beta precompute dispatch unless the
recurrent pipeline is reorganized enough to absorb it without another node or
barrier.

A separate tiled2 variant that explicitly retained its two DeltaNet state
columns in registers across the token loop was also logits-safe but neutral to
negative: `633.73 tok/s` versus the nearby `638.54 tok/s` default, with the
profiled GDN total at `1.48 s`. The Metal compiler likely already promotes the
loop-carried state, or Q/K/V traffic and recurrence latency dominate. This
variant was fully reverted and should not be retried as a source-level
load/store hoist.

The remaining credible margin is therefore in a deeper GDN decomposition or
in the routed IQ3_S/IQ4_XS matmuls and their intermediate representation.
Micro-flags and additional standalone dispatches are unlikely to close the
gap.

Two follow-up MoE probes were also rejected and fully reverted:

- Writing the fused pair-MPP SwiGLU intermediate directly as F16 for the
  IQ4_XS layers produced a non-finite final layer value in the 64-token batch
  logits test (`layer_rmsdiff=nan`). It was rejected before benchmarking. The
  compact F32 intermediate was retained. This rejected that implementation;
  it does not establish that every F16 intermediate implementation is unsafe.
- A `float4` version of the eight-slot expert reduction was logits-identical,
  but its profiled reduction remained `2.87 ms/layer` versus about
  `2.84 ms/layer` for the scalar source. The compiler or memory system already
  extracts the available vectorization.

The batch logits regression now rejects every non-finite max/RMS metric. This
closes a test hole where IEEE NaN comparisons could previously leave the test
status as `ok`.

Two deeper MPP occupancy/tile probes were evaluated and fully reverted:

- Splitting the fused IQ3_S gate/up MPP kernel into two phases shortened the
  lifetime of each cooperative-tensor accumulator, but lost the shared X load.
  Gate/up increased from about `24.78` to `25.43 ms/layer` on the IQ4_XS
  layers.
- An IQ4_XS down MPP tile with 128 output rows used eight simdgroups and a
  matching 128-row indirect dispatch. It did not preserve the expected output
  layout: the 64-token batch ended at `layer_maxdiff=2.21` and
  `layer_rmsdiff=0.126`. The validated 64x32/four-simdgroup tile was retained;
  this result alone does not establish a hardware limitation on wider tiles.

The batch regression now also enforces final-layer tolerances of `0.05`
maximum difference and `0.005` RMS difference. Previously a finite but badly
divergent vector could still be printed as `ok`.

## 2026-09-05 MoE Reduction-Dimension Experiments

Tested larger K tiles in the existing MPP kernels, preserving the 64 output
rows, 32 routed token slots, four simdgroups, F32 intermediate, and F16 KV
cache. This differs from the earlier 128-output-row experiment. The local
llama.cpp routed MPP kernel also uses the original 64x32x32 shape.

All model processes ran sequentially on the 24 GB M5. Measurements used:

```sh
./qw3-bench -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
  --llama-style -p 4096 -n 0 -r 3
```

The benchmark's untimed warmup remained enabled.

| Configuration | Repetitions | Mean tok/s | Standard deviation |
| --- | ---: | ---: | ---: |
| Original, beginning of session | 3 | 636.92 | 4.29 |
| IQ4_XS down, K=64 | 3 | 632.87 | 5.81 |
| IQ4_XS down, K=128 | 3 | 612.93 | 7.18 |
| IQ3_S fused gate/up, K=64 | 3 | 576.25 | 83.24 |
| IQ3_S fused gate/up, K=64, repeated | 5 | 611.23 | 13.35 |
| Original, restored at end | 3 | 608.41 | 7.45 |

The unchanged baseline drifted by about 4.5% during the session. These runs
therefore do not isolate a small speedup or slowdown from changing machine
conditions. None demonstrated a reliable improvement, and all experimental
kernel and host changes were removed. Do not promote a larger K tile on the
basis of these measurements.

Implementation and correctness findings:

- For non-square B tiles, the tensor extents must match the K-contiguous
  token rows: `(NK, NR1)`. Keeping `(NR1, NK)` from the square original caused
  the first K=64 down attempt to fail with `layer_maxdiff=0.05067945` and
  `layer_rmsdiff=0.005437171`. Correcting the extents restored the baseline
  results. A and B also need disjoint shared-memory ranges: the down trials
  used 12 KiB at K=64 and 24 KiB at K=128.
- All three corrected variants passed the 64-token single-layer comparison
  with `layer_maxdiff=0.0009708405` and `layer_rmsdiff=1.66781e-05`.
- Down K=64 also matched the original JSON dump byte-for-byte for the top 64
  logits at each of eight greedy generation steps after the 6399-token
  `prompt_perf.txt` input. This covers the full model but only the saved top
  logits, not every vocabulary entry. The other discarded variants were not
  subjected to this long-prompt logits comparison.

On the restored original, the 6399-token prompt generated a coherent
128-token explanation of partial GPU offloading, with no observed garbage.
That real-text run measured 491.61 tok/s prefill and 29.99 tok/s generation;
its prompt length and content differ from synthetic pp4096.

A live model-driven agent check also passed: asked to execute
`printf 'qw3-prefill-tool-ok'`, the agent selected the native `bash` tool,
executed it with exit status 0, and correctly reported the returned marker.
This check used a separate temporary conversation store and did not inject
a prewritten tool call.

## 2026-09-05 Decode Readback Batching And Measurement Controls

The CPU-logits decode path now keeps the last layer, final RMSNorm, output
projection, and logits readback in the same command batch. Previously it
waited after the layer graph and again after the final norm. The Q6_K and
Q8_0 output helpers now wait for caller-owned batches when a CPU output is
requested; they must not return before that output is ready. No kernels,
sampling parameters, cache formats, or agent/tool behavior were changed.
The explicitly synchronized profiling path remains available.

Pure-generation benchmark warmup now evaluates one token at position zero,
matching the local llama-bench implementation. It no longer processes the
entire depth prefix once just for warmup. Each measured repetition still
prepares its depth outside the timer. Unlike llama-bench, qw3 currently
recomputes this prefix instead of restoring a saved state between repetitions.
Use one repetition per invocation for long-depth cross-engine comparisons to
avoid that additional difference in thermal history.

Some earlier measurements in this session were invalidated by repeated macOS
Sleep/DarkWake transitions, confirmed in `pmset -g log`. The unchanged binary
fell to 15-19 tok/s and returned to about 42 tok/s with sleep prevention.
Those samples must not be used to judge an optimization or be attributed
solely to thermal throttling. For unattended measurements on this machine use:

```sh
caffeinate -disu ./qw3-bench \
  -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
  --llama-style -p 0 -n 128 -d 16000 -r 1 -t 4

caffeinate -disu llama-bench \
  -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
  -p 0 -n 128 -d 0,4096,16000 -r 1 -t 4 -ngl 99 -fa 1 \
  -ctk f16 -ctv f16 -mmp 0 -o json
```

Only one model process ran at a time. The sleep assertion ends with the
command. The corrected runs used F16 KV, full Metal offload and normal warmup.
Thermal-state spot checks were nominal; these are short benchmarks, not proof
of sustained throughput on a hot fanless laptop.

| Effective depth | qw3 tok/s | llama-bench tok/s |
| --- | ---: | ---: |
| 0 | 42.44 (3 repetitions, SD 0.33) | 35.85 (1 repetition) |
| 4096 | 35.83 (1 repetition) | 34.60 (1 repetition) |
| 16000 | 23.04 (1 repetition) | 30.58 (1 repetition) |

The system llama-bench is build 8765, commit `9e209c5ae`; the inspected local
source is `6b80c74f2`. Both benchmarks evaluate synthetic tokens without
sampling, but their token streams differ. This is an engine comparison, not
an identical-conversation or agent-throughput comparison.

A controlled short-context A/B/A series measured the original at 41.97 and
41.84 tok/s and the first readback-batched implementation at 42.50 tok/s
(three repetitions each). Removing its redundant empty synchronization then
measured 42.44 tok/s. A later original/final pair measured 40.82 (SD 0.77)
and 41.07 (SD 0.24) tok/s. The observed benefit is small and variable,
approximately 0.6-1.6% across these comparisons, with noise comparable to
the smallest difference. It is not a guaranteed throughput gain and does
not close the long-context attention gap.

Three attention prototypes were removed: per-SIMD query-head ownership,
shared split-reduction softmax weights, and a context-balanced split count.
They passed the tested isolated F16 attention comparisons but did not
establish a reliable end-to-end improvement. The shared-weight reduction
also preserved eight selected generation tokens after the long prompt, with
maximum saved-logit difference 0.0001078. Some of their timings preceded the
sleep investigation, so they are inconclusive, not evidence of a hardware
limit. Production attention kernels and split selection remain unchanged.

Validation of the final readback-batching change:

- `caffeinate -disu make test-metal-logits`: passed. CPU/Metal final logits,
  short-session logits, four greedy steps, an agent-like prompt, and the
  64-token batch comparison all passed. The latter retained
  `layer_maxdiff=0.0009708405` and `layer_rmsdiff=1.66781e-05`.
- With `--ctx 16000 --nothink --prompt-file prompt_perf.txt --temp 0 -n 8
  --dump-logprobs ... --logprobs-top-k 64`, the JSON matched the pre-change
  reference byte-for-byte. This compares the saved top 64 logits for eight
  steps, not the entire vocabulary at long context.
- The real 6399-token prompt generated a coherent 128-token explanation of
  partial GPU offloading, without observed garbage: 539.20 tok/s prefill and
  32.68 tok/s generation. This is not the synthetic benchmark above.
- A live agent run, using a temporary store, selected `bash`, executed
  `printf 'qw3-decode-tool-ok'` with exit status 0 and correctly reported its
  output. No tool call was injected.
- A profiled `-p 0 -n 1 -d 64 -r 1` benchmark logged the warmup at position
  1 and the measured token at position 65, verifying that the timed decode
  still follows the full depth prefix.
- `caffeinate -disu make test-prefill-bench` passed at 498.20 tok/s without
  warmup (test threshold 450). A separate warmed pp4096 check measured
  583.63 tok/s (SD 27.43), followed by 565.52 (SD 29.18) on the original
  executable, three repetitions each. Both are below the earlier session's
  prefill figures and variable; they do not establish a prefill speedup or
  a code-induced slowdown. The prefill kernels were not changed.

The main remaining decode target is long-context full attention, based on
the growing depth-dependent gap. Synchronized per-stage profiling points
in the same direction but adds substantial overhead; its timings should not
be treated as normal execution costs. A future attention change needs both
controlled end-to-end timing and the long-prompt/tool regressions above.

## 2026-09-06 Four-Row Decode Attention Experiments

Compared the current grouped-head split kernel with llama.cpp's vector
FlashAttention implementation (`kernel_flash_attn_ext_vec` in the local
`ggml-metal.metal`). The latter processes blocks of cache rows and maintains
a blockwise online softmax. Tested that scheduling idea in qw3 while keeping
KV-head reuse, the split count, F16 cache, and the existing output reduction.
This was not another split-count flag or the previous one-head-per-SIMD
prototype.

Three implementations were evaluated and then completely removed:

1. Four cache rows per iteration, with separate dot-product scratch and
   score storage. This reduced three threadgroup barriers per row to three
   per block, preserving the per-row softmax update order.
2. The same layout with blockwise softmax, rescaling the previous accumulator
   once per block instead of once per row.
3. Fixed-size, explicitly unrolled loops for the blockwise version, to expose
   constant indexing to the compiler as in llama's vector implementation.

The partial kernel needed 288 shared floats instead of 64. No extra global
buffers, cache quantization, sampling changes, or prefill changes were used.

| Variant | Isolated attend+out at 4097 (ms) | Full tg128 at depth 16000 (tok/s) |
| --- | ---: | ---: |
| Original, initial | not measured | 22.89 |
| Four-row barriers, per-row softmax | 1.0227 | 22.51 |
| Four-row blockwise softmax | 1.2879 | not measured |
| Blockwise softmax, unrolled | 1.0937 | 20.93 |
| Original, restored | 1.0121 | 22.81 |

Full generation used `--llama-style -p 0 -n 128 -d 16000 -r 1 -t 4`, normal
warmup, and `caffeinate -disu`. Model processes ran sequentially. These are
single full-run samples, not statistical confidence intervals, but no variant
demonstrated a useful gain and the restored baseline returned near its initial
speed. The isolated test averages 64 attend+out calls on a cache constructed
from a repeated input token; it is not a full-model throughput benchmark.

All three variants passed the isolated CPU comparison at 4097 tokens, with
maximum error about 0.00016946 and RMS about 0.00004361. The first and third
also preserved all eight selected tokens and the top-64 token ordering after
the 6399-token prompt. Saved-logit differences versus the original were:

- Per-row softmax: maximum 0.0001040, RMS 0.00001999.
- Unrolled blockwise softmax: maximum 0.0001297, RMS 0.00002958.

The long-prompt comparisons checked finite saved logits with a separate JSON
parser. They cover top-64 entries, not every vocabulary entry. The intermediate
blockwise implementation was not long-prompt tested or promoted.

Added `make test-metal-gqa-decode`, also included in `test-regression-full`.
It forces F16 KV and checks 1023, 1024, 4097, and 16385 tokens sequentially,
covering unsplit attention, the split threshold, uneven chunks and the
256-split path. It checks the CPU reference and rejects missing/non-finite
or excessive max/RMS metrics outside the engine's `-ffast-math` build.
The parser was also tested with synthetic NaN, infinity, excessive error and
an empty context list; all were rejected, while valid metrics passed.

On the restored kernel, the new boundary suite and `make test-metal-logits`
passed. The boundary comparisons stayed below 0.000170 maximum and 0.000044
RMS difference. Do not repeat these block-four variants without a materially
different work distribution: amortizing barriers alone did not improve this
grouped-head implementation.

The restored build also generated a coherent 128-token explanation from
`prompt_perf.txt` without observed garbage (525.93 tok/s prefill, 32.07 tok/s
generation). A live agent run selected `bash`, executed
`printf 'qw3-sept6-tool-ok'` with exit status 0 and reported the marker correctly.
No tool call was injected; a temporary conversation store was used.

## 2026-09-06 Vector FlashAttention Decode

The long-context F16 GQA decode path now uses the vector FlashAttention kernel
from the local llama.cpp-derived Metal source. This is a structural change,
not another tuning flag for the old split kernel. It uses 32 cache rows per
workgroup, blockwise online softmax, 1/2/4 SIMD groups selected from context
depth, and the corresponding vector reduction kernel. A small native kernel
pads the final partial cache block and a second one applies the Qwen attention
gate. The six specialized pipeline variants are cached so crossing a 32-token
boundary does not trigger repeated Metal compilation.

The path is enabled for F16 KV, head dimension 256, and contexts of at least
1024 tokens. Unsupported configurations retain the previous implementation.
Setting `QW3_METAL_GQA_FLASH_DECODE=0` provides an explicit diagnostic fallback.
Q8 KV behavior was not changed.

Isolated `make test-metal-gqa-decode` results after the final pipeline-cache
change:

| Context | Attend+out (ms) | Max difference | RMS difference |
| ---: | ---: | ---: | ---: |
| 1023, old unsplit path | 1.9725 | 0.000169933 | 0.0000436123 |
| 1024 | 0.5852 | 0.000169337 | 0.0000436142 |
| 4097 | 0.6627 | 0.000169337 | 0.0000436080 |
| 16385 | 1.4305 | 0.000169754 | 0.0000436227 |

Before this change the restored split kernel measured 1.0121 ms at 4097 and
about 2.43 ms at 16385 in the same regression harness. The vector path is
therefore about 35% faster at 4097 and 41% faster at 16385 for the isolated
full-attention operation.

Controlled full-model samples using `--llama-style -p 0 -n 128`:

| Depth | Previous qw3 (tok/s) | Vector path (tok/s) |
| ---: | ---: | ---: |
| 1024 | 40.92 | 41.97 |
| 4096 | 35.83 | 40.93 |
| 16000 | 22.89 | 32.97 and 34.82 |

The local system llama benchmark previously measured 30.58 tok/s at depth
16000. This comparison indicates that the former long-context attention gap
has been removed on the tested M5, but the single-run values are not confidence
intervals. A later three-repetition qw3 run measured 28.32 tok/s with a very
large 8.48 tok/s standard deviation after repeatedly rebuilding a 16k prefix;
that thermally stressed run is retained here rather than discarded. Disabling
per-layer command-buffer flushes was also tested separately and rejected:
22.60 tok/s versus a 22.89 tok/s baseline.

Final validation:

- `make test-metal-gqa-decode test-metal-logits` passed. The boundary test
  covers 1023, 1024, 4097, and 16385 tokens sequentially.
- At the 6399-token real prompt, all eight selected tokens matched the saved
  pre-change top-64 reference. Per-step maximum saved-logit difference stayed
  at or below 0.0016632. This is a top-64 comparison, not full-vocabulary.
- Two final 128-token runs of `prompt_perf.txt` were coherent, without observed
  garbage. The last measured 446.23 tok/s prefill and 33.84 tok/s generation.
- A final live agent run selected and executed `bash`, returned exit status 0,
  and reported the expected marker. No tool call was injected.
- `make test-prefill-bench` passed at 634.66 tok/s; the final full regression
  invocation measured 589.85 tok/s. Both clear the 450 tok/s guard and confirm
  that this decode change did not reduce the pp4096 path.
