# Metal Prefill Validation

This project treats logits as the regression boundary for Metal prefill changes.
Performance-only checks are not enough: every prefill optimization must preserve
CPU/Metal final logits and greedy top-1 choices before it can be enabled by
default.

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
- GQA batch prefill fuses RMSNorm, Q gate copy, and RoPE by default. Set
  `QW3_METAL_GQA_NORM_ROPE_SPLIT=1` for the legacy split-kernel comparison.
- GQA cached prefill attention uses the llama/ds4 FlashAttention kernel by
  default when the shared Metal source is available. Set
  `QW3_METAL_GQA_FLASH_ATTN=0` or `QW3_METAL_GQA_FLASH_ATTN_DISABLE=1` to use
  the native `block4` fallback. In fallback mode, set
  `QW3_METAL_GQA_ATTEND_BLOCK2=1` for the two-query comparison path, or
  `QW3_METAL_GQA_ATTEND_BLOCK1=1` for the legacy one-query kernel.
- Metal session reset does not zero the GQA KV buffers by default. The valid
  KV range is controlled by the session position and every prefill/decode step
  writes the entries it can later read. Set `QW3_METAL_FORCE_KV_CLEAR=1` only
  when bisecting memory bugs against the legacy full-clear behavior.
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

## 2026-06-07 Lazy KV Clear

Metal session reset now skips full GQA KV zero-fill by default. The cached
attention kernels only read positions below the current logical context, and
prefill/decode writes those positions before they are visible to attention.
Avoiding the eager clear prevents the large f16 KV allocation from polluting
GPU memory residency immediately before a long prompt run, especially with
`--ctx 32000` and larger.

Debug fallback:
- `QW3_METAL_FORCE_KV_CLEAR=1` restores the old full K/V blit clear.

Validation on Apple M5, Qwen3.6 35B A3B IQ4_XS:
- `make qw3-metal`
- `make test-metal-logits`
- `make test-metal-logits-concurrent`
- `./qw3-bench-metal --llama-style ... --ctx-alloc 16000 -p 4096 -n 0 -r 1`:
  `415.92 tok/s`
- `./qw3-bench-metal --llama-style ... --ctx-alloc 32000 -p 4096 -n 0 -r 1`:
  `380.41 tok/s`
- `./qw3-metal ... --ctx 16000 --nothink --prompt-file ./prompt_perf.txt -n 128`
  produced coherent Italian output with `prefill=6399` at `306.46 tok/s` and
  generation at `23.98 tok/s`.

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

## 2026-06-02 Partial Metal Layer Offload

`QW3_METAL_NGL=N`, exposed as `--ngl N` on `qw3-metal` and `qw3-agent`, keeps
only the first `N` transformer layers active on Metal and evaluates the
remaining layers on the CPU reference path. This mirrors the operational role
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
