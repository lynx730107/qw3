# Metal Prefill Validation

This project treats logits as the regression boundary for Metal prefill changes.
Performance-only checks are not enough: every prefill optimization must preserve
CPU/Metal final logits and greedy top-1 choices before it can be enabled by
default.

Default safety policy:
- `QW3_METAL_KV_Q8_0` is not part of the default validation path.
- `QW3_METAL_PREFILL_CONCURRENT=1` enables the llama.cpp-style concurrent
  Metal encoder for prefill frontiers. It is opt-in until it shows a real speed
  win on long prompts.
- `QW3_METAL_PREFILL_BATCH` defaults to 4096, the current Metal batch cap.
- Expert-major MoE gate/up is enabled inside batch prefill with at least 32
  tokens; set `QW3_METAL_MOE_MAP_GATEUP_DISABLE=1` for legacy comparisons.
- Expert-major MoE down is enabled inside IQ4_XS batch prefill with at least 32
  tokens; set `QW3_METAL_MOE_MAP_DOWN_DISABLE=1` for legacy comparisons.
- GQA batch prefill fuses RMSNorm, Q gate copy, and RoPE by default. Set
  `QW3_METAL_GQA_NORM_ROPE_SPLIT=1` for the legacy split-kernel comparison.
- GQA cached prefill attention uses the `block4` query-grouped kernel by
  default. Set `QW3_METAL_GQA_ATTEND_BLOCK2=1` for the two-query comparison
  path, or `QW3_METAL_GQA_ATTEND_BLOCK1=1` for the legacy one-query kernel.
- `QW3_METAL_PROFILE_PREFILL_GQA_SYNC=1` is a diagnostic-only sync profiler
  for GQA prefill stages: attention norm, qkv projection, norm/RoPE, cache
  write, attend, output projection, residual norm.
- `QW3_METAL_PROFILE_PREFILL_LINEAR_SYNC=1` is a diagnostic-only sync profiler
  for linear-attention prefill stages: attention norm, qkv/gate/alpha/beta
  projection, conv1d, q/k l2norm, DeltaNet GDN, output projection, residual
  norm.
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
- Linear-attention batch DeltaNet uses the tiled recurrent core plus a separate
  gated RMSNorm node by default. Set `QW3_METAL_BATCH_GDN_LEGACY=1` for the
  old scalar fused GDN kernel.
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
