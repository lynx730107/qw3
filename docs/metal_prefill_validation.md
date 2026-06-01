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
- `QW3_METAL_MOE_MAP_GATEUP_PAIR=1` enables the experimental fused mapped
  gate/up/SwiGLU kernel. It is not default because the current version is
  correct but slower on the validation prompt.
- `QW3_METAL_MOE_MID_F16=1` enables the DS4-style F16 routed-MoE intermediate.
  It is correct under logits regression, but remains opt-in until it shows a
  repeatable speed win on long prompts.
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
