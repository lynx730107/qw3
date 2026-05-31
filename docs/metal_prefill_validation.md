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
- Expert-major MoE gate/up is the default for batch prefill with at least 32
  tokens; set `QW3_METAL_MOE_MAP_GATEUP_DISABLE=1` for legacy comparisons.
- Expert-major MoE down is the default for IQ4_XS batch prefill with at least
  32 tokens; set `QW3_METAL_MOE_MAP_DOWN_DISABLE=1` for legacy comparisons.
- `QW3_METAL_PREFILL_BATCH` defaults to 1024, the current Metal batch cap;
  larger caps must pass the logits regression before becoming available.
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
4. Only then benchmark with `--metal-run` or `qw3-agent`.

Useful commands:

```sh
make qw3-metal
make test-metal-logits
make test-metal-logits-concurrent
./qw3-metal -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 1024 \
  --metal-session-decode-test -p "ciao"
env QW3_METAL_PREFILL_TEST_TOKENS=64 ./qw3-metal \
  -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 1024 \
  --metal-session-prefill-q8-batch-test 66
```
