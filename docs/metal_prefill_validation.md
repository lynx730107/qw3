# Metal Prefill Validation

This project treats logits as the regression boundary for Metal prefill changes.
Performance-only checks are not enough: every prefill optimization must preserve
CPU/Metal final logits and greedy top-1 choices before it can be enabled by
default.

Default safety policy:
- `QW3_METAL_KV_Q8_0` is not part of the default validation path.
- Expert-major MoE gate/up is opt-in with `QW3_METAL_MOE_MAP_GATEUP=1`.
- Expert-major MoE down is opt-in with `QW3_METAL_MOE_MAP_DOWN=1`.
- `QW3_METAL_PREFILL_BATCH` defaults to the conservative batch size in code;
  larger batches must pass the logits regression before becoming default.

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
./qw3-metal -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 1024 \
  --metal-session-decode-test -p "ciao"
env QW3_METAL_PREFILL_TEST_TOKENS=64 ./qw3-metal \
  -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 1024 \
  --metal-session-prefill-q8-batch-test 66
```
