# qw3

Standalone Qwen3.6-35B-A3B inference engine, split out from the parent
`ds4-main` tree so development can proceed without mixing build artifacts or
targets with `ds4`.

Useful sibling paths from this workspace:

- `../../models` contains the model files, including
  `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`.
- `../../llama.cpp` is available as the reference project for GGUF naming,
  tokenizer behavior, kernels, and Qwen architecture details.

Build the current CPU reference binary:

```sh
make
```

Run the current CPU reference path:

```sh
./qw3-cpu --cpu -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
```

`qw3-cpu` is compiled with `QW3_NO_METAL`; `--metal` is reserved for the
future Metal target and is rejected by this binary instead of silently falling
back to CPU.

Build the initial Metal bring-up target on macOS:

```sh
make metal
make test-metal-smoke
./qw3-metal --metal -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --inspect --ctx 128
./qw3-metal --metal-session-test -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-session-embed-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-session-rmsnorm-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-session-qkv-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-session-z-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-session-conv-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-session-l2norm-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-session-gates-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-session-recur-zero-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-session-gated-rmsnorm-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-session-attn-out-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-session-gqa-project-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-session-gqa-single-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-session-gqa-cached2-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-moe-real-layer-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-rmsnorm-test -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-rmsnorm-weight-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-embed-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-matvec-q8-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-deltanet-proj-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-deltanet-conv-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-deltanet-conv-step-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-deltanet-l2-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-deltanet-gates-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-deltanet-recur-zero-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-deltanet-recur-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-deltanet-recur-step-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-deltanet-gated-norm-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-deltanet-out-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-deltanet-branch-step-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-deltanet-layer-step-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-deltanet-layer2-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-deltanet-layer4-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-deltanet-layer8-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-deltanet-branch-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-deltanet-resid-norm-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-moe-router-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-moe-shared-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-moe-iq4-down-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-moe-iq3-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-moe-sparse-top1-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-moe-sparse-top8-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-moe-layer-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-moe-real-layer-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-deltanet3-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-mixed4-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-mixed8-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-logits-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-decode-test -p ciao -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-greedy-test 4 -p ciao -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-run 1 -p ciao -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-gqa-project-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-gqa-single-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-gqa-attend2-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-gqa-attend4-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-gqa-branch4-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-gqa-layer4-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
./qw3-metal --metal-gqa-real-layer-test 66 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 128
```

At this stage `qw3-metal` initializes the Metal device, maps the GGUF
tensor-data range into one or more shared Metal views, and carries a growing
set of kernel diagnostics for the vertical graph bring-up.

The current full-depth diagnostic is `--metal-mixed40-test ID`: on token 66 it
walks all 40 layers in first-token mode and currently reports
`maxdiff=1.335144e-05`, `rmsdiff=1.340667e-06`, `first_bad_layer=-1`.
`--metal-logits-test ID` extends that path through `output_norm` and the
`q6_K` `lm_head`; token 66 currently reports identical CPU/Metal top-8 logits,
`maxdiff=1.341105e-05`, `rmsdiff=2.278345e-06`, top0 `992`.
`--metal-decode-test -p ciao` runs the real tokenized ChatML prompt through
stateful Metal DeltaNet/GQA layers, then output norm and `lm_head`; on Apple M5
it currently reports identical CPU/Metal top-8 logits for the 12-token prompt,
`maxdiff=2.074242e-05`, `rmsdiff=4.185742e-06`, top0 `8160`.
`--metal-greedy-test 4 -p ciao` repeats that stateful path for greedy
continuation diagnostics; the first four generated-token decisions currently
match CPU/Metal (`8160`, `579`, `264`, `7047`) with per-step maxdiff at or
below `2.1e-05`.
The Metal decode diagnostics print `cpu_ms`, `metal_ms` and `total_ms`. For the
12-token `ciao` prompt on Apple M5, `--metal-greedy-test 1` currently spends
about `8153 ms` in the CPU reference pass and `8585 ms` in the diagnostic Metal
pass, so the command is intentionally measuring correctness with double work,
not optimized generation latency.
`--metal-run 1 -p ciao` is the first Metal-only greedy generation diagnostic:
it skips the CPU reference pass, uses Metal logits to append token `8160`, and
currently takes about `10194 ms` for the 12-token prompt on Apple M5.
Plain greedy generation through the Metal backend is now wired as well:
`./qw3-metal --metal -p ciao -n 2 ...` uses the Metal graph and emits
`Here's`.
This is still an early M6 path built on full reprefill per generated token, not
the final persistent Metal session/decode implementation.
`make test-metal-smoke` exercises the Metal RMSNorm diagnostic, CPU-vs-Metal
decode for `ciao`, and the plain Metal generation path with an exact two-token
output check. Inside Codex's sandbox, run that target outside the sandbox/with
approval so Metal device enumeration is visible.
Session payload commands (`--save-session`, `--load-session` and
`--session-roundtrip`) are intentionally CPU-only for now; pass `--cpu` for
those until the Metal session owns persistent device-side KV/DeltaNet state.
The first persistent Metal session scaffold is now present:
`--metal-session-test` allocates and clears device-side buffers for GQA KV,
DeltaNet recurrent state, DeltaNet conv state, logits and scratch. At
`--ctx 128` it currently reports about `68.97 MiB` total.
`--metal-session-embed-test 66` is the first diagnostic that writes a real
kernel result into a persistent session buffer (`x0`); it matches the CPU
embedding row with `maxdiff=0`.
`--metal-session-rmsnorm-test 66` extends that chain from session `x0` through
layer-0 `attn_norm` into session `x1`; it currently reports
`maxdiff=9.536743e-07`.
`--metal-session-qkv-test 66` continues from `x1` through layer-0
`attn_qkv.weight` into session scratch; it currently reports
`maxdiff=7.629395e-06`.
`--metal-session-z-test 66` writes the layer-0 `attn_gate.weight` projection
after QKV in the same scratch buffer (`scratch_offset=8192`); it currently
reports `maxdiff=2.861023e-06`.
`--metal-session-conv-test 66` runs layer-0 zero-state DeltaNet conv1d from
session scratch into the persistent conv output buffer; it currently reports
`maxdiff=1.907349e-06`.
`--metal-session-l2norm-test 66` continues from that conv output through
persistent Q/K L2Norm buffers; it currently reports `maxdiff=1.490116e-07`.
`--metal-session-gates-test 66` writes the layer-0 SSM alpha/beta f32
projections from `x1` into session scratch; it currently reports
`maxdiff=1.907349e-06`.
`--metal-session-recur-zero-test 66` runs the zero-state DeltaNet recurrence
from persistent Q/K/V buffers into persistent recurrent/core buffers; it
currently reports `core_maxdiff=5.960464e-08` and
`state_maxdiff=1.192093e-06`.
`--metal-session-gated-rmsnorm-test 66` continues from persistent core plus
the session Z projection through DeltaNet gated RMSNorm; it currently reports
`maxdiff=3.576279e-07`.
`--metal-session-attn-out-test 66` runs the persistent DeltaNet `inner` buffer
through layer-0 `ssm_out` into session `x1`; it currently reports
`maxdiff=3.72529e-08`.
`--metal-session-gqa-project-test 66` runs layer-3 GQA projection from session
`x1`, per-head Q/K RMSNorm, RoPE, and K/V cache write; it currently reports
`q_max=2.384186e-06`, `k_max=1.192093e-06`, `v_max=4.768372e-07`.
`--metal-session-gqa-single-test 66` continues from that persistent GQA state
through single-token attention and `attn_o`; it currently reports
`maxdiff=5.960464e-07`.
`--metal-session-gqa-cached2-test 66` exercises persistent GQA KV cache over
two cached tokens and then `attn_o`; it currently reports
`maxdiff=1.072884e-06`.
`--metal-moe-real-layer-test 66` validates the complete layer path through
residual, `ffn_norm`, router, sparse top-8 MoE, shared expert and final
residual; it currently reports `layer_max=9.23872e-07`.
`--metal-mixed40-test 66` is the current 40-layer Metal runner diagnostic;
it traverses layers 0..39 with `first_bad_layer=-1`.
`--metal-logits-test 66` adds final `output_norm` and `lm_head`; CPU/Metal
top-8 logits match exactly and top0 is `992`.
For correctness during bring-up, model-weight matvec wrappers copy the needed
tensor slice from the mmap into a small Metal buffer before dispatch; this
avoids zero reads from high-offset tensors in the second GGUF Metal view and is
expected to be optimized later.

Qwen3.5-MoE has one important GGUF-layout wrinkle: Hugging Face stores linear
attention V-heads grouped by K head and uses `repeat_interleave`, while
`llama.cpp` reorders those V-heads to tiled order during GGUF conversion. In
this GGUF path the DeltaNet head mapping is therefore `hv % num_k_heads`, not
the raw HF grouped mapping.

On the local Apple M5, the bring-up path initializes `Apple M5` and maps the
Qwen GGUF tensor-data range in 2 Metal views. Inside Codex's default sandbox,
Metal device enumeration can be hidden; running the same binary outside the
sandbox exposes the device correctly.

The first diagnostic kernel is `--metal-rmsnorm-test`: a plain RMSNorm over a
synthetic 2048-float row, compared against CPU. On Apple M5 it currently reports
`maxdiff=0`.
`--metal-rmsnorm-weight-test ID` applies real layer-0 `attn_norm.weight` from the
mapped GGUF model view to a real token embedding; token 66 currently reports
`maxdiff=9.536743e-07`.
`--metal-embed-test ID` dequantizes one `q8_0` token embedding row directly from
the mapped GGUF model view and compares it with the CPU tensor reader; token 66
currently reports `maxdiff=0`.
`--metal-matvec-q8-test ID` runs `blk.0.attn_qkv.weight` as a real `q8_0`
matvec after embedding + weighted RMSNorm; token 66 currently reports
`maxdiff=9.536743e-06`.
`--metal-deltanet-proj-test ID` runs layer-0 DeltaNet QKV and Z projections;
token 66 currently reports CPU-matching Q/K/V/Z RMS values with
`maxdiff=9.536743e-06`.
`--metal-deltanet-conv-test ID` runs the layer-0 short convolution in zero-state
mode after the QKV projection; token 66 currently reports `maxdiff=2.384186e-06`.
`--metal-deltanet-conv-step-test ID` runs the layer-0 short convolution with
a deterministic non-zero previous conv state and returns the shifted state;
token 66 currently reports `conv_max=4.768372e-07` and `state_max=0`.
`--metal-deltanet-l2-test ID` normalizes layer-0 post-conv Q/K heads; token 66
currently reports `maxdiff=1.192093e-07` with Q/K head-0 norms equal to 1.
`--metal-deltanet-gates-test ID` runs layer-0 `linear_ssm_alpha` and
`linear_ssm_beta` f32 matvecs and compares both raw outputs and transformed
gates; token 66 currently reports `raw_maxdiff=1.907349e-06` and
`gate_maxdiff=4.768372e-07`.
`--metal-deltanet-recur-zero-test ID` runs the first DeltaNet recurrence kernel
for a zero initial state, writing both `core` and the recurrent state; token 66
currently reports `core_maxdiff=2.980232e-08` and
`state_maxdiff=4.768372e-07`.
`--metal-deltanet-recur-test ID` runs the full single-token recurrence against
a deterministic non-zero state, including decay and `K @ S_old`; token 66
currently reports `core_maxdiff=5.820766e-10` and
`state_maxdiff=1.490116e-08`.
`--metal-deltanet-recur-step-test ID` composes non-zero conv state, Metal
conv-step, L2 normalization and non-zero recurrent state; token 66 currently
reports `core_max=2.980232e-08` and `state_max=4.768372e-07`.
`--metal-deltanet-gated-norm-test ID` runs the layer-0 gated RMSNorm block after
DeltaNet recurrence, using `linear_ssm_norm` and `SiLU(z)`; token 66 currently
reports `maxdiff=4.768372e-07`.
`--metal-deltanet-out-test ID` runs the layer-0 `linear_ssm_out` q8_0
projection from DeltaNet `inner` to the residual-sized `attn` vector; token 66
currently reports `maxdiff=4.172325e-07`.
`--metal-deltanet-branch-step-test ID` composes non-zero conv state, non-zero
recurrent state, gated RMSNorm and `linear_ssm_out`; token 66 currently reports
`attn_max=1.072884e-06`, `state_max=4.768372e-07` and `conv_state_max=0`.
`--metal-deltanet-layer-step-test ID` completes the stateful layer-0 DeltaNet
path through residual, `ffn_norm`, router/top-8 and MoE; token 66 currently
reports `layer_max=1.192093e-06`.
`--metal-deltanet-layer2-test ID` runs layer-0 DeltaNet for two sequential
tokens while carrying conv and recurrent state between steps; token 66
currently reports `layer_max=1.311302e-06`.
`--metal-deltanet-layer4-test ID` runs layer-0 DeltaNet for four sequential
tokens while carrying conv and recurrent state between steps; token 66
currently reports `layer_max=3.874302e-07`.
`--metal-deltanet-layer8-test ID` runs layer-0 DeltaNet for eight sequential
tokens while carrying conv and recurrent state between steps; token 66
currently reports `layer_max=4.768372e-07`.
`--metal-deltanet-branch-test ID` composes the verified Metal primitives from
embedding + RMSNorm through DeltaNet `attn`; token 66 currently reports
`attn_maxdiff=6.854534e-07` and `state_maxdiff=1.430511e-06`.
`--metal-deltanet-resid-norm-test ID` runs residual `x + attn` followed by
layer-0 `ffn_norm.weight`; token 66 currently reports `maxdiff=4.768372e-07`.
`--metal-moe-router-test ID` runs layer-0 MoE router f32 matvec from the
normalized FFN input and compares top-8 routing; token 66 currently reports
`maxdiff=9.536743e-07` with identical top-8 experts.
`--metal-moe-shared-test ID` runs the layer-0 shared expert q8_0 path
(`gate`, `up`, SiLU product, `down`, scalar shared gate); token 66 currently
reports `maxdiff=5.960464e-08`.
`--metal-moe-iq4-down-test ID` runs the layer-0 sparse expert `IQ4_XS` down
matvec on Metal, using CPU `IQ3_S` gate/up to feed the hidden vector; token 66
currently reports `maxdiff=3.72529e-09`.
`--metal-moe-iq3-test ID` runs layer-0 sparse expert `IQ3_S` gate/up matvecs
on Metal; token 66 currently reports `gate_max=4.768372e-07`,
`up_max=6.556511e-07`.
`--metal-moe-sparse-top1-test ID` composes the layer-0 sparse expert top-1
path (`IQ3_S` gate/up, SiLU product, `IQ4_XS` down); token 66 currently reports
`maxdiff=1.583248e-08`.
`--metal-moe-sparse-top8-test ID` runs the layer-0 sparse top-8 weighted path
with the verified expert kernels; token 66 currently reports
`maxdiff=1.210719e-08`.
`--metal-moe-layer-test ID` composes sparse top-8 plus the shared expert on the
same normalized FFN input; token 66 currently reports `maxdiff=5.960464e-08`.
`--metal-moe-real-layer-test ID` composes layer-0 DeltaNet through post-attn
`ffn_norm`, Metal router/top-8, MoE sparse+shared and final `x + attn + moe`;
token 66 currently reports `moe_max=2.980232e-07` and
`layer_max=9.23872e-07`.
`--metal-deltanet3-test ID` runs layers 0, 1 and 2 as a first-token DeltaNet
sequence with zero initial recurrent/conv state; token 66 currently reports
`maxdiff=2.384186e-07`.
`--metal-mixed4-test ID` runs the first mixed sequence, DeltaNet layers 0..2
plus GQA layer 3, as a single Metal diagnostic; token 66 currently reports
`maxdiff=1.788139e-07`.
`--metal-mixed8-test ID` runs the first two mixed cycles, DeltaNet layers
0..2 and 4..6 plus GQA layers 3 and 7, as a single Metal diagnostic; token 66
currently reports `maxdiff=1.66893e-06`.
`--metal-mixed40-test ID` runs all 40 layers in first-token mode; token 66
currently reports `maxdiff=1.335144e-05` and `first_bad_layer=-1`.
`--metal-logits-test ID` runs `mixed40`, output RMSNorm and the `q6_K`
`output.weight` projection; token 66 currently reports identical CPU/Metal
top-8 logits and `maxdiff=1.341105e-05`.
`--metal-decode-test -p ciao` runs the 12-token ChatML prompt through the
stateful Metal decode path, carrying all DeltaNet recurrent states and GQA KV
caches; it currently reports identical CPU/Metal top-8 logits and
`maxdiff=2.074242e-05`.
`--metal-greedy-test 4 -p ciao` verifies greedy continuation by appending the
Metal top token and re-running the growing prompt; the first four decisions
currently match CPU (`8160`, `579`, `264`, `7047`).
`--metal-gqa-project-test ID` runs layer-3 GQA `q/k/v` projections, Q/K
RMSNorm and RoPE on Metal; token 66 currently reports `q_max=2.384186e-06`,
`k_max=1.192093e-06`, `v_max=4.768372e-07`.
`--metal-gqa-single-test ID` runs layer-3 single-token GQA attention inner
(`V * sigmoid(gate)`) plus the `attn_o` q8_0 projection on Metal; token 66
currently reports `inner_max=4.768372e-07` and `out_max=5.960464e-07`.
`--metal-gqa-attend2-test ID` runs layer-3 GQA attention over a 2-token
KV cache, including `q.k`, softmax and gated value mixing; token 66 currently
reports `maxdiff=2.384186e-07`.
`--metal-gqa-attend4-test ID` runs the generic `attend_n` Metal kernel over
a 4-token KV cache; token 66 currently reports `maxdiff=8.34465e-07`.
`--metal-gqa-branch4-test ID` runs layer-3 GQA projections for 4 tokens,
builds the KV cache, runs `attend_n`, and applies `attn_o`; token 66 currently
reports `inner_max=8.940697e-07` and `attn_max=1.192093e-06`.
`--metal-gqa-layer4-test ID` completes layer-3 over a 4-token sequence through
residual, `ffn_norm`, router/top-8 and MoE; token 66 currently reports
`maxdiff=9.536743e-07`.
`--metal-gqa-real-layer-test ID` composes layer-3 GQA `attn_o`, residual,
`ffn_norm`, router/top-8 and MoE sparse+shared; token 66 currently reports
top-8 identical, `moe_max=3.576279e-07` and `layer_max=5.960464e-07`.

Quick smoke tests:

```sh
./qw3-cpu --cpu -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf -p ciao -n 8 --ctx 128
./qw3-cpu --cpu -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf -p ciao -n 8 --ctx 128 --temp 1.5 --sample-top-k 50 --top-p 1 --seed 7
./qw3-cpu --cpu -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf -p ciao --top-k 5 --ctx 128
./qw3-cpu --cpu -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf -p ciao -n 4 --dump-logprobs /tmp/qw3-ciao.logits.json --logprobs-top-k 8 --ctx 128
./qw3-cpu --cpu -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf -p ciao --dump-trace /tmp/qw3-ciao.trace.json --ctx 128
./qw3-cpu --cpu -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf -p ciao --session-roundtrip --ctx 128
./qw3-cpu --cpu -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf -p ciao --trace-layers --ctx 128
./qw3-cpu --cpu -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf -p ciao --save-session /tmp/qw3-ciao.session --ctx 128
./qw3-cpu --cpu -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --load-session /tmp/qw3-ciao.session -n 8 --ctx 128
```

## Test Vectors

`tests/test-vectors` contains early local/reference vectors for the current
CPU path. They are not official Qwen vectors yet; they are regression
guardrails in the same spirit as `ds4`: tokenization/template identity, top
logit ordering, short greedy continuation, and session payload restore.
`--dump-logprobs` can also emit a compact JSON fixture with prompt tokens,
selected tokens, decoded bytes and top logits for each generated step.
`--dump-trace` emits a CPU reference JSON trace for the final prompt token:
embedding stats, each layer output stats and final top logits. It is intended
as the comparison surface for future Metal kernels.

Run:

```sh
make test-vectors
```
