#!/bin/sh
set -eu

MODEL="${QW3_DENSE_MODEL:-../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf}"
BIN="${QW3_METAL_BIN:-./qw3-test}"
CTX="${QW3_CTX:-32}"

fail() {
    echo "test-dense-metal-bringup: FAIL: $*" >&2
    exit 1
}

[ -f "$MODEL" ] || fail "model not found: $MODEL"
[ -x "$BIN" ] || fail "binary not found: $BIN"

OUT="$(mktemp -t qw3-dense-metal.XXXXXX)"
trap 'rm -f "$OUT"' EXIT HUP INT TERM

QW3_METAL_NGL=64 "$BIN" -m "$MODEL" --ctx "$CTX" \
    --metal-session-test --metal-session-embed-test 66 \
    --metal-session-rmsnorm-test 66 --metal-session-qkv-test 66 \
    --metal-session-z-test 66 --metal-session-conv-test 66 \
    --metal-session-l2norm-test 66 --metal-session-gates-test 66 \
    --metal-session-recur-zero-test 66 --metal-session-recur-step-test 66 \
    --metal-session-gated-rmsnorm-test 66 \
    --metal-session-attn-out-test 66 \
    --metal-session-ffn-norm-test 66 \
    --metal-session-layer0-test 66 \
    --metal-session-gqa-project-test 66 \
    --metal-session-gqa-single-test 66 \
    --metal-session-gqa-cached2-test 66 \
    --metal-session-quant-types-test 66 \
    --metal-gqa-real-layer-test 66 \
    --metal-dense-logits-test 66 \
    --metal-dense-greedy-test 66 8 \
    >"$OUT" 2>&1 || {
    cat "$OUT" >&2
    fail "dense Metal differential diagnostics failed"
}

grep -q "metal session: ok ctx=$CTX full=16 linear=48" "$OUT" || {
    cat "$OUT" >&2
    fail "profile-sized Metal session was not created"
}
grep -q "metal session embed: ok profile=Qwen3.8-27B type=q3_K token=66 n=5120 maxdiff=0 rmsdiff=0" "$OUT" || {
    cat "$OUT" >&2
    fail "Metal Q3_K embedding did not match the CPU reference"
}
grep -q "metal session dense staging: ok maxdiff=0" "$OUT" || {
    cat "$OUT" >&2
    fail "Metal dense row staging did not preserve x0/x1"
}
grep -q "metal session dense iq3_s batch probe: ok .* tokens=128 " "$OUT" || {
    cat "$OUT" >&2
    fail "Metal 128-token IQ3_S batch projection did not match the CPU reference"
}
grep -q "metal session dense iq4_xs batch probe: ok .* tokens=128 " "$OUT" || {
    cat "$OUT" >&2
    fail "Metal 128-token IQ4_XS batch projection did not match the CPU reference"
}
grep -q "metal session dense iq2_s batch probe: ok .* tokens=128 " "$OUT" || {
    cat "$OUT" >&2
    fail "Metal 128-token IQ2_S batch projection did not match the CPU reference"
}
grep -q "metal session dense iq3_xxs batch probe: ok .* tokens=128 " "$OUT" || {
    cat "$OUT" >&2
    fail "Metal 128-token IQ3_XXS batch projection did not match the CPU reference"
}
grep -q "metal session dense q3_K batch probe: ok .* tokens=128 " "$OUT" || {
    cat "$OUT" >&2
    fail "Metal 128-token Q3_K batch projection did not match the CPU reference"
}
grep -q "metal session dense q4_K batch probe: ok .* tokens=128 " "$OUT" || {
    cat "$OUT" >&2
    fail "Metal 128-token Q4_K batch projection did not match the CPU reference"
}
grep -q "metal session dense q5_K batch probe: ok .* tokens=128 " "$OUT" || {
    cat "$OUT" >&2
    fail "Metal 128-token Q5_K batch projection did not match the CPU reference"
}
grep -q "metal session dense iq3_s FFN batch probe: ok .* tokens=128 " "$OUT" || {
    cat "$OUT" >&2
    fail "Metal 128-token IQ3_S batch FFN did not match the CPU reference"
}
grep -q "metal session dense iq4_xs FFN batch probe: ok .* tokens=128 " "$OUT" || {
    cat "$OUT" >&2
    fail "Metal 128-token IQ4_XS batch FFN did not match the CPU reference"
}
grep -q "metal session rmsnorm: ok profile=Qwen3.8-27B token=66 n=5120" "$OUT" || {
    cat "$OUT" >&2
    fail "Metal dense RMSNorm did not match the CPU reference"
}
grep -q "metal session qkv: ok profile=Qwen3.8-27B type=iq4_xs token=66 n_in=5120 n_out=10240" "$OUT" || {
    cat "$OUT" >&2
    fail "Metal dense IQ4_XS QKV projection did not match the CPU reference"
}
grep -q "metal session z: ok profile=Qwen3.8-27B type=iq4_xs token=66 n_in=5120 n_out=6144" "$OUT" || {
    cat "$OUT" >&2
    fail "Metal dense IQ4_XS Z projection did not match the CPU reference"
}
grep -q "metal session conv1d: ok profile=Qwen3.8-27B token=66 n=10240" "$OUT" || {
    cat "$OUT" >&2
    fail "Metal dense causal convolution did not match the CPU reference"
}
grep -q "metal session l2norm: ok profile=Qwen3.8-27B token=66 n=2048" "$OUT" || {
    cat "$OUT" >&2
    fail "Metal dense Q/K L2 normalization did not match the CPU reference"
}
grep -q "metal session gates: ok profile=Qwen3.8-27B alpha_type=q8_0 beta_type=q8_0 token=66 heads=48" "$OUT" || {
    cat "$OUT" >&2
    fail "Metal dense alpha/beta projections did not match the CPU reference"
}
grep -q "metal session recur zero: ok profile=Qwen3.8-27B token=66 heads=48" "$OUT" || {
    cat "$OUT" >&2
    fail "Metal dense zero-state recurrence did not match the CPU reference"
}
grep -q "metal session recur step: ok profile=Qwen3.8-27B token=66,67 heads=48 stage=recur-step" "$OUT" || {
    cat "$OUT" >&2
    fail "Metal dense persistent recurrence did not match the CPU reference"
}
grep -q "metal session gated RMSNorm: ok profile=Qwen3.8-27B token=66 heads=48 n=6144" "$OUT" || {
    cat "$OUT" >&2
    fail "Metal dense gated RMSNorm did not match the CPU reference"
}
grep -q "metal session attn out: ok profile=Qwen3.8-27B type=q5_K token=66 n_in=6144 n_out=5120" "$OUT" || {
    cat "$OUT" >&2
    fail "Metal dense Q5_K DeltaNet output projection did not match the CPU reference"
}
grep -q "metal session ffn norm: ok profile=Qwen3.8-27B token=66 n=5120" "$OUT" || {
    cat "$OUT" >&2
    fail "Metal dense residual FFN RMSNorm did not match the CPU reference"
}
grep -q "metal session layer0: ok profile=Qwen3.8-27B ffn=iq2_xs/iq2_s/iq3_xxs token=66" "$OUT" || {
    cat "$OUT" >&2
    fail "complete dense DeltaNet plus FFN layer did not match the CPU reference"
}
grep -q "metal session gqa project: ok profile=Qwen3.8-27B types=iq4_nl/q4_K/q5_K token=66 layer=3" "$OUT" || {
    cat "$OUT" >&2
    fail "dense GQA Q/K/V projection, norm or RoPE did not match CPU"
}
grep -q "metal session gqa single: ok profile=Qwen3.8-27B out_type=q4_K token=66 layer=3" "$OUT" || {
    cat "$OUT" >&2
    fail "dense GQA single-token output did not match CPU"
}
grep -q "metal session gqa cached2: ok profile=Qwen3.8-27B tokens=66,67 layer=3" "$OUT" || {
    cat "$OUT" >&2
    fail "dense GQA two-token KV cache output did not match CPU"
}
grep -q "metal session quant types: ok x1=13 inner=8 output=1 token=66" "$OUT" || {
    cat "$OUT" >&2
    fail "dense quantized matvec formats did not match CPU"
}
grep -q "metal dense logits: ok tokens=66,67" "$OUT" || {
    cat "$OUT" >&2
    fail "dense 64-layer Metal decode did not match CPU logits"
}
grep -q "metal dense greedy: ok start=66 steps=8 ids=8,561,1118,321,2018,41131,314,279" "$OUT" || {
    cat "$OUT" >&2
    fail "dense eight-step Metal greedy sequence did not match CPU"
}
grep -q "metal gqa real layer: ok profile=Qwen3.8-27B layer=3 ffn=q3_K/iq3_s/iq4_xs token=66 stage=FFN SwiGLU/down" "$OUT" || {
    cat "$OUT" >&2
    fail "complete dense GQA plus FFN layer did not match the CPU reference"
}

QW3_METAL_NGL=64 "$BIN" -m "$MODEL" --ctx "$CTX" --kv-f32 \
    --metal-session-gqa-cached2-test 66 \
    --metal-dense-logits-test 66 >"$OUT" 2>&1 || {
    cat "$OUT" >&2
    fail "dense GQA two-token F32 KV cache diagnostic failed"
}
grep -q "metal session gqa cached2: ok profile=Qwen3.8-27B tokens=66,67 layer=3 stage=cached attention/out kv_type=0" "$OUT" || {
    cat "$OUT" >&2
    fail "dense GQA two-token F32 KV cache output did not match CPU"
}
grep -q "metal dense logits: ok tokens=66,67" "$OUT" || {
    cat "$OUT" >&2
    fail "dense F32 KV cache decode did not match CPU logits"
}
grep -q "metal dense batch prefill: ok tokens=66..69 .* state0_ok=1" "$OUT" || {
    cat "$OUT" >&2
    fail "dense batch prefill logits or DeltaNet state diverged"
}
grep -q "metal dense batch continuation: ok " "$OUT" || {
    cat "$OUT" >&2
    fail "dense batch prefill broke the next generated token"
}

QW3_METAL_NGL=64 QW3_METAL_DENSE_Q3_K_BATCH=1 \
    QW3_METAL_DENSE_GQA_BATCH=1 \
    "$BIN" -m "$MODEL" --ctx "$CTX" --metal-dense-logits-test 66 \
    >"$OUT" 2>&1 || {
    cat "$OUT" >&2
    fail "experimental Q3_K batch prefill failed"
}
grep -q "metal dense batch prefill: ok tokens=66..69 " "$OUT" || {
    cat "$OUT" >&2
    fail "experimental Q3_K batch prefill changed logits"
}
grep -q "metal dense batch continuation: ok " "$OUT" || {
    cat "$OUT" >&2
    fail "experimental Q3_K batch prefill broke continuation"
}

QW3_METAL_NGL=64 QW3_METAL_DENSE_BATCH_TEST_TOKENS=128 \
    "$BIN" -m "$MODEL" --ctx 130 --metal-dense-logits-test 66 \
    >"$OUT" 2>&1 || {
    cat "$OUT" >&2
    fail "default 128-token dense prefill failed"
}
grep -q "metal dense batch prefill: ok tokens=66..193 " "$OUT" || {
    cat "$OUT" >&2
    fail "default 128-token dense prefill changed logits"
}
grep -q "metal dense batch continuation: ok " "$OUT" || {
    cat "$OUT" >&2
    fail "default 128-token dense prefill broke continuation"
}

QW3_METAL_NGL=64 "$BIN" -m "$MODEL" --ctx "$CTX" --nothink \
    -p ciao -n 1 >"$OUT" 2>&1 || {
    cat "$OUT" >&2
    fail "dense Metal CLI generation failed"
}
grep -q "Metal argmax repetition timing: prefill=" "$OUT" || {
    cat "$OUT" >&2
    fail "dense Metal CLI generation did not complete"
}

QW3_METAL_NGL=64 "$BIN" -m "$MODEL" --ctx "$CTX" --nothink \
    -p ciao --session-roundtrip >"$OUT" 2>&1 || {
    cat "$OUT" >&2
    fail "dense Metal session roundtrip failed"
}
grep -q "session roundtrip:.*logits=exact decode_steps=4.*corrupt=rejected" "$OUT" || {
    cat "$OUT" >&2
    fail "dense Metal session state did not restore exactly"
}

if QW3_METAL_NGL=63 "$BIN" -m "$MODEL" --ctx "$CTX" --nothink \
    -p ciao -n 1 >"$OUT" 2>&1; then
    cat "$OUT" >&2
    fail "partial dense Metal offload unexpectedly ran"
fi
grep -q "dense Metal decode requires full 64-layer offload; use --cpu" "$OUT" || {
    cat "$OUT" >&2
    fail "partial dense Metal offload was not rejected cleanly"
}

if QW3_METAL_NGL=64 "$BIN" -m "$MODEL" --ctx "$CTX" --kv-q8 \
    --nothink -p ciao -n 1 >"$OUT" 2>&1; then
    cat "$OUT" >&2
    fail "unvalidated dense Q8 KV unexpectedly ran"
fi
grep -q "dense Metal Q8 KV is not validated; use --kv-f16 or --kv-f32" "$OUT" || {
    cat "$OUT" >&2
    fail "unvalidated dense Q8 KV was not rejected cleanly"
}

echo "test-dense-metal-bringup: ok"
