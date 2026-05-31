#!/bin/sh
set -eu

MODEL="${QW3_MODEL:-../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf}"
BIN="${QW3_METAL_BIN:-./qw3-metal}"
CTX="${QW3_CTX:-128}"
PROMPT="${QW3_PROMPT:-ciao}"

fail() {
    echo "test-metal-smoke: FAIL: $*" >&2
    exit 1
}

run_metal() {
    "$BIN" -m "$MODEL" --ctx "$CTX" "$@"
}

run_metal --metal-rmsnorm-test ||
    fail "rmsnorm diagnostic failed"

run_metal --metal-decode-test -p "$PROMPT" ||
    fail "decode diagnostic failed"

run_metal --metal-session-decode-test -p "$PROMPT" ||
    fail "persistent Metal session decode failed"

run_metal --metal-session-test ||
    fail "Metal session allocation failed"

run_metal --metal-session-embed-test 66 ||
    fail "Metal session embedding failed"

run_metal --metal-session-rmsnorm-test 66 ||
    fail "Metal session RMSNorm failed"

run_metal --metal-session-qkv-test 66 ||
    fail "Metal session QKV projection failed"

run_metal --metal-session-z-test 66 ||
    fail "Metal session Z projection failed"

run_metal --metal-session-conv-test 66 ||
    fail "Metal session conv1d failed"

run_metal --metal-session-l2norm-test 66 ||
    fail "Metal session L2Norm failed"

run_metal --metal-session-gates-test 66 ||
    fail "Metal session SSM gates failed"

run_metal --metal-session-recur-zero-test 66 ||
    fail "Metal session recurrent zero failed"

run_metal --metal-session-recur-step-test 66 ||
    fail "Metal session recurrent persistent step failed"

run_metal --metal-session-gated-rmsnorm-test 66 ||
    fail "Metal session gated RMSNorm failed"

run_metal --metal-session-attn-out-test 66 ||
    fail "Metal session attention output failed"

run_metal --metal-session-ffn-norm-test 66 ||
    fail "Metal session FFN norm failed"

run_metal --metal-session-layer0-test 66 ||
    fail "Metal session layer0 failed"

run_metal --metal-session-gqa-project-test 66 ||
    fail "Metal session GQA project/cache failed"

run_metal --metal-session-gqa-single-test 66 ||
    fail "Metal session GQA single-token failed"

run_metal --metal-session-gqa-cached2-test 66 ||
    fail "Metal session GQA cached attention failed"

if [ "${QW3_TEST_KV_Q8:-0}" = "1" ]; then
    run_metal -ctk q8_0 -ctv q8_0 --metal-session-gqa-cached-bench 66 64 ||
        fail "Metal session q8 GQA cached benchmark failed"
fi

run_metal --metal-moe-real-layer-test 66 ||
    fail "Metal full layer failed"

run_metal --metal-mixed40-test 66 ||
    fail "Metal 40-layer runner failed"

run_metal --metal-logits-test 66 ||
    fail "Metal final logits failed"

run_metal --metal-greedy-test 2 -p "$PROMPT" ||
    fail "Metal greedy logits regression failed"

echo "test-metal-smoke: ok"
