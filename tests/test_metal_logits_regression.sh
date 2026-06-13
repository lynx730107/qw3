#!/bin/sh
set -eu

MODEL="${QW3_MODEL:-../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf}"
BIN="${QW3_METAL_BIN:-./qw3}"
CTX="${QW3_CTX:-1024}"

fail() {
    echo "test-metal-logits-regression: FAIL: $*" >&2
    exit 1
}

run_metal() {
    "$BIN" -m "$MODEL" --ctx "$CTX" "$@"
}

# Keep KV-cache q8 out of the default regression path until it is validated.
unset QW3_METAL_KV_Q8_0

run_metal --metal-logits-test 66 ||
    fail "single-token final logits diverged"

run_metal --metal-session-decode-test -p "ciao" ||
    fail "short prompt session logits diverged"

run_metal --metal-greedy-test 4 -p "ciao" ||
    fail "short prompt greedy tokens diverged"

run_metal --metal-session-decode-test -p "crea un file c helloworld.c, mettilo nella cartella corrente" ||
    fail "agent-like prompt session logits diverged"

env QW3_METAL_PREFILL_TEST_TOKENS=64 \
    "$BIN" -m "$MODEL" --ctx "$CTX" --metal-session-prefill-q8-batch-test 66 ||
    fail "64-token batch prefill layer/logit vector path diverged"

echo "test-metal-logits-regression: ok"
