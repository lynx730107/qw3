#!/bin/sh
set -eu

MODEL="${QW3_MODEL:-../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf}"
BIN="${QW3_BIN:-./qw3-cpu}"
CTX="${QW3_CTX:-128}"
FIXTURES_DIR="${QW3_FIXTURES_DIR:-tests/test-vectors}"

fail() {
    echo "test-vectors: FAIL: $*" >&2
    exit 1
}

run() {
    "$BIN" --cpu -m "$MODEL" "$@"
}

expected_tokens='[248045, 846, 198, 66, 21817, 248046, 198, 248045, 74455, 198, 248068, 198]'
tokens=$(run -p ciao --chat-tokenize 2>/dev/null | tail -n 1)
[ "$tokens" = "$expected_tokens" ] || fail "chat-tokenize mismatch: $tokens"

top=$(run -p ciao --top-k 5 --ctx "$CTX" 2>/dev/null)
echo "$top" | grep -q ' 1  id=8160 .* text="Here"' || fail "top1 is not Here/id=8160"
echo "$top" | grep -q ' 2  id=1596 .* text="We"' || fail "top2 is not We/id=1596"
echo "$top" | grep -q ' 3  id=90700 .* text="Thinking"' || fail "top3 is not Thinking/id=90700"

gen=$(run -p ciao -n 8 --ctx "$CTX" 2>/dev/null)
case "$gen" in
    "Here's a thinking process:"*) ;;
    *) fail "generation prefix mismatch: $gen" ;;
esac

session_file=$(mktemp /tmp/qw3-vector.XXXXXX.session)
logits_file=$(mktemp /tmp/qw3-vector.XXXXXX.json)
trace_file=$(mktemp /tmp/qw3-vector.XXXXXX.trace.json)
trap 'rm -f "$session_file" "$logits_file" "$trace_file"' EXIT

save=$(run -p ciao --save-session "$session_file" --ctx "$CTX" 2>/dev/null)
echo "$save" | grep -q 'saved session:' || fail "session save failed"

load=$(run --load-session "$session_file" -n 8 --ctx "$CTX" 2>/dev/null)
case "$load" in
    "Here's a thinking process:"*) ;;
    *) fail "loaded-session generation mismatch: $load" ;;
esac

roundtrip=$(run -p ciao --session-roundtrip --ctx "$CTX" 2>/dev/null)
echo "$roundtrip" | grep -q 'top5=ok maxdiff=0' || fail "session roundtrip mismatch"

run -p ciao -n 2 --dump-logprobs "$logits_file" --logprobs-top-k 5 --ctx "$CTX" >/dev/null 2>/dev/null
cmp -s "$FIXTURES_DIR/ciao-logits-v1.json" "$logits_file" ||
    fail "dump-logprobs fixture mismatch: $logits_file"

run -p ciao --dump-trace "$trace_file" --ctx "$CTX" >/dev/null 2>/dev/null
grep -q '"schema":"qw3-local-trace-v1"' "$trace_file" || fail "dump-trace schema missing"
grep -q '"name":"embedding","layer":-1' "$trace_file" || fail "dump-trace embedding missing"
grep -q '"name":"full","layer":39' "$trace_file" || fail "dump-trace final layer missing"
grep -q '"top_logits":\[{"id":8160,"logit":25.60219}' "$trace_file" ||
    fail "dump-trace top logit mismatch"

echo "test-vectors: ok"
