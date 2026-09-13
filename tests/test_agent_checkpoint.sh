#!/bin/sh
set -eu

MODEL="${QW3_MODEL:-../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf}"
AGENT="${QW3_AGENT_BIN:-./qw3-agent}"
CTX="${QW3_CTX:-4096}"

fail() {
    echo "test-agent-checkpoint: FAIL: $*" >&2
    exit 1
}

[ -x "$AGENT" ] || fail "missing executable: $AGENT"
[ -r "$MODEL" ] || fail "missing model: $MODEL"

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/qw3-agent-checkpoint.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT
first="$tmpdir/first.out"
second="$tmpdir/second.out"

unset QW3_METAL_KV_Q8_0
export QW3_METAL_KV_F16=1
export QW3_METAL_KERNEL_DIR="${QW3_METAL_KERNEL_DIR:-metal}"

"$AGENT" -m "$MODEL" --ctx "$CTX" --nothink --temp 0 -n 64 \
    --store-dir "$tmpdir" --conversation checkpoint-test \
    -p "Rispondi soltanto con CHECKPOINT_SEED_OK" >"$first" 2>&1 ||
    fail "initial conversation failed"

grep -q 'KV ready' "$first" || fail "checkpoint was not saved"
[ -s "$tmpdir/checkpoint-test.kvc" ] || fail "checkpoint file is missing"

"$AGENT" -m "$MODEL" --ctx "$CTX" --nothink --temp 0 -n 256 \
    --store-dir "$tmpdir" --conversation checkpoint-test \
    -p "Usa il tool bash per eseguire esattamente: printf QW3_KV_RESUME_TOOL_OK. Poi riferisci il risultato." \
    >"$second" 2>&1 || fail "resumed conversation failed"

cat "$second"
grep -q 'KV resumed' "$second" || fail "checkpoint was not restored"
grep -q '\[tool\] bash' "$second" || fail "resumed agent did not call bash"
grep -q 'QW3_KV_RESUME_TOOL_OK' "$second" || fail "resumed tool output is missing"

echo "test-agent-checkpoint: ok"
