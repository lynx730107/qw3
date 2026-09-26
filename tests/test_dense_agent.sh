#!/bin/sh
set -eu

MODEL="${QW3_DENSE_MODEL:-../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf}"
AGENT="${QW3_AGENT_BIN:-./qw3-agent}"

fail() {
    [ ! -f "${out:-}" ] || cat "$out" >&2
    echo "test-dense-agent: FAIL: $*" >&2
    exit 1
}

[ -r "$MODEL" ] || fail "model not found: $MODEL"
[ -x "$AGENT" ] || fail "agent binary not found: $AGENT"

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/qw3-dense-agent.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT HUP INT TERM
mkdir "$tmpdir/store" "$tmpdir/prefix" "$tmpdir/work"
out="$tmpdir/agent.out"

"$AGENT" -m "$MODEL" --ngl 64 --ctx 256 --no-prefix-cache \
    --no-tools --nothink --dump-prompt -p ciao \
    --store-dir "$tmpdir/store" >"$out" 2>&1 ||
    fail "full-offload prompt rendering failed"
grep -q '<|im_start|>assistant' "$out" ||
    fail "Qwen chat assistant prefix is missing"
grep -q '<think>' "$out" || fail "Qwen thinking prefix is missing"

(unset QW3_METAL_KERNEL_DIR QW3_METAL_KERNEL_SOURCE \
    QW3_METAL_FLASH_ATTN_SOURCE
 "$AGENT" -m "$MODEL" --ngl 64 --ctx 256 --no-prefix-cache \
    --compact-tools --nothink --dump-prompt -p ciao \
    --chdir "$tmpdir/work" --store-dir "$tmpdir/store") \
    >"$out" 2>&1 || fail "agent could not resolve paths before --chdir"
grep -q '"name":"read"' "$out" || fail "compact read tool is missing"
grep -q '"name":"write"' "$out" || fail "compact write tool is missing"
grep -q '"name":"edit"' "$out" || fail "compact edit tool is missing"
grep -q '"name":"bash"' "$out" || fail "compact bash tool is missing"
if grep -q '"name":"google_search"' "$out"; then
    fail "compact mode advertised a non-local tool"
fi
"$AGENT" --compact-tools --store-dir "$tmpdir/store" \
    --tool-native '<tool_call><function=google_search><parameter=query>x</parameter></function></tool_call>' \
    >"$out" 2>&1 || fail "compact tool rejection failed"
grep -q 'tool is not available with --compact-tools' "$out" ||
    fail "compact mode executed an undeclared tool"

if "$AGENT" -m "$MODEL" --ngl 63 --ctx 256 --no-prefix-cache \
    --no-tools --dump-prompt -p ciao --store-dir "$tmpdir/store" \
    >"$out" 2>&1; then
    fail "partial dense offload unexpectedly passed"
fi
grep -q 'dense Metal requires full 64-layer offload' "$out" ||
    fail "partial offload was not rejected clearly"

if "$AGENT" -m "$MODEL" --ngl 64 --kv-q8 --ctx 256 \
    --no-prefix-cache --no-tools --dump-prompt -p ciao \
    --store-dir "$tmpdir/store" >"$out" 2>&1; then
    fail "dense Q8 KV unexpectedly passed"
fi
grep -q 'dense Metal Q8 KV is not validated' "$out" ||
    fail "Q8 KV was not rejected clearly"

if "$AGENT" -m "$MODEL" --ngl 65 --ctx 256 --no-prefix-cache \
    --no-tools --dump-prompt -p ciao --store-dir "$tmpdir/store" \
    >"$out" 2>&1; then
    fail "out-of-range layer count unexpectedly passed"
fi
grep -q -- '--ngl must be in the range 0..64' "$out" ||
    fail "layer count was not validated against the loaded model"

"$AGENT" -m "$MODEL" --ngl 64 --ctx 256 --no-prefix-cache \
    --no-tools --nothink --temp 0 -n 8 \
    --conversation dense-smoke --store-dir "$tmpdir/store" \
    -p 'Rispondi solo con OK.' >"$out" 2>&1 ||
    fail "dense agent generation failed"
grep -q '^OK$' "$out" || fail "dense agent did not answer"
grep -q 'KV ready' "$out" || fail "dense agent checkpoint was not saved"

"$AGENT" -m "$MODEL" --ngl 64 --ctx 256 --no-prefix-cache \
    --no-tools --nothink --dump-prompt \
    --conversation dense-smoke --store-dir "$tmpdir/store" \
    >"$out" 2>&1 || fail "dense agent checkpoint resume failed"
grep -q 'KV resumed' "$out" || fail "dense agent checkpoint was not resumed"

"$AGENT" -m "$MODEL" --ngl 64 --ctx 256 --no-tools --nothink \
    --dump-prompt -p ciao --store-dir "$tmpdir/prefix" \
    >"$out" 2>&1 || fail "dense prefix cache creation failed"
grep -q 'prefix cache created' "$out" ||
    fail "dense prefix cache was not created"

"$AGENT" -m "$MODEL" --ngl 64 --ctx 256 --no-tools --nothink \
    --dump-prompt -p ciao --store-dir "$tmpdir/prefix" \
    >"$out" 2>&1 || fail "dense prefix cache resume failed"
grep -q 'prefix cache resumed' "$out" ||
    fail "dense prefix cache was not resumed"

echo "test-dense-agent: ok"
