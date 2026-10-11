#!/bin/sh
set -eu

MODEL="${QW3_MODEL:-../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf}"
AGENT="${QW3_AGENT_BIN:-./qw3-agent}"
CTX="${QW3_CTX:-4096}"

fail() {
    [ ! -f "${last_output:-}" ] || cat "$last_output" >&2
    echo "test-agent-checkpoint-compatibility: FAIL: $*" >&2
    exit 1
}

[ -x "$AGENT" ] || fail "missing executable: $AGENT"
[ -r "$MODEL" ] || fail "missing model: $MODEL"
tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/qw3-checkpoint-compat.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT
mkdir "$tmpdir/kernels" "$tmpdir/store"
cp metal/*.metal "$tmpdir/kernels/"
unset QW3_METAL_KERNEL_SOURCE QW3_METAL_KV_Q8_0
export QW3_METAL_KERNEL_DIR="$tmpdir/kernels"
export QW3_METAL_FLASH_ATTN_SOURCE="$tmpdir/kernels/flash_attn.metal"
export QW3_METAL_KV_F16=1

last_output="$tmpdir/save.out"
"$AGENT" -m "$MODEL" --ctx "$CTX" --kv-f16 --nothink --temp 0 \
    --no-prefix-cache --store-dir "$tmpdir/store" --conversation compatibility \
    -p "Rispondi soltanto con Ciao." -n 32 >"$last_output" 2>&1 ||
    fail "initial conversation failed"
grep -q 'KV ready' "$last_output" || fail "checkpoint was not saved"

last_output="$tmpdir/resume.out"
"$AGENT" -m "$MODEL" --ctx "$CTX" --kv-f16 --nothink \
    --no-prefix-cache --store-dir "$tmpdir/store" --conversation compatibility \
    --dump-prompt >"$last_output" 2>&1 || fail "compatible load failed"
grep -q 'KV resumed' "$last_output" || fail "unchanged kernels did not resume"

mv "$tmpdir/kernels" "$tmpdir/relocated-kernels"
export QW3_METAL_KERNEL_DIR="$tmpdir/relocated-kernels"
export QW3_METAL_FLASH_ATTN_SOURCE="$tmpdir/relocated-kernels/flash_attn.metal"
last_output="$tmpdir/relocated.out"
"$AGENT" -m "$MODEL" --ctx "$CTX" --kv-f16 --nothink \
    --no-prefix-cache --store-dir "$tmpdir/store" --conversation compatibility \
    --dump-prompt >"$last_output" 2>&1 || fail "relocated source load failed"
grep -q 'KV resumed' "$last_output" || fail "identical source at another path was rejected"

printf '\n// Checkpoint compatibility regression: changed loaded source.\n' \
    >>"$QW3_METAL_KERNEL_DIR/qw3_core_common.metal"
last_output="$tmpdir/rebuild.out"
"$AGENT" -m "$MODEL" --ctx "$CTX" --kv-f16 --nothink --temp 0 \
    --no-prefix-cache --store-dir "$tmpdir/store" --conversation compatibility \
    -p "Rispondi soltanto con Ciao." -n 32 >"$last_output" 2>&1 ||
    fail "transcript fallback failed"
grep -q 'ignoring incompatible KV checkpoint' "$last_output" ||
    fail "changed kernels accepted the old checkpoint"
grep -q 'runtime/kernel fingerprint mismatch' "$last_output" ||
    fail "checkpoint was rejected for an unexpected reason"
if grep -q 'KV resumed' "$last_output"; then
    fail "incompatible state was resumed"
fi
grep -q 'rebuild pending' "$last_output" || fail "transcript rebuild was not selected"
grep -q 'Ciao' "$last_output" || fail "no coherent response after rebuild"
grep -q 'KV ready' "$last_output" || fail "rebuilt checkpoint was not saved"

last_output="$tmpdir/rebuilt-resume.out"
"$AGENT" -m "$MODEL" --ctx "$CTX" --kv-f16 --nothink \
    --no-prefix-cache --store-dir "$tmpdir/store" --conversation compatibility \
    --dump-prompt >"$last_output" 2>&1 || fail "rebuilt checkpoint load failed"
grep -q 'KV resumed' "$last_output" || fail "rebuilt state did not resume"
echo "test-agent-checkpoint-compatibility: ok"
