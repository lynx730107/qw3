#!/bin/sh
set -eu

MODEL="${QW3_MODEL:-../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf}"
AGENT="${QW3_AGENT_BIN:-./qw3-agent}"
CTX="${QW3_CTX:-4096}"

fail() {
    [ ! -f "${out:-}" ] || cat "$out" >&2
    echo "test-agent-coding-smoke: FAIL: $*" >&2
    exit 1
}

[ -x "$AGENT" ] || fail "missing executable: $AGENT"
[ -r "$MODEL" ] || fail "missing model: $MODEL"

project_dir=$(pwd -P)
case "$MODEL" in
    /*) model_path="$MODEL" ;;
    *) model_path="$project_dir/$MODEL" ;;
esac
case "$AGENT" in
    /*) agent_path="$AGENT" ;;
    *) agent_path="$project_dir/${AGENT#./}" ;;
esac

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/qw3-agent-coding.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT
mkdir "$tmpdir/work" "$tmpdir/store"
out="$tmpdir/agent.out"

unset QW3_METAL_KV_Q8_0
export QW3_METAL_KV_F16=1
export QW3_METAL_KERNEL_DIR="${QW3_METAL_KERNEL_DIR:-$project_dir/metal}"

prompt='Work only in the current directory. You MUST call the write tool to create agent_smoke.c with exactly this C program (without Markdown fences):
#include <stdio.h>
int main(void) { puts("QW3_AGENT_CODING_SMOKE"); return 0; }
Then you MUST call the bash tool with exactly: cc agent_smoke.c -o agent_smoke && ./agent_smoke
Do not use bash, Python, sed, or another script to create or edit the file. After the real tool output is available, finish with CODING_SMOKE_OK.'

"$agent_path" -m "$model_path" --ctx "$CTX" --kv-f16 --nothink \
    --temp 0 -n 512 --max-tool-rounds 8 --chdir "$tmpdir/work" \
    --store-dir "$tmpdir/store" -p "$prompt" >"$out" 2>&1 ||
    fail "agent run failed"

grep -q '\[tool\] write' "$out" || fail "write tool was not called"
grep -q '\[tool\] bash' "$out" || fail "bash tool was not called"
grep -q 'QW3_AGENT_CODING_SMOKE' "$out" || fail "program output is missing"
grep -q 'CODING_SMOKE_OK' "$out" || fail "final confirmation is missing"
[ -x "$tmpdir/work/agent_smoke" ] || fail "program was not compiled"

actual=$("$tmpdir/work/agent_smoke")
[ "$actual" = "QW3_AGENT_CODING_SMOKE" ] || fail "compiled program output differs"

expected_source='#include <stdio.h>
int main(void) { puts("QW3_AGENT_CODING_SMOKE"); return 0; }'
actual_source=$(cat "$tmpdir/work/agent_smoke.c")
[ "$actual_source" = "$expected_source" ] ||
    fail "write tool did not preserve the requested source"

echo "test-agent-coding-smoke: ok"
