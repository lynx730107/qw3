#!/bin/sh
set -eu

MODEL="${QW3_DENSE_MODEL:-../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf}"
AGENT="${QW3_AGENT_BIN:-./qw3-agent}"
project_dir=$(pwd -P)
case "$MODEL" in /*) ;; *) MODEL="$project_dir/$MODEL" ;; esac
case "$AGENT" in /*) ;; *) AGENT="$project_dir/${AGENT#./}" ;; esac

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/qw3-agent-python.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT HUP INT TERM
mkdir "$tmpdir/work" "$tmpdir/store"
out="$tmpdir/agent.out"
export QW3_METAL_KERNEL_DIR="${QW3_METAL_KERNEL_DIR:-$project_dir/metal}"

fail() {
    cat "$out" >&2
    echo "test-agent-python-smoke: FAIL: $*" >&2
    exit 1
}

prompt='Use the write tool to create demo.py containing exactly: print("vado a casa".split()). Then execute it with the bash tool using: cd "'"$tmpdir/work"'" && python3 demo.py. This is a didactic whitespace splitter, not the model tokenizer. After successful execution, answer PYTHON_RUN_OK. Do not create other files.'
followup='Which tokens did the script print? Answer exactly: FOLLOWUP_OK: vado, a, casa. Do not call tools or change files.'

# Two turns in one conversation: a successful Python run must release the
# edit guard, including for the subsequent question.
printf '%s\n%s\n/quit\n' "$prompt" "$followup" |
    "$AGENT" --metal -m "$MODEL" --ctx 8192 --kv-f16 --nothink \
        --temp 0 -n 512 --max-tool-rounds 8 --no-prefix-cache \
        --chdir "$tmpdir/work" --store-dir "$tmpdir/store" >"$out" 2>&1 ||
    fail "agent conversation failed"

grep -q '\[tool\] write' "$out" || fail "write was not called"
grep -q '\[tool\] bash' "$out" || fail "bash was not called"
grep -q "\['vado', 'a', 'casa'\]" "$out" || fail "script output missing"
grep -q 'PYTHON_RUN_OK' "$out" || fail "first final answer missing"
grep -q 'FOLLOWUP_OK: vado, a, casa' "$out" || fail "follow-up answer missing"
if grep -Eq 'pending verification|unverified file changes|skipped repeated bash' "$out"; then
    fail "unnecessary verification/repetition guard triggered"
fi
[ "$(grep -c '\[tool\] bash' "$out")" = 1 ] || fail "extra checks executed"
[ "$(python3 "$tmpdir/work/demo.py")" = "['vado', 'a', 'casa']" ] ||
    fail "saved script output differs"
echo "test-agent-python-smoke: ok"
