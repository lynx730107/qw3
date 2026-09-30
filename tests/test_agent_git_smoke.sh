#!/bin/sh
set -eu

MODEL="${QW3_DENSE_MODEL:-../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf}"
AGENT="${QW3_AGENT_BIN:-./qw3-agent}"
project_dir=$(pwd -P)
case "$MODEL" in /*) ;; *) MODEL="$project_dir/$MODEL" ;; esac
case "$AGENT" in /*) ;; *) AGENT="$project_dir/${AGENT#./}" ;; esac
[ -r "$MODEL" ] && [ -x "$AGENT" ] || exit 1

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/qw3-agent-git.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT HUP INT TERM
mkdir "$tmpdir/work" "$tmpdir/store"
out="$tmpdir/agent.out"
trace="$tmpdir/trace.jsonl"
fail() {
    [ ! -f "$out" ] || cat "$out" >&2
    echo "test-agent-git-smoke: FAIL: $*" >&2
    exit 1
}

# Use an isolated repository, not the real project, as the agent's cwd.
git init -q "$tmpdir/work"
for label in FIRST SECOND THIRD; do
    git -C "$tmpdir/work" -c user.name=qw3-test \
        -c user.email=qw3-test@example.invalid \
        -c commit.gpgsign=false -c core.hooksPath=/dev/null commit -q --allow-empty \
        -m "QW3_GIT_SMOKE_$label"
done
expected=$(git -C "$tmpdir/work" log -3 --format=%h)
unset QW3_AGENT_DEBUG_TOOLS QW3_METAL_KV_Q8_0
export QW3_METAL_KERNEL_DIR="${QW3_METAL_KERNEL_DIR:-$project_dir/metal}"

# Use the natural request, default sampling/seed, and the full tool prompt.
# Do not coach the model with a command or a tool-call example.
for cache in cold warm disabled; do
    extra_arg=""
    [ "$cache" != disabled ] || extra_arg="--no-prefix-cache"
    "$AGENT" --metal -m "$MODEL" --ctx 8192 --kv-f16 --nothink \
        -n 384 --max-tool-rounds 4 $extra_arg \
        --chdir "$tmpdir/work" --store-dir "$tmpdir/store" \
        --trace "$tmpdir/$cache.jsonl" \
        -p "dimmi le ultime 3 commit del progetto corrente" >"$out" 2>&1 ||
        fail "$cache conversation failed"
    trace="$tmpdir/$cache.jsonl"
    if grep -q '"event":"empty_response"' "$trace" ||
       grep -q 'retrying\|stopped repeated\|skipped repeated' "$out"; then
        fail "$cache required a retry or loop recovery"
    fi
    grep -q '"event":"tool".*"name":"bash"' "$trace" ||
        fail "$cache did not execute bash"
    grep -q '"event":"session_end","status":0' "$trace" ||
        fail "$cache did not finish successfully"
    # Check the final answer, not merely the git output in the tool response.
    answer=$(awk '/^exit=0$/ {text=""; final=1; next}
                   final {text=text $0 "\n"} END {printf "%s", text}' "$out")
    for id in $expected; do
        printf '%s\n' "$answer" | grep -Fq "$id" ||
            fail "$cache final answer omitted commit $id"
    done
    for label in FIRST SECOND THIRD; do
        printf '%s\n' "$answer" | grep -Fq "QW3_GIT_SMOKE_$label" ||
            fail "$cache final answer omitted a commit message"
    done
    case "$cache" in
        cold) grep -q '"event":"prefix_cache","result":"written"' "$trace" ||
                  fail "cold prefix was not created" ;;
        warm) grep -q '"event":"prefix_cache","result":"hit"' "$trace" ||
                  fail "warm prefix was not reused" ;;
    esac
    echo "test-agent-git-smoke: $cache ok"
done
