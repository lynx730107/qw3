#!/bin/sh
set -eu

MODEL="${QW3_MODEL:-../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf}"
AGENT="${QW3_AGENT_BIN:-./qw3-agent}"
CTX="${QW3_CTX:-4096}"

fail() {
    echo "test-agent-prefix-cache: FAIL: $*" >&2
    exit 1
}

[ -x "$AGENT" ] || fail "missing executable: $AGENT"
[ -r "$MODEL" ] || fail "missing model: $MODEL"

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/qw3-agent-prefix.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT
first="$tmpdir/first.out"
second="$tmpdir/second.out"
corrupt="$tmpdir/corrupt.out"
disabled="$tmpdir/disabled.out"

unset QW3_METAL_KV_Q8_0
export QW3_METAL_KV_F16=1
export QW3_METAL_KERNEL_DIR="${QW3_METAL_KERNEL_DIR:-metal}"

"$AGENT" -m "$MODEL" --ctx "$CTX" --nothink --store-dir "$tmpdir" \
    --dump-prompt -p "PREFIX_CACHE_TEST" >"$first" 2>&1 ||
    fail "initial prefix-cache run failed"

grep -q 'prefix cache created' "$first" || fail "cache was not created"
cache_count=$(find "$tmpdir" -type f -name '.prefix-v*.kvc' | wc -l | tr -d ' ')
[ "$cache_count" = 1 ] || fail "expected one private cache file, found $cache_count"
cache_path=$(find "$tmpdir" -type f -name '.prefix-v*.kvc' -print -quit)

"$AGENT" -m "$MODEL" --ctx "$CTX" --nothink --store-dir "$tmpdir" \
    --dump-prompt -p "PREFIX_CACHE_TEST" >"$second" 2>&1 ||
    fail "prefix-cache resume failed"

grep -q 'prefix cache resumed' "$second" || fail "cache was not resumed"

: >"$cache_path"
"$AGENT" -m "$MODEL" --ctx "$CTX" --nothink --store-dir "$tmpdir" \
    --dump-prompt -p "PREFIX_CACHE_TEST" >"$corrupt" 2>&1 ||
    fail "corrupt prefix-cache recovery failed"

grep -q 'prefix cache created' "$corrupt" || fail "corrupt cache was not rebuilt"
[ -s "$cache_path" ] || fail "rebuilt cache is empty"

"$AGENT" -m "$MODEL" --ctx "$CTX" --nothink --store-dir "$tmpdir" \
    --no-prefix-cache --dump-prompt -p "PREFIX_CACHE_TEST" \
    >"$disabled" 2>&1 || fail "disabled prefix-cache run failed"

if grep -q 'prefix cache' "$disabled"; then
    fail "--no-prefix-cache still used the cache"
fi

echo "test-agent-prefix-cache: ok"
