#!/bin/sh
set -eu

MODEL="${QW3_MODEL:-../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf}"
BIN="${QW3_BENCH_BIN:-./qw3-bench}"
N_PROMPT="${QW3_PREFILL_BENCH_TOKENS:-4096}"
REPS="${QW3_PREFILL_BENCH_REPS:-1}"
MIN_TPS="${QW3_PREFILL_MIN_TPS:-450}"

fail() {
    echo "test-prefill-bench-regression: FAIL: $*" >&2
    exit 1
}

[ -x "$BIN" ] || fail "missing executable: $BIN"
[ -r "$MODEL" ] || fail "missing model: $MODEL"

tmp=$(mktemp /tmp/qw3-prefill-bench.XXXXXX)
trap 'rm -f "$tmp"' EXIT

"$BIN" -m "$MODEL" --llama-style -p "$N_PROMPT" -n 0 -r "$REPS" \
    --no-warmup >"$tmp" 2>&1 ||
    fail "qw3-bench failed"

tps=$(awk -F, '$1 == "pp" { print $6; found=1 } END { if (!found) exit 1 }' "$tmp") ||
    fail "could not parse pp avg_tps"

awk -v tps="$tps" -v min="$MIN_TPS" '
    BEGIN {
        if ((tps + 0) < (min + 0)) {
            printf("pp avg_tps %.2f below minimum %.2f\n", tps, min) > "/dev/stderr";
            exit 1;
        }
    }' ||
    fail "prefill throughput regression"

printf "test-prefill-bench-regression: ok pp%s %.2f tok/s (min %.2f)\n" \
    "$N_PROMPT" "$tps" "$MIN_TPS"
