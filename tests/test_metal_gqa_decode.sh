#!/bin/sh
set -eu

MODEL="${QW3_MODEL:-../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf}"
BIN="${QW3_METAL_BIN:-./qw3-test}"
CONTEXTS="${QW3_GQA_DECODE_CONTEXTS:-1023 1024 4097 16385}"

fail() {
    echo "test-metal-gqa-decode: FAIL: $*" >&2
    exit 1
}

[ -x "$BIN" ] || fail "missing executable: $BIN"
[ -r "$MODEL" ] || fail "missing model: $MODEL"
[ -n "$CONTEXTS" ] || fail "empty context list"
unset QW3_METAL_KV_Q8_0
export QW3_METAL_KV_F16=1 QW3_METAL_GQA_SPLIT_ATTN=1
tmp=$(mktemp "${TMPDIR:-/tmp}/qw3-gqa-decode.XXXXXX")
trap 'rm -f "$tmp"' EXIT

# Exercise the split threshold, uneven chunks, and the 256-split path.
tested=0
for ctx in $CONTEXTS; do
    case "$ctx" in
        ''|*[!0-9]*) fail "invalid context: $ctx" ;;
    esac
    [ "$ctx" -gt 0 ] || fail "context must be positive"
    if ! "$BIN" -m "$MODEL" --ctx "$ctx" \
        --metal-session-gqa-cached-bench 66 "$ctx" >"$tmp" 2>&1; then
        cat "$tmp" >&2
        fail "CPU/Metal comparison failed at context $ctx"
    fi

    # Validate finite metrics outside the engine's -ffast-math build as well.
    if ! awk -v ctx="$ctx" '
        /^metal session gqa cached bench: ok / {
            found++;
            for (i = 1; i <= NF; i++) {
                split($i, pair, "=");
                if (pair[1] == "n_ctx") actual = pair[2];
                if (pair[1] == "maxdiff") maxdiff = pair[2];
                if (pair[1] == "rmsdiff") rmsdiff = pair[2];
            }
            number = "^[0-9]+([.][0-9]*)?([eE][-+]?[0-9]+)?$";
            if (actual != ctx || maxdiff !~ number || rmsdiff !~ number ||
                maxdiff + 0 > 0.05 || rmsdiff + 0 > 0.005) bad = 1;
            print;
        }
        END { if (found != 1 || bad) exit 1; }
    ' "$tmp"; then
        cat "$tmp" >&2
        fail "missing, non-finite or divergent metrics at context $ctx"
    fi
    tested=$((tested + 1))
done

[ "$tested" -gt 0 ] || fail "empty context list"
echo "test-metal-gqa-decode: ok"
