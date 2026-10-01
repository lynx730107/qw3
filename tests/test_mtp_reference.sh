#!/bin/sh
set -eu

MODEL="${QW3_DENSE_MODEL:-../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf}"
BIN="${QW3_MTP_BIN:-./qw3-mtp-acceptance}"
STEPS="${QW3_MTP_STEPS:-32}"
fail() { echo "test-mtp-reference: FAIL: $*" >&2; exit 1; }
case "$STEPS" in ''|*[!0-9]*) fail "invalid steps" ;; esac
[ "$STEPS" -ge 1 ] && [ "$STEPS" -le 128 ] || fail "steps must be 1..128"
[ -f "$MODEL" ] || fail "model not found: $MODEL"
[ -x "$BIN" ] || fail "binary not found: $BIN"
[ -z "${QW3_MTP_TRACE:-}" ] || fail "unset QW3_MTP_TRACE for uninstrumented runs"

OUT="$(mktemp -t qw3-mtp-reference.XXXXXX)"
trap 'rm -f "$OUT"' EXIT HUP INT TERM
run() {
    label="$1"
    shift
    "$BIN" "$MODEL" "$STEPS" "$@" >"$OUT" 2>&1 || { cat "$OUT" >&2; fail "$label"; }
    # Independent finite checks. Agreement with the target is measured, not
    # prescribed: this is not a speculative acceptance-rate or speedup test.
    awk -v steps="$STEPS" '
        /^native_reference / {
            found++;
            for (i=1; i<=NF; i++) { split($i,p,"="); value[p[1]]=p[2]; }
            number="^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$";
            if (value["greedy_match"] != steps "/" steps ||
                value["rmse"] !~ number || value["max_abs"] !~ number ||
                value["native_mtp_readback_ms"] !~ number ||
                value["rmse"]+0 > 0.001 || value["max_abs"]+0 > 0.005 ||
                value["native_mtp_readback_ms"]+0 <= 0) bad=1;
        }
        END { if (found != 1 || bad) exit 1; }
    ' "$OUT" || { cat "$OUT" >&2; fail "$label metrics"; }
    echo "test-mtp-reference: $label"
    sed -n '/^greedy_agreement=/p; /^native_reference /p' "$OUT"
}
run code
run math 'Risolvi passo per passo: tre scatole contengono rispettivamente 12, 18 e 24 matite. Quanti gruppi uguali si possono formare usando tutte le matite?'
run story 'Scrivi un breve racconto in italiano su una biblioteca che si sveglia di notte.'
echo "test-mtp-reference: ok"
