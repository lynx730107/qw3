#!/bin/sh
set -eu

MODEL="${QW3_DENSE_MODEL:-../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf}"
BIN="${QW3_MTP_DECODE_BIN:-./qw3-mtp-decode}"
fail() { echo "test-mtp-decode: FAIL: $*" >&2; exit 1; }
[ -f "$MODEL" ] || fail "model not found: $MODEL"
[ -x "$BIN" ] || fail "binary not found: $BIN"
OUT="$(mktemp -t qw3-mtp-decode.XXXXXX)"
trap 'rm -f "$OUT"' EXIT HUP INT TERM
run() {
    label="$1"
    steps="$2"
    shift 2
    "$BIN" "$MODEL" "$steps" 8 serial "$@" >"$OUT" 2>&1 || { cat "$OUT" >&2; fail "$label"; }
    awk -v steps="$steps" '
        /^mtp_summary / {
            found++;
            for (i=1; i<=NF; i++) { split($i,p,"="); value[p[1]]=p[2]; }
            if (value["tokens"] != steps || value["matched"] != steps ||
                value["serial_state_exact"] != 1 || value["continuation"] != 8 ||
                value["all_rows"]+0 < steps || value["max"] != 0 ||
                value["rms"] != 0 || value["hidden_max"] != 0 || value["hidden_rms"] != 0 ||
                value["rollback_checks"] != value["rejected_cycles"]) bad=1;
        }
        END { if (found != 1 || bad) exit 1; }
    ' "$OUT" || { cat "$OUT" >&2; fail "$label metrics"; }
    case "$label" in
        reject-*)
            boundary="${label#reject-}"
            sed -n '/^mtp_cycle=1 /p' "$OUT" | \
                awk -v expected="$boundary" '{
                    for(i=1;i<=NF;i++){split($i,p,"=");v[p[1]]=p[2]}
                    if(v["committed"] != expected || v["rejected"] != 1) exit 1;
                    found++;
                } END {if(found != 1) exit 1;}' || fail "$label boundary"
            ;;
    esac
    echo "test-mtp-decode: $label"
    sed -n '/^mtp_summary /p' "$OUT"
}
run code 32 'Spiega dettagliatamente come implementare una tabella hash in C.'
run math 16 'Risolvi passo per passo: tre scatole contengono rispettivamente 12, 18 e 24 matite. Quanti gruppi uguali si possono formare usando tutte le matite?'
run story 16 'Scrivi un breve racconto in italiano su una biblioteca che si sveglia di notte.'
# First, middle and last proposal; the last crosses the 32-token F16 tail
# boundary for this prompt. Fault injection is not acceptance-rate evidence.
for boundary in 1 4 7; do
    run "reject-$boundary" 8 'Spiega dettagliatamente come implementare una tabella hash in C.' "$boundary"
done
echo 'test-mtp-decode: ok'
