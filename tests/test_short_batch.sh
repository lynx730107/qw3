#!/bin/sh
set -eu

MODEL="${QW3_DENSE_MODEL:-../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf}"
BIN="${QW3_SHORT_BATCH_BIN:-./qw3-short-batch-bench}"
fail() { echo "test-short-batch: FAIL: $*" >&2; exit 1; }
[ -f "$MODEL" ] || fail "model not found: $MODEL"
[ -x "$BIN" ] || fail "binary not found: $BIN"
unset QW3_BLOCK_BASELINE_BATCH QW3_BLOCK_QUANT_ONLY

OUT="$(mktemp -t qw3-short-batch.XXXXXX)"
trap 'rm -f "$OUT"' EXIT HUP INT TERM
run() {
    label="$1"
    block="$2"
    prompt="$3"
    QW3_BLOCK_PROMPT="$prompt" "$BIN" "$MODEL" 1 "$block" >"$OUT" 2>&1 || {
        cat "$OUT" >&2; fail "$label";
    }
    # Independent finite/tolerance checks. Timing is measured, never prescribed.
    awk -v block="$block" '
        function fields() {
            for (i=1; i<=NF; i++) { split($i,p,"="); value[p[1]]=p[2]; }
        }
        BEGIN { number="^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$"; }
        /^short_policy_checks=ok$/ { policy++; }
        /^short_quant / {
            quant++; fields();
            if (value["guards"] != "ok" || value["max"] !~ number ||
                value["rms"] !~ number || value["cpu_max"] !~ number ||
                value["max"]+0 > 0.0001 || value["rms"]+0 > 0.00001 ||
                value["cpu_max"]+0 > 0.0001) bad=1;
        }
        /^block_probe / {
            fields(); if (value["baseline_kind"] != "serial") bad=1;
        }
        /^block_summary / {
            summary++; fields();
            if (value["n"] != block || value["quality_ok"] != "1" ||
                value["last_logits_only"] != "1" ||
                value["worst_max"] !~ number || value["worst_rms"] !~ number ||
                value["worst_max"]+0 > 0.005 || value["worst_rms"]+0 > 0.001 ||
                value["baseline_ms"] !~ number || value["batch_ms"] !~ number ||
                value["baseline_ms"]+0 <= 0 || value["batch_ms"]+0 <= 0) bad=1;
        }
        END { if (policy != 1 || quant != 84 || summary != 1 || bad) exit 1; }
    ' "$OUT" || { cat "$OUT" >&2; fail "$label metrics"; }
    echo "test-short-batch: $label"
    sed -n '/^block_summary /p' "$OUT"
}
CODE='Spiega dettagliatamente come implementare una tabella hash in C.'
run code2 2 "$CODE"
run code4 4 "$CODE"
run code8 8 "$CODE"
run math8 8 'Ho 24 matite e le distribuisco in gruppi uguali da 6. Spiega quanti gruppi ottengo e verifica il risultato.'
run story8 8 'Scrivi un breve racconto su una biblioteca che apre soltanto durante i temporali.'
echo "test-short-batch: ok"
