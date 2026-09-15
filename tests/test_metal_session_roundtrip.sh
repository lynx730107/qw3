#!/bin/sh
set -eu

MODEL="${QW3_MODEL:-../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf}"
BIN="${QW3_METAL_BIN:-./qw3-test}"
CTX="${QW3_CTX:-256}"
PROMPT="Analizza con attenzione questa richiesta tecnica. Elenca i componenti principali del progetto, descrivi il ruolo del motore di inferenza, della cache KV, dello stato ricorrente DeltaNet, dei kernel Metal e del client interattivo. Mantieni la risposta concreta e verifica ogni conclusione prima di procedere."

fail() {
    echo "test-metal-session-roundtrip: FAIL: $*" >&2
    exit 1
}

[ -x "$BIN" ] || fail "missing executable: $BIN"
[ -r "$MODEL" ] || fail "missing model: $MODEL"

tmp=$(mktemp "${TMPDIR:-/tmp}/qw3-session-roundtrip.XXXXXX")
trap 'rm -f "$tmp"' EXIT

unset QW3_METAL_KV_Q8_0
export QW3_METAL_KV_F16=1
if ! "$BIN" -m "$MODEL" --ctx "$CTX" --nothink \
    --session-roundtrip -p "$PROMPT" >"$tmp" 2>&1; then
    cat "$tmp" >&2
    fail "Metal snapshot restore failed"
fi

cat "$tmp"
grep -Eq 'session roundtrip: .* logits=exact decode_steps=4 maxdiff=0 rmsdiff=0 corrupt=rejected incompatible=3$' "$tmp" ||
    fail "restored logits were not bit-identical"

echo "test-metal-session-roundtrip: ok"
