#!/bin/sh
set -eu

BIN=${BIN:-./qw3-cli}
MODEL=${MODEL:-../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf}
CTX=${CTX:-32000}
TARGET_MEMORY=${TARGET_MEMORY:-16gb}
PROMPT=${PROMPT:-datasets/synthetic-c/mixed_profile_prompt.txt}
PROFILE=${PROFILE:-profiles/synthetic-c.tsv}
PRELOAD=${PRELOAD:-1024}
GEN=${GEN:-0}
OUTDIR=${OUTDIR:-bench/ssd-streaming-$(date -u +%Y%m%dT%H%M%SZ)}

if pgrep -fl '[/]qw3-(cli|test|agent|bench)' >/dev/null 2>&1; then
    echo "qw3: refusing to start benchmark while another qw3 model process is active" >&2
    pgrep -fl '[/]qw3-(cli|test|agent|bench)' >&2 || true
    exit 3
fi

if [ ! -x "$BIN" ]; then
    echo "qw3: $BIN not found or not executable; run make first" >&2
    exit 2
fi

if [ ! -f "$MODEL" ]; then
    echo "qw3: model not found: $MODEL" >&2
    exit 2
fi

if [ ! -f "$PROMPT" ]; then
    echo "qw3: prompt not found: $PROMPT" >&2
    echo "qw3: run: make code-profile-c-dataset" >&2
    exit 2
fi

mkdir -p "$OUTDIR"

summarize_log() {
    log=$1
    grep -E 'prefill=|SSD streaming cache stats|SSD streaming cache churn|preload reuse|working set exceeds' "$log" || true
}

run_case() {
    label=$1
    shift
    log="$OUTDIR/$label.log"
    echo "== $label =="
    "$BIN" -m "$MODEL" \
        --ctx "$CTX" --nothink \
        --target-memory "$TARGET_MEMORY" \
        --prompt-file "$PROMPT" -n "$GEN" \
        "$@" >"$log" 2>&1
    summarize_log "$log"
    echo "log: $log"
    echo
}

run_case built-in
run_case cold --ssd-streaming-cold

if [ -f "$PROFILE" ]; then
    run_case external-hotlist \
        --streaming-hotlist "$PROFILE" \
        --streaming-preload "$PRELOAD"

    if [ "${RUN_HOT_EVICTION_AB:-0}" = "1" ]; then
        label=external-hotlist-lru
        log="$OUTDIR/$label.log"
        echo "== $label =="
        QW3_METAL_STREAMING_HOT_EVICTION_DISABLE=1 "$BIN" -m "$MODEL" \
            --ctx "$CTX" --nothink \
            --target-memory "$TARGET_MEMORY" \
            --streaming-hotlist "$PROFILE" \
            --streaming-preload "$PRELOAD" \
            --prompt-file "$PROMPT" -n "$GEN" >"$log" 2>&1
        summarize_log "$log"
        echo "log: $log"
        echo
    fi
else
    echo "qw3: skipping external-hotlist; profile not found: $PROFILE" >&2
fi

echo "logs written to $OUTDIR"
