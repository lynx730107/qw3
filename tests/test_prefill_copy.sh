#!/bin/sh
set -eu
model="${QW3_DENSE_MODEL:-../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf}"
out=$(mktemp -t qw3-copy.XXXXXX)
trap 'rm -f "$out"' EXIT HUP INT TERM
if ! ./qw3-prefill-staging-bench "$model" production_copy >"$out" 2>&1; then
    cat "$out" >&2
    exit 1
fi
cat "$out"
awk '
    /^copy_quant / { copies++; if ($0 !~ /guards=ok bits_exact=1$/) bad=1; }
    /^prefill_pair / { pairs++; if ($0 !~ /logits_exact=1 continuation=8$/) bad=1; }
    /^prefill_pp128 / {
        summary++;
        number="^[-+]?[0-9]+([.][0-9]*)?([eE][-+]?[0-9]+)?$";
        metrics=0;
        for (i=1;i<=NF;i++) {
            split($i,p,"=");
            if (p[1] ~ /^(baseline_tps|candidate_tps|gain_pct|saved_ms|ci95_ms)$/) {
                metrics++; if (p[2] !~ number) bad=1;
            }
        }
        if (metrics!=5 || $0 !~ /logits_exact=1$/) bad=1;
    }
    /^prefill_state deltanet=exact conv=exact live_kv=exact tokens=136$/ { states++; }
    END { if (copies!=10 || pairs!=8 || summary!=1 || states!=1 || bad) exit 1; }
' "$out"
echo 'test-prefill-copy: ok'
