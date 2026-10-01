#!/bin/sh
set -eu
MODEL="${QW3_DENSE_MODEL:-../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf}"
BIN="${QW3_IQ3_STREAM_BIN:-./qw3-iq3-stream-bench}"
out=$(mktemp /tmp/qw3-iq3-stream-quant.XXXXXX)
trap 'rm -f "$out"' EXIT HUP INT TERM
"$BIN" "$MODEL" >"$out" 2>&1 || { cat "$out" >&2; exit 1; }
awk '
    /^stream_quant / {
        checks++; max=""; rms=""; cpu=""; guard="";
        for (i=1;i<=NF;i++) {
            split($i,p,"=");
            if (p[1]=="max") max=p[2];
            if (p[1]=="rms") rms=p[2];
            if (p[1]=="cpu_max") cpu=p[2];
            if (p[1]=="guards") guard=p[2];
        }
        number="^[0-9]+([.][0-9]*)?([eE][-+]?[0-9]+)?$";
        if (max !~ number || rms !~ number || cpu !~ number ||
            max+0>0.0001 || rms+0>0.00001 || cpu+0>0.0001 || guard!="ok") bad=1;
    }
    END { if (checks!=36 || bad) exit 1; }
' "$out" || { cat "$out" >&2; exit 1; }
cat "$out"
echo "test-iq3-stream-quant: ok checks=36 (performance not asserted)"
