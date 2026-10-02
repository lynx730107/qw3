#!/bin/sh
set -eu
out=$(mktemp -t qw3-decode-trials.XXXXXX)
trap 'rm -f "$out"' EXIT HUP INT TERM
if ! ./qw3-decode-trials-bench "${QW3_DENSE_MODEL:-../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf}" >"$out" 2>&1; then
    cat "$out" >&2
    exit 1
fi
cat "$out"
awk '
    /^lut_quant / {
        n++; max=""; rms=""; cpu="";
        for(i=1;i<=NF;i++){split($i,a,"=");if(a[1]=="max")max=a[2];if(a[1]=="rms")rms=a[2];if(a[1]=="cpu_max")cpu=a[2];}
        number="^[0-9]+([.][0-9]*)?([eE][-+]?[0-9]+)?$";
        if(max !~ number || rms !~ number || cpu !~ number || max+0!=0 || rms+0!=0 || cpu+0>0.0001 || $0 !~ /guards=ok$/) bad=1;
    }
    END {if(n!=42 || bad)exit 1;}
' "$out"
echo 'test-decode-trials-quant: ok (42 cases)'
