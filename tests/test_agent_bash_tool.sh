#!/bin/sh
set -eu

BIN="${QW3_AGENT_BIN:-./qw3-agent}"

fail() {
    echo "test-agent-bash: FAIL: $*" >&2
    exit 1
}

[ -x "$BIN" ] || fail "missing executable: $BIN"

tmpdir=$(mktemp -d /tmp/qw3-agent-bash.XXXXXX)
trap 'rm -rf "$tmpdir"' EXIT
store="$tmpdir/store"
tool_file="$tmpdir/tool.xml"
mkdir -p "$store"

run_tool() {
    "$BIN" --store-dir "$store" --tool-native-file "$tool_file" 2>&1
}

contains() {
    case "$1" in
        *"$2"*) return 0 ;;
        *) return 1 ;;
    esac
}

cat > "$tool_file" <<'EOF'
<tool_call>
<function=bash>
<parameter=cmd>
printf 'STDOUT_MARKER\n'
printf 'STDERR_MARKER\n' >&2
exit 3
</parameter>
</function>
</tool_call>
EOF
out=$(run_tool)
contains "$out" 'STDOUT_MARKER' || fail "stdout was not captured"
contains "$out" 'STDERR_MARKER' || fail "stderr was not captured"
contains "$out" 'exit=3' || fail "non-zero exit status was lost"

cat > "$tool_file" <<'EOF'
<tool_call>
<function=bash>
<parameter=cmd>
i=0
while [ "$i" -lt 5000 ]; do
    printf 'BODY-%04d-xxxxxxxxxxxxxxxx\n' "$i"
    i=$((i + 1))
done
printf 'TAIL_MARKER\n'
</parameter>
</function>
</tool_call>
EOF
out=$(run_tool)
contains "$out" 'BODY-0000-' || fail "large-output head is missing"
contains "$out" 'TAIL_MARKER' || fail "large-output tail is missing"
contains "$out" 'bytes omitted; full output saved to' ||
    fail "large output was not spilled"
spill=$(echo "$out" |
    sed -n 's/.*full output saved to \([^ ]*\)\. Use.*/\1/p' |
    head -n 1)
[ -n "$spill" ] || fail "spill path was not reported"
[ -f "$spill" ] || fail "reported spill file does not exist"
grep -q 'BODY-2500-' "$spill" || fail "spill lost its middle"
grep -q 'TAIL_MARKER' "$spill" || fail "spill lost its tail"
mode=$(stat -f '%Lp' "$spill")
[ "$mode" = "600" ] || fail "spill mode is $mode, expected 600"

i=0
while [ "$i" -lt 10 ]; do
    run_tool >/dev/null
    i=$((i + 1))
done
spill_count=$(find "$store/spills" -type f -name 'bash-*' | wc -l | tr -d ' ')
[ "$spill_count" -le 8 ] ||
    fail "spill retention kept $spill_count files, expected at most 8"

cat > "$tool_file" <<'EOF'
<tool_call>
<function=bash>
<parameter=cmd>sleep 5</parameter>
</function>
</tool_call>
EOF
started=$(date +%s)
out=$(QW3_AGENT_BASH_TIMEOUT_SEC=1 run_tool)
elapsed=$(($(date +%s) - started))
contains "$out" 'timed_out=1s' || fail "timeout status is missing"
[ "$elapsed" -le 4 ] || fail "timeout took ${elapsed}s"

echo "test-agent-bash: ok"
