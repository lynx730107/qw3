#!/bin/sh
set -eu

BIN="${QW3_AGENT_BIN:-./qw3-agent}"

fail() {
    echo "test-agent-trace: FAIL: $*" >&2
    exit 1
}

[ -x "$BIN" ] || fail "missing executable: $BIN"

tmpdir=$(mktemp -d /tmp/qw3-agent-trace.XXXXXX)
trap 'rm -rf "$tmpdir"' EXIT
trace="$tmpdir/agent.jsonl"
tool_file="$tmpdir/tool.xml"

cat > "$tool_file" <<'EOF'
<tool_call>
<function=bash>
<parameter=cmd>printf 'TRACE_TOOL_OK\n'</parameter>
</function>
</tool_call>
EOF

"$BIN" --trace "$trace" --tool-native-file "$tool_file" >/dev/null 2>&1

[ -f "$trace" ] || fail "trace file was not created"
[ "$(stat -f '%Lp' "$trace")" = "600" ] || fail "trace is not private"
grep -q '"event":"session_start"' "$trace" ||
    fail "session_start event is missing"
grep -q '"event":"tool".*"name":"bash"' "$trace" ||
    fail "tool event is missing"
grep -q '"output_bytes":[1-9]' "$trace" ||
    fail "tool output size was not recorded"
if grep -v '}$' "$trace" >/dev/null; then
    fail "trace contains an incomplete JSONL record"
fi

echo "test-agent-trace: ok"
