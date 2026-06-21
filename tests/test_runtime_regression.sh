#!/bin/sh
set -eu

MODEL="${QW3_MODEL:-../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf}"
CLI="${QW3_CLI_BIN:-./qw3-cli}"
AGENT="${QW3_AGENT_BIN:-./qw3-agent}"
CTX="${QW3_CTX:-4096}"

fail() {
    echo "test-runtime-regression: FAIL: $*" >&2
    exit 1
}

[ -x "$CLI" ] || fail "missing executable: $CLI"
[ -x "$AGENT" ] || fail "missing executable: $AGENT"
[ -r "$MODEL" ] || fail "missing model: $MODEL"

tmpdir=$(mktemp -d /tmp/qw3-runtime-regression.XXXXXX)
trap 'rm -rf "$tmpdir"' EXIT

cli_out="$tmpdir/cli.out"
agent_out="$tmpdir/agent.out"
tool_file="$tmpdir/bash_tool.xml"

"$CLI" -m "$MODEL" --ctx "$CTX" --nothink -p "ciao" -n 48 >"$cli_out" 2>&1 ||
    fail "qw3-cli failed"

grep -q 'Ciao' "$cli_out" ||
    fail "qw3-cli did not produce the expected greeting"
grep -Eq 'aiut|assist|help' "$cli_out" ||
    fail "qw3-cli greeting looked malformed"
if grep -Eq 'non-specificazione|_BINARY|一直|った|�' "$cli_out"; then
    fail "qw3-cli output contains known garbage markers"
fi

cat >"$tool_file" <<'EOF'
<tool_call>
<function=bash>
<parameter=cmd>printf 'qw3-tool-ok'</parameter>
</function>
</tool_call>
EOF

"$AGENT" --tool-native-file "$tool_file" >"$agent_out" 2>&1 ||
    fail "qw3-agent native bash tool failed"

grep -q '\[tool\] bash' "$agent_out" ||
    fail "qw3-agent did not call the bash tool"
grep -q 'qw3-tool-ok' "$agent_out" ||
    fail "qw3-agent tool result missing"
if grep -Eq 'non-specificazione|_BINARY|一直|った|�' "$agent_out"; then
    fail "qw3-agent output contains known garbage markers"
fi

echo "test-runtime-regression: ok"
