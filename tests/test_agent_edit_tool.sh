#!/bin/sh
set -eu

BIN="${QW3_AGENT_BIN:-./qw3-agent}"

fail() {
    echo "test-agent-edit: FAIL: $*" >&2
    exit 1
}

[ -x "$BIN" ] || fail "missing executable: $BIN"

tmpdir=$(mktemp -d /tmp/qw3-agent-edit.XXXXXX)
trap 'rm -rf "$tmpdir"' EXIT

tool_file="$tmpdir/tool.xml"
target="$tmpdir/target.txt"

run_tool() {
    "$BIN" --tool-native-file "$tool_file" 2>&1
}

cat > "$target" <<'EOF'
alpha
TARGET = old
omega
EOF

cat > "$tool_file" <<EOF
<tool_call>
<function=edit>
<parameter=path>$target</parameter>
<parameter=old>TARGET = old</parameter>
<parameter=new>TARGET = new</parameter>
</function>
</tool_call>
EOF
run_tool >/dev/null
grep -q '^TARGET = new$' "$target" || fail "replace did not update target line"

cat > "$target" <<'EOF'
old one
keep
old two
EOF

cat > "$tool_file" <<EOF
<tool_call>
<function=edit>
<parameter=path>$target</parameter>
<parameter=mode>replace_all</parameter>
<parameter=old>old</parameter>
<parameter=new>new</parameter>
</function>
</tool_call>
EOF
run_tool >/dev/null
[ "$(grep -c '^new ' "$target")" -eq 2 ] || fail "replace_all count mismatch"
! grep -q '^old ' "$target" || fail "replace_all left old text"

cat > "$target" <<'EOF'
all: qw3-cli qw3-agent
	cc foo
EOF

cat > "$tool_file" <<EOF
<tool_call>
<function=edit>
<parameter=path>$target</parameter>
<parameter=mode>insert_after</parameter>
<parameter=anchor>all: qw3-cli qw3-agent</parameter>
<parameter=new>tools: codenav</parameter>
</function>
</tool_call>
EOF
run_tool >/dev/null
sed -n '2p' "$target" | grep -q '^tools: codenav$' ||
    fail "insert_after did not insert after anchor"

cat > "$tool_file" <<EOF
<tool_call>
<function=edit>
<parameter=path>$target</parameter>
<parameter=mode>insert_before</parameter>
<parameter=anchor>tools: codenav</parameter>
<parameter=new>.PHONY: tools</parameter>
</function>
</tool_call>
EOF
run_tool >/dev/null
sed -n '2p' "$target" | grep -q '^.PHONY: tools$' ||
    fail "insert_before did not insert before anchor"

cat > "$target" <<'EOF'
middle
EOF

cat > "$tool_file" <<EOF
<tool_call>
<function=edit>
<parameter=path>$target</parameter>
<parameter=mode>prepend</parameter>
<parameter=new>first</parameter>
</function>
</tool_call>
EOF
run_tool >/dev/null

cat > "$tool_file" <<EOF
<tool_call>
<function=edit>
<parameter=path>$target</parameter>
<parameter=mode>append</parameter>
<parameter=new>last</parameter>
</function>
</tool_call>
EOF
run_tool >/dev/null
sed -n '1p' "$target" | grep -q '^first$' || fail "prepend failed"
tail -n 1 "$target" | grep -q '^last$' || fail "append failed"

cat > "$target" <<'EOF'
all: qw3-cli qw3-agent
	cc foo
EOF

cat > "$tool_file" <<EOF
<tool_call>
<function=edit>
<parameter=path>$target</parameter>
<parameter=old>all: qw3 qw3-agent</parameter>
<parameter=new>all: qw3-cli qw3-agent</parameter>
</function>
</tool_call>
EOF
out=$(run_tool)
echo "$out" | grep -q 'error: old text not found' ||
    fail "missing old-text error"
echo "$out" | grep -q 'nearby candidate lines:' ||
    fail "missing candidate-line hint"
echo "$out" | grep -q 'all: qw3-cli qw3-agent' ||
    fail "missing candidate line"

cat > "$tool_file" <<EOF
<tool_call>
<function=edit>
<parameter=path>$target</parameter>
<parameter=mode>sideways</parameter>
<parameter=new>x</parameter>
</function>
</tool_call>
EOF
out=$(run_tool)
echo "$out" | grep -q 'error: unsupported edit mode' ||
    fail "missing unsupported-mode error"

echo "test-agent-edit: ok"
