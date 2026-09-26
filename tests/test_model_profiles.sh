#!/bin/sh
set -eu

MOE_MODEL="${QW3_MOE_MODEL:-../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf}"
DENSE_MODEL="${QW3_DENSE_MODEL:-../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf}"
BIN="${QW3_PROFILE_BIN:-./qw3-cpu-test}"
PUBLIC_BIN="${QW3_PROFILE_PUBLIC_BIN:-./qw3-cpu}"

fail() {
    echo "test-model-profiles: FAIL: $*" >&2
    exit 1
}

[ -x "$BIN" ] || fail "missing executable: $BIN"
[ -x "$PUBLIC_BIN" ] || fail "missing executable: $PUBLIC_BIN"
[ -r "$MOE_MODEL" ] || fail "missing MoE model: $MOE_MODEL"
[ -r "$DENSE_MODEL" ] || fail "missing dense model: $DENSE_MODEL"

moe=$($BIN --cpu -m "$MOE_MODEL" --inspect -n 0 -p test 2>/dev/null)
echo "$moe" | grep -q 'profile: Qwen3.6-35B-A3B ffn=moe trunk_layers=40 nextn_layers=0' ||
    fail "MoE profile mismatch"

dense=$($BIN --cpu -m "$DENSE_MODEL" --inspect -n 0 -p test 2>/dev/null)
echo "$dense" | grep -q 'profile: Qwen3.8-27B ffn=dense trunk_layers=64 nextn_layers=1' ||
    fail "dense profile mismatch"
echo "$dense" | grep -q 'dense_ffn: dim=17408' ||
    fail "dense FFN metadata mismatch"
echo "$dense" | grep -q 'tensor_types:.*q3_K=12.*iq3_s=111.*iq4_xs=156' ||
    fail "dense tensor type inventory mismatch"

public_dense=$($PUBLIC_BIN --cpu -m "$DENSE_MODEL" --inspect 2>/dev/null)
echo "$public_dense" | grep -q 'profile: Qwen3.8-27B ffn=dense trunk_layers=64 nextn_layers=1' ||
    fail "public --inspect does not recognize dense profile"

layer0=$($BIN --cpu -m "$DENSE_MODEL" --layer-types 0 -n 0 -p test 2>/dev/null)
echo "$layer0" | grep -q 'layer 0: kind=deltanet' ||
    fail "dense layer 0 kind mismatch"
echo "$layer0" | grep -q 'gate=iq2_xs up=iq2_s down=iq3_xxs' ||
    fail "dense layer 0 FFN types mismatch"

mtp=$($BIN --cpu -m "$DENSE_MODEL" --layer-types 64 -n 0 -p test 2>/dev/null)
echo "$mtp" | grep -q 'layer 64: kind=gqa nextn/mtp' ||
    fail "dense MTP layer mismatch"

echo "test-model-profiles: ok"
