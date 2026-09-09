#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
tmp=$(mktemp -d "${TMPDIR:-/tmp}/qw3-sampling.XXXXXX")
trap 'rm -rf "$tmp"' EXIT HUP INT TERM
"${CC:-clang}" -O2 -fobjc-arc tests/test_metal_sampling.m \
    -framework Foundation -framework Metal -o "$tmp/test"
"$tmp/test"
