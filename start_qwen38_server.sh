#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${QW3_MODEL:-$DIR/../../models/Qwen3.8-27B-UD-IQ3_XXS.gguf}"
HOST="${QW3_HOST:-127.0.0.1}"
PORT="${QW3_PORT:-8080}"
CTX="${QW3_CTX:-32768}"

exec "$DIR/qw3-server" \
  -m "$MODEL" \
  --alias qwen3.8-27b \
  --host "$HOST" \
  --port "$PORT" \
  --ctx "$CTX" \
  --kv-f16 \
  --warm-weights \
  --nothink \
  "$@"
