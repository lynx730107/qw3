#!/bin/sh
# Build a pinned, isolated audio service; does not install system packages/models.
set -eu
task_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
audio_source="$task_root/.deps/audio.cpp"
audio_commit=41c1f8ffa0d13a1b7e366e30ea5fe5ab8d169d48
if [ "$(uname -s)" != Darwin ]; then
    echo "This setup currently targets macOS/Metal." >&2
    exit 1
fi
for dependency in git cmake; do
    command -v "$dependency" >/dev/null 2>&1 || {
        echo "Missing build dependency: $dependency" >&2
        exit 1
    }
done
if [ ! -e "$audio_source" ]; then
    mkdir -p "$task_root/.deps"
    git -c advice.detachedHead=false clone --depth 1 --branch v0.9.3 \
        https://github.com/0xShug0/audio.cpp.git "$audio_source"
fi
if [ "$(git -C "$audio_source" rev-parse HEAD)" != "$audio_commit" ]; then
    echo "Expected audio.cpp v0.9.3 at $audio_commit; existing checkout was left intact." >&2
    exit 1
fi
if [ -n "$(git -C "$audio_source" status --porcelain --untracked-files=no)" ]; then
    echo "The audio.cpp checkout contains tracked changes; review them before building." >&2
    exit 1
fi
cmake -S "$audio_source" -B "$audio_source/build/qw3-metal" \
    -DCMAKE_BUILD_TYPE=Release \
    -DAUDIOCPP_VERSION=0.9.3 \
    -DAUDIOCPP_MODEL_SET=custom \
    -DAUDIOCPP_MODELS=qwen3_asr,kokoro_tts \
    -DAUDIOCPP_DEPLOYMENT_BUILD=ON \
    -DAUDIOCPP_STATIC_ESPEAK=ON \
    -DENGINE_ENABLE_METAL=ON \
    -DENGINE_ENABLE_CUDA=OFF \
    -DENGINE_ENABLE_VULKAN=OFF \
    -DENGINE_ENABLE_OPENMP=OFF \
    -DGGML_OPENMP=OFF
cmake --build "$audio_source/build/qw3-metal" --parallel "${QW3_AUDIO_BUILD_JOBS:-4}" \
    --target audiocpp_server audiocpp_cli
echo "Audio service built: $audio_source/build/qw3-metal/bin/audiocpp_server"
