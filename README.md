# qw3

`qw3` is a minimalist, zero-dependency C/Objective-C inference runtime for Apple
Silicon (Metal / Metal 4) supporting **Inkling 2 Mini** (text + native Metal
vision), **Qwen3.6-35B-A3B** (MoE), and **Qwen3.8-27B** (dense) GGUF models.

Inspired by Salvatore Sanfilippo's DwarfStar/ds4 philosophy, `qw3` focuses on a
small model set, direct Metal kernels, local checkpoints, native agent tooling,
and a self-contained OpenAI-compatible HTTP server (`qw3-server`). Model
weights are not included; `llama.cpp` serves as the reference for GGUF loading
and numerical validation.

## Supported Models

| Model | Validated GGUF | Backend & Highlights |
| :--- | :--- | :--- |
| **Inkling 2 Mini** (26.4B looped MoE) | `Inkling-2-Mini-UD-Q3_K_XL.gguf` | Metal 4 / NAX prefill (batch 256 default), zero-sync GPU-indexed MoE decode, FP16 KV, zero env vars |
| **Inkling 2 Mini Vision** (`inkling2v`) | `mmproj-Inkling-2-Mini-F16.gguf` | Native Metal ViT (`metal/qw3_vision.metal`) + `ImageIO`/`CoreGraphics`, 2D RoPE, `simdgroup_matrix` FlashAttention, LRU cache |
| **Qwen3.6 35B-A3B** (Hybrid DeltaNet/GQA MoE) | `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` | Full Metal offload, FP16 KV, ~633 tok/s warmed `pp4096` prefill |
| **Qwen3.8 27B** (Hybrid DeltaNet/GQA dense) | `Qwen3.8-27B-UD-Q3_K_XL.gguf`, `UD-IQ3_XXS` | Full Metal offload (`--ngl 64`), staged Metal 4 FFN/GQA prefill, packed8 DeltaNet |

All measurements and tests were run on a fanless **MacBook Air M5 (24 GB
unified memory)** up to **32,768 tokens**. Other quantizations and hardware
configurations should be treated as unvalidated until tested.

## Build

Requires macOS with Xcode Command Line Tools and Python 3:

```sh
make metal      # qw3-cli, qw3-agent, qw3-server (+ vision), qw3-bench, qw3-eval
make cpu        # CPU-only builds (*-cpu)
make voice-setup # Optional local voice mode (see docs/voice.md)
```

## Quick Start

### Inkling 2 Mini (Zero Environment Variables)

```sh
# Single-turn CLI
./qw3-cli --metal -m ../../models/Inkling-2-Mini-UD-Q3_K_XL.gguf \
  --ctx 32768 --nothink --temp 0 -n 128 -p "Reply with OK."

# Interactive coding agent (32K context)
./start-inkling-agent.sh

# OpenAI-compatible HTTP server (+ automatic --mmproj vision if present)
./start-inkling-server.sh
```

### Qwen3.6 / Qwen3.8

```sh
# Qwen3.6 MoE CLI & Agent
./qw3-cli -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 16000 --nothink -p "hello"
./qw3-agent -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --ctx 16000 --nothink

# Qwen3.8 Dense CLI & Agent
./qw3-cli -m ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf --ngl 64 --ctx 2048 --nothink -p "ciao"
./qw3-agent -m ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf --ngl 64 --kv-f16 --ctx 2048 --compact-tools --nothink
```

## OpenAI API Server & Vision (`qw3-server`)

`qw3-server` is a zero-dependency C99 POSIX-socket HTTP/1.1 server built
directly on `qw3.h`, inheriting all Metal 4 / NAX optimizations.

- **Endpoints**: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions`, `POST /tokenize`, CORS `OPTIONS`, optional `--api-key KEY`.
- **Chat & Tools**: SSE streaming (`stream: true`, `stream_options.include_usage`), `timings`, `reasoning_content` split (`reasoning_effort` / `enable_thinking`), OpenAI `tools` / `tool_calls`, `stop` sequences, client disconnect cancellation, and multi-turn KV prefix reuse.
- **Native Metal Vision (`--mmproj`)**: supports `inkling2v` (`F16`, `BF16`, `F32`) with native `ImageIO` decoding (PNG, JPEG, WebP, GIF, HEIC, BMP, TIFF) via `image_url` (`data:image/...;base64,...` or `file://...`). Single command buffer for all 23 ViT layers + projector (`cosine = 0.9999996` vs FP32 CPU). Encoded images are LRU-cached (8 slots) and bound to content-hashed negative sentinels in the session timeline, skipping both ViT encoding (`0 ms`) and visual prefill on follow-up turns.
- **Memory footprint (M5 24 GB, `--ctx 32768`)**: `mmproj-F16` is zero-copy `mmap`-backed (~610 MiB); a `1280×720` image adds ~201 MiB of GPU scratch (`~1.53 GiB` total `phys_footprint` for KV + scratch, 0 swap). Bound max visual tokens with `--vision-max-tokens N` (default `4096`).

```sh
./qw3-server --metal --ctx 32768 --port 8080 \
  -m ../../models/Inkling-2-Mini-UD-Q3_K_XL.gguf \
  --mmproj ../../models/mmproj-Inkling-2-Mini-F16.gguf

# Multimodal OpenAI request
B64=$(base64 -i image.png | tr -d '\n')
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"inkling\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$B64\"}},{\"type\":\"text\",\"text\":\"Describe this image concisely.\"}]}]}"
```

## Performance (Apple M5 Air, 24 GB)

### Inkling 2 Mini (`Inkling-2-Mini-UD-Q3_K_XL.gguf`, FP16 KV)

Measured November 22, 2026 against `llama.cpp` (`b12588-mix-c0c3de8`) with default settings:

| Workload | `qw3` Metal 4 | `llama.cpp` b12588 | Delta |
| :--- | ---: | ---: | ---: |
| Prefill `pp128`, depth 0 | 326.42 tok/s | 198.77 tok/s | +64.2% |
| Prefill `pp512`, depth 0 (default batch 256) | 407.28 tok/s | 216.55 tok/s | +88.1% |
| Prefill `pp512`, depth 0 (`batch=512`) | 438.90 tok/s | 216.55 tok/s | +102.7% |
| 8,283-token needle prompt (`qw3-cli`) | 295.65 tok/s | 127.6–146.0 tok/s | >2.0× |
| Decode `tg32`, depth 0 | 29.65 tok/s | 19.78 tok/s | +49.9% |
| Decode `tg32`, depth 1,024 | 26.29 tok/s | 18.95 tok/s | +38.7% |

`qw3-agent --ctx 32768` also passes tool-call, resume, and 20,234-token needle
tests (128.9 tok/s sustained warm prefill, 1,324.5 MiB peak private footprint,
0 swap).

### Qwen3.6 35B-A3B (`UD-IQ4_XS`, FP16 KV)

| Workload | Context | Result |
| :--- | ---: | ---: |
| Prefill `pp4096`, warmed (`r=3`) | 16,000 | 633.49 tok/s |
| Decode `tg128`, depth 4,096 (`r=3`) | 16,000 | 39.08 tok/s |
| Decode `tg128`, depth 16,384 | 17,000 | 32.84 tok/s |

## Local Agent (`qw3-agent`)

- **Native tools**: `read`, `more`, `list`, `write`, `edit`, `search`, `bash`, `search_web`, `visit_url`, plus optional `get_skeleton` / `get_function` (`codenav`) and `semantic_search` (`colgrep`).
- **Session persistence**: atomic Metal KV/state checkpoints with strict SHA-256 build/kernel fingerprinting (automatic fallback to transcript replay on mismatch) and reusable system+tool prefix caches (`--no-prefix-cache` to disable).
- **Context management**: interruptible prefill (`/interrupt`), automatic tool-output spilling, and context compaction (`/compact`).
- **Commands**: `/help`, `/status`, `/tools on|off`, `/think`, `/nothink`, `/chdir`, `/compact`, `/save`, `/list`, `/switch`, `/del`, `/quit`.

## Validation

```sh
make test-runtime-fingerprint  # Metal source/flag build fingerprint check
make test-inkling-metal        # Inkling logits vs llama.cpp + rewind replay
make test-inkling-window       # Inkling sliding-window ring wrap
make test-server               # 9-part OpenAI API suite (SSE, tools, KV reuse)
make test-server-vision        # 5-part Inkling vision suite (PNG, OCR, JPEG, KV reuse)
make test-regression           # Core Qwen3 Metal regression suite
```

## Documentation

Detailed design notes, kernel maps, and historical M5 experiment records live
in `docs/`:

- [Inkling 2 Mini architecture & roadmap](docs/inkling-2-mini.md)
- [Metal kernel layout & overrides](docs/metal_kernels.md)
- [Metal prefill validation & benchmark history](docs/metal_prefill_validation.md)
- [Local voice mode setup](docs/voice.md)

## License

MIT License. See `LICENSE` and `THIRD_PARTY_NOTICES.md`.
