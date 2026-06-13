# qw3

`qw3` is a small experimental C/Objective-C runtime for Qwen3.6 35B-A3B GGUF
models, with an Apple Metal backend, benchmark tools, evaluation code, and an
integrated local agent.

The project started as an attempt to reproduce, on a smaller and narrower
scale, some of the ideas behind Salvatore Sanfilippo's DwarfStar/ds4 project:
a minimal runtime, tight model-specific integration, local sessions, native
tools, and low latency without an HTTP server boundary. It is not a replacement
for llama.cpp and does not aim for the same generality. It is a focused
laboratory for this model family and Apple Silicon.

This experiment would not have been possible without GPT-5.5-assisted
development and llama.cpp as the practical reference implementation for GGUF
loading, Qwen behavior, Metal performance expectations, and validation
discipline.

## Status

- Main backend: Metal on macOS/Apple Silicon.
- Target model: `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`.
- The only supported GGUF quantization is currently Unsloth's `IQ4_XS`
  release from Hugging Face.
- Metal generation: stable.
- Metal prefill: still being optimized; the local `pp4096` benchmark is around
  600 tok/s on the tested Apple M5 setup under the conditions documented in
  `docs/metal_prefill_validation.md`.
- Agent: usable, with local tools, context compaction, and code-navigation
  helpers.

All tests and performance measurements so far were run only on a MacBook Air
M5 with 24 GB of unified memory. Because of that hardware limit, the largest
context tested in practice is 32,000 tokens. Larger contexts, different Apple
Silicon machines, and other memory sizes should be considered unvalidated until
tested.

Model weights are not included in this repository. At the moment, other GGUF
quantizations should be considered unsupported unless explicitly validated.

## Build

On macOS with Xcode Command Line Tools:

```sh
make
```

The default Darwin build produces Metal binaries without a `metal` suffix:

```text
qw3        generation CLI
qw3-agent  local coding agent
qw3-bench  benchmark tool
qw3-eval   evaluation tool
```

CPU-only targets are still available explicitly:

```sh
make cpu
```

Clean generated files:

```sh
make clean
```

## Quick Start

Generation:

```sh
./qw3 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
  --ctx 16000 --nothink -p "hello"
```

Interactive agent:

```sh
./qw3-agent -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
  --ctx 16000 --nothink
```

llama-bench-style benchmark:

```sh
./qw3-bench -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
  --llama-style -p 4096 -n 0 -r 3 --no-warmup
```

Metal logits regression:

```sh
make test-metal-logits
```

## Agent

`qw3-agent` uses Qwen-native tool calling and keeps the session local. The main
tools are:

- `read`, `more`, `list`: file reading and navigation.
- `write`, `edit`: file creation and modification.
- `search`: text search.
- `bash`: local shell commands.
- `get_skeleton`: file outline through `codenav`.
- `get_function`: function or method body through `codenav`.
- `semantic_search`: semantic code search through `colgrep`.

To avoid filling the context by reading large source files, the agent is
instructed to prefer `get_skeleton`, `get_function`, and `semantic_search`
before sequential file reads.

Useful interactive commands:

```text
/help
/status
/tools on
/tools off
/think
/nothink
/compact
/quit
```

## External Agent Tools

The `codenav` sources are included in the repository. The target downloads the
required tree-sitter parsers into `codenavsrc/third_party` when needed and then
builds the binary:

```sh
make tools
export PATH="$PWD/codenavsrc:$PATH"
codenav get_skeleton qw3_agent.c
```

`colgrep` is not vendored. Install it from its upstream channel and make sure
it is available in `PATH`:

```sh
colgrep --help
colgrep init .
```

Without `codenav` or `colgrep`, the agent still runs, but `get_skeleton`,
`get_function`, and `semantic_search` will return tool errors.

## Metal Notes

Useful options:

- `--ctx N`: context size.
- `--kv-f16`: f16 KV cache.
- `--ngl N`: number of layers executed on Metal; the remaining layers use the
  CPU path.
- `--nothink`: disable thinking mode in the prompt.

The q8 KV cache and several experimental environment flags should still be
considered unstable. Before promoting any Metal optimization, run logits
regressions and a no-garbage test on a real prompt.

## Validation

Minimum checks before trusting a Metal change:

```sh
make
make test-metal-logits
./qw3 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
  --ctx 16000 --nothink --prompt-file ./prompt_perf.txt -n 128
```

Development notes and benchmark history are kept in:

```text
docs/metal_prefill_validation.md
```

## License

`qw3` is released under the MIT License. See `LICENSE` and
`THIRD_PARTY_NOTICES.md` for attribution notes covering ds4, llama.cpp/ggml,
linenoise, optional tree-sitter parsers, colgrep, and external Qwen model
artifacts.
