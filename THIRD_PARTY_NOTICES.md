# Third-Party Notices

`qw3` is released under the MIT License. Some code, algorithms, validation
strategy, and implementation ideas were developed with reference to other open
source projects.

## DwarfStar / ds4

This project was inspired by Salvatore Sanfilippo's DwarfStar/ds4 project and
reuses the same broad idea of a small, model-specific local runtime and native
agent.

The parent `ds4-main` project is MIT licensed:

- Copyright (c) 2026 The ds4.c authors
- Copyright (c) 2023-2026 The ggml authors

## llama.cpp / ggml

`llama.cpp` and `ggml` were used as practical references for GGUF loading,
Qwen behavior, quantization conventions, benchmarking expectations, and Metal
implementation direction.

`llama.cpp` is MIT licensed:

- Copyright (c) 2023-2026 The ggml authors

## linenoise

The repository includes `linenoise.c` and `linenoise.h` for terminal line
editing. Those files carry their own BSD-style license header:

- Copyright (c) 2010-2023, Salvatore Sanfilippo
- Copyright (c) 2010-2013, Pieter Noordhuis

Redistribution conditions and disclaimer are preserved in the source files.

## tree-sitter parsers

The `codenavsrc` helper can download tree-sitter and language parsers into
`codenavsrc/third_party` during local builds. Those downloaded sources are not
vendored in this repository and remain governed by their upstream licenses.

## colgrep

`colgrep` is an optional external runtime tool for semantic search. It is not
vendored in this repository.

## Qwen / model weights

Qwen model weights, tokenizer assets, and model cards are not included in this
repository. Users must download model artifacts separately and comply with the
corresponding upstream license and terms.
