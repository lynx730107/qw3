# Agent Notes

`qw3.c` is a standalone Qwen3.6-35B-A3B inference engine. Keep it isolated
from the parent `ds4` build: source, build outputs, and future Metal files
belong under this directory unless explicitly shared as reference material.

Reference paths from this directory:

- `../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`: target model.
- `../../models/chat_template.jinja`: local Qwen chat template reference.
- `../../llama.cpp`: upstream/reference implementation for GGUF tensor names,
  tokenizer behavior, Qwen model code, conversion/layout details, and
  quantization behavior.
- `..`: `ds4-main`, primary project reference for architecture, vertical coding
  style, runtime/session integration, CLI/server patterns, and Metal bring-up
  conventions.
-  

Build with `make` from this directory, or `make qw3` from the parent directory.
