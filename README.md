# qw3

`qw3` is a small experimental C/Objective-C runtime for Qwen3.6 35B-A3B and
Qwen3.8 27B GGUF models, with an Apple Metal backend, benchmark tools,
evaluation code, and an integrated local agent.

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
- Validated GGUFs: `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` and
  `Qwen3.8-27B-UD-Q3_K_XL.gguf`. Other quantizations remain unvalidated.
- Qwen3.6 Metal generation is stable; its warmed `pp4096` prefill reaches
  about 633 tok/s on the tested Apple M5 setup.
- Qwen3.8 CPU and full-offload Metal decode pass differential logits and
  eight-step greedy tests. Dense Metal prefill batches FFN projections using
  IQ3_S/IQ4_XS in groups of up to 128 tokens, including mixed layers with
  two supported projections. Attention and other quantizations still use
  tokenwise projections.
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

On macOS with Xcode Command Line Tools and Python 3 (for the Metal build identity):

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
./qw3-cli -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
  --ctx 16000 --nothink -p "hello"
```

Qwen3.8 dense generation (all 64 trunk layers on Metal; F16 KV by default):

```sh
./qw3-cli -m ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf \
  --ngl 64 --ctx 32 --nothink -p "ciao" -n 1
```

Use `--cpu` if full Metal offload is unavailable. Dense Q8 KV is not yet
validated; use F16 or `--kv-f32`.

Interactive agent:

```sh
./qw3-agent -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
  --ctx 16000 --nothink
```

Qwen3.8 dense agent (full Metal offload required):

```sh
./qw3-agent -m ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf \
  --ngl 64 --kv-f16 --ctx 2048 --compact-tools --nothink
```

`--compact-tools` advertises only `read`, `write`, `edit`, and `bash` to keep
the initial prompt shorter while dense prefill remains expensive. The full
tool set remains the default. Dense agent generation, checkpoint restore, and
prefix-cache restore are covered by `make test-dense-agent`.

llama-bench-style benchmark:

```sh
./qw3-bench -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
  --llama-style -p 4096 -n 0 -r 3 --no-warmup
```

Metal logits regression:

```sh
make test-metal-logits
```

Qwen3.8 dense Metal regression (all GGUF matvec types, representative complete
DeltaNet/GQA layers, 64-layer F16/F32 logits, eight greedy tokens, CLI
generation, and session roundtrip):

```sh
make test-dense-metal-bringup
```

Dense Metal decode is functional. Per-layer prefill batches FFN projections
in 61 of 64 layers with IQ3_S, IQ4_XS, IQ2_S, or IQ3_XXS. On Metal 4 and
batches of at least 16 tokens, Q3_K, Q4_K, and Q5_K also use tiled tensor
matmuls, raising FFN batch coverage to 63 layers. Smaller batches and devices
without the Tensor API retain the tokenwise path for these formats. Set
`QW3_METAL_DENSE_Q3_K_BATCH=0` to disable the new path, or `=1` to force
the experimental four-token fallback. Set
`QW3_METAL_DENSE_Q4_Q5_K_BATCH=0` to disable Q4_K/Q5_K batch. Compatible
GQA Q/K/V projections are also batched for at least 16 tokens; set
`QW3_METAL_DENSE_GQA_BATCH=0` for an A/B control or `=1` to force smaller
batches. GQA causal attention and recurrent state updates remain sequential.

On the fanless Apple M5 test machine, the accumulated dense batch kernels and
per-layer command batching measured 108.95 tok/s for `pp128` versus 97.66
tok/s with `QW3_METAL_DENSE_LAYER_BATCH=0` in a matched three-repetition A/B.
Four-simdgroup staged prefill kernels now reuse each dequantized 64x32 weight
tile across 128 input rows for Q3_K, Q4_K, Q5_K, IQ2_S, IQ3_XXS, IQ3_S, and
IQ4_XS. Interleaved runs improved the default from about 109.4-110.2 tok/s to
136.3-137.6 tok/s. Set `QW3_METAL_STAGED128=0` to select the previous 64-row
cooperative kernels. Per-layer batching
keeps the individual compute/blit encoders ordered in one retained Metal
command buffer and synchronizes once after the FFN.

Full 128-token chunks now also use separate staged IQ3_S gate/up projections
instead of the older 32-token mapped pair tile, preserving its F16 activation
rounding. Matched warmed seven-sample `pp128` measurements were 142.66 ± 0.60
tok/s versus 136.57 ± 0.89 tok/s with the pair tile (about 4.5% higher
throughput). Smaller chunks keep the pair tile.
Set `QW3_METAL_DENSE_PREFILL_IQ3_PAIR=1` to restore paired prefill for A/B runs;
`=0` forces separate projections. Decode's pair setting is independent.
Individual cold runs vary; these are not sustained-speed guarantees.
The earlier local `llama.cpp` baseline measured
159.22 ± 1.13 tok/s for `pp128` with the same GGUF and full GPU offload, so
the remaining gap is still substantial. Set
`QW3_METAL_DENSE_LAYER_BATCH=0` to restore separate stage submissions. The
128-token batch-vs-tokenwise regression preserves the greedy choice and
next-token continuation (latest default: logit max difference 0.02285, RMS
0.00456; DeltaNet state max difference 0.00076).
Reproduce the `qw3` measurement with:

```sh
./qw3-bench -m ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf --ngl 64 \
  --llama-style -p 128 -n 32 -r 1 --no-warmup --ctx-alloc 256
```

Set `QW3_METAL_DENSE_PREFILL_BATCH=1` to disable dense batching for comparison,
or a value from 2 to 128 to choose the batch size. The next speed step is
reducing the remaining quantized-matmul and DeltaNet cost while preserving the
single-command-buffer layer path.
Set
`QW3_METAL_DENSE_PROFILE=1` for per-stage decode diagnostics (it synchronizes
between stages and therefore slows the run).
Set `QW3_METAL_DENSE_PROBE_REPS=20` when running `--metal-session-embed-test`
to time repeated quantized projections separately from their first invocation.
The same option now times warmed input/inner matvecs and IQ3_S gate/up pairs with
`--metal-session-quant-types-test`.

The two-column DeltaNet prefill kernel now retains FP32 state in registers
through the chunk and stores it once, preserving the state layout and math.
A matched five-repetition `pp128` comparison improved from 144.55 to 152.84
tok/s (+5.7%). Greedy continuation and session round-trip regressions pass
without wider tolerances.

Dense decode now reuses vector-loaded IQ4_XS activations and workgroup-local
lookup tables for IQ4_XS and IQ3_S matvecs. Short warmed M5 measurements at
depth 128 (`tg32`, ctx 256) improved from 6.74-6.76 to 7.16-7.21 tok/s;
the same-GGUF llama.cpp reference measured 8.88 tok/s. Longer repeated runs
showed baseline drift, so these observations are not sustained-speed promises.
IQ3_S gate/up pairs additionally use exact, expanded FP32 lookup magnitudes;
single matvec retains the smaller packed lookup table. The warmed pair probe
improved from 0.9682 to 0.8448 ms, but this is not a whole-model decode result.
Grouped FlashAttention also supports dense 6:1 Q/KV heads, automatically from
8192 context tokens. `QW3_METAL_GQA_FLASH_DECODE_GROUPED=0` disables this path;
`=1` permits it from 1024 tokens for testing. The MoE 8:1 threshold is unchanged.
Use `make test-dense-gqa-decode` for CPU comparisons at threshold and tail
boundaries with non-constant K/V input. See
[the dated measurement record](docs/dense_metal_perf_2026-09-30.md) for A/B
results, current llama.cpp comparisons, and rejected experiments.
FP32 activations/accumulators and the existing numerical test thresholds are
retained. This does not enable the model's optional NextN/MTP block.

A further decode pass splits IQ3_XXS and IQ2_XXS/XS/S quant blocks into
32-value lane units, uses vector dot products, and reuses activations across
two output rows per SIMD group. IQ3_S gate/up pairs also use vector products.
The IQ2 lookup cache stores exact small integer magnitudes in bytes; scaled
weights, activations, products and accumulators remain FP32. This reassociates
floating-point sums, so it is not bit-identical to the older kernels.
A sequential five-repeat comparison at depth 128 (`tg32`, ctx 256) measured
7.18 -> 8.29 tok/s (+15.5%), versus 8.69 for the same-GGUF llama.cpp reference.
These remain short-run measurements, not a claim of sustained parity.
At depth 1024, three-repeat `tg32` runs gave 7.39 -> 8.63 tok/s (+16.8%);
llama.cpp measured 8.66 tok/s in the following run, essentially tied in this
short sample.
`QW3_METAL_DENSE_IQ2_VECTOR=0` restores the old IQ2 decode kernels for A/B tests.
Dense regression additionally checks both paired outputs directly and 60
odd-row/partial-input cases, with independent finite-metric validation.
The subsequent byte-cache refinement halves IQ2 workgroup lookup memory
without changing that arithmetic. Alternating five-repeat `tg64` runs show
a small, variable gain (about 2-3% when bracketing the old-cache runs), not a
sustained overtake of llama.cpp; see the dated measurements above.
`make bench-sampled-decode` measures 128 generated tokens with sampling and
repetition penalty, separately reports model/sampling time, and hashes the
token sequence; prompt prefill is outside the measurement.

On the base M5, dense IQ2_XS/S projections with at least 4096 inputs and
outputs now share their exact byte lookup cache across eight SIMD groups,
instead of two. Per-thread arithmetic and precision are unchanged; other
formats, small matrices and other GPUs retain their previous launches.
`QW3_METAL_DENSE_IQ2_SIMDGROUPS=2` disables this policy; `=8` opts in on other
GPUs. In-process alternating-token A/B runs measured only about +0.5-0.6%
decode throughput, with identical sampled token IDs. See
[the M5 investigation](docs/m5_decode_2026-10-01.md), including rejected
TensorOps experiments. `make qw3-decode-ab-bench` builds that two-session
comparison; run it with the dense model path as its only argument.

The later [Splish-inspired M5 trials](docs/m5_splish_trials_2026-10-01.md)
add optional `qw3-iq3-stream-bench` and `qw3-prefill-staging-bench` diagnostics.
Compact IQ3 tables, payload prefetch, smaller stage allocation and vector
stage writes did not demonstrate a useful whole-model gain; none was
promoted. The report includes fresh, separately measured pp128/tg32 llama
references and exact-logit continuation checks. `make test-iq3-stream-quant`
checks the experimental kernels, not a production performance threshold.

Qwen text IMRoPE now pairs the two halves of the rotated prefix on CPU,
single-token Metal and batched/fused prefill. Earlier adjacent-pair rotation
could pass position-zero and greedy-ID checks while drifting from llama.cpp
in the full logit vector at later positions. `make test-rope test-rope-metal`
checks nonzero positions without a model; `make test-dense-metal-logits`
compares multi-token prefixes against sequential llama.cpp evaluation with
matching KV precision. See [the correction and MTP probe](docs/mtp_probe_2026-10-01.md).

`make test-mtp-reference` builds an optional, separate native Metal MTP probe
against the local `LLAMA_CPP` build. It measures teacher-forced greedy
agreement and checks every logit against llama.cpp MTP; it does **not** enable
speculation in the agent or measure a generation speedup.

The optional `qw3-q5-head-bench` and `qw3-block-cost-bench` targets isolate
LM-head geometry and short-block backbone cost. They do not enable MTP in
the CLI or agent; the block probe computes **only the last logit row**.
See [the measured limitations](docs/decode_block_cost_2026-10-01.md).

On the base M5, large IQ3_XXS/IQ2_S projections now use FP32 vector kernels
for 2..8-row batches. An alternating old/new comparison measured +13.3%
throughput for an eight-token backbone block, **not** ordinary decode.
One-token generation and larger prefill retain their existing kernels.
`QW3_METAL_DENSE_SHORT_VECTOR=0` restores the old short-batch policy;
`make test-short-batch` checks quantization/tails, logits and continuation.
See [the short-batch measurements and scope](docs/short_batch_kernels_2026-10-01.md).

## Performance

Measurements below were taken on September 14, 2026, on a MacBook Air M5 with
24 GB unified memory. They use the supported
`Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`, full Metal offload, F16 KV cache, and
`caffeinate -disu`. Results are local observations, not hardware-independent
guarantees.

| Test | Context allocation | Repetitions | Result |
| --- | ---: | ---: | ---: |
| Prefill `pp4096`, warmed | 16,000 | 3 | 633.49 tok/s (SD 2.02) |
| Decode `tg128`, depth 4,096, warmed | 16,000 | 3 | 39.08 tok/s (SD 0.88) |
| Decode `tg128`, depth 16,384, no warmup | 17,000 | 1 | 32.84 tok/s |

The warmed prefill command was:

```sh
caffeinate -disu ./qw3-bench -m MODEL.gguf --ctx-alloc 16000 \
  --llama-style -p 4096 -n 0 -r 3
```

The M5 Air is fanless. Long repeated runs can reduce sustained decode speed;
the depth-16,384 row is deliberately reported as a single observation rather
than a stable average. Full benchmark history, rejected experiments, and
validation details are recorded in `docs/metal_prefill_validation.md`.

## Agent

`qw3-agent` uses Qwen-native tool calling and keeps the session local. On a
full-Metal F16 run, saved conversations include an atomic inference checkpoint:
occupied KV rows, recurrent DeltaNet and convolution state, tokens, and logits.
`/switch` and `--conversation` can therefore resume without replaying the saved
prefix. A missing, stripped, incompatible, or damaged checkpoint is ignored and
rebuilt from the persisted transcript.

Fresh sessions also keep a private reusable Metal checkpoint for the system and
tool prefix. The first launch creates it; later launches on the same model,
context, configuration, and local date restore it instead of prefilling that
prefix again. At most four variants are retained. Use `--no-prefix-cache` for a
direct A/B comparison or troubleshooting. Full conversation checkpoints always
take precedence over the prefix cache.

The main tools are:

- `read`, `more`, `list`: file reading and navigation.
- `write`, `edit`: file creation and modification.
- `search`: text search.
- `bash`: local shell commands.
- `get_skeleton`: file outline through `codenav`.
- `get_function`: function or method body through `codenav`.
- `semantic_search`: semantic code search through `colgrep`.
- `google_search`: web search with compact Markdown links.
- `visit_page`: fetch a URL and return readable text with a temp full-output path.

To avoid filling the context by reading large source files, the agent is
instructed to prefer `get_skeleton`, `get_function`, and `semantic_search`
before sequential file reads.

For performance or tool-loop diagnostics, `--trace PATH` appends private JSONL
events with prefill/decode timing, cache reuse, tool-result sizes, and context
compaction decisions. The regular terminal output is unchanged.

Useful interactive commands:

```text
/help
/status
/tools on
/tools off
/think
/nothink
/chdir PATH
/compact
/save [name]
/list
/switch ID
/del ID
/strip [ID]
/quit
```

Checkpoint compatibility is deliberately strict: the model fingerprint,
backend layout, context allocation, KV type, and token sequence must match, and
a payload checksum rejects damaged inference state before it can be resumed.
Metal payload v3 also checks SHA-256 identities of the built runtime sources,
compiler/build flags, and the actual assembled Metal source (including overrides
and optional flash attention), effective compile options, OS/device, and runtime
execution environment. Build identity is generated by `make` using Python 3;
source edits and flag changes trigger regeneration without relying on git or
timestamps. Loaded-source identity is captured at compilation, not reread at save.
Identical source at another override path remains compatible. Runtime environment
checks are conservative, including diagnostic switches referenced by the runtime.
Older Metal payloads reject as unsupported; a changed identity reports
`Metal session runtime/kernel fingerprint mismatch`. Both conversation and prefix
checkpoints fall back to rebuilding from tokens/transcript on rejection.
Partial `--ngl` sessions and experimental q8 KV currently keep the transcript
fallback but do not write a resumable Metal checkpoint.

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

The Metal runtime loads split kernel sources from `metal/` by default. See
`docs/metal_kernels.md` for the file order and environment overrides such as
`QW3_METAL_KERNEL_DIR`.

## Validation

Minimum checks before trusting a Metal change:

```sh
make
make test-regression
```

For larger Metal changes:

```sh
make test-regression-full
```

For prefill optimization work, also run the throughput guard:

```sh
make test-prefill-bench
```

Development notes and benchmark history are kept in:

```text
docs/metal_prefill_validation.md
docs/metal_kernels.md
```

## License

`qw3` is released under the MIT License. See `LICENSE` and
`THIRD_PARTY_NOTICES.md` for attribution notes covering ds4, llama.cpp/ggml,
linenoise, optional tree-sitter parsers, colgrep, and external Qwen model
artifacts.
