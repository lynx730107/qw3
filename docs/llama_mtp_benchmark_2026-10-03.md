# Local llama.cpp MTP benchmark — Apple M5

The local MTP implementation does not demonstrate a compelling generation
speedup on this model and machine. MTP code, diagnostic kernels, tests and build targets were removed from qw3
on 2026-10-03 at the user's request. This report records the external
llama.cpp benchmark and remains historical evidence. Ordinary inference
continues to accept the same GGUF and executes only its 64 trunk layers.

## Configuration and measurement

Target: the existing `Qwen3.8-27B-UD-Q3_K_XL.gguf`, 13,146,393,504 bytes
(the manifests retain the actual stat values). Engine: local llama-server,
build 11041, commit `4fea119de`, AppleClang 21.0.0.21000101, Apple M5.
All runs request full Metal offload for target and draft, Flash Attention,
context 512, batch/ubatch 128, one slot, and F16 attention K/V for both
contexts. Verbosity-4 logs confirm 66/66 GPU layers, GPU output head and
F16 target/draft KV. MTP uses the head inside the same GGUF.

Greedy sampling: temperature 0, top-k 1, top-p 1, min-p 0, repeat penalty 1,
seed 42, ignore EOS, prompt caching disabled per request. Numeric token IDs
come from the qw3 diagnostic tokenizer with thinking disabled and are
identical across configurations. Warmups are excluded. GPU processes run
sequentially and are terminated after each block.

`pp128` processes exactly 128 input tokens and measures the server's prompt
phase, including final prompt logits. A single output is requested to obtain
that phase's timing. `tg32` emits exactly 32 greedy tokens from the 26-token
code prompt. The server times 31 decode tokens because the first output comes
from prefill. Rates use **31 / generation seconds**, matching its reported
`predicted_per_second`. These are server workloads, not direct replacements
for pure `llama-bench` synthetic pp/tg results from earlier experiments.

## Results

Rates below are total timed tokens divided by total timed seconds. The first
series uses OFF/1/3/3/1/OFF, three repetitions per block. The second uses
OFF/1/7/7/1/OFF, two repetitions per block.

| Series | Configuration | pp128 tok/s | tg32 tok/s | Code draft acceptance |
|---|---|---:|---:|---:|
| 1 | OFF | 90.79 | 6.71 | — |
| 1 | MTP, max 1 | 91.24 | 6.67 | 13/18, 72.2% |
| 1 | MTP, max 3 | 96.94 | 5.57 | 18/37, 48.6% |
| 2 | OFF | 79.67 | 6.05 | — |
| 2 | MTP, max 1 | 77.06 | 5.58 | 13/18, 72.2% |
| 2 | MTP, max 7 | 68.75 | 2.86 | 21/63, 33.3% |

**There is substantial performance drift even with MTP OFF.** Do not read
aggregate differences as an isolated MTP effect or a statistically confirmed
speedup. Neither series achieved stable machine performance; the second is a
repeat, not a steady-state measurement. A specific thermal cause was not
established. `pmset -g therm` reported no warning. The initial/final OFF
generation blocks change 9.05 → 5.33 tok/s in series 1 and 7.34 → 5.14 in
series 2. The apparent series-1 pp128 improvement with MTP is confounded.

For transparency, the complete block rates are:

| Series | Block | Configuration | pp128 tok/s | tg32 tok/s |
|---|---:|---|---:|---:|
| 1 | 0 | OFF | 121.63 | 9.05 |
| 1 | 1 | MTP 1 | 112.87 | 8.65 |
| 1 | 2 | MTP 3 | 106.10 | 6.21 |
| 1 | 3 | MTP 3 | 89.23 | 5.05 |
| 1 | 4 | MTP 1 | 76.57 | 5.42 |
| 1 | 5 | OFF | 72.43 | 5.33 |
| 2 | 0 | OFF | 94.02 | 7.34 |
| 2 | 1 | MTP 1 | 89.08 | 6.22 |
| 2 | 2 | MTP 7 | 73.82 | 2.92 |
| 2 | 3 | MTP 7 | 64.32 | 2.81 |
| 2 | 4 | MTP 1 | 67.90 | 5.06 |
| 2 | 5 | OFF | 69.12 | 5.14 |

All 32 output token IDs match OFF exactly for code, math and story, including
both MTP-7 blocks' measured code outputs. Code's hash is
`c25b6505f1acb4cf`, also matching the qw3 private greedy diagnostic recorded before its removal. This
benchmark establishes token equality on these prompts; it does not establish
full-vocabulary logit equality, session persistence or agent correctness.

## Evidence and reproduction

- [Series 1 raw results](measurements/llama_mtp_2026-10-03/results.json)
- [Series 2 raw results](measurements/llama_mtp_repeat_2026-10-03/results.json)
- [Sequential benchmark script](../tools/bench_llama_mtp.py)

Each result retains its exact command and full HTTP response. Neighboring
files retain server logs and excluded warmup responses. Startup/model loading
and HTTP overhead are excluded from reported phase timings. The script was
run before adding explicit baseline-drift summaries and fixing version capture;
those metadata fields were reconstructed from saved responses and the same
binary's version output without changing any measured timing.

```sh
python3 tools/bench_llama_mtp.py --output /tmp/llama-mtp-new --verbosity 3
python3 tools/bench_llama_mtp.py --output /tmp/llama-mtp-seven-new \
  --order off,mtp1,mtp7,mtp7,mtp1,off --repetitions 2 --verbosity 4
```

## DFlash2 next experiment

The [official Qwen3.8-27B DFlash2 draft](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
is trained for this target. Its five-layer parallel drafter reads target
layers 5/19/33/47/61, uses block size 8 and a top-16/rank-256 selector.
The [official GGUF repository](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF)
provides a Q4_K_M draft of about 1.14 GB, plus Q8_0 and BF16 variants. This
avoids converting or changing the existing target quantization.

The local source implements DFlash2 in `src/models/dflash.cpp` and
`common/speculative.cpp`; the existing `libllama.0.4.1.dylib` also contains
its DFlash2 metadata validation and selector strings. Actual compatibility
and Metal execution still require a load/generation test. The candidate
comparison is ordinary versus `--spec-type draft-dflash`, with the official
draft supplied by `-md`, max 7 proposals and the same target/offload/F16 KV.
Test greedy equality and memory use first, then pp128 and tg32, before porting
the five hidden-state taps, parallel draft graph or selector to qw3. DFlash2
has not been downloaded or benchmarked in this task yet.
