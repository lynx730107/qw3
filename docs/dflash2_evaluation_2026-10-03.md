# DFlash2 evaluation — Apple M5

Goal: measure the official Qwen3.8-27B DFlash2 drafter against ordinary
llama.cpp using the existing UD-Q3_K_XL target. A native qw3 port is
conditional on a stable, substantial gain and greedy correctness. MTP
remains removed from qw3.

The Q4_K_M draft is downloaded from `incoai/Qwen3.8-27B-DFlash2-GGUF`,
revision `51962825493a48b846b40126d35c799ac4093ad0`, and verified against
the repository's SHA-256. See
[download provenance](measurements/dflash2_draft_download_2026-10-03.json).
The target is unchanged. Initial engine: local build 11041 / `4fea119de`.

The [benchmark script](../tools/bench_llama_dflash.py) uses full target and
draft Metal offload, F16 attention KV, context 512, batch/ubatch 128, one
slot and four CPU threads. Per-request prompt reuse and server prompt-cache
RAM are disabled; additional server checkpoints are disabled consistently.
Native recurrent speculative history is retained as required by llama.cpp.
No synthetic acceptance is used. All GPU jobs run sequentially.

The pilot checks model loading, DFlash2 identity, complete offload, greedy
32-token equality on code/math/story and process RSS. Subsequent trials
interleave ordinary and speculative configurations with equal cooldowns.
The initial candidate drafts seven tokens, matching the trained block size
of eight. Smaller verify widths can be evaluated if this candidate is slow.

`pp128` measures 128 numeric input IDs and final prompt logits. `tg32`
emits 32 greedy tokens from the same 26-token code prefix; the reported
generation rate follows llama-server's convention of 31 timed decode
tokens, since the first output comes from prefill. Raw phase timings,
commands, responses, excluded warmups and server logs are retained.

Decision criteria: exact greedy equality on the tested prompts, full
offload without disruptive memory pressure, reproducible generation gains
across interleaved trials, and no unresolved baseline drift. A marginal or
confounded aggregate mean is insufficient to justify the native port.

If the port is justified, implement hidden-state taps, parallel draft graph,
selector and accepted-prefix state handling behind `--dflash`, then check
logits, greedy generation, rollback, session persistence and agent behavior.
Otherwise retain the negative result and profile ordinary Metal prefill and
decode stages to identify the main cost centers.

## Pilot and verify-width screening

[Pilot raw measurements](measurements/dflash2_pilot_2026-10-03/results.json)
confirmed that this local Metal build loads the actual DFlash2 convolution
and selector graph. All 32 output IDs matched ordinary decoding on the
three prompts; the pp128 output ID matched too. With seven proposals,
code generation fell from 9.16 to 5.79 tokens/s, math from 9.03 to 6.64,
and story from 8.91 to 2.83. Code accepted 22 of 51 proposed tokens.
This two-block pilot had no cooldown and establishes feasibility and a
poor initial result, not a stable speed comparison.

[Width-screening raw measurements](measurements/dflash2_widths_2026-10-03/results.json)
interleaved OFF/DF1/OFF/DF3/OFF/DF5/OFF with 20 seconds of idle time
before each process. All greedy checks passed. OFF code rates were
8.77–9.04 tokens/s, much steadier than the historical MTP experiments.

| Mode | pp128 tokens/s | Code tg32 tokens/s |
| --- | ---: | ---: |
| OFF, four blocks pooled | 166.41 | 8.91 |
| DFlash2, one proposal | 165.53 | 9.89 |
| DFlash2, three proposals | 164.99 | 8.58 |
| DFlash2, five proposals | 159.23 | 6.44 |

One proposal is promising on this prompt, with approximately 11% higher
generation throughput and 15/16 proposals accepted. It needs confirmation
across repeated blocks and different workloads. Larger proposal widths do
not justify a native port on these measurements.

## Confirmation and decision

[Confirmation raw measurements](measurements/dflash2_confirmation_2026-10-03/results.json)
use OFF/DF1/DF2/DF2/DF1/OFF, two measured repetitions of each workload in
every block, excluded warmups for all workloads, and equal 20-second
cooldowns. Each cell below is throughput calculated from the mean phase
latency within that block, not an arithmetic mean of throughput samples.

| Block | Mode | pp128 | Code tg32 | Math tg32 | Story tg32 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | OFF | 161.30 | 9.01 | 8.96 | 8.93 |
| 1 | DF1 | 148.63 | 8.88 | 8.34 | 6.11 |
| 2 | DF2 | 116.81 | 6.69 | 6.59 | 3.96 |
| 3 | DF2 | 112.37 | 6.23 | 6.11 | 4.00 |
| 4 | DF1 | 103.14 | 5.81 | 5.32 | 4.07 |
| 5 | OFF | 97.32 | 5.48 | 5.05 | 5.25 |

Units are tokens/s. The baseline fell 40% in pp128 and 39% in code tg32.
Equal cooldowns did not stabilize this longer experiment. No temperature,
clock or swap trace was collected, so the cause is not established. The
pooled DF1 code rate is superficially 3.1% above pooled OFF, but this is
**not evidence of a stable speedup** under the observed drift.

All measured output IDs matched OFF exactly, including all four pp128
outputs per mode. DF1 acceptance is 15/16 on code, 14/16 on math and
10/21 on story in every repetition. DF2 acceptance is 19/22, 20/21 and
14/33 respectively. The story case is slower in both DF1 blocks than the
neighboring OFF block, showing why the initial code-only result cannot be
generalized to agent workloads. These short greedy checks do not establish
full-logit equivalence or a correct native rollback/session implementation.

The width-screening server reports about 12,419 MiB for the ordinary target.
DF1 reports 12,582 MiB for the target plus 1,458 MiB of additional device
allocation, approximately 1.6 GiB more overall, with about 4,145 MiB of
device-budget headroom. Raw RSS is retained separately; mapped-file RSS is
not an exclusive physical-memory measurement.

**Decision: do not implement native DFlash2 on the current evidence.** The
best short code trial gains about 11%, but a stable, substantial benefit
across the tested workloads was not demonstrated. qw3 has no `--dflash`
option; client, agent and sessions continue through the ordinary path.
MTP remains removed. A future positive result must first reproduce gains
with nearby OFF controls and stable baselines before incurring the port.

The existing qw3 short-block cost probe also measured a two-row backbone
at 399 ms versus 215 ms for two serial forwards, even before draft and
all-row logit costs. Therefore even a llama.cpp gain cannot be assumed to
transfer directly to qw3's current verifier. See
[short-block limitations](decode_block_cost_2026-10-01.md).

The fallback work is complete: [ordinary Metal profiling](dense_metal_profile_2026-10-03.md)
identifies FFN projections and DeltaNet prefill as the next candidates.
The profiler is intrusive and establishes cost centers, not a new speedup.

## Reproduce

Run from the repository root with the downloaded draft and existing local
llama.cpp build:

```sh
python3 tools/bench_llama_dflash.py \
  --output /tmp/dflash-width-screen \
  --order off,df1,off,df3,off,df5,off --cooldown 20
python3 tools/bench_llama_dflash.py \
  --output /tmp/dflash-confirmation \
  --order off,df1,df2,df2,df1,off --cooldown 20 \
  --repetitions 2 --all-workloads
```

Output directories must be new. The server listens only on localhost and
is stopped after each block, including on request/startup failure. The
draft download was verified against the pinned official artifact:
[GGUF repository](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF).
