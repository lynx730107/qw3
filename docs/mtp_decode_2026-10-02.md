# First autonomous MTP decoder (2026-10-02)

Base Apple M5, dense Qwen3.8-27B UD-Q3_K_XL, 512-token sessions, full
64-layer target offload and F16 KV. This is an optional diagnostic executable,
not a public inference API, CLI option or agent change. Production inference
and model/prompt/tool behavior are unchanged.

## What is new

`tests/bench_mtp_decode.m` reuses the existing reference-checked native MTP
graph from `bench_mtp_acceptance.m`. That graph can now skip the unused LM
head during catch-up. The older teacher-forced reference test retains its
normal full-head path.

The new decoder actually rolls out its draft: first feed the target's next
greedy token and previous normalized target hidden; subsequent steps consume
the previous draft token and normalized draft hidden. No future target tokens
are supplied to these autonomous proposals. This growing-KV arrangement and
accepted-prefix catch-up follow the local
[llama.cpp MTP driver](https://github.com/ggml-org/llama.cpp/blob/579ef425aba8563eee4e85c769e66543c7dac323/common/speculative.cpp).
That is a source reference, not a throughput measurement of its binary.

Each cycle verifies up to eight inputs (one target token plus seven proposals).
Every target output-norm hidden row and every full-vocabulary logit row is
exposed, not just the final row. The target always decides acceptance. Tokens
after the first mismatch are never emitted.

Before verification, GPU snapshots preserve full K/V, F16 FlashAttention
tails, DeltaNet, convolution, logits and final hidden buffers. A rejected
cycle restores them and serially replays only accepted inputs. Position,
history and CPU-logit validity are repaired too. The draft is separately
restored and caught up using true accepted target hidden rows.

This deliberately conservative snapshot is 190.61 MiB for the target at
ctx512. It includes unused KV and derived tails. It is not a proposed
long-context checkpoint format. Replaying accepted tokens is an expensive
baseline; efficient per-position recurrent checkpoints remain future work.

## Correctness checks and current limitation

An independent serial oracle checks **all** rows, including proposed rows
after a rejection. Full logits require finite values, max error <=0.005,
RMS <=0.001 and identical argmax. Hidden rows require max <=0.001 and
RMS <=0.0001. These checks are not relaxed for MTP.

After each rejection, independently restoring/replaying the oracle must
reproduce every saved persistent GPU buffer byte for byte. For the serial
verifier, the final state must also exactly equal the untouched ordinary
serial generator. All emitted token IDs, position and history are checked;
eight additional ordinary-decode continuations compare full logits.

An initial logit-only batch-8 probe failed the 32-token code case on cycle
index 2, row 2: max logit difference **0.0424070**, RMS **0.00456111** against
a serial oracle starting from the same snapshot and consuming the same
proposals. The final executable checks hidden rows too and rejects this
case earlier, on cycle 0, row 0: hidden max **0.00333309**, RMS **0.000534521**.
Both tests exited nonzero. Earlier last-row-only cost checks had not covered
these intermediate rows or these autonomously proposed prefixes. This does
not establish a quality regression in ordinary decode; it blocks adopting
this batch path as an equivalent MTP verifier.

The default is therefore the serial correctness path. It is **slower**, not
an optimization to enable in the agent. Draft agreement in the previous
teacher-forced benchmark must not be presented as acceptance of these
autonomous chains.

The complete serial regression passed six cases: code (32 emitted tokens),
math (16), story (16), and forced rejection indices 1/4/7 (8 each). All
88 emitted tokens matched ordinary greedy generation. Across these cases,
179 proposed verification rows had exactly equal full logits and hidden
rows, 21 rejected cycles passed byte-exact restore/replay checks, final
serial states were identical, and all six eight-token continuations passed.

The autonomous code run accepted 21 of 65 draft proposals across 11 cycles
(including one final singleton). That is not the previous teacher-forced
agreement of roughly 81%. Actual recursive drafting must be measured on
the requested workload; the old percentage cannot justify a speed estimate.

## Timing scope

The executable prints component times including snapshots, draft generation,
all target heads/readbacks, restore/replay and draft catch-up. Independent
oracle/full-state/continuation checks are excluded. Their interleaving
perturbs the system, and the ordinary generator runs before speculation;
therefore rates are diagnostic costs, **not** an interleaved promotion
benchmark or proof of a speedup against llama.cpp.

In the serial code regression the ordinary baseline was 9.125 tok/s and
the instrumented speculative segments totaled 2.911 tok/s. Component
totals were 46.4 ms snapshots/argmax, 699.4 ms drafting, 8444.0 ms verification,
1691.0 ms restore/replay and 112.0 ms draft catch-up. This demonstrates the
current cost problem, not a useful inference setting. Early-stopping serial
verification could remove wasted work, but does not provide the parallel
backbone speedup that motivated MTP.

Fault injection uses known ordinary tokens only to put the first rejection
at a chosen boundary. It is clearly labeled and is not draft acceptance or
performance evidence. The final boundary crosses position 32 in the chosen
26-token prompt, exercising F16 tail restoration.

The modified native graph also passed a fresh three-prompt llama.cpp MTP
differential check (four rows per prompt): 12/12 native/reference greedy
IDs matched, worst max error 0.00212336, worst RMS 0.000285068. This tests
the shared graph against the reference, not speculative sampling or
equivalence of complete recursive chains against the llama.cpp driver.

## Reproduce

```sh
make qw3-mtp-decode
caffeinate -disu make test-mtp-decode
caffeinate -disu ./qw3-mtp-decode \
  ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf 32 8 serial
# Experimental: currently expected to fail the strict all-row check.
caffeinate -disu ./qw3-mtp-decode \
  ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf 32 8 batch
```

Arguments: model, fixed output token count (1..128), block width (2..8),
`serial|batch`, optional prompt, optional forced rejection index (1..block-1).
The executable does not stop on EOS and is greedy-only. Exact stochastic
acceptance/resampling, efficient rollback, cancellation, EOS handling and
CLI/agent integration are still absent. Next priority is correct, fast
all-row verification before promoting any MTP generation path.

An additional `serial-replay` mode now replaces full-model accepted-prefix
replay with compact recurrent-input replay. It does not solve the batch
numerical failures above or enable production speculation. See
[the compact replay report](mtp_compact_replay_2026-10-02.md).
