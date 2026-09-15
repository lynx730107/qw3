# Agent reliability roadmap

This document tracks the agent and session improvements inspired by measured
behavior in other local, in-process coding agents. Each phase must land with
model-free unit or integration coverage. Changes that affect inference must
also pass the Metal logit, no-garbage, and native tool-call checks.

## Phase 1: Tool execution and editing

- [completed] Make `bash` interruptible and time-bounded, merge stderr, drain
  large outputs without deadlocking, retain head and tail, and spill the full
  output to a bounded private directory.
- [completed] Require unique exact matches for replacements and insertion
  anchors, reject no-op edits, and recover literal newline/tab escapes only
  when the recovered target is valid and unambiguous.
- [completed] Add safe whitespace-flexible edit recovery and exact mismatch
  diagnostics while preserving the unique-target invariant.
- [completed] Track edits and require a build or test before a coding turn can
  finish.

## Phase 2: Context ownership

- [completed] Keep a structured role/content/tool-call message ledger alongside
  the rendered token transcript.
- [completed] Persist the message ledger and compact only at complete user-turn
  boundaries, never in the middle of assistant output or a tool exchange.
- [completed] Archive the complete pre-compaction message ledger in a bounded,
  private spill directory and retain its exact reload path in the summary.
- [completed] Remove stale thinking first, shorten old tool results, and
  preserve the recent working set before invoking model-generated compaction.
- [completed] Reject low-yield compaction and defer automatic retries so it
  cannot force recurrent-cache rebuilds on consecutive steps.

## Phase 3: Persistent inference state

- [completed] Add Metal export/import for occupied F16 KV rows, the F16 flash
  tail, GatedDeltaNet state, convolution state, token ids, and logits, with a
  bit-exact multi-step logit round-trip test.
- [completed] Store full session checkpoints atomically with model, backend,
  context, cache-type, layout, token-sequence, payload-checksum, built-runtime,
  and loaded-kernel validation. Keep transcript replay as the fallback for
  missing, stripped, damaged, or otherwise rejected state.
- [completed] Add a bounded, private system/tool-prefix checkpoint keyed by its
  exact tokens, model path, context, and backend configuration, with an A/B
  switch and full-conversation checkpoint precedence.
- [pending] Add the second warm-prefix tier for a project-specific environment
  tail.

## Phase 4: Measurement and safety

- [completed] Add an opt-in private JSONL trace for session configuration, user
  turns, tool results, compaction decisions, prefill, and decode metrics.
- [completed] Extend the trace with checkpoint save/load outcomes and retain
  per-inference cached-token versus appended-prefill metrics, making resume,
  incremental extension, rejection, and rebuild decisions observable.
- [completed] Add a reproducible model-driven coding smoke task that requires
  native `write`, native `bash`, compilation, execution, and exact source
  verification in an isolated workspace.
- [pending] Add broader feature A/B switches for agent behavior experiments.
- [pending] Add permission modes, pre-edit workspace checkpoints, session forks,
  atomic private session files, and optional `--ctx auto` memory governance.

## Phase 5: Decode fast path

- [completed, rejected] Profiled two fused Q8 GDN input projections for
  `linear_qkv_proj` + `linear_gate_proj`; the best candidate was neutral and
  the llama-style multi-row candidate regressed decode by 1-2% on M5.
- [completed] Keep decode logits on Metal through default argmax/top-k
  sampling, with lazy CPU readback for diagnostics and unsupported samplers.
- [deferred] A pre-encoded single-token Metal layer pipeline has a measured
  0.45-0.67 ms/token CPU-submit ceiling versus 22.1-22.5 ms of GPU wait;
  revisit only after the dominant GPU kernels improve substantially.
- [deferred] Consider model-specific speculative decoding only after exact GDN
  rollback and batched verification are available.
