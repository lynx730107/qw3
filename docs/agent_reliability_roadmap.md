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

- [pending] Add Metal export/import for occupied F16 KV rows, GatedDeltaNet
  state, convolution state, token ids, and logits.
- [pending] Store full session checkpoints atomically with model/backend/cache
  compatibility metadata.
- [pending] Add two warm-prefix checkpoint tiers: stable model/tool behavior and
  the project-specific environment tail.

## Phase 4: Measurement and safety

- [completed] Add an opt-in private JSONL trace for session configuration, user
  turns, tool results, compaction decisions, prefill, and decode metrics.
- [pending] Extend the trace with explicit incremental-append and full-rebuild
  cache decisions once persistent backend checkpoints are available.
- [pending] Add reproducible agent coding smoke tasks and feature A/B switches.
- [pending] Add permission modes, pre-edit workspace checkpoints, session forks,
  atomic private session files, and optional `--ctx auto` memory governance.

## Phase 5: Decode fast path

- [pending] Profile a fused Q8 GDN input projection for
  `linear_qkv_proj` + `linear_gate_proj`.
- [pending] Explore a pre-encoded single-token Metal layer pipeline while
  keeping the validated prefill path unchanged.
- [deferred] Consider model-specific speculative decoding only after exact GDN
  rollback and batched verification are available.
