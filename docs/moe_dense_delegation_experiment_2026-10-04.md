# Abandoned MoE/dense delegation experiment — 2026-10-04

Development on `codex/moe-dense-delegation` was stopped at the user's
request. This commit preserves the experiment and the existing working
baseline; it does not declare the feature ready for release.

The optional supervisor alternates a MoE primary process and a dense expert
process, using separate IPC, model-specific checkpoints, tool journals and
durable result identifiers. The ordinary path remains available without an
expert model. Default expert thinking CLI serialization was corrected after
a real launch exposed `invalid expert thinking mode`.

Observed validation includes model-free IPC/state/guard tests, real handoff,
default-thinking launch, isolated edits and verification, timeout, unavailable
expert, interruption during generation/tools, crash recovery, duplicate
result recovery and incompatible-checkpoint fallback. Both model session
roundtrips matched full-vocabulary logits exactly and preserved the tested
greedy continuations. The latest changes to expert response completion,
packet fallback and generation budgets have not had the entire live suite
rerun. Repeated TUI/memory cycles and the frozen eight-task pilot were
prepared but not executed.

A private, read-only diagnosis test with an explicitly authorized symptom
hint showed a useful partial diagnosis from the dense expert, but it reached
the 1200-second timeout without a final answer. The MoE resumed its own KV
checkpoint, and model-owning processes did not overlap. The whole handoff
took about 1491 seconds. The MoE's final proposed fixes failed an isolated
counterexample. A fresh MoE-only control took about 289 seconds, pursued an
incorrect explanation and exhausted its per-response token limit. Neither
run establishes a completed, verified task or a general quality advantage.
Private source, transcripts and checkpoints remain outside this repository.

Known remaining work includes auditing hybrid `/save`, `/del` and `/strip`
sidecars, eliminating irrelevant evidence navigation in manual delegation,
completing the live regression suite, and measuring repeated handoffs and
the frozen quality pilot. Automatic routing has no demonstrated quality
advantage. These are recorded limitations, not follow-up development tasks:
the experiment is abandoned.
