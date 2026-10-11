# Immutable IQ3_S weight/grid descriptors on M5, 2026-10-11

Neither immutable-buffer descriptor variant improves ordinary decode in
this screening. Both preserve exact matrix and measured model results,
but their nearby-control point gains are −0.060% and −0.281%. Both lose
against each adjacent OFF run. No variant is adopted; production inference
sources, shaders and executables remain unchanged. The latest direct
[qw3/llama comparison](m5_native_reference_after_store_cleanup_2026-10-11.md)
still requires about 2.962% more qw3 throughput to reach llama's decode rate.

This is a distinct hypothesis from the rejected
[threadgroup pipeline hints](m5_iq3_hint_2026-10-11.md). It applies
`MTLMutabilityImmutable` to the original IQ3_S single-matvec function's
weight buffer at index 1; variant 2 also applies it to the existing grid
buffer at index 4. Both use the original `qw3_matvec_iq3_s_fast` function
from the production Metal library, without a new shader, weight layout,
weight copy or session data buffer. Paired matvecs remain native.

[Apple's buffer-mutability API](https://developer.apple.com/documentation/metal/mtlcomputepipelinedescriptor/buffers)
allows this declaration when neither CPU nor GPU changes the bound buffer
between binding and command-buffer completion. The model views and lookup
grid satisfy that contract. Activations and outputs retain their default
mutability; earlier/later GPU stages can change them. The
[default behavior](https://developer.apple.com/documentation/metal/mtlpipelinebufferdescriptor/mutability)
for ordinary buffers is mutable. The documented opportunity for additional
optimizations does not guarantee a speed gain.

Maximum-thread and SIMD-multiple descriptor hints retain their defaults.
All three compiled pipelines report SIMD width 32, maximum 1,024 threads
and 1,024 static threadgroup bytes. Dispatch remains eight rows and 64
threads, with original arithmetic, loop/reduction order and grid staging.
The optional runner selects the original or descriptor-created pipeline
and counts actual candidate selections.

## Numerical gate

Builds are warning-free. With Metal API validation enabled, 36 guarded
matrix cases pass: six input/output shapes, three input patterns and two
variants. Outputs are finite and bit-identical to the original pipeline;
input contents and output guards remain intact.

The 27B model check uses one reused ctx256 session and a disk oracle.
Baseline and both variants process the same fixed-seed prefix of 128 tokens
and 32 greedy continuations, sequentially. Each phase checks the complete
finite vocabulary logits 33 times, tokens, positions and history, plus
canonical live GQA KV and full DeltaNet/convolution state before and after
continuation. Temporary Flash Attention tail scratch is excluded. Each
variant selects its pipeline 2,336 times; baseline selects it zero times.
This screen is not the eight-prefix/checkpoint quality suite. No production
integration is proposed, so that broader integration gate is not run.

## Ordinary decoding

Five sequential OFF/v1/OFF/v2/OFF processes use tg32, depth128, ctx256,
F16 KV, full 64-layer offload, three repetitions and warmup. Pipeline
compilation precedes warmup. Each process has 60 seconds idle before launch;
Metal API validation is disabled for timing. All 15 samples are retained.

| Block | Mode | Mean time for 32 tokens (ms) |
| ---: | --- | ---: |
| 0 | OFF | 3501.683 |
| 1 | Immutable weight | 3502.738 |
| 2 | OFF | 3499.599 |
| 3 | Immutable weight and grid | 3511.594 |
| 4 | OFF | 3503.860 |

The OFF max/min mean spread is 0.122%, passing the ≤5% stability gate.
Relative to the mean of the two adjacent controls, variant 1 gives
−0.060% throughput and variant 2 −0.281%. Variant 1 gives −0.030% versus
its left control and −0.090% versus its right; variant 2 gives −0.342%
and −0.220%. Each variant has one ON process; this is a negative screening
result, not a confidence interval or proof of universal slowdown. There
is no positive result to replicate or integrate.

Candidate pipeline counts are 7,081 in every ON run and zero in each OFF
run. Every timing process reports 176,761,680 session bytes. Swap remains
1,165.06 MiB before and after all three numerical and five timing processes.
The store remains history-only at 30,109 bytes with unchanged hash.
These observations do not measure peak RAM or establish a cleanup speed
effect. Sources, binaries, snapshots and model size/mtime match after
completion. Existing source snapshots are reused by hash, and the new
optional runner and generated includes are frozen in this archive.

[Numerical manifest](measurements/iq3_immutable_2026-10-11/manifest.json),
[timing manifest](measurements/iq3_immutable_2026-10-11/native/manifest.json),
[analysis](measurements/iq3_immutable_2026-10-11/analysis.json),
[verification script](measurements/iq3_immutable_2026-10-11/verify_campaign.py),
build logs, every raw stream and immutable source references are retained.
