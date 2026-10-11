# IQ4_XS packed integer lookup trial on Apple M5, 2026-10-11

Both private lookup variants fail the first guarded matrix check by one
ULP. They are rejected before performance screening. Production inference
and its floating-point expressions remain unchanged.

The trial stores the 16 exact IQ4 nonlinear magnitudes in four literal
32-bit words, selects a byte with integer operations and converts the
decoded integer to FP32. Variant 1 sign-extends an int8 representation;
variant 2 subtracts 128 from a biased unsigned byte. The generator verifies
all 16 decoded integers against the native table. This removes the shared
FP32 table initialization, barrier and per-quant table reads, while keeping
the native activation loads, vector accumulations, block scaling, K loop,
two-row geometry and SIMD reductions in the source.

Both compiled variants differ in the 256→1 single-matvec case with mixed
finite input. The reference output is −0.126425833 (`be0175c6`) and each
candidate gives −0.126425818 (`be0175c5`). No case passes before failure.
The intended suite contains six shapes, three input patterns and three
operations (single, pair and residual add), with finite byte equality,
output guards and unchanged input checks. It stops at the first difference.

The original failure streams are retained. A second build adds only
first-difference diagnostics to the matrix harness, reproducing the same
failure for both variants and recording the hex values. Its shader hash
matches the initial build; the diagnostic source map references the original
snapshot for unchanged files and stores the two changed test files separately.
Both builds complete without warnings. Metal API validation is enabled;
the real model's 156 IQ4_XS tensors pass the existing alignment check.

Both the production library and private library use freshly constructed
default `MTLCompileOptions`; production additionally enables its available
tensor API macro. The trial does not establish why compiled evaluation
changes. Identical integer table values and retained FP32 source expressions
are insufficient to establish identical outputs. The byte comparison remains
the acceptance gate.

No full-model, checkpoint or timing run is made for these rejected candidates.
The optional full-quality and paired-decode entry points are present for
reproduction. No speed or state-correctness claim follows from this trial.

The [initial manifest](measurements/iq4_packed_lut_2026-10-11/manifest.json),
[diagnostic manifest](measurements/iq4_packed_lut_2026-10-11/diagnostic/manifest.json),
[first-failure analysis](measurements/iq4_packed_lut_2026-10-11/analysis.json),
build/failure logs and immutable source snapshots are retained.

```sh
make qw3-iq4-packed-lut-test qw3-iq4-packed-lut-bench
# Clear QW3/GGML/MTL overrides first, then set:
export QW3_METAL_KV_F16=1 QW3_METAL_NGL=64 MTL_DEBUG_LAYER=1
export QW3_METAL_DENSE_PREFILL_FAST=1 QW3_METAL_DENSE_PREFILL_K64=1
export QW3_METAL_DENSE_GDN_CACHE=1
./qw3-iq4-packed-lut-test ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf 1
./qw3-iq4-packed-lut-test ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf 2
```
