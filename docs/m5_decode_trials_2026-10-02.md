# Dense M5 lookup and submission trials (2026-10-02)

Starting production revision: `52d32d7`. Base Apple M5, 24 GB, dense
Qwen3.8-27B UD-Q3_K_XL, full 64-layer offload and F16 KV. All inference
ran sequentially under `caffeinate -disu`. No agent, sampling, precision,
GGUF layout or MTP change. **No candidate below is enabled in production.**

The optional `qw3-decode-trials-bench` includes the bridge/core privately,
compiles selected modified kernels in memory and swaps private pipelines.
A compile-time layer-completion hook permits private submission experiments;
the normal build defines it as constant success and does not submit prefixes.
There is no new production environment knob or dispatch policy.

## IQ4_XS: table location is not an acceleration

Single, gate/up pair and residual-add kernels currently stage a 16-entry
FP32 magnitude table in threadgroup memory. Two experiments remove that
initialization/barrier while retaining the existing dot grouping:

- Read the same exact magnitudes from the constant table.
- Keep one exact magnitude per lane and use SIMD shuffles for lookup.

The 36 IQ4 checks (two alternatives, three operations, six shapes) were
bit-identical to the original GPU outputs. Independent double CPU dots
on first/middle/last rows passed max 1e-4. Buffer offsets are nonzero;
all output guards are checked. The residual-add test uses nonzero initial
outputs and a negative scale, reset outside each timed dispatch.

At actual large projection shapes the shuffle version was roughly 42-46%
slower in synchronous measurements. The constant table was generally neutral
or slightly slower; one isolated paired projection showed +2.57%, but
the full model did not benefit. Do not promote the isolated winner.

## IQ3_S: two-byte payload reads

Paired gate/up reads quant pairs and sign pairs through scalar `ushort`
loads, rather than separate byte loads, retaining its original FP32 grid,
scales, signed dot products and accumulators. GGUF IQ3_S blocks are 110
bytes: they are two-byte aligned, not necessarily four-byte aligned. No
`uint`, `ushort2` or other unsafe aligned weight load is introduced.

Six additional pair tests passed exact original/candidate outputs and the
independent CPU double dot (worst sampled-row difference 2.50e-7). The
5120->17408 paired projection measured 0.8288/0.8075 ms (+2.64% synchronous
throughput), but whole-model replication changed sign. It is not retained.
Shapes shared by both format probes are 256->1, 4352->33,
5120->4097/4111/17408 and 17408->5120. Alternate shapes reinterpret a bounded
real tensor prefix; they are not all actual model tensor dimensions.

## Full sampled decode and prefix submission

Two independent ctx-512 sessions use the same chat prefix, seed 42,
temperature 0.6, top-k 20, top-p 0.95 and repetition penalty 1.06 over at
most 1024 previous tokens. Each generates 128 sampled tokens. First eight
forwards are untimed; remaining 120 alternate AB/BA order. Sampling is
separately measured and included in reported throughput. Entire vocabulary
logits are checked after every forward, outside timing; all compared vectors
and sampled IDs were identical. Every run produced hash `6e27f5726999f2da`.
These two-session rates are not ordinary single-session throughput claims.

Submission alternatives split the ordinary one-token stream after every
8, 16 or 32 complete layers, submitting a prefix without a CPU wait. They
reuse the existing command queue/pending-buffer synchronization and submit
the output head with the last layer. No tensor operation is changed.

| Trial | Baseline / candidate tok/s | Forward change | Saved ms/token |
| --- | ---: | ---: | ---: |
| IQ4 constant table | 9.054 / 9.004 | -0.54% | -0.595958 |
| Submit every 16 layers | 9.024 / 9.043 | +0.23% | 0.250717 |
| Submit every 8 layers | 8.919 / 8.945 | +0.32% | 0.357067 |
| Submit every 32 layers | 8.497 / 8.488 | -0.08% | -0.091308 |
| IQ3_S word reads | 8.754 / 8.783 | +0.35% | 0.402583 |
| Submit every 8 layers, replication | 7.859 / 7.873 | +0.22% | 0.280375 |
| IQ3_S word reads, replication | 7.630 / 7.614 | -0.18% | -0.237033 |
| Submit every 8 layers, final replication | 8.990 / 9.012 | +0.27% | 0.304025 |

Absolute rates drift strongly across series; infer effects only from pairs,
not from different rows. The harness prints per-token differences and eight
block aggregates, so uncertainty can be inspected rather than hidden.
Its `naive_ci95_ms` uses Student-t df=119 but **assumes independent tokens**;
adjacent tokens are correlated, so this is descriptive and not evidence for
promotion. IQ3_S first/replica naive intervals were respectively +/-0.392046
and +/-0.642020 ms. The first +0.35% alone does not justify a speed claim.

The final instrumented scheduler separates CPU encoding up to synchronization,
the final CPU wait, and summed command-buffer GPU durations. GPU and CPU
intervals overlap for prefix submission and must not be added naively.

| Schedule | CPU encoding baseline / candidate ms | Final wait ms | GPU command durations ms |
| --- | ---: | ---: | ---: |
| Every 8 layers, first | 0.349293 / 0.451105 | 111.457926 / 110.997014 | 111.237632 / 111.184545 |
| Every 8 layers, replica | 0.351395 / 0.448505 | 126.510558 / 126.131361 | 126.257473 / 126.288785 |
| Every 8 layers, final | 0.343859 / 0.437091 | 110.579220 / 110.180417 | 110.357492 / 110.353928 |
| Every 32 layers | 0.352838 / 0.369271 | 116.986099 / 117.060157 | 116.758549 / 117.026186 |

Only about 0.35 ms/token is spent encoding the complete dense stream,
against over 110 ms of GPU work. This places a measured bound of roughly
0.3% on eliminating that host delay alone. Prefix submissions recover only
a small part while adding buffers and synchronization bookkeeping. They
cannot close the remaining gap by themselves, so no default change is made.
The next substantial target remains GPU projection/recurrence work, not
agent prompts or executor complexity.

## Reproduce

```sh
make qw3-decode-trials-bench
caffeinate -disu make test-decode-trials-quant
caffeinate -disu ./qw3-decode-trials-bench \
  ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf --full constant
# Other private modes: shuffle, iq3s_words, flush8, flush16, flush32.
```

Source extraction and replacement counts are asserted. The shell quant test
requires 42 finite exact GPU comparisons, CPU max <=1e-4 and intact guards,
without a speed threshold. The final full harness additionally checks
initial prefix logits, exact token history and canonical live K/V,
DeltaNet and convolution state after generation. Unused KV capacity and
derived FlashAttention scratch are not compared; all state readbacks are
outside timing.

The final `flush8` replication passed these canonical-state checks at
154 committed tokens, as well as all prefix/continuation logits. The 42-case
quant test passed. Production regressions also passed:
`test-runtime-fingerprint` (two tests), `test-dense-metal-bringup`,
`test-dense-agent`, `test-agent-verify`, and `test-metal-logits` (MoE).
These validate the unchanged default decode, not a promoted candidate.

Logs: `/tmp/qw3-decode-trials-quant-final.log`,
`/tmp/qw3-iq4-lookup-full-constant.log`,
`/tmp/qw3-iq4-lookup-full-flush16.log`,
`/tmp/qw3-iq4-lookup-full-flush8.log`,
`/tmp/qw3-iq4-lookup-full-flush32.log`,
`/tmp/qw3-iq4-lookup-full-flush8-repeat.log`,
`/tmp/qw3-iq4-lookup-full-iq3words.log`,
`/tmp/qw3-iq4-lookup-full-iq3words-repeat.log`,
`/tmp/qw3-decode-trials-flush8-final.log`,
`/tmp/qw3-decode-trials-production-regressions.log`.

The target remains unachieved. None of these decode trials proves matching
or beating local llama.cpp on both pp128 and tg32. The preceding production
prefill-copy improvement remains enabled and is not reverted here.
