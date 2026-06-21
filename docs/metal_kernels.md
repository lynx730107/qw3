# Metal Kernel Sources

The core Metal backend is loaded from source files at runtime. The default
layout is the repository `metal/` directory.

## Core Load Order

The split core kernel files are concatenated in this exact order:

1. `qw3_core_common.metal`
2. `qw3_core_linear.metal`
3. `qw3_core_sequence.metal`
4. `qw3_core_deltanet.metal`
5. `qw3_core_moe.metal`
6. `qw3_core_argmax.metal`

The order matters because later files use helpers, constants, and type
definitions from earlier files.

## Optional FlashAttention Source

`flash_attn.metal` is optional. It is appended after the core source when
FlashAttention is enabled or when an explicit FlashAttention source is provided.

Useful environment variables:

- `QW3_METAL_KERNEL_DIR=/path/to/metal`: load the split core files from a custom
  directory.
- `QW3_METAL_KERNEL_SOURCE=/path/to/qw3_kernels.metal`: load one monolithic core
  source file instead of the split files.
- `QW3_METAL_FLASH_ATTN_SOURCE=/path/to/flash_attn.metal`: load a specific
  FlashAttention source.
- `QW3_METAL_GQA_FLASH_ATTN=0`: disable the optional GQA FlashAttention path.

`QW3_METAL_KERNEL_SOURCE` takes precedence over `QW3_METAL_KERNEL_DIR`.

If `QW3_METAL_KERNEL_DIR` is set and one of the split files is missing, startup
fails immediately and prints the exact missing path. This is intentional: a
custom kernel directory should never silently fall back to another source tree.

## Regression Checks

After touching any Metal source file or the runtime loader, run:

```sh
make test-regression
```

For larger loader or kernel moves, run the full suite:

```sh
make test-regression-full
```
