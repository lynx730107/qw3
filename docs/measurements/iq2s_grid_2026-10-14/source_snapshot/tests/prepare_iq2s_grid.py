"""Private IQ2_S sg8 packed-grid candidates; preserve native FP32 dot order."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent.parent
out = root / 'tests/.iq2s_grid'
out.mkdir(exist_ok=True)
source = (root / 'metal/qw3_core_linear.metal').read_text()


def extract(name, prefix):
    start = source.index(prefix + name)
    return source[start:source.index('\n}\n', start) + 3]


helper = extract('qw3_iq2_dot32_vector(', 'template<ushort Kind>\ninline float ')
kernel = extract('qw3_matvec_iq2_fast(', 'template<ushort Kind, ushort Simdgroups = 2>\nkernel void ')
original_grid = '''    threadgroup uchar4 grid[NG * 2u];'''
stage_start = kernel.index('    for (uint i = tid; i < NG;')
stage_end = kernel.index('    if (Kind != 2u)', stage_start)
original_stage = kernel[stage_start:stage_end]
original_loads = [
    'float4(grid[2u * idx])', 'float4(grid[2u * idx + 1u])'
]
expanded = '''inline uchar4 private_iq2s_grid4(ushort packed, uint half_index) {
    constexpr uchar values[4] = { 8, 25, 43, 0 };
    const uint q = uint(packed) >> (8u * half_index);
    return uchar4(values[q & 3u], values[(q >> 2u) & 3u],
                  values[(q >> 4u) & 3u], values[(q >> 6u) & 3u]);
}
'''
shader = expanded
for variant in (1, 2):
    helper_name = f'private_iq2s_grid_dot{variant}'
    kernel_name = f'private_iq2s_grid_kernel{variant}'
    pointer = ('threadgroup' if variant == 1 else 'device') + ' const ushort *grid'
    new_loads = [f'float4(private_iq2s_grid4(grid[idx], {i}u))' for i in (0, 1)]
    h = helper.replace('qw3_iq2_dot32_vector', helper_name)
    h = h.replace('threadgroup const uchar4 *grid', pointer)
    for original, new in zip(original_loads, new_loads):
        assert h.count(original) == 1
        h = h.replace(original, new)
    restored = h.replace(helper_name, 'qw3_iq2_dot32_vector').replace(pointer, 'threadgroup const uchar4 *grid')
    for original, new in zip(original_loads, new_loads):
        restored = restored.replace(new, original)
    assert restored == helper
    k = kernel.replace('qw3_matvec_iq2_fast', kernel_name).replace('qw3_iq2_dot32_vector', helper_name)
    if variant == 1:
        declaration = '    threadgroup ushort grid[NG];'
        stage = '''    for (uint i = tid; i < NG; i += Simdgroups * 32u)
        grid[i] = grid_table[i];
'''
    else:
        declaration = '    device const ushort *grid = grid_table;'
        stage = ''
    assert k.count(original_grid) == 1 and k.count(original_stage) == 1
    k = k.replace(original_grid, declaration).replace(original_stage, stage)
    # Keep the original barrier in both candidates: isolate grid representation.
    restored = k.replace(kernel_name, 'qw3_matvec_iq2_fast').replace(helper_name, 'qw3_iq2_dot32_vector')
    restored = restored.replace(declaration, original_grid)
    if stage:
        restored = restored.replace(stage, original_stage)
    else:
        restored = restored.replace('    if (Kind != 2u)', original_stage + '    if (Kind != 2u)')
    assert restored == kernel
    shader += h + k + f'''typedef decltype({kernel_name}<2, 8>) private_iq2s_grid_type{variant};
template [[host_name("private_iq2s_grid_fast{variant}")]] kernel private_iq2s_grid_type{variant} {kernel_name}<2, 8>;
'''
(out / 'kernels.metal').write_text(shader)

base = root / 'docs/measurements/iq3xxs_sign_2026-10-14'
manifest = json.loads((base / 'manifest.json').read_text())


def frozen(name):
    key = 'tests/.iq3xxs_sign/' + name
    data = (base / manifest['source_snapshot_paths'][key]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == manifest['source_sha256'][key]
    return data.decode()


def rename(text):
    return (text.replace('iq3xxs_sign', 'iq2s_grid').replace('IQ3XXSSign', 'IQ2SGrid')
            .replace('iq3xxsSign', 'iq2sGrid').replace('IQ3XXS sign', 'IQ2S grid')
            .replace('IQ3XXS_SIGN', 'IQ2S_GRID').replace('sign_variant', 'grid_variant')
            .replace('g_matvec_iq3_xxs_pipeline', 'g_matvec_iq2_s_sg8_pipeline')
            .replace('sign-parity/direct-sign-table', 'packed-grid/device-grid'))


hooks = rename(frozen('hooks.inc'))
hooks = hooks.replace('IQ3_XXS sign parity/direct table; original FP32 vector dot and data buffers.',
                      'IQ2_S sg8 packed/device grid; original FP32 vector dot and data buffers.')
hooks = hooks.replace('maxTotalThreadsPerThreadgroup<64', 'maxTotalThreadsPerThreadgroup<256')
hooks = hooks.replace('staticThreadgroupMemoryLength!=4096u', 'staticThreadgroupMemoryLength!=(v==1 ? 2048u : 0u)')
hooks = hooks.replace('threads=64', 'threads=256')
(out / 'hooks.inc').write_text(hooks)
(out / 'model.inc').write_text(rename(frozen('model.inc')).replace('32u*34u', '32u*15u'))
(out / 'native.inc').write_text(rename(frozen('native.inc')))
print('IQ2_S candidates generated; source restoration exact; no build or GPU run')
