"""Aligned integer loads only: retain the native IQ3_S FP32 expressions."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / 'tests/.iq3_words'
out.mkdir(exist_ok=True)
source = (root / 'metal/qw3_core_moe.metal').read_text()


def extract(marker):
    first = source.index(marker)
    last = source.index('\n}\n', first) + 3
    return source[first:last]


value = extract('inline float qw3_iq3s_grid_value(ushort packed, uint j)')
helper = extract('template<typename Grid>\ninline float qw3_iq3s_dot32_shared(')
kernel = extract('kernel void qw3_matvec_iq3_s_fast(')
indices = '''        const uint idx1 = uint(qs[2u * l]) | ((uint(qhb) << (8u - 2u * l)) & 256u);
        const uint idx2 = uint(qs[2u * l + 1u]) | ((uint(qhb) << (7u - 2u * l)) & 256u);'''
words = '''        const uint packed_qs = uint(reinterpret_cast<device const ushort *>(qs)[l]);
        const uint idx1 = (packed_qs & 255u) | ((uint(qhb) << (8u - 2u * l)) & 256u);
        const uint idx2 = (packed_qs >> 8u) | ((uint(qhb) << (7u - 2u * l)) & 256u);'''
assert helper.count(indices) == 1
shader = '#include <metal_stdlib>\nusing namespace metal;\nstruct qw3_matvec_q8_0_args {uint n_in,n_out,row_bytes;};\n' + value
for variant in (1, 2):
    dot = helper.replace(indices, words)
    if variant == 2:
        signs = '    device const uchar *signs = blk + 74 + ib * 4u;'
        assert dot.count(signs) == 1
        dot = dot.replace(signs, signs + '''
    const uint packed_signs0 = uint(reinterpret_cast<device const ushort *>(signs)[0]);
    const uint packed_signs1 = uint(reinterpret_cast<device const ushort *>(signs)[1]);''')
        dot = dot.replace('        const uchar s = signs[l];', '''        const uint packed_signs = l < 2u ? packed_signs0 : packed_signs1;
        const uchar s = uchar((packed_signs >> (8u * (l & 1u))) & 255u);''')
    # 110-byte blocks and both byte offsets (2, 74) are ushort-aligned.
    # No uint/ushort2/ushort4 cast is safe for all original block addresses.
    name = f'private_iq3_words_dot{variant}'
    dot = dot.replace('qw3_iq3s_dot32_shared', name)
    body = kernel.replace('qw3_iq3s_dot32_shared', name)
    body = body.replace('qw3_matvec_iq3_s_fast', f'private_iq3_words_fast{variant}')
    shader += dot + body
(out / 'kernels.metal').write_text(shader)

# Reuse the dispatch-counting bridge and the one-session disk-oracle runners.
def rename(text):
    return (text.replace('iq3_grid', 'iq3_words').replace('IQ3Grid', 'IQ3Words')
            .replace('iq3Grid', 'iq3Words').replace('grid_variant', 'words_variant')
            .replace('IQ3 grid', 'IQ3 words').replace('QW3_PRIVATE_IQ3_GRID', 'QW3_PRIVATE_IQ3_WORDS'))


base = root / 'tests/.iq3_grid'
hooks = rename((base / 'hooks.inc').read_text())
hooks = hooks.replace('Direct reads of the existing packed grid; no extra weight/session buffers.',
                      'Aligned integer loads from original IQ3_S weights; native grid and FP32 math.')
(out / 'hooks.inc').write_text(hooks)
(out / 'matrix.inc').write_text(rename((base / 'matrix.inc').read_text()))
for filename, target in [('test_iq3_grid_model.m', 'model.inc'), ('bench_iq3_grid_native.m', 'native.inc')]:
    text = (root / 'tests' / filename).read_text().replace('#include ".iq3_grid/hooks.inc"\n', '')
    if target == 'native.inc':
        text = text.replace('#include "../qw3_bench.c"', '#include "../../qw3_bench.c"')
    text = text.replace('baseline/device/constant', 'baseline/QS-word/QS-and-sign-word')
    (out / target).write_text(rename(text))
