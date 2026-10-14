"""Predecode four odd magnitudes into the same ushort[512] local grid."""
from pathlib import Path
import hashlib
import json
import struct

root = Path(__file__).resolve().parent.parent
out = root / 'tests/.iq3_nibble'
out.mkdir(exist_ok=True)
source = (root / 'metal/qw3_core_moe.metal').read_text()


def extract(name):
    a = source.index(name)
    return source[a:source.index('\n}\n', a) + 3]


helper = extract('template<typename Grid>\ninline float qw3_iq3s_dot32_shared(')
kernel = extract('kernel void qw3_matvec_iq3_s_fast(')
packing = '''inline ushort private_iq3_nibble_pack(ushort packed) {
    const uint p = uint(packed);
    return ushort((2u * (p & 7u) + 1u) |
        ((2u * ((p >> 3u) & 7u) + 1u) << 4u) |
        ((2u * ((p >> 6u) & 7u) + 1u) << 8u) |
        ((2u * ((p >> 9u) & 7u) + 1u) << 12u));
}
inline float private_iq3_nibble_value(ushort packed, uint j) {
    return float((uint(packed) >> (4u * j)) & 15u);
}
'''
h = helper.replace('qw3_iq3s_dot32_shared', 'private_iq3_nibble_dot')
h = h.replace('qw3_iq3s_grid_value', 'private_iq3_nibble_value')
assert h.replace('private_iq3_nibble_dot', 'qw3_iq3s_dot32_shared').replace('private_iq3_nibble_value', 'qw3_iq3s_grid_value') == helper
k = kernel.replace('qw3_matvec_iq3_s_fast', 'private_iq3_nibble_fast1')
k = k.replace('qw3_iq3s_dot32_shared', 'private_iq3_nibble_dot')
original = 'grid[i] = kgrid[i];'
changed = 'grid[i] = private_iq3_nibble_pack(kgrid[i]);'
assert k.count(original) == 1
k = k.replace(original, changed)
assert k.replace('private_iq3_nibble_fast1', 'qw3_matvec_iq3_s_fast').replace('private_iq3_nibble_dot', 'qw3_iq3s_dot32_shared').replace(changed, original) == kernel
(out / 'kernels.metal').write_text(packing + h + k)

base = root / 'docs/measurements/iq3xxs_sign_2026-10-14'
m = json.loads((base / 'manifest.json').read_text())


def frozen(name):
    key = 'tests/.iq3xxs_sign/' + name
    data = (base / m['source_snapshot_paths'][key]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == m['source_sha256'][key]
    return data.decode()


def rename(text):
    return (text.replace('iq3xxs_sign', 'iq3_nibble').replace('IQ3XXSSign', 'IQ3Nibble')
            .replace('iq3xxsSign', 'iq3Nibble').replace('IQ3XXS sign', 'IQ3 nibble')
            .replace('IQ3XXS_SIGN', 'IQ3_NIBBLE').replace('sign_variant', 'nibble_variant')
            .replace('g_matvec_iq3_xxs_pipeline', 'g_matvec_iq3_s_pipeline')
            .replace('sign-parity/direct-sign-table', 'predecoded-nibbles'))


hooks = rename(frozen('hooks.inc'))
hooks = hooks.replace('pipeline[3]', 'pipeline[2]')
hooks = hooks.replace('pipeline[1] || pipeline==iq3_nibble_pipeline[2]', 'pipeline[1]')
hooks = hooks.replace('if (iq3_nibble_pipeline[2])', 'if (iq3_nibble_pipeline[1])')
hooks = hooks.replace('v<=2', 'v<=1').replace('staticThreadgroupMemoryLength!=4096u', 'staticThreadgroupMemoryLength!=1024u')
hooks = hooks.replace('IQ3_XXS sign parity/direct table; original FP32 vector dot and data buffers.',
                      'IQ3_S predecoded nibbles; same 1024-byte local grid and original FP32 scalar order.')
(out / 'hooks.inc').write_text(hooks)
(out / 'model.inc').write_text(rename(frozen('model.inc')).replace('phase<=2', 'phase<=1').replace('32u*34u', '32u*73u'))
native = rename(frozen('native.inc')).replace(' && strcmp(mode,"2")', '')
(out / 'native.inc').write_text(native)

for p in range(65536):
    packed = sum((2 * ((p >> (3 * j)) & 7) + 1) << (4 * j) for j in range(4))
    for j in range(4):
        assert struct.pack('f', 2 * ((p >> (3 * j)) & 7) + 1) == struct.pack('f', (packed >> (4 * j)) & 15)
(out / 'grid_check.json').write_text(json.dumps({
    'packed_words': 65536, 'scalar_magnitudes': 262144,
    'cpu_float32_bits_equal': True, 'static_threadgroup_bytes_before_after': [1024, 1024],
    'gpu_or_model_equivalence_proven': False}, indent=2) + '\n')
print('IQ3_S nibble sources generated; CPU magnitudes exact; GPU/model gate still required')
