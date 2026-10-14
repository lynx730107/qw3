"""Verify frozen IQ2_S grid candidates and their complete raw numerical gate."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--index', action='store_true')
args = parser.parse_args()
archive = Path(__file__).resolve().parent
root = archive.parents[2]
m = json.loads((archive / 'manifest.json').read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


for name, digest in m['source_sha256'].items():
    frozen = (archive / m['source_snapshot_paths'][name]).resolve()
    assert sha(root / name) == sha(frozen) == digest, name
    if args.index:
        data = subprocess.run(['git', 'show', ':' + str(frozen.relative_to(root))],
                              cwd=root, capture_output=True, check=True).stdout
        assert hashlib.sha256(data).hexdigest() == digest, name
for name, digest in {**m['binary_sha256'], **m['production_binary_sha256']}.items():
    assert sha(root / name) == digest, name
assert all(m[k] for k in ('source_unchanged', 'binary_unchanged', 'model_unchanged',
                         'store_unchanged', 'all_passed'))
assert not (archive / 'build.stderr').read_text()
assert m['completed_utc'] and len(m['runs']) == 3
assert all(r['exit_code'] == 0 for r in m['runs'])
assert [r['name'] for r in m['runs']] == ['matrix_v1', 'matrix_v2', 'model']
model = Path(m['model'])
assert (model.stat().st_size, model.stat().st_mtime_ns) == (m['model_size'], m['model_mtime_ns'])
for v in (1, 2):
    out = (archive / f'matrix_v{v}.stdout').read_text()
    assert len(re.findall(r'^matrix_exact .*native_threads=256 candidate_threads=256 guards=exact input=unchanged$', out, re.M)) == 192
    assert f'PASS iq2s_grid={v} matrix_cases=192 finite_bit_exact=all' in out
    log = (archive / f'matrix_v{v}.stderr').read_text()
    assert 'Metal API Validation Enabled' in log
    for variant, size in ((0, 8192), (1, 2048), (2, 0)):
        assert re.search(rf'iq2s_grid_metadata variant={variant} width=32 max_threads=\d+ static_threadgroup_bytes={size} dynamic_threadgroup_bytes=128 threads=256', log)
out = (archive / 'model.stdout').read_text()
assert re.findall(r'^PASS iq2s_grid_model phase=(\d) prefix=128 continuation=32 finite_logits=bit_exact canonical_state=exact history=exact calls=(\d+) sessions=1 additional_weight_buffer_bytes=0$', out, re.M) == [('0', '0'), ('1', '480'), ('2', '480')]
assert 'Metal API Validation Enabled' in (archive / 'model.stderr').read_text()

native = (root / 'metal/qw3_core_linear.metal').read_text()
shader = (root / 'tests/.iq2s_grid/kernels.metal').read_text()


def extract(text, prefix):
    a = text.index(prefix)
    return text[a:text.index('\n}\n', a) + 3]


helper = extract(native, 'template<ushort Kind>\ninline float qw3_iq2_dot32_vector(')
kernel = extract(native, 'template<ushort Kind, ushort Simdgroups = 2>\nkernel void qw3_matvec_iq2_fast(')
start = kernel.index('    for (uint i = tid; i < NG;')
end = kernel.index('    if (Kind != 2u)', start)
stage = kernel[start:end]
for v in (1, 2):
    h = extract(shader, f'template<ushort Kind>\ninline float private_iq2s_grid_dot{v}(')
    space = 'threadgroup' if v == 1 else 'device'
    h = h.replace(f'private_iq2s_grid_dot{v}', 'qw3_iq2_dot32_vector')
    h = h.replace(space + ' const ushort *grid', 'threadgroup const uchar4 *grid')
    h = h.replace('float4(private_iq2s_grid4(grid[idx], 0u))', 'float4(grid[2u * idx])')
    h = h.replace('float4(private_iq2s_grid4(grid[idx], 1u))', 'float4(grid[2u * idx + 1u])')
    assert h == helper
    k = extract(shader, f'template<ushort Kind, ushort Simdgroups = 2>\nkernel void private_iq2s_grid_kernel{v}(')
    k = k.replace(f'private_iq2s_grid_kernel{v}', 'qw3_matvec_iq2_fast').replace(f'private_iq2s_grid_dot{v}', 'qw3_iq2_dot32_vector')
    if v == 1:
        k = k.replace('    threadgroup ushort grid[NG];', '    threadgroup uchar4 grid[NG * 2u];')
        k = k.replace('    for (uint i = tid; i < NG; i += Simdgroups * 32u)\n        grid[i] = grid_table[i];\n', stage)
    else:
        k = k.replace('    device const ushort *grid = grid_table;', '    threadgroup uchar4 grid[NG * 2u];')
        k = k.replace('    if (Kind != 2u)', stage + '    if (Kind != 2u)')
    assert k == kernel
assert 'private_iq2s_grid_kernel1<2, 8>;' in shader
assert 'private_iq2s_grid_kernel2<2, 8>;' in shader
ng = extract(native, 'inline float qw3_iq2_grid_value(')
cg = extract(shader, 'inline uchar4 private_iq2s_grid4(')
old = [float(x.strip().rstrip('f')) for x in re.search(r'values\[4\] = \{(.*?)\}', ng).group(1).split(',')]
new = [int(x) for x in re.search(r'values\[4\] = \{(.*?)\}', cg).group(1).split(',')]
assert len(old) == len(new) == 4
for packed in range(65536):
    for j in range(8):
        half_index, component = divmod(j, 4)
        q = packed >> (8 * half_index)
        assert struct.pack('f', old[(packed >> (2 * j)) & 3]) == struct.pack('f', new[(q >> (2 * component)) & 3])
result = {'index_verified': args.index, 'source_references_verified': len(m['source_sha256']),
          'guarded_matrix_cases': 384, 'model_phases': 3, 'candidate_dispatches_per_continuation': 480,
          'single_reused_ctx256_session': True, 'cpu_grid_scalar_magnitudes_exact': 524288,
          'native_floating_point_expressions_loops_reductions_preserved': True,
          'static_threadgroup_bytes': [8192, 2048, 0], 'dynamic_threadgroup_bytes': 128,
          'threads_per_group': 256, 'additional_weight_or_session_data_buffers': 0,
          'scope': 'private prefix128/greedy32 gate; eight prefixes/checkpoints needed before adoption'}
(archive / ('verification_index.json' if args.index else 'verification.json')).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
