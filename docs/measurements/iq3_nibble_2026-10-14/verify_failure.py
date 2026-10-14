"""Verify the first finite bit mismatch and the absence of later model/timing runs."""
import argparse
import hashlib
import json
import math
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
assert not m['all_passed']
assert all(m[k] for k in ('source_unchanged', 'binary_unchanged', 'model_unchanged', 'store_unchanged'))
assert not (archive / 'build.stderr').read_text()
assert len(m['runs']) == 1 and m['runs'][0]['name'] == 'matrix_v1'
assert m['runs'][0]['exit_code'] == 1 and m['runs'][0]['matrix_cases_passed'] == 2
assert len(re.findall(r'^matrix_exact .*guards=exact input=unchanged$', (archive / 'matrix_v1.stdout').read_text(), re.M)) == 2
log = (archive / 'matrix_v1.stderr').read_text()
assert 'Metal API Validation Enabled' in log
for v in (0, 1):
    assert f'iq3_nibble_metadata variant={v} width=32 max_threads=1024 static_threadgroup_bytes=1024 dynamic_threadgroup_bytes=128 threads=64' in log
failure = re.search(r'IQ3 nibble mismatch n_in=(\d+) n_out=(\d+) pattern=(\d+) index=(\d+) bits=([0-9a-f]+)/([0-9a-f]+)', log)
assert failure and failure.groups() == ('256', '1', '0', '5', 'c15ab463', 'c15ab465')
bits = [int(v, 16) for v in failure.groups()[-2:]]
assert all(math.isfinite(struct.unpack('<f', struct.pack('<I', v))[0]) for v in bits)
assert abs(bits[0] - bits[1]) == 2
assert not (archive / 'model.stdout').exists() and not (archive / 'native').exists()
native = (root / 'metal/qw3_core_moe.metal').read_text()
shader = (root / 'tests/.iq3_nibble/kernels.metal').read_text()


def extract(text, prefix):
    a = text.index(prefix)
    return text[a:text.index('\n}\n', a) + 3]


helper = extract(native, 'template<typename Grid>\ninline float qw3_iq3s_dot32_shared(')
k = extract(shader, 'template<typename Grid>\ninline float private_iq3_nibble_dot(')
assert k.replace('private_iq3_nibble_dot', 'qw3_iq3s_dot32_shared').replace('private_iq3_nibble_value', 'qw3_iq3s_grid_value') == helper
kernel = extract(native, 'kernel void qw3_matvec_iq3_s_fast(')
k = extract(shader, 'kernel void private_iq3_nibble_fast1(')
assert k.replace('private_iq3_nibble_fast1', 'qw3_matvec_iq3_s_fast').replace('private_iq3_nibble_dot', 'qw3_iq3s_dot32_shared').replace('grid[i] = private_iq3_nibble_pack(kgrid[i]);', 'grid[i] = kgrid[i];') == kernel
result = {'index_verified': args.index, 'source_references_verified': len(m['source_sha256']),
          'completed_exact_matrix_cases': 2, 'first_failure_ulp_distance': 2,
          'first_failure_bits': failure.groups()[-2:], 'finite_failure': True,
          'first_failure_shape': [256, 1], 'pattern': 0, 'output_index': 5,
          'static_threadgroup_bytes_before_after': [1024, 1024],
          'model_test_executed': False, 'timing_executed': False,
          'production_unchanged': True, 'adopted': False,
          'scope': 'CPU magnitude equality does not establish compiled GPU bit equality; no codegen cause established'}
(archive / ('verification_index.json' if args.index else 'verification.json')).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
