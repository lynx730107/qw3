"""Verify frozen IQ3 shape sources and numerical evidence, including failures."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--index', action='store_true')
args = parser.parse_args()
archive = Path(__file__).resolve().parent
root = archive.parents[2]
manifest = json.loads((archive / 'manifest.json').read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

for name, expected in manifest['source_sha256'].items():
    frozen = (archive / manifest['source_snapshot_paths'][name]).resolve()
    assert sha(root / name) == sha(frozen) == expected, name
    if args.index:
        staged = subprocess.run(['git', 'show', ':'+str(frozen.relative_to(root))],
                                cwd=root, capture_output=True, check=True).stdout
        assert hashlib.sha256(staged).hexdigest() == expected, name
for name, expected in {**manifest['binary_sha256'], **manifest['production_binary_sha256']}.items():
    assert sha(root / name) == expected, name
assert all(manifest[k] for k in ('source_unchanged', 'binary_unchanged', 'model_unchanged', 'store_unchanged'))
assert manifest['store_before'] == manifest['store_after']
model = Path(manifest['model'])
assert (model.stat().st_size, model.stat().st_mtime_ns) == (manifest['model_size'], manifest['model_mtime_ns'])
assert not (archive / 'build.stderr').read_text()
source = (root / 'metal/qw3_core_moe.metal').read_text()
name = 'qw3_matvec_iq3_s_fast'
a = source.index('kernel void '+name+'(')
original = source[a:source.index('\n}\n', a)+3]
shader = (root / 'tests/.iq3_shape/kernels.metal').read_text()
for variant in (1, 2):
    private = f'private_iq3_shape_fast{variant}'
    a = shader.index('kernel void '+private+'(')
    body = shader[a:shader.index('\n}\n', a)+3]
    restored = body.replace(private, name).replace('private_iq3_shape_nb32',
        '(args.n_in / 256u) * 8u').replace('private_iq3_shape_stride', 'args.row_bytes')
    assert restored == original

assert [r['name'] for r in manifest['runs'][:2]] == ['matrix_v1', 'matrix_v2']
passed_cases = 0
for variant, run in zip((1, 2), manifest['runs']):
    output = (archive / f'matrix_v{variant}.stdout').read_text()
    rows = re.findall(r'^matrix_exact .*guards=exact input=unchanged$', output, re.M)
    assert len(rows) == run['matrix_cases_passed']
    passed_cases += len(rows)
    if run['exit_code'] == 0:
        assert len(rows) == 48
        assert output.count('input_offset=4 guards=') == output.count('input_offset=16 guards=') == 24
        assert f'PASS iq3_shape={variant} matrix_cases=48 finite_bit_exact=all' in output
        metadata = (archive / f'matrix_v{variant}.stderr').read_text()
        assert len(re.findall(r'^iq3_shape_metadata .*width=32 .*static_threadgroup_bytes=1024 .*threads=64$', metadata, re.M)) == 10

matrices_passed = all(r['exit_code'] == 0 for r in manifest['runs'][:2])
if matrices_passed:
    assert len(manifest['runs']) == 3 and manifest['runs'][2]['name'] == 'model'
    if manifest['runs'][2]['exit_code'] == 0:
        output = (archive / 'model.stdout').read_text()
        rows = re.findall(r'^PASS iq3_shape_model phase=(\d) prefix=128 continuation=32 finite_logits=bit_exact canonical_state=exact history=exact calls=(\d+) sessions=1 additional_weight_buffer_bytes=0$', output, re.M)
        assert rows == [('0', '0'), ('1', '2336'), ('2', '2336')]
else:
    assert len(manifest['runs']) == 2 and not (archive / 'model.stdout').exists()
assert manifest['all_passed'] == (matrices_passed and manifest['runs'][-1]['exit_code'] == 0)
result = {'index_verified': args.index, 'source_references_verified': len(manifest['source_sha256']),
          'integer_only_shader_changes_verified': True, 'matrix_cases_completed': passed_cases,
          'numerical_gate_passed': manifest['all_passed'],
          'production_binaries_unchanged_during_candidate_gate': True,
          'additional_weight_and_session_data_buffers': 0,
          'scope': 'private numerical screen; extended prefix/checkpoint gate required before adoption'}
(archive / ('verification_index.json' if args.index else 'verification.json')).write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
