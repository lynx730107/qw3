"""Verify unchanged shaders, the bounded scheduling edit, and raw model equality."""
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
m = json.loads((archive / 'manifest.json').read_text())


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


for name, digest in m['source_sha256'].items():
    frozen = (archive / m['source_snapshot_paths'][name]).resolve()
    assert sha(root / name) == sha(frozen) == digest, name
    if args.index:
        data = subprocess.run(['git', 'show', ':' + str(frozen.relative_to(root))], cwd=root,
                              capture_output=True, check=True).stdout
        assert hashlib.sha256(data).hexdigest() == digest, name
assert sha(Path(m['binary'])) == m['binary_sha256']
for name, digest in m['production_binary_sha256'].items():
    assert sha(root / name) == digest
assert all(m[k] for k in ('all_passed', 'source_unchanged', 'binary_unchanged',
                         'production_unchanged', 'model_unchanged', 'store_unchanged'))
assert m['exit_code'] == 0 and m['completed_utc']
assert not (archive / 'build.stderr').read_text()
assert not (archive / 'native_build.stderr').read_text()
out = (archive / 'model.stdout').read_text()
assert re.findall(r'^PASS ffn_concurrent_model phase=(\d) prefix=128 continuation=32 finite_logits=bit_exact canonical_state=exact history=exact calls=(\d+) sessions=1 additional_weight_buffer_bytes=0$', out, re.M) == [('0', '0'), ('1', '992')]
assert 'Metal API Validation Enabled' in (archive / 'model.stderr').read_text()
native = (root / 'qw3.c').read_text()
private = (root / 'tests/.decode_ffn_concurrent/qw3_private.c').read_text()
private = private.replace('            ok = private_ffn_window_begin() &&\n                 qw3_metal_session_matvec_quant_x1_to_scratch(', '            ok = qw3_metal_session_matvec_quant_x1_to_scratch(')
private = private.replace('\n            private_ffn_window_end();', '')
assert private == native
hooks = (root / 'tests/.decode_ffn_concurrent/hooks.inc').read_text()
assert 'newBuffer' not in hooks and 'newCommandBuffer' not in hooks
assert 'g_batch_concurrent=1;' in hooks and 'g_batch_concurrent=0;' in hooks
assert hooks.count('qw3_metal_close_batch_encoder();') == 2
result = {'index_verified': args.index, 'source_references_verified': len(m['source_sha256']),
          'model_phases': 2, 'candidate_windows_per_continuation': 992,
          'windows_per_token': 31, 'single_reused_ctx256_session': True,
          'full_finite_vocabularies_bit_exact': 66, 'canonical_state_history_exact': True,
          'metal_shaders_changed': False, 'new_weight_or_session_data_buffers': 0,
          'scope': 'prefix128 plus greedy32; eight prefixes/checkpoints and agent regressions remain required before adoption'}
(archive / ('verification_index.json' if args.index else 'verification.json')).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
