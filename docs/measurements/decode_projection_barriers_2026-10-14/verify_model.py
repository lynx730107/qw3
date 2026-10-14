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
assert re.findall(r'^PASS projection_barriers_model phase=(\d) prefix=128 continuation=32 finite_logits=bit_exact canonical_state=exact history=exact calls=(\d+) sessions=1 additional_weight_buffer_bytes=0$', out, re.M) == [('0', '0'), ('1', '2528')]
assert 'Metal API Validation Enabled' in (archive / 'model.stderr').read_text()
native = (root / 'qw3.c').read_text()
private = (root / 'tests/.decode_projection_barriers/qw3_private.c').read_text()
private = private.replace('            ok = private_ffn_window_begin(2u) &&\n                 qw3_metal_session_matvec_quant_x1_to_scratch(', '            ok = qw3_metal_session_matvec_quant_x1_to_scratch(')
private = private.replace('            ok = private_ffn_window_begin(4u) &&\n                 qw3_metal_session_matvec_quant_x1_to_scratch(', '            ok = qw3_metal_session_matvec_quant_x1_to_scratch(')
private = private.replace('\n            private_ffn_window_end();', '')
private = private.replace('int ok = private_decode_begin() &&', 'int ok = qw3_metal_begin_commands() &&')
private = private.replace('{private_decode_end(); return -1;}', 'return -1;').replace('private_decode_end(); return 0;', 'return 0;')
assert private == native
hooks = (root / 'tests/.decode_projection_barriers/hooks.inc').read_text()
assert 'newBuffer' not in hooks and 'newCommandBuffer' not in hooks
assert 'qw3_metal_close_batch_encoder' not in hooks
assert 'qw3_metal_begin_commands_concurrent()' in hooks
bridge = (root/'tests/.decode_projection_barriers/qw3_private_metal.m').read_text()
bridge = re.sub(r'\[\(id<QW3FFNBarrierHooks>\)([A-Za-z_]\w*) privateDispatch(Threadgroups|Threads):',lambda m:'['+m[1]+' dispatch'+m[2]+':',bridge)
assert bridge==(root/'qw3_metal.m').read_text()
assert re.findall(r'^dispatch_stats phase=(\d) dispatches=(\d+) barriers=(\d+) independent_skips=(\d+)$',out,re.M)==[('0','0','0','0'),('1','31296','25696','5600')]
assert re.findall(r'^scope_stats phase=(\d) two_windows=(\d+) four_windows=(\d+)$',out,re.M)==[('0','0','0'),('1','992','1536')]
result = {'index_verified': args.index, 'source_references_verified': len(m['source_sha256']),
          'model_phases': 2, 'candidate_windows_per_continuation': 2528,
          'windows_per_token': 79,'two_dispatch_windows_per_token':31,'four_dispatch_windows_per_token':48, 'single_reused_ctx256_session': True,
          'full_finite_vocabularies_bit_exact': 66, 'canonical_state_history_exact': True,
          'dispatches_per_token':978,'barriers_per_token':803,'independent_skips_per_token':175,
          'extra_compute_encoders_per_token':0,'metal_shaders_changed': False, 'new_weight_or_session_data_buffers': 0,
          'scope': 'prefix128 plus greedy32; eight prefixes/checkpoints and agent regressions remain required before adoption'}
(archive / ('verification_index.json' if args.index else 'verification.json')).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
