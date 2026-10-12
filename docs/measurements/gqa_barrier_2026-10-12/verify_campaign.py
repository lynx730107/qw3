"""Verify frozen inputs, the numerical gate and every recorded timing sample."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--index', action='store_true')
args = parser.parse_args()
base = Path(__file__).resolve().parent
root = base.parents[2]
m = json.loads((base / 'manifest.json').read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for name, expected in m['source_sha256'].items():
    frozen = (base / m['source_snapshot_paths'][name]).resolve()
    assert sha(frozen) == expected, ('snapshot', name)
    assert sha(root / name) == expected, ('current', name)
    if args.index:
        content = subprocess.check_output(['git', 'show', ':' + str(frozen.relative_to(root))], cwd=root)
        assert hashlib.sha256(content).hexdigest() == expected, ('index', name)
for name, expected in {**m['binary_sha256'], **m['production_binary_sha256']}.items():
    assert sha(root / name) == expected, ('binary', name)
model = Path(m['model'])
assert (model.stat().st_size, model.stat().st_mtime_ns) == (m['model_size'], m['model_mtime_ns'])
assert all(m[k] for k in ('source_unchanged', 'binary_unchanged', 'model_unchanged', 'store_unchanged', 'all_passed'))
assert not (base / 'build.stderr').read_text()
assert len(m['runs']) == 3 and all(r['exit_code'] == 0 for r in m['runs'])
for v in (1, 2):
    output = (base / f'matrix_v{v}.stdout').read_text()
    assert len(re.findall(r'^matrix_exact .*guards=exact input=unchanged$', output, re.M)) == 288
    assert f'PASS gqa_barrier={v} matrix_cases=288 finite_bit_exact=all' in output
out = (base / 'model.stdout').read_text()
assert len(re.findall(r'^PASS gqa_barrier_model .*finite_logits=bit_exact canonical_state=exact history=exact .*sessions=1', out, re.M)) == 3
assert 'phase=0' in out and 'calls=0' in out
assert out.count('calls=512') == 2
assert (base / 'model.stderr').read_text().count('calls=512 fallbacks=0') == 2
source=(root/'metal/qw3_core_sequence.metal').read_text()
name='qw3_gqa_attend_n_inner'
a=source.index('kernel void '+name+'(')
original=source[a:source.index('\n}\n',a)+3]
shader=(root/'tests/.gqa_barrier/kernels.metal').read_text()
for v in (1,2):
    private_name=f'private_gqa_barrier{v}'
    a=shader.index('kernel void '+private_name+'(')
    body=shader[a:shader.index('\n}\n',a)+3]
    assert body.count('threadgroup_barrier(')==2
    restored=body.replace(private_name,name).replace('group_heads > 7u','group_heads > 8u')
    restored=restored.replace('if (lane == 0 && gh < group_heads) sh[gh * 8u + uint(simd_idx)] = part;', 'if (lane == 0) sh[gh * 8u + uint(simd_idx)] = part;')
    restored=restored.replace('if (tid == 0 && gh < group_heads) sh[56u + gh] = dot;', 'if (tid == 0) sh[gh] = dot;')
    restored=restored.replace('float score = sh[56u + gh] * scale;', 'float score = sh[gh] * scale;')
    restored=restored.replace('    }\n    if (i < args.head_dim)', '        threadgroup_barrier(mem_flags::mem_threadgroup);\n    }\n    if (i < args.head_dim)')
    if v==2: restored=restored.replace('gh < group_heads; gh++','gh < 8u; gh++')
    assert restored==original
    assert 'calls=216 fallbacks=72' in (base/f'matrix_v{v}.stderr').read_text()
native = json.loads((base / 'native/manifest.json').read_text())
assert native['order'] == [0, 1, 0, 2, 0] and len(native['runs']) == 5
assert all(native[k] for k in ('source_unchanged', 'binary_unchanged', 'model_unchanged'))
assert native['source_sha256'] == m['source_sha256']
assert native['production_binary_sha256'] == m['production_binary_sha256']
session_bytes = []
for i, run in enumerate(native['runs']):
    assert run['exit_code'] == 0 and run['variant'] == native['order'][i]
    assert len(run['samples']) == 3 and run['candidate_calls'] == (1552 if run['variant'] else 0)
    assert run['environment'].get('MTL_DEBUG_LAYER') is None
    log = (base / 'native' / (run['name'] + '.stderr')).read_text()
    parsed = [{'repetition': int(n), 'tps': float(t), 'ms': float(ms)}
              for n, t, ms in re.findall(r'qw3-bench: sample (\d+)/3 ([\d.]+) tok/s ([\d.]+) ms', log)]
    assert parsed == run['samples']
    counts = re.search(r'gqa_barrier_dispatch_counts calls=(\d+) fallbacks=(\d+)', log)
    assert counts and tuple(map(int, counts.groups())) == ((1552, 0) if run['variant'] else (0, 0))
    for row in (base / 'native' / (run['name'] + '.stdout')).read_text().splitlines():
        if row.startswith('tg,'):
            session_bytes.append(int(row.split(',')[-1]))
assert session_bytes == [176761680] * 5
analysis = json.loads((base / 'native/analysis.json').read_text())
assert analysis['samples_retained'] == 15
replica = json.loads((base / 'native_retry/manifest.json').read_text())
assert replica['order'] == [0, 1, 0, 1, 0, 1, 0] and len(replica['runs']) == 7
assert all(replica[k] for k in ('source_unchanged', 'binary_unchanged', 'model_unchanged'))
assert replica['initial_manifest_sha256'] == sha(base / 'native/manifest.json')
assert replica['initial_analysis_sha256'] == sha(base / 'native/analysis.json')
assert replica['production_binary_sha256'] == m['production_binary_sha256']
for name, expected in replica['source_sha256'].items():
    frozen = (base / 'native_retry' / replica['source_snapshot_paths'][name]).resolve()
    assert sha(frozen) == expected and sha(root / name) == expected, ('replica source', name)
    if args.index:
        content = subprocess.check_output(['git', 'show', ':' + str(frozen.relative_to(root))], cwd=root)
        assert hashlib.sha256(content).hexdigest() == expected, ('replica index', name)
for i, run in enumerate(replica['runs']):
    assert run['exit_code'] == 0 and run['variant'] == replica['order'][i]
    assert len(run['samples']) == 3 and run['candidate_calls'] == (1552 if run['variant'] else 0)
    assert run['environment'].get('MTL_DEBUG_LAYER') is None
    log = (base / 'native_retry' / (run['name'] + '.stderr')).read_text()
    parsed = [{'repetition': int(n), 'tps': float(t), 'ms': float(ms)}
              for n, t, ms in re.findall(r'qw3-bench: sample (\d+)/3 ([\d.]+) tok/s ([\d.]+) ms', log)]
    assert parsed == run['samples']
    counts = re.search(r'gqa_barrier_dispatch_counts calls=(\d+) fallbacks=(\d+)', log)
    assert counts and tuple(map(int, counts.groups())) == ((1552, 0) if run['variant'] else (0, 0))
    rows = [r for r in (base / 'native_retry' / (run['name'] + '.stdout')).read_text().splitlines() if r.startswith('tg,')]
    assert len(rows) == 1 and int(rows[0].split(',')[-1]) == 176761680
replica_analysis = json.loads((base / 'native_retry/analysis.json').read_text())
assert replica_analysis['samples_retained'] == 21
result = {'index_verified': args.index, 'frozen_source_references_verified': len(m['source_sha256']),
          'replica_source_references_verified': len(replica['source_sha256']),
          'matrix_cases_passed': 576, 'one_session_model_phases_passed': 3,
          'ordinary_samples_verified': 36, 'production_unchanged': True,
          'numerical_store_unchanged': True, 'analysis': analysis,
          'replica_analysis': replica_analysis,
          'scope': 'private screen; no production adoption, no eight-prefix/checkpoint gate, no fresh llama comparison'}
(base / ('verification_index.json' if args.index else 'verification.json')).write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
