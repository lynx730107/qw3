"""Validate the exact numerical rejection; no timing/model evidence is claimed."""
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
manifest = json.loads((archive/'manifest.json').read_text())
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

for name, expected in manifest['source_sha256'].items():
    frozen = (archive/manifest['source_snapshot_paths'][name]).resolve()
    assert sha(root/name) == sha(frozen) == expected, name
    if args.index:
        data = subprocess.run(['git','show',':'+str(frozen.relative_to(root))],cwd=root,
                              capture_output=True,check=True).stdout
        assert hashlib.sha256(data).hexdigest() == expected, name
for name, expected in {**manifest['binary_sha256'],**manifest['production_binary_sha256']}.items():
    assert sha(root/name) == expected, name
assert all(manifest[k] for k in ('source_unchanged','binary_unchanged','model_unchanged','store_unchanged'))
assert manifest['completed_utc'] and manifest['all_passed'] is False
assert not (archive/'build.stderr').read_text()
assert [r['name'] for r in manifest['runs']] == ['matrix_v1','matrix_v2']
assert not (archive/'model.stdout').exists() and not (archive/'native').exists()
model = Path(manifest['model'])
assert (model.stat().st_size,model.stat().st_mtime_ns) == (manifest['model_size'],manifest['model_mtime_ns'])
failures = []
for variant, run in zip((1,2),manifest['runs']):
    assert run['exit_code'] == 1
    out = (archive/f'matrix_v{variant}.stdout').read_text()
    err = (archive/f'matrix_v{variant}.stderr').read_text()
    completed = len(re.findall(r'^matrix_exact .*guards=exact input=unchanged$',out,re.M))
    assert completed == run['matrix_cases_passed'] == (0 if variant==1 else 24)
    assert 'Metal API Validation Enabled' in err
    match = re.search(r'Q5 words mismatch n_in=(\d+) n_out=(\d+) pattern=(\d+) index=(\d+) bits=([0-9a-f]+)/([0-9a-f]+)',err)
    assert match
    ni,no,pattern,index,reference,candidate = match.groups()
    bits = [int(reference,16),int(candidate,16)]
    assert abs(bits[0]-bits[1]) == 2 and (bits[0]>>31)==(bits[1]>>31)
    failures.append({'variant':variant,'completed_guarded_cases':completed,'n_in':int(ni),
                     'n_out':int(no),'pattern':int(pattern),'output_index_including_guard':int(index),
                     'reference_bits':reference,'candidate_bits':candidate,'ulp_distance':2})

# Reverse only integer payload loads and unpacking; FP32 source remains original.
source = (root/'metal/qw3_core_linear.metal').read_text()
name = 'qw3_matvec_q5_k_fast'
a = source.index('kernel void '+name+'(')
original = source[a:source.index('\n}\n',a)+3]
shader = (root/'tests/.q5_words/kernels.metal').read_text()
for variant in (1,2):
    private = f'private_q5_words_fast{variant}'
    a = shader.index('kernel void '+private+'(')
    body = shader[a:shader.index('\n}\n',a)+3].replace(private,name)
    for symbol,pointer in (('q1_words','q1'),('q2_words','q2'),('qh_words','qh')):
        body = body.replace(f'        const uint2 {symbol} = *((device const uint2 *){pointer});\n','')
    for symbol,pointer,words in (('q1_byte','q1','q1_words'),('q2_byte','q2','q2_words')):
        body = body.replace(f'            const uchar {symbol} = uchar(({words}[l >> 2u] >> (8u * (l & 3u))) & 255u);\n','')
        body = body.replace(f'float({symbol} &',f'float({pointer}[l] &')
    body = body.replace('const uchar h = uchar((qh_words[l >> 2u] >> (8u * (l & 3u))) & 255u);','const uchar h = qh[l];')
    assert body == original
result = {'index_verified':args.index,'source_references_verified':len(manifest['source_sha256']),
          'integer_only_shader_changes_verified':True,'numerical_failures':failures,
          'timing_samples':0,'model_gate_run':False,'production_binaries_unchanged':True,
          'adopted':False,'reason':'required finite bit-for-bit outputs differ by two ULP'}
(archive/('verification_index.json' if args.index else 'verification.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
