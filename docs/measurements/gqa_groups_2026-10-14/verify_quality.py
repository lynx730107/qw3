"""Recheck private extended numerical evidence and its frozen dependencies."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import argparse

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--index',action='store_true')
args=parser.parse_args()
b=Path(__file__).resolve().parent
root=b.parents[2]
base=b/'quality_v2'
m=json.loads((base/'manifest.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert m['exit_code']==0 and m['variant']==2 and m['context']==256 and m['sessions']==1
assert all(m[k] for k in ('source_unchanged','binary_unchanged','model_unchanged'))
assert m['additional_weight_and_session_data_buffers']==0
for n,h in m['source_sha256'].items():
    frozen=(base/m['source_snapshot_paths'][n]).resolve()
    assert sha(root/n)==sha(frozen)==h,n
    if args.index:
        data=subprocess.check_output(['git','show',':'+str(frozen.relative_to(root))],cwd=root)
        assert hashlib.sha256(data).hexdigest()==h,n
assert sha(m['command'][2])==m['binary_sha256']
model=Path(m['command'][3])
assert (model.stat().st_size,model.stat().st_mtime_ns)==(m['model_size'],m['model_mtime_ns'])
assert sha(b/m['performance_analysis'])==m['performance_analysis_sha256']
subprocess.run(['python3',str(b/'verify_replica.py')],check=True,capture_output=True)
text=(base/'quality.stdout').read_text()
rows=re.findall(r'full_exact phase=(\d) prefix=(\d+) extended_prefix=(\d+) continuation=32 finite_logits=bit_exact canonical_state=exact history=exact checkpoint=exact calls=(\d+) sessions=1',text)
assert rows==[(str(phase),str(length),str(159 if length==128 else length),str(512 if phase else 0))
    for length in (1,15,16,31,127,128,129,159) for phase in (0,1)]
assert 'PASS gqa_groups_quality eight_prefixes=exact checkpoint_roundtrips=16 sessions=1 additional_weight_buffer_bytes=0' in text
result={'passed':True,'index_verified':args.index,'prefix_phases':16,'checkpoint_roundtrips':16,
        'source_references_verified':len(m['source_sha256']),'scope':m['scope']}
(base/('verification_index.json' if args.index else 'verification.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
