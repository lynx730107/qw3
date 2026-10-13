"""Verify the eight-prefix gate without treating compile-only evidence as runtime proof."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
p=argparse.ArgumentParser();p.add_argument('--index',action='store_true');args=p.parse_args()
b=Path(__file__).resolve().parent;root=b.parents[2];q=b/'quality'
m=json.loads((q/'manifest.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert m['exit_code']==0 and m['context']==256 and m['sessions']==1
assert all(m[k] for k in ('source_unchanged','binary_unchanged','model_unchanged'))
assert m['environment']['MTL_DEBUG_LAYER']=='1'
for name,h in m['source_sha256'].items():
    frozen=(q/m['source_snapshot_paths'][name]).resolve()
    assert sha(frozen)==h and sha(root/name)==h,name
    if args.index:
        content=subprocess.check_output(['git','show',':'+str(frozen.relative_to(root))],cwd=root)
        assert hashlib.sha256(content).hexdigest()==h,('index',name)
assert sha(Path('/tmp/qw3-q4-words-quality-test'))==m['binary_sha256']
model=Path(m['command'][-1]);assert (model.stat().st_size,model.stat().st_mtime_ns)==(m['model_size'],m['model_mtime_ns'])
log=(q/'quality.stdout').read_text()
rows=re.findall(r'^full_exact phase=(\d) prefix=(\d+) extended_prefix=(\d+) continuation=32 finite_logits=bit_exact canonical_state=exact history=exact checkpoint=exact calls=(\d+) fallbacks=(\d+) sessions=1$',log,re.M)
expected=[]
for length in (1,15,16,31,127,128,129,159):
    for phase in (0,1):expected.append((str(phase),str(length),str(159 if length==128 else length),str(1152 if phase else 0),'0'))
assert rows==expected
assert 'PASS q4_words_quality eight_prefixes=exact checkpoint_roundtrips=16 sessions=1 additional_weight_buffer_bytes=0' in log
result={'index_verified':args.index,'source_references_verified':len(m['source_sha256']),
        'prefix_lengths':[1,15,16,31,127,128,129,159],'extension_checked':[128,159],
        'baseline_candidate_phases':16,'checkpoint_roundtrips_passed':16,
        'continuation_tokens_checked_per_phase':32,'sessions':1,
        'full_finite_logits':'bit_exact','canonical_state_history':'exact',
        'scope':'private uint candidate; no production or agent regression claim'}
(q/('verification_index.json' if args.index else 'verification.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
