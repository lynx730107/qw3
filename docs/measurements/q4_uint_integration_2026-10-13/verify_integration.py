"""Verify isolated production sources, tests and actual public pipeline selections."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
p=argparse.ArgumentParser();p.add_argument('--index',action='store_true');args=p.parse_args()
b=Path(__file__).resolve().parent;repo=b.parents[2]
m=json.loads((b/'manifest.json').read_text());work=Path(m['integration_worktree'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,h in m['source_sha256'].items():
 frozen=b/'source_snapshot'/name
 assert sha(frozen)==h and sha(work/name)==h,name
 if args.index:
  content=subprocess.check_output(['git','show',':'+str(frozen.relative_to(repo))],cwd=repo)
  assert hashlib.sha256(content).hexdigest()==h,('index',name)
assert m['source_unchanged'] and m['binary_unchanged']
for name,h in m['binary_sha256'].items():assert sha(work/name)==h,name
assert not (b/'build.stderr').read_text() and not (b/'quality_build.stderr').read_text()
assert len(m['tests'])==4 and all(t['exit_code']==0 for t in m['tests'])
assert [t['name'] for t in m['tests']]==['agent_bash','agent_edit','runtime_fingerprint','dense_agent']
log=(b/'quality.stdout').read_text()
rows=re.findall(r'^full_exact phase=(\d) prefix=(\d+) extended_prefix=(\d+) continuation=32 finite_logits=bit_exact canonical_state=exact history=exact checkpoint=exact calls=(\d+) fallbacks=(\d+) sessions=1$',log,re.M)
expected=[]
for length in (1,15,16,31,127,128,129,159):
 for phase in (0,1):expected.append((str(phase),str(length),str(159 if length==128 else length),str(1152 if phase else 0),'0'))
assert rows==expected
assert 'PASS q4_words_quality eight_prefixes=exact checkpoint_roundtrips=16 sessions=1 additional_weight_buffer_bytes=0' in log
# The public shader must match the numerically validated uint candidate exactly.
original=(repo/'docs/measurements/q4_words_2026-10-13/source_snapshot/tests/.q4_words/kernels.metal').read_text()
a=original.index('kernel void private_q4_words_fast2(')
validated=original[a:original.index('\n}\n',a)+3].replace('private_q4_words_fast2','qw3_matvec_q4_k_words')
public=(work/'metal/qw3_core_linear.metal').read_text()
a=public.index('kernel void qw3_matvec_q4_k_words(')
assert public[a:public.index('\n}\n',a)+3]==validated
result={'index_verified':args.index,'source_references_verified':len(m['source_sha256']),
 'public_shader_matches_validated_candidate':True,'full_quality_phases':16,'checkpoint_roundtrips':16,
 'agent_checks_passed':[t['name'] for t in m['tests']],
 'scope':'isolated integrated policy; performance/reference and main-branch merge remain separate'}
(b/('verification_index.json' if args.index else 'verification.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
