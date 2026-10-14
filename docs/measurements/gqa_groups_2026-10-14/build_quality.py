"""Build extended checkpoint runner only after verified positive replication."""
import hashlib
import json
from pathlib import Path
import subprocess

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--variant',type=int,choices=(1,2),default=2)
args=parser.parse_args()
assert args.variant==2
archive=Path(__file__).resolve().parent
root=archive.parents[2]
analysis=json.loads((archive/f'replica_v{args.variant}/analysis.json').read_text())
replica=json.loads((archive/f'replica_v{args.variant}/manifest.json').read_text())
assert replica['completed_utc'] and len(replica['runs'])==12
assert all(r['exit_code']==0 for r in replica['runs'])
assert analysis['ci_lower_positive'] and analysis['both_order_groups_positive']
assert analysis['off_spread_percent']<=5 and analysis['on_spread_percent']<=5
verified=subprocess.run(['python3',str(archive/'verify_replica.py'),'--variant',str(args.variant)],cwd=root,
                        capture_output=True,text=True,check=True)
verification=json.loads(verified.stdout)
assert verification['performance_gate_passed']
binary=Path('/tmp/qw3-gqa-groups-quality-test')
source=root/'tests/test_gqa_groups_quality.m'
command=['cc','-O3','-ffast-math','-mcpu=native','-Wall','-Wextra','-fobjc-arc',
         '-fno-finite-math-only','-I.','-o',str(binary),str(source),'-lm','-pthread',
         '-framework','Foundation','-framework','Metal']
with (archive/'quality_build.stdout').open('w') as out,(archive/'quality_build.stderr').open('w') as err:
    result=subprocess.run(command,cwd=root,stdout=out,stderr=err,check=True)
assert not (archive/'quality_build.stderr').read_text()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
meta={'command':command,'exit_code':result.returncode,'warning_free':True,
      'binary':str(binary),'binary_sha256':sha(binary),'source_sha256':sha(source),
      'performance_analysis_sha256':sha(archive/f'replica_v{args.variant}/analysis.json'),
      'variant':args.variant,'scope':'quality binary only; runtime gate remains separate'}
(archive/'quality_build.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta,indent=2))
