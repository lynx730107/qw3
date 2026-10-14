"""Recompute all close-pair samples; this is a within-process diagnostic."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics

b = Path(__file__).resolve().parent
root = b.parents[2]
m = json.loads((b / 'manifest.json').read_text())
assert m['completed_utc'] and m['exit_code'] == 0
assert all(m[k] for k in ('source_unchanged', 'binary_unchanged', 'production_unchanged', 'model_unchanged'))
assert m['order'] == [0,1,1,0,0,1,1,0,0,1,1,0]


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


for name, digest in m['source_sha256'].items():
    assert sha(root / name) == sha((b / m['source_snapshot_paths'][name]).resolve()) == digest
assert sha(Path(m['binary'])) == m['binary_sha256']
for name, digest in m['production_binary_sha256'].items():
    assert sha(root / name) == digest
model = Path(m['model'])
assert (model.stat().st_size, model.stat().st_mtime_ns) == (m['model_size'], m['model_mtime_ns'])
log = (b / 'pairs.stderr').read_text()
assert 'Metal API Validation Enabled' not in log
samples = [{'repetition': int(n), 'tps': float(t), 'ms': float(ms)}
           for n,t,ms in re.findall(r'qw3-bench: sample (\d+)/12 ([\d.]+) tok/s ([\d.]+) ms', log)]
assert samples == m['samples'] and [s['repetition'] for s in samples] == list(range(1,13))
assert all(math.isfinite(s[k]) and s[k] > 0 for s in samples for k in ('tps','ms'))
markers = re.findall(r'paired_sample repetition=(\d+) variant=(\d) calls=(\d+) dispatches=(\d+) barriers=(\d+) independent_skips=(\d+) sessions=1', log)
assert len(markers) == 12
for i, fields in enumerate(markers):
    n,v,c,d,bar,skip = map(int, fields)
    assert (n,v) == (i+1,m['order'][i])
    assert (c,d,bar,skip) == ((2528,31296,25696,5600) if v else (0,0,0,0))
assert 'PASS projection_pairs runs=12 sessions=1 additional_weight_buffer_bytes=0' in log
rows = list(csv.DictReader(io.StringIO((b / 'pairs.stdout').read_text())))
assert len(rows) == 1
assert all(int(rows[0][k]) == v for k,v in {'n_prompt':0,'n_gen':32,'n_depth':128,'reps':12,'session_bytes':176761680}.items())
assert abs(float(rows[0]['avg_ms']) - statistics.mean(s['ms'] for s in samples)) < .006
pairs = []
for i in range(6):
    values = {m['order'][2*i+j]: samples[2*i+j]['ms'] for j in (0,1)}
    pairs.append({'pair':i+1,'order':m['order'][2*i:2*i+2], 'off_ms':values[0], 'on_ms':values[1],
                  'saved_ms':values[0]-values[1], 'rate_delta_pct':100*(values[0]/values[1]-1)})
saved = [p['saved_ms'] for p in pairs]
mean = statistics.mean(saved)
radius = 2.571 * statistics.stdev(saved) / math.sqrt(6)
off = [p['off_ms'] for p in pairs]
on = [p['on_ms'] for p in pairs]
result = {'scope':'one-process close-pair diagnostic; original timer/reset/prefill and one session; no independent process replication or llama comparison',
          'samples_retained':12,'samples_discarded':0,'pairs':pairs,
          'control_mean_ms':statistics.mean(off),'candidate_mean_ms':statistics.mean(on),
          'rate_delta_pct':100*(statistics.mean(off)/statistics.mean(on)-1),
          'saved_ms_mean':mean,'saved_ms_ci95':[mean-radius,mean+radius],
          'ci_assumption':'Student t df5 critical2.571; six pairs treated independent; same-process time correlation can weaken this assumption',
          'order_group_saved_ms':{str(first):statistics.mean(p['saved_ms'] for p in pairs if p['order'][0]==first) for first in (0,1)},
          'off_spread_pct':100*(max(off)-min(off))/statistics.mean(off),
          'on_spread_pct':100*(max(on)-min(on))/statistics.mean(on),
          'sources_snapshots_binaries_model_match':True,'session_bytes':176761680,
          'adoption_evidence':False}
(b / 'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
