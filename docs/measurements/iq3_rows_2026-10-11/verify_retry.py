"""Check all seven replica blocks against the frozen numerical/pilot evidence."""
import contextlib
import csv
import io
import json
import math
from pathlib import Path
import re
import runpy
import statistics

screen = Path(__file__).resolve().parent
with contextlib.redirect_stdout(io.StringIO()):
    common = runpy.run_path(str(screen/'verify_campaign.py'))
path = screen/'native_retry'
m = common['validate_manifest'](path/'manifest.json')
assert m['order']==[0,1,0,1,0,1,0] and len(m['runs'])==7
assert m['source_sha256']==common['numeric']['source_sha256']
assert m['cooldown_seconds_before_every_process']==60
assert m['context']==256 and m['repetitions']==3
means=[]
for run in m['runs']:
    assert run['exit_code']==0
    log=(path/(run['name']+'.stderr')).read_text()
    assert common['metadata'](log)==common['expected_metadata']
    assert 'Metal API Validation Enabled' not in log
    samples=[{'repetition':int(n),'tps':float(tps),'ms':float(ms)} for n,tps,ms in
             re.findall(r'qw3-bench: sample (\d+)/3 ([\d.]+) tok/s ([\d.]+) ms',log)]
    assert samples==run['samples'] and [s['repetition'] for s in samples]==[1,2,3]
    assert all(math.isfinite(s[k]) and s[k]>0 for s in samples for k in ('ms','tps'))
    counts=re.findall(r'PASS iq3_rows_native variant=(\d+) calls=(\d+) additional_weight_buffer_bytes=0',log)
    assert counts==[(str(run['variant']),str(run['candidate_calls']))]
    assert run['candidate_calls']==(7081 if run['variant'] else 0)
    assert run['environment']['QW3_PRIVATE_IQ3_ROWS']==str(run['variant'])
    rows=list(csv.DictReader((path/(run['name']+'.stdout')).read_text().splitlines()))
    assert len(rows)==1 and rows[0]['test']=='tg'
    assert all(int(rows[0][k])==v for k,v in {
        'n_prompt':0,'n_gen':32,'n_depth':128,'reps':3,'session_bytes':176761680}.items())
    mean=statistics.mean(s['ms'] for s in samples)
    assert abs(float(rows[0]['avg_ms'])-mean)<=.006
    means.append(mean)
off=[means[i] for i in (0,2,4,6)]
on=[means[i] for i in (1,3,5)]
spreads={'off':100*(max(off)/min(off)-1),'on':100*(max(on)/min(on)-1)}
comparisons=[]
for i in (1,3,5):
    ref=(means[i-1]+means[i+1])/2
    comparisons.append({'block':i,'candidate_mean_ms':means[i],'adjacent_off_mean_ms':ref,
                        'point_gain_pct':100*(ref/means[i]-1),
                        'gain_vs_left_off_pct':100*(means[i-1]/means[i]-1),
                        'gain_vs_right_off_pct':100*(means[i+1]/means[i]-1),
                        'saved_ms':ref-means[i]})
control=statistics.mean(c['adjacent_off_mean_ms'] for c in comparisons)
candidate=statistics.mean(on)
all_runs=[(stage,r) for stage,data in (('numeric',common['numeric']),('pilot',common['native']),('retry',m))
          for r in data['runs']]
swap=[{'stage':stage,**{k:r[k] for k in ('name','swap_before','swap_after')}} for stage,r in all_runs]
analysis={'adopted':False,'replica_variant':1,'rows_per_simd':2,'runs':7,
          'samples_retained':21,'samples_discarded':0,'block_means_ms':means,
          'run_mean_spread_pct':spreads,'control_stability_gate_passed':spreads['off']<=5,
          'both_modes_stable_5pct':all(v<=5 for v in spreads.values()),'comparisons':comparisons,
          'neighbor_weighted_control_mean_ms':control,'candidate_mean_ms':candidate,
          'gain_vs_neighbor_weighted_controls_pct':100*(control/candidate-1),
          'all_three_local_comparisons_positive':all(c['point_gain_pct']>0 for c in comparisons),
          'all_three_candidates_faster_than_both_neighbors':all(
              c['gain_vs_left_off_pct']>0 and c['gain_vs_right_off_pct']>0 for c in comparisons),
          'additional_weight_buffer_bytes':0,'additional_session_data_buffers':0,
          'native_session_bytes':176761680,'swap_observations':swap,
          'all_numeric_and_timing_swap_observations_equal':len({s[k] for s in swap for k in ('swap_before','swap_after')})==1,
          'scope':'seven-block replication; adjacent interpolated controls are shared; no independent-sample CI asserted',
          'numeric_checkpoint_suite_repeated':False}
(path/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
print(json.dumps(analysis,indent=2))
