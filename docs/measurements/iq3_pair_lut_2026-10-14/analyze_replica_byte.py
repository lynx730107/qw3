"""Six fresh paired process means; AB/BA order balanced, samples retained."""
import json
import math
from pathlib import Path
import statistics
b=Path(__file__).resolve().parent
m=json.loads((b/'replica_byte/manifest.json').read_text())
assert m['order']==[0,2,2,0,0,2,2,0,0,2,2,0] and len(m['runs'])==12
assert all(r['exit_code']==0 and len(r['samples'])==3 for r in m['runs'])
assert all(m[k] for k in ('source_unchanged','binary_unchanged','model_unchanged'))
means=[statistics.mean(s['ms'] for s in r['samples']) for r in m['runs']]
pairs=[]
for i in range(6):
    r=m['runs'][2*i:2*i+2]
    values={v['variant']:means[2*i+j] for j,v in enumerate(r)}
    pairs.append({'pair':i,'order':[v['variant'] for v in r],
                  'off_ms':values[0],'on_ms':values[2],
                  'saved_ms':values[0]-values[2],
                  'throughput_change_percent':(values[0]/values[2]-1)*100})
saved=[p['saved_ms'] for p in pairs]
mean=statistics.mean(saved)
half=2.571*statistics.stdev(saved)/math.sqrt(6)
off=[p['off_ms'] for p in pairs];on=[p['on_ms'] for p in pairs]
order_means={str(first):statistics.mean(p['saved_ms'] for p in pairs if p['order'][0]==first) for first in (0,2)}
result={'scope':'independent six-pair replica; process-run means are units, not the 36 individual repetitions',
        'samples_retained':36,'run_means_ms':means,'pairs':pairs,
        'candidate_mean_ms':statistics.mean(on),'control_mean_ms':statistics.mean(off),
        'throughput_change_percent':(statistics.mean(off)/statistics.mean(on)-1)*100,
        'off_spread_percent':(max(off)-min(off))/statistics.mean(off)*100,
        'on_spread_percent':(max(on)-min(on))/statistics.mean(on)*100,
        'saved_ms_mean':mean,'saved_ms_ci95':[mean-half,mean+half],
        'ci_assumption':'Student t df=5, critical 2.571; pairs treated as independent, time/thermal correlation may weaken this assumption',
        'order_group_saved_ms':order_means,
        'candidate_wins_all_pairs':all(p['saved_ms']>0 for p in pairs),
        'both_order_groups_positive':all(x>0 for x in order_means.values()),
        'ci_lower_positive':mean-half>0}
(b/'replica_byte/analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
