"""Validate each recorded sample and produce paired latency summaries."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import re
from statistics import mean

out=Path(__file__).resolve().parent
data={}
def save():
    (out/'analysis.json').write_text(json.dumps(data,indent=2)+'\n')
def log(n):
    return (out/(n+'.log')).read_text()
def summarize(samples):
    assert all(math.isfinite(ms) and ms>0 for _,ms in samples)
    a=[x for m,x in samples if m==0]
    b=[x for m,x in samples if m==1]
    assert len(a)==len(b)
    deltas=[]
    for i in range(0,len(samples),2):
        pair=dict(samples[i:i+2])
        deltas.append(100*(1-pair[1]/pair[0]))
    return dict(baseline_ms=mean(a),candidate_ms=mean(b),reduction_pct=100*(1-mean(b)/mean(a)),paired_reductions_pct=deltas,positive_pairs=sum(x>0 for x in deltas))

depth=json.loads((out/'depth.json').read_text())
for d in (1024,4096,8192):
    runs=[r for r in depth['runs'] if r['depth']==d and 'ms' in r]
    if len(runs)!=4:continue
    for r in runs:
        stderr=(out/f"d{d}-{r['order']}-{r['engine']}.stderr").read_text()
        if r['engine']=='llama':
            assert re.search(rf'n_ctx\s*=\s*{d+256}\b',stderr)
            offload=re.search(r'offloaded (\d+)/(\d+) layers',stderr)
            assert offload and offload[1]==offload[2]
        else:
            assert 'Apple M5' in stderr and 'sample 1/1' in stderr
    data[f'depth_{d}']=summarize([(int(r['engine']=='qw3'),r['ms']) for r in runs])
if (out/'screen.json').exists():
    screen=json.loads((out/'screen.json').read_text())
    data['failures']=[r['name'] for r in screen if r['exit']!=0]
else: screen=[]
for r in screen:
    n=r['name']
    if r['exit']:continue
    if n.startswith('turn-'):
        hits=re.findall(r'sample=(\d+) warmup=(\d+) cached=(\d+) added=(\d+) mode=(\d+) ms=([\d.]+) first=(\d+)',log(n))
        assert len(hits)==10 and len({h[-1] for h in hits})==1
        assert [int(h[0]) for h in hits]==list(range(10)) and [int(h[1]) for h in hits]==[1,1]+[0]*8
        data[n]=summarize([(int(h[4])!=0,float(h[5])) for h in hits[2:]])
    if n.startswith('profile-'):
        report=[json.loads(l.split(' ',1)[1]) for l in log(n).splitlines() if l.startswith('projection_profile ')]
        assert len(report)==2 and 'full vocabulary comparisons bit-exact' in log(n)
        measures=[]
        for step in report:
            groups=Counter()
            kernels=Counter()
            for p in step['passes']:
                names=list(p['dispatches'])
                name=names[0] if len(names)==1 else 'mixed' if names else 'empty'
                kernels[name]+=p['gpu_ms']
                category=next((s for s in ('iq4_xs','iq3_s','iq3_xxs','q4_k','q5_k','iq2_s','q3_k','gdn') if s in name),None)
                if not category:
                    if any(s in name for s in ('gqa','flash_attn','flash_group','fattn')):category='GQA/Flash'
                    else:category=name
                groups[category]+=p['gpu_ms']
            measures.append(dict(step=step['rep'],wall_ms=step['wall_ms'],groups_ms=dict(groups),kernels_ms=dict(kernels)))
        data[n]=measures
for kind in ('nibble','gqa'):
    series=[r for r in screen if r['name'].startswith(kind+'-tg-') and not r['exit']]
    if len(series)==8:
        samples=[]
        for r in series:
            matches=re.findall(r'qw3-bench: sample (\d)/2 .*? ([\d.]+) ms',log(r['name']))
            assert [i for i,_ in matches]==['1','2']
            variant=int(r['name'][-1])
            assert re.search(r'PASS .*native variant='+str(variant)+r' calls=\d+',log(r['name']))
            samples.append((variant,mean(float(ms) for _,ms in matches)))
        data[kind+'_decode']=summarize(samples)
save()
