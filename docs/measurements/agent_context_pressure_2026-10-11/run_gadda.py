"""Natural Wikipedia request, isolated workspace, live title-list oracle."""
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time
import unicodedata
import urllib.request

archive=Path(__file__).resolve().parent
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def store_inventory():
    store=Path('/Users/pasqualecapasso/.qw3/agent_store')
    return {str(p.relative_to(store)):{'bytes':p.stat().st_size,'sha256':sha(p)}
            for p in store.rglob('*') if p.is_file()}
root=archive.parents[2]
model='/Users/pasqualecapasso/Downloads/models/Qwen3.8-27B-UD-Q3_K_XL.gguf'
url='https://it.wikipedia.org/wiki/Carlo_Emilio_Gadda'
request=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
raw=urllib.request.urlopen(request,timeout=45).read(2000000)
page=raw.decode('utf-8')
start=page.index('id="Romanzi,_racconti,_saggi"')
start=page.index('<ul',start);end=page.index('</ul>',start)+5
titles=[]
for item in re.findall(r'<li\b[^>]*>(.*?)</li>',page[start:end],re.S):
    italic=re.search(r'<i\b[^>]*>(.*?)</i>',item,re.S)
    # Four entries in this live list use plain text instead of italics;
    # their first comma separates the title from publication metadata.
    title_html=italic.group(1) if italic else item.split(',',1)[0]
    title=html.unescape(re.sub(r'<[^>]+>','',title_html)).strip()
    titles.append(' '.join(title.split()))
assert len(titles)>=30
canonical_titles=[re.sub(r'\s*\([^)]*\)', '',t.split('. ',1)[0]).strip() for t in titles]
(archive/'gadda-oracle.json').write_text(json.dumps({
    'url':url,'fetched_utc':datetime.now(timezone.utc).isoformat(),
    'page_bytes':len(raw),'page_sha256':hashlib.sha256(raw).hexdigest(),
    'category':'Romanzi, racconti, saggi','titles':titles,
    'canonical_titles':canonical_titles,
    'title_validation':'Main titles in separate list entries; subtitles and parenthetical supplements may be omitted.'},indent=2,ensure_ascii=False)+'\n')
env={k:v for k,v in os.environ.items() if not k.startswith(('QW3_','GGML_','MTL_'))}
env.update(QW3_METAL_KV_F16='1',QW3_METAL_KV_Q8_0='0',QW3_METAL_NGL='64',
           QW3_METAL_KERNEL_DIR=str(root/'metal'),MTL_DEBUG_LAYER='1',NO_COLOR='1')
started_utc=datetime.now(timezone.utc).isoformat()
start_wall=time.time()
source_names=json.loads((archive.parent/'agent_regressions_retry_2026-10-11/manifest.json').read_text())['source_sha256']
source_sha256={name:sha(root/name) for name in source_names}
binary_sha256={name:sha(root/name) for name in ('qw3-agent','qw3-cli','qw3-bench','qw3-agent-context-pressure-test')}
store_before=store_inventory()
outside_before={str(p) for p in Path('/tmp').glob('gadda*')}
query='Cerca su Wikipedia le opere di Carlo Emilio Gadda ed elencale.'
with tempfile.TemporaryDirectory(prefix='qw3-gadda-web-') as tmp:
    work=Path(tmp)/'work';work.mkdir();store=Path(tmp)/'store';store.mkdir()
    trace=Path(tmp)/'trace.jsonl'
    cmd=[str(root/'qw3-agent'),'--metal','-m',model,'--ctx','8192','--nothink',
         '--no-prefix-cache','--temp','0','-n','4096',
         '--chdir',str(work),'--store-dir',str(store),'--trace',str(trace),'-p',query]
    start=time.monotonic()
    timed_out=False
    with (archive/'gadda.stdout').open('w') as out,(archive/'gadda.stderr').open('w') as err:
        process=subprocess.Popen(cmd,cwd=root,env=env,stdout=out,stderr=err,start_new_session=True)
        power_command=['caffeinate','-disu','-w',str(process.pid)]
        power=subprocess.Popen(power_command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        try:
            process.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            timed_out=True
            os.killpg(process.pid,signal.SIGKILL)
            process.wait()
        try:
            power.wait(timeout=10)
        except subprocess.TimeoutExpired:
            power.terminate();power.wait(timeout=10)
    events=trace.read_text() if trace.exists() else ''
    (archive/'gadda.trace.jsonl').write_text(events)
    answer=(archive/'gadda.stdout').read_text()
    def norm(s):
        return re.sub(r'[^a-z0-9]+',' ',unicodedata.normalize('NFKD',s).encode('ascii','ignore').decode().lower()).strip()
    listed=[]
    for line in answer.splitlines():
        m=re.match(r'^\s*(?:[-*+]\s+|\d+[.)]\s+|\|)(.*)',line)
        if m:
            listed.append(re.sub(r'^(?:\d+\s+)+','',norm(m.group(1))))
    matched=set()
    for line in listed:
        choices=[t for t in canonical_titles if line==norm(t) or line.startswith(norm(t)+' ')]
        if choices: matched.add(max(choices,key=lambda t:len(norm(t))))
    missing=[t for t in canonical_titles if t not in matched]
    files=[{'path':str(p.relative_to(work)),'bytes':p.stat().st_size,
            'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
           for p in work.rglob('*') if p.is_file()]
    readable_pages=[{'path':str(p.relative_to(store)), 'bytes':p.stat().st_size,
                     'sha256':sha(p), 'main_titles_present':[
                         t for t in canonical_titles if norm(t) in norm(p.read_text())]}
                    for p in (store/'spills').glob('web-*') if p.is_file()]
    compact_archives=[]
    for p in (store/'spills').glob('context-*'):
        dest=archive/'gadda_context'/p.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(p,dest)
        contents=p.read_text()
        compact_archives.append({'path':str(dest.relative_to(archive)),'bytes':p.stat().st_size,
                                 'sha256':sha(p),'user_requests':re.findall(
                                     r'--- message \d+ kind=user role=user tokens=\d+\.\.\d+ ---\n(.*?)\n--- end message ---',contents,re.S)})
    metadata={'query':query,'command':cmd,'exit_code':process.returncode,
              'timed_out':timed_out,'timeout_seconds':1800,'power_assertion_command':power_command,
              'seconds':time.monotonic()-start,'started_utc':started_utc,'completed_utc':datetime.now(timezone.utc).isoformat(),
              'source_sha256':source_sha256,'binary_sha256':binary_sha256,
              'source_unchanged':all(sha(root/n)==h for n,h in source_sha256.items()),
              'binary_unchanged':all(sha(root/n)==h for n,h in binary_sha256.items()),
              'model_size':Path(model).stat().st_size,'model_mtime_ns':Path(model).stat().st_mtime_ns,
              'store_before':store_before,'store_after':store_inventory(),
              'store_unchanged':store_before==store_inventory(),
              'tool_round_limit':24,'max_generation_tokens':4096,'context':8192,'prefix_cache':False,
              'oracle_category':'Romanzi, racconti, saggi','oracle_titles':len(titles),
              'missing_titles':missing,'matched_main_titles':sorted(matched),
              'title_validation':'Separate list entries; canonical title prefixes, longest match; subtitles optional',
              'downloaded_workspace_files':files,
              'downloaded_readable_pages':readable_pages,
              'compaction_archives':compact_archives,
              'one_model_at_a_time':True,'temporary_store_removed':True}
    outside=[]
    stderr=(archive/'gadda.stderr').read_text()
    outside_paths={Path(s) for s in re.findall(r'/tmp/gadda[A-Za-z0-9_.-]*',stderr)}
    for p in sorted(outside_paths):
        if p.is_file() and p.stat().st_mtime >= start_wall and str(p) not in outside_before:
            outside.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),
                            'removed_after_test':True})
            p.unlink()
    metadata['downloaded_files_outside_workspace']=outside
    (archive/'gadda-result.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(metadata,indent=2,ensure_ascii=False),flush=True)
    assert metadata['source_unchanged'] and metadata['binary_unchanged'] and metadata['store_unchanged']
    assert process.returncode==0 and not timed_out
    assert '"event":"tool"' in events and ('"name":"visit_url"' in events or '"name":"search_web"' in events)
    assert not missing, missing
    assert 'it.wikipedia.org/wiki/Carlo_Emilio_Gadda' in answer
    print('PASS gadda live_wikipedia_titles=all source=cited',flush=True)
