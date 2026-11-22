"""Compile a row-level readback diagnostic and compare the final prefix position by layer."""
from pathlib import Path
from array import array
import hashlib
import json
import math
import os
import re
import subprocess
import sys

out=Path(__file__).resolve().parent
root=out.parents[2]
work=Path(sys.argv[1]).resolve()
build=work/'layers'
old=out.parent/'staged128_partial_2026-11-17'
lib=(root/'../../llama.cpp/build/bin').resolve()
model=(root/'../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf').resolve()
def replace(src,a,b):
    assert src.count(a)==1, a
    return src.replace(a,b)
def compile_probe():
    build.mkdir()
    src=(work/'qw3.c').read_text()
    anchor='static int qw3_metal_session_eval_dense_prefill_batch(\n    qw3_session *s, const int *tokens, int n_tokens,\n    char *err, size_t errlen, int logits_mode) {'
    helper=r"""
static int capture_residual(qw3_session *s, int layer, const char *stage, int rows, int dim, int batch) {
    float *data = malloc((size_t)rows * dim * sizeof(float));
    if (!data) return 0;
    int ok = qw3_metal_synchronize() && (batch ?
        qw3_metal_session_read_batch_x0(s->metal, data, rows, dim) :
        qw3_metal_session_read_x0(s->metal, data, dim));
    if (ok) {
        char path[2048];
        snprintf(path, sizeof(path), "%s/qw3-%s-%d", getenv("ROW_DIR"), stage, layer);
        FILE *file = fopen(path, "wb");
        ok = file && fwrite(data+(rows-1)*dim, sizeof(float), dim, file)==(size_t)dim;
        if (file && fclose(file)) ok=0;
    }
    free(data);
    return ok;
}
"""
    src=replace(src,anchor,helper+anchor)
    src=replace(src,'    const uint32_t dense_dn_stride = dense_dn_inner_offset + inner_n;\n    for',
        '    const uint32_t dense_dn_stride = dense_dn_inner_offset + inner_n;\n    if (ok && pos0+ntok==127) ok=capture_residual(s,-1,"embed",ntok,n_embd,1);\n    for')
    src=replace(src,'        const double attention_done = profile ? qw3_now_sec() : 0.0;\n        if (ok && batch_ffn)',
        '        const double attention_done = profile ? qw3_now_sec() : 0.0;\n        if (ok && pos0+ntok==127 && batch_ffn) ok=capture_residual(s,il,"attn",ntok,n_embd,1) && (!layer_batch || qw3_metal_begin_dense_layer_commands());\n        if (ok && batch_ffn)')
    src=replace(src,'        if (profile) {\n            const double layer_done = qw3_now_sec();',
        '        if (ok && pos0+ntok==127) ok=capture_residual(s,il,"out",ntok,n_embd,1);\n        if (profile) {\n            const double layer_done = qw3_now_sec();')
    for anchor,stage,layer in [('    QW3_DENSE_PROFILE_STAGE(0);','embed','-1'),('        QW3_DENSE_PROFILE_STAGE(4);','attn','il'),('        QW3_DENSE_PROFILE_STAGE(6);','out','il')]:
        src=replace(src,anchor,anchor+f'\n        if (ok && pos==126) ok=capture_residual(s,{layer},"{stage}",1,n_embd,0) && qw3_metal_begin_commands();')
    (build/'core.c').write_text(src)
    c=(root/'tests/test_dense_logits.c').read_text().replace('TEST_CTX = 128','TEST_CTX = 256').replace('TEST_THREADS = 16','TEST_THREADS = 4').replace('#include "../qw3.h"','#include "qw3.h"')
    a=c.index('static bool trace_llama_tensor'); b=c.index('static int ranked_logit_desc',a)
    c=c[:a]+r"""
static int capture_llama;
static bool trace_llama_tensor(struct ggml_tensor *t,bool ask,void *userdata) {
    (void)userdata;
    if (!capture_llama) return false;
    const char *stage=NULL;
    int layer=-1;
    if (!strncmp(t->name,"model.input_embed",17)) stage="embed";
    else if (sscanf(t->name,"attn_residual-%d",&layer)==1) stage="attn";
    else if (sscanf(t->name,"l_out-%d",&layer)==1) stage="out";
    if (ask) return stage!=NULL;
    if (!stage) return true;
    if (t->type!=GGML_TYPE_F32 || t->ne[0]!=5120 || t->ne[2]!=1 || t->ne[3]!=1) abort();
    float data[5120];
    ggml_backend_tensor_get(t,data,(t->ne[1]-1)*t->nb[1],sizeof(data));
    char path[2048];snprintf(path,sizeof(path),"%s/llama-%s-%d",getenv("ROW_DIR"),stage,layer);
    FILE *f=fopen(path,"wb");if (!f || fwrite(data,1,sizeof(data),f)!=sizeof(data) || fclose(f)) abort();
    return true;
}
"""+c[b:]
    c=replace(c,'        batch = llama_batch_get_one((llama_token *)tokens, n_tokens);','        capture_llama=1;\n        batch = llama_batch_get_one((llama_token *)tokens, n_tokens);')
    c=replace(c,'            batch = llama_batch_get_one((llama_token *)tokens + i, 1);','            capture_llama=(i+1==n_tokens);\n            batch = llama_batch_get_one((llama_token *)tokens + i, 1);')
    c=replace(c,'    if (decode_result != 0) {','    capture_llama=0;\n    if (decode_result != 0) {')
    a=c.index('    for (int i = 0; i < n_tokens; i++) {\n        if (qw3_session_eval(')
    b=c.index('    if (getenv("QW3_DENSE_TRACE"))',a)
    c=c[:a]+'''    qw3_tokens ids={.v=(int *)tokens,.len=n_tokens,.cap=n_tokens};
    if(qw3_session_sync(session,&ids,err,sizeof(err))) { fprintf(stderr,"%s\\n",err);return 1;}
'''+c[b:]
    (build/'main.c').write_text(c)
    flags=['cc','-O3','-ffast-math','-mcpu=native','-I'+str(work)]
    subprocess.run([*flags,'-std=c99','-c','core.c','-o','core.o'],cwd=build,check=True)
    subprocess.run([*flags,'-fobjc-arc','-c',str(work/'qw3_metal.m'),'-o','metal.o'],cwd=build,check=True)
    subprocess.run(['cc','-O3','-std=c11','-I'+str(work),'-I'+str(lib.parents[1]/'include'),'-I'+str(lib.parents[1]/'ggml/include'),'main.c','core.o','metal.o','-L'+str(lib),'-Wl,-rpath,'+str(lib),'-lllama','-lggml','-lggml-base','-lm','-pthread','-framework','Foundation','-framework','Metal','-o','trace'],cwd=build,check=True)

if sys.argv[2]=='build':
    compile_probe()
    sys.exit()
ids=re.search(r'tokens=([0-9,]+)',(old/'frontier-v0.log').read_text()).group(1).split(',')
assert len(ids)==127
for name in ('serial','batch'):
    folder=build/name
    folder.mkdir()
    env={k:v for k,v in os.environ.items() if not k.startswith(('QW3_','MTL_','GGML_'))}
    env.update(QW3_DENSE_METAL='1',QW3_METAL_NGL='64',QW3_METAL_KV_F16='1',ROW_DIR=str(folder),QW3_METAL_DENSE_PREFILL_BATCH='1' if name=='serial' else '128')
    if name=='batch': env['QW3_DENSE_LLAMA_BATCH']='1'
    with (out/f'layers-{name}.log').open('w') as log:
        p=subprocess.run([str(build/'trace'),str(model),*ids],cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT)
    assert p.returncode==(0 if name=='serial' else 1) and 'top_match=8/8' in (out/f'layers-{name}.log').read_text()

def read(mode,engine,stage,index):
    p=build/mode/f'{engine}-{stage}-{index}'
    if not p.exists(): return None
    v=array('f');v.frombytes(p.read_bytes());assert len(v)==5120 and all(map(math.isfinite,v));return v
rows=[]
for index,stage in [(-1,'embed')]+[(i,s) for i in range(64) for s in ('attn','out')]:
    vectors={f'{e}_{m}':read(m,e,stage,index) for e in ('qw3','llama') for m in ('serial','batch')}
    row=dict(layer=index,stage=stage)
    for key1,key2 in [('qw3_serial','llama_serial'),('qw3_batch','llama_batch'),('qw3_serial','qw3_batch'),('llama_serial','llama_batch')]:
        v,w=vectors[key1],vectors[key2]
        if v is None or w is None: continue
        error=[abs(x-y) for x,y in zip(v,w)]
        row[key1+'_vs_'+key2]=dict(rmse=math.sqrt(sum(d*d for d in error)/len(error)),max_abs=max(error),relative_rms=math.sqrt(sum(d*d for d in error)/max(1e-300,sum(x*x for x in v))))
    rows.append(row)
assert all('qw3_serial_vs_qw3_batch' in r for r in rows if r['stage']=='out')
(out/'layers.json').write_text(json.dumps(rows,indent=2)+'\n')
print('layer capture complete')
