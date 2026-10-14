"""IQ3_XXS sign parity/direct lookup; original FP32 vector dot and buffers."""
from pathlib import Path
import hashlib
import json
import re

root=Path(__file__).resolve().parent.parent
out=root/'tests/.iq3xxs_sign'
out.mkdir(exist_ok=True)
core=(root/'qw3.c').read_text()
table=re.search(r'const uint8_t qw3_ksigns_iq2xs\[128\] = \{(.*?)\};',core,re.S)
assert table
values=[int(v) for v in re.findall(r'\d+',table[1])]
assert len(values)==128 and values==[i|((bin(i).count('1')&1)<<7) for i in range(128)]
source=(root/'metal/qw3_core_linear.metal').read_text()
def extract(name):
    a=source.index(name)
    return source[a:source.index('\n}\n',a)+3]
helper=extract('inline float qw3_iq3xxs_dot32_vector(')
kernel=extract('kernel void qw3_matvec_iq3_xxs_fast(')
shader=''
for variant in (1,2):
    name=f'private_iq3xxs_sign_dot{variant}'
    h=helper.replace('qw3_iq3xxs_dot32_vector',name)
    h=h.replace('threadgroup const uchar *signs','device const uchar *signs')
    old='        const uint s = uint(signs[(packed >> (7u * l)) & 127u]);'
    new='''        const uint sign_index = (packed >> (7u * l)) & 127u;
        const uint s = sign_index | ((popcount(sign_index) & 1u) << 7u);'''
    if variant==1:
        assert h.count(old)==1;h=h.replace(old,new)
    assert h.replace(name,'qw3_iq3xxs_dot32_vector').replace('device const uchar *signs','threadgroup const uchar *signs').replace(new,old)==helper
    k=kernel.replace('qw3_matvec_iq3_xxs_fast',f'private_iq3xxs_sign_fast{variant}').replace('qw3_iq3xxs_dot32_vector',name)
    old_decl='    threadgroup uchar signs[128];'
    new_decl='    device const uchar *signs = sign_table;'
    old_stage='    for (uint i = tid; i < 128u; i += 64u) signs[i] = sign_table[i];\n'
    assert k.count(old_decl)==1 and k.count(old_stage)==1
    k=k.replace(old_decl,new_decl).replace(old_stage,'')
    restored=k.replace(f'private_iq3xxs_sign_fast{variant}','qw3_matvec_iq3_xxs_fast').replace(name,'qw3_iq3xxs_dot32_vector').replace(new_decl,old_decl)
    restored=restored.replace('    threadgroup_barrier(mem_flags::mem_threadgroup);',old_stage+'    threadgroup_barrier(mem_flags::mem_threadgroup);')
    assert restored==kernel
    shader+=h+k
(out/'kernels.metal').write_text(shader)

previous=root/'docs/measurements/iq3_input_2026-10-12'
m=json.loads((previous/'manifest.json').read_text())
def read_base(name):
    key='tests/.iq3_input/'+name;data=(previous/m['source_snapshot_paths'][key]).read_bytes()
    assert hashlib.sha256(data).hexdigest()==m['source_sha256'][key]
    return data.decode()
def rename(s):
    return (s.replace('iq3_input','iq3xxs_sign').replace('IQ3Input','IQ3XXSSign').replace('iq3Input','iq3xxsSign')
            .replace('input_variant','sign_variant').replace('IQ3 input','IQ3XXS sign').replace('IQ3_INPUT','IQ3XXS_SIGN')
            .replace('g_matvec_iq3_s_pipeline','g_matvec_iq3_xxs_pipeline')
            .replace('baseline/packed-input/packed-input-unroll2','baseline/sign-parity/direct-sign-table'))
hooks=rename(read_base('hooks.inc'))
hooks=hooks.replace('Packed input loads; native weights, grid, FP32 arithmetic and data buffers.',
                    'IQ3_XXS sign parity/direct table; original FP32 vector dot and data buffers.')
hooks=hooks.replace('if (!qw3_metal_compile_kernels()) return 0;',
                    'if (!qw3_metal_init() || !qw3_metal_compile_kernels()) return 0;')
hooks=hooks.replace('iq3xxs_sign_pipeline[v].staticThreadgroupMemoryLength!=iq3xxs_sign_pipeline[0].staticThreadgroupMemoryLength',
                    'iq3xxs_sign_pipeline[v].staticThreadgroupMemoryLength!=4096u')
hooks=hooks.replace('dynamic_threadgroup_bytes=256','dynamic_threadgroup_bytes=128')
(out/'hooks.inc').write_text(hooks)
(out/'model.inc').write_text(rename(read_base('model.inc')).replace('32u*73u','32u*34u'))
(out/'native.inc').write_text(rename(read_base('native.inc')))
