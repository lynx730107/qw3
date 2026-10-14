"""Private exact lookup representations; products and reductions stay FP32."""
from pathlib import Path
import hashlib
import json
import struct

root=Path(__file__).resolve().parent.parent
out=root/'tests/.iq3_pair_lut'
out.mkdir(exist_ok=True)
# All four odd magnitudes are exactly representable as binary16 and as bytes.
for code in range(4096):
    values=[2*((code>>(3*j))&7)+1 for j in range(4)]
    word=sum(value<<(8*j) for j,value in enumerate(values))
    for j,value in enumerate(values):
        assert struct.pack('<f',struct.unpack('<e',struct.pack('<e',value))[0])==struct.pack('<f',value)
        assert ((word>>(8*j))&255)==value
source=(root/'metal/qw3_core_moe.metal').read_text()
def extract(name):
    a=source.index(name)
    return source[a:source.index('\n}\n',a)+3]
helper=extract('inline float qw3_iq3s_dot32_vector(')
kernel=extract('kernel void qw3_matvec_iq3_s_pair_fast(')
shader='''inline uint private_iq3_pair_lut_pack(ushort packed) {
    const uint p=uint(packed);
    return (2u*(p&7u)+1u) | ((2u*((p>>3u)&7u)+1u)<<8u) |
           ((2u*((p>>6u)&7u)+1u)<<16u) | ((2u*((p>>9u)&7u)+1u)<<24u);
}
inline float4 private_iq3_pair_lut_decode(uint packed) {
    return float4((uint4(packed)>>uint4(0u,8u,16u,24u))&uint4(255u));
}
'''
for variant in (1,2):
    name=f'private_iq3_pair_lut_dot{variant}'
    dtype='half4' if variant==1 else 'uint'
    body=helper.replace('qw3_iq3s_dot32_vector',name)
    body=body.replace('threadgroup const float4 *grid',f'threadgroup const {dtype} *grid')
    for index in (1,2):
        expanded=f'float4(grid[idx{index}])' if variant==1 else f'private_iq3_pair_lut_decode(grid[idx{index}])'
        body=body.replace(f'dot(grid[idx{index}] * sign{index}',f'dot({expanded} * sign{index}')
    restored=body.replace(name,'qw3_iq3s_dot32_vector').replace(f'threadgroup const {dtype} *grid','threadgroup const float4 *grid')
    for index in (1,2):
        expanded=f'float4(grid[idx{index}])' if variant==1 else f'private_iq3_pair_lut_decode(grid[idx{index}])'
        restored=restored.replace(expanded,f'grid[idx{index}]')
    assert restored==helper
    k=kernel.replace('qw3_matvec_iq3_s_pair_fast',f'private_iq3_pair_lut_fast{variant}').replace('qw3_iq3s_dot32_vector',name)
    old='grid[i] = qw3_iq3s_float_grid(kgrid[i]);'
    new='grid[i] = half4(qw3_iq3s_float_grid(kgrid[i]));' if variant==1 else 'grid[i] = private_iq3_pair_lut_pack(kgrid[i]);'
    assert k.count(old)==1
    k=k.replace('threadgroup float4 grid[512];',f'threadgroup {dtype} grid[512];').replace(old,new)
    assert k.replace(f'private_iq3_pair_lut_fast{variant}','qw3_matvec_iq3_s_pair_fast').replace(name,'qw3_iq3s_dot32_vector').replace(f'threadgroup {dtype} grid[512];','threadgroup float4 grid[512];').replace(new,old)==kernel
    shader+=body+k
(out/'kernels.metal').write_text(shader)

previous=root/'docs/measurements/iq3_pair_grid_2026-10-14'
m=json.loads((previous/'manifest.json').read_text())
def read_base(name):
    key='tests/.iq3_pair_grid/'+name
    content=(previous/m['source_snapshot_paths'][key]).read_bytes()
    assert hashlib.sha256(content).hexdigest()==m['source_sha256'][key]
    return content.decode()
def rename(s):
    return (s.replace('iq3_pair_grid','iq3_pair_lut').replace('IQ3PairGrid','IQ3PairLut')
            .replace('iq3PairGrid','iq3PairLut').replace('pair_grid_variant','pair_lut_variant')
            .replace('IQ3 pair grid','IQ3 pair lut').replace('IQ3_PAIR_GRID','IQ3_PAIR_LUT')
            .replace('baseline/packed-threadgroup-grid/direct-device-grid','baseline/exact-half4-grid/expanded-byte-grid'))
hooks=rename(read_base('hooks.inc')).replace('Paired packed grid; native weights, geometry and FP32 dot expressions.',
                                          'Exact half4 or expanded-byte paired grid; native FP32 dot expressions.')
hooks=hooks.replace('(v==1 ? 1024u : 0u)','(v==1 ? 4096u : 2048u)')
(out/'hooks.inc').write_text(hooks)
for name in ('model.inc','native.inc'):(out/name).write_text(rename(read_base(name)))
