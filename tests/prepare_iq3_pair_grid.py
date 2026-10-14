"""Private paired IQ3_S lookup: packed threadgroup or existing device grid."""
from pathlib import Path
import hashlib
import json

root=Path(__file__).resolve().parent.parent
out=root/'tests/.iq3_pair_grid'
out.mkdir(exist_ok=True)
source=(root/'metal/qw3_core_moe.metal').read_text()
def extract(name):
    first=source.index(name)
    return source[first:source.index('\n}\n',first)+3]
helper=extract('inline float qw3_iq3s_dot32_vector(')
kernel=extract('kernel void qw3_matvec_iq3_s_pair_fast(')
shader=''
for variant in (1,2):
    private=f'private_iq3_pair_grid_dot{variant}'
    body=helper.replace('qw3_iq3s_dot32_vector',private)
    body=body.replace('threadgroup const float4 *grid',
                      ('threadgroup' if variant==1 else 'device')+' const ushort *grid')
    for index in (1,2):
        body=body.replace(f'dot(grid[idx{index}] * sign{index}',
                          f'dot(qw3_iq3s_float_grid(grid[idx{index}]) * sign{index}')
    restored=body.replace(private,'qw3_iq3s_dot32_vector')
    restored=restored.replace(('threadgroup' if variant==1 else 'device')+' const ushort *grid',
                              'threadgroup const float4 *grid')
    for index in (1,2):
        restored=restored.replace(f'qw3_iq3s_float_grid(grid[idx{index}])',f'grid[idx{index}]')
    assert restored==helper
    k=kernel.replace('qw3_matvec_iq3_s_pair_fast',f'private_iq3_pair_grid_fast{variant}')
    k=k.replace('qw3_iq3s_dot32_vector',private)
    old='''    threadgroup float4 grid[512];
    const uint tid = uint(simd_idx) * 32u + uint(lane);
    for (uint i = tid; i < 512u; i += 64u) grid[i] = qw3_iq3s_float_grid(kgrid[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);'''
    new=('''    threadgroup ushort grid[512];
    const uint tid = uint(simd_idx) * 32u + uint(lane);
    for (uint i = tid; i < 512u; i += 64u) grid[i] = kgrid[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);''' if variant==1 else
         '    device const ushort *grid = kgrid;')
    assert k.count(old)==1
    k=k.replace(old,new)
    assert k.replace(f'private_iq3_pair_grid_fast{variant}','qw3_matvec_iq3_s_pair_fast').replace(private,'qw3_iq3s_dot32_vector').replace(new,old)==kernel
    shader+=body+k
(out/'kernels.metal').write_text(shader)

previous=root/'docs/measurements/iq3_input_2026-10-12'
manifest=json.loads((previous/'manifest.json').read_text())
def read_base(name):
    key='tests/.iq3_input/'+name
    content=(previous/manifest['source_snapshot_paths'][key]).read_bytes()
    assert hashlib.sha256(content).hexdigest()==manifest['source_sha256'][key]
    return content.decode()
def rename(s):
    return (s.replace('iq3_input','iq3_pair_grid').replace('IQ3Input','IQ3PairGrid')
            .replace('iq3Input','iq3PairGrid').replace('input_variant','pair_grid_variant')
            .replace('IQ3 input','IQ3 pair grid').replace('IQ3_INPUT','IQ3_PAIR_GRID')
            .replace('baseline/packed-input/packed-input-unroll2','baseline/packed-threadgroup-grid/direct-device-grid'))
hooks=rename(read_base('hooks.inc')).replace('g_matvec_iq3_s_pipeline','g_matvec_iq3_s_pair_pipeline')
hooks=hooks.replace('if (!qw3_metal_compile_kernels()) return 0;',
                    'if (!qw3_metal_init() || !qw3_metal_compile_kernels()) return 0;')
hooks=hooks.replace('Packed input loads; native weights, grid, FP32 arithmetic and data buffers.',
                    'Paired packed grid; native weights, geometry and FP32 dot expressions.')
hooks=hooks.replace('iq3_pair_grid_pipeline[v].staticThreadgroupMemoryLength!=iq3_pair_grid_pipeline[0].staticThreadgroupMemoryLength',
                    'iq3_pair_grid_pipeline[v].staticThreadgroupMemoryLength!=(v==1 ? 1024u : 0u)')
hooks=hooks.replace('dynamic_threadgroup_bytes=256','dynamic_threadgroup_bytes=128')
(out/'hooks.inc').write_text(hooks)
(out/'model.inc').write_text(rename(read_base('model.inc')).replace('32u*73u','32u*19u'))
(out/'native.inc').write_text(rename(read_base('native.inc')))
