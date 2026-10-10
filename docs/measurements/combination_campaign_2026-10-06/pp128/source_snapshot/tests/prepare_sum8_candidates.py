"""Keep Q5_K geometry/arithmetic; share exact input sums across all rows."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / 'tests/.sum8_candidates'
out.mkdir(exist_ok=True)
shader = (root / 'metal/qw3_core_linear.metal').read_text()
start = shader.index('kernel void qw3_matvec_q5_k_fast(')
end = shader.index('kernel void qw3_matvec_q4_k(', start)
shader = shader[start:end].replace('qw3_matvec_q5_k_fast', 'sum8_q5_head')
anchor = '                                 threadgroup float *sh,'
assert shader.count(anchor) == 1
shader = shader.replace(anchor, '                                 device const float *partials,\n' + anchor)
anchor = '        float4 sumx = float4(0.0f);'
assert shader.count(anchor) == 1
shader = shader.replace(anchor, '''        uint si = (ib * 256u + x_offset) / 8u;
        float4 sumx = float4(partials[si], partials[si+4u], partials[si+16u], partials[si+20u]);''')
for i, term in enumerate(['yl[l]', 'yl[l + 8u]', 'yh[l]', 'yh[l + 8u]']):
    anchor = f'sumx[{i}] += {term};'
    assert shader.count(anchor) == 1
    shader = shader.replace(anchor, '')
shader += '''
kernel void sum8_prepare(constant uint &n, device const float *x,
    device float *out, uint gid [[thread_position_in_grid]]) {
    if (gid >= n/8u) return;
    float ss = 0.0f;
#pragma unroll
    for (uint l=0; l<8u; ++l) ss += x[gid*8u+l];
    out[gid] = ss;
}
'''
(out / 'kernels.metal').write_text(shader)
bridge = (root / 'qw3_metal.m').read_text()
start = bridge.index('int qw3_metal_session_matvec_q5_k_x1_to_logits(')
end = bridge.index('int qw3_metal_session_argmax_logits(', start)
body = bridge[start:end]
anchor = '    if (!s || !s->obj || n_in == 0'
assert body.count(anchor) == 1
body = body.replace(anchor, '''    if (!(combo_mode & 4)) return sum8_base_logits(s,tensor_offset,n_in,n_out,out);
    if (!sum8_compile()) return 0;
''' + anchor)
anchor = '    int owned = 0;'
assert body.count(anchor) == 1
body = body.replace(anchor, '''    uint64_t partial_bytes=(uint64_t)(n_in/8u)*sizeof(float);
    if (!sum8_buffer || sum8_buffer.length<partial_bytes)
        sum8_buffer=qw3_metal_new_private_buffer(partial_bytes);
    if (!sum8_buffer) return 0;
''' + anchor)
anchor = '    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);'
assert body.count(anchor) == 1
body = body.replace(anchor, '''    if (!cb) return 0;
    id<MTLComputeCommandEncoder> prep=qw3_metal_compute_encoder(cb);
    [prep setComputePipelineState:sum8_prepare_pipeline];
    [prep setBytes:&n_in length:sizeof(n_in) atIndex:0];
    [prep setBuffer:obj.x1 offset:0 atIndex:1];
    [prep setBuffer:sum8_buffer offset:0 atIndex:2];
    [prep dispatchThreads:MTLSizeMake(n_in/8u,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
    qw3_metal_end_compute_encoder(cb,prep);
''' + anchor)
body = body.replace('setComputePipelineState:g_matvec_q5_k_pipeline', 'setComputePipelineState:sum8_head_pipeline')
anchor = '    [enc setBuffer:obj.logits offset:0 atIndex:3];'
assert body.count(anchor) == 1
body = body.replace(anchor, anchor + '\n    [enc setBuffer:sum8_buffer offset:0 atIndex:4];')
body = body.replace('    return 1;\n}', '    sum8_head_calls++;\n    return 1;\n}')
(out / 'head.inc').write_text(body)

adapter = (root / 'tests/combination_candidates.inc').read_text()
anchor = '#include ".combination_candidates/bridge.inc"'
assert adapter.count(anchor) == 1
adapter = adapter.replace(anchor, '''#define qw3_metal_session_matvec_q5_k_x1_to_logits sum8_base_logits
#include "../.combination_candidates/bridge.inc"
#undef qw3_metal_session_matvec_q5_k_x1_to_logits
''')
anchor = '#define qw3_metal_env_n_gpu_layers combo_core_env_n_gpu_layers'
assert adapter.count(anchor) == 1
adapter = adapter.replace(anchor, '#include "../sum8_dispatch.inc"\n' + anchor)
adapter = adapter.replace('".combination_candidates/core.inc"', '"../.combination_candidates/core.inc"')
(out / 'candidates.inc').write_text(adapter)
quality = (root / 'tests/.combination_candidates/quality.inc').read_text()
quality = quality.replace('#include "../combination_candidates.inc"', '#include "candidates.inc"')
quality = quality.replace('return name && !strcmp(name,"residual") ? 1 : name && !strcmp(name,"round") ? 2 : name && !strcmp(name,"both") ? 3 : -1;',
    'return name && !strcmp(name,"logits") ? 4 : name && !strcmp(name,"all") ? 7 : -1;')
quality = quality.replace('        rc=0;', '        if (!sum8_head_calls) goto done;\n        printf("sum8_head=%llu\\n",sum8_head_calls);\n        rc=0;')
(out / 'quality.inc').write_text(quality)
