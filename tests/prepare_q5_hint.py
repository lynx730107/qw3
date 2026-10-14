"""Original Q5_K function; descriptor hints only, no shader/data changes."""
from pathlib import Path
import hashlib
import json

root=Path(__file__).resolve().parent.parent
out=root/'tests/.q5_hint'
out.mkdir(exist_ok=True)
previous=root/'docs/measurements/iq3_input_2026-10-12'
m=json.loads((previous/'manifest.json').read_text())
def read_base(name):
    key='tests/.iq3_input/'+name
    data=(previous/m['source_snapshot_paths'][key]).read_bytes()
    assert hashlib.sha256(data).hexdigest()==m['source_sha256'][key]
    return data.decode()
def rename(s):
    return (s.replace('iq3_input','q5_hint').replace('IQ3Input','Q5Hint').replace('iq3Input','q5Hint')
            .replace('input_variant','hint_variant').replace('IQ3 input','Q5 hint')
            .replace('IQ3_INPUT','Q5_HINT').replace('g_matvec_iq3_s_pipeline','g_matvec_q5_k_pipeline')
            .replace('baseline/packed-input/packed-input-unroll2','baseline/max64/max64-and-SIMD-multiple'))
hooks=rename(read_base('hooks.inc'))
hooks=hooks.replace('Packed input loads; native weights, grid, FP32 arithmetic and data buffers.',
                    'Original Q5_K shader/data buffers; descriptor hints only.')
a=hooks.index('static int q5_hint_compile(void) {')
hooks=hooks[:a]+'''static int q5_hint_compile(void) {
    if (q5_hint_pipeline[2]) return 1;
    if (!qw3_metal_init() || !qw3_metal_compile_kernels()) return 0;
    q5_hint_pipeline[0]=g_matvec_q5_k_pipeline;
    id<MTLFunction> fn=[g_library newFunctionWithName:@"qw3_matvec_q5_k_fast"];
    if (!fn) return 0;
    NSError *error=nil;
    for (int v=1;v<=2;v++) {
        MTLComputePipelineDescriptor *descriptor=[MTLComputePipelineDescriptor new];
        descriptor.computeFunction=fn;
        descriptor.maxTotalThreadsPerThreadgroup=64u;
        descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth=(v==2);
        q5_hint_pipeline[v]=[g_device newComputePipelineStateWithDescriptor:descriptor
            options:MTLPipelineOptionNone reflection:nil error:&error];
        if (!q5_hint_pipeline[v] || q5_hint_pipeline[v].threadExecutionWidth!=32u ||
            q5_hint_pipeline[v].maxTotalThreadsPerThreadgroup!=64u ||
            q5_hint_pipeline[v].staticThreadgroupMemoryLength!=q5_hint_pipeline[0].staticThreadgroupMemoryLength) {
            fprintf(stderr,"Q5 hint compile variant=%d: %s\\n",v,error.localizedDescription.UTF8String);return 0;
        }
    }
    for (int v=0;v<=2;v++) fprintf(stderr,"q5_hint_metadata variant=%d width=%lu max_threads=%lu static_threadgroup_bytes=%lu dynamic_threadgroup_bytes=128 threads=64\\n",v,(unsigned long)q5_hint_pipeline[v].threadExecutionWidth,(unsigned long)q5_hint_pipeline[v].maxTotalThreadsPerThreadgroup,(unsigned long)q5_hint_pipeline[v].staticThreadgroupMemoryLength);
    return 1;
}
static __attribute__((unused)) void scratch_select(int mode) {
    g_matvec_q5_k_pipeline=q5_hint_pipeline[mode ? hint_variant : 0];
}
'''
(out/'hooks.inc').write_text(hooks)
(out/'model.inc').write_text(rename(read_base('model.inc')).replace('32u*73u','32u*26u'))
(out/'native.inc').write_text(rename(read_base('native.inc')))
