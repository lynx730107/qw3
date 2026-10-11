"""Compile the original IQ3_S function with explicit threadgroup hints only."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / 'tests/.iq3_hint'
out.mkdir(exist_ok=True)
base = root / 'tests/.iq3_words'


def rename(text):
    return (text.replace('iq3_words', 'iq3_hint').replace('IQ3Words', 'IQ3Hint')
            .replace('iq3Words', 'iq3Hint').replace('words_variant', 'hint_variant')
            .replace('IQ3 words', 'IQ3 hint').replace('QW3_PRIVATE_IQ3_WORDS', 'QW3_PRIVATE_IQ3_HINT'))


hooks = rename((base / 'hooks.inc').read_text())
hooks = hooks.replace('Aligned integer loads from original IQ3_S weights; native grid and FP32 math.',
                      'Original shader and buffers; explicit compute-pipeline threadgroup hints only.')
first = hooks.index('static int iq3_hint_compile(void) {')
hooks = hooks[:first] + '''static int iq3_hint_compile(void) {
    if (iq3_hint_pipeline[2]) return 1;
    if (!qw3_metal_compile_kernels()) return 0;
    iq3_hint_pipeline[0]=g_matvec_iq3_s_pipeline;
    id<MTLFunction> fn=[g_library newFunctionWithName:@"qw3_matvec_iq3_s_fast"];
    if (!fn) return 0;
    NSError *error=nil;
    for (int v=1;v<=2;v++) {
        MTLComputePipelineDescriptor *descriptor=[MTLComputePipelineDescriptor new];
        descriptor.computeFunction=fn;
        descriptor.maxTotalThreadsPerThreadgroup=64;
        descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth=(v==2);
        iq3_hint_pipeline[v]=[g_device newComputePipelineStateWithDescriptor:descriptor
            options:MTLPipelineOptionNone reflection:nil error:&error];
        if (!iq3_hint_pipeline[v] || iq3_hint_pipeline[v].maxTotalThreadsPerThreadgroup!=64 ||
            iq3_hint_pipeline[v].threadExecutionWidth!=32 ||
            iq3_hint_pipeline[v].staticThreadgroupMemoryLength!=iq3_hint_pipeline[0].staticThreadgroupMemoryLength) {
            fprintf(stderr,"IQ3 hint compile variant=%d: %s\\n",v,error.localizedDescription.UTF8String);return 0;
        }
    }
    for (int v=0;v<=2;v++) {
        id<MTLComputePipelineState> p=iq3_hint_pipeline[v];
        fprintf(stderr,"iq3_hint_metadata variant=%d thread_width=%lu max_threads=%lu static_threadgroup_bytes=%lu\\n",
            v,(unsigned long)p.threadExecutionWidth,(unsigned long)p.maxTotalThreadsPerThreadgroup,
            (unsigned long)p.staticThreadgroupMemoryLength);
    }
    return 1;
}
static __attribute__((unused)) void scratch_select(int mode) {
    g_matvec_iq3_s_pipeline=iq3_hint_pipeline[mode ? hint_variant : 0];
}
'''
(out / 'hooks.inc').write_text(hooks)
for name in ['matrix.inc', 'model.inc', 'native.inc']:
    text = rename((base / name).read_text())
    text = text.replace('baseline/QS-word/QS-and-sign-word', 'baseline/max64/max64-and-SIMD-multiple')
    (out / name).write_text(text)
