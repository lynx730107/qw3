"""Private disjoint GQA partial/dot scratch; original 64-float allocation."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / 'tests/.gqa_barrier'
out.mkdir(exist_ok=True)
source = (root / 'metal/qw3_core_sequence.metal').read_text()
name = 'qw3_gqa_attend_n_inner'
a = source.index('kernel void ' + name + '(')
original = source[a:source.index('\n}\n', a)+3]
shader = '/* Original declarations and cache helper come from production source. */\n'
for variant in (1, 2):
    body = original.replace(name, f'private_gqa_barrier{variant}')
    body = body.replace('group_heads > 8u', 'group_heads > 7u')
    body = body.replace('if (lane == 0) sh[gh * 8u + uint(simd_idx)] = part;',
                        'if (lane == 0 && gh < group_heads) sh[gh * 8u + uint(simd_idx)] = part;')
    body = body.replace('if (tid == 0) sh[gh] = dot;',
                        'if (tid == 0 && gh < group_heads) sh[56u + gh] = dot;')
    body = body.replace('float score = sh[gh] * scale;', 'float score = sh[56u + gh] * scale;')
    last_barrier = '        threadgroup_barrier(mem_flags::mem_threadgroup);\n    }\n    if (i < args.head_dim)'
    assert body.count(last_barrier) == 1
    body = body.replace(last_barrier, '    }\n    if (i < args.head_dim)')
    if variant == 2:
        assert body.count('gh < 8u') == 5
        body = body.replace('gh < 8u', 'gh < group_heads')
    assert body.count('threadgroup_barrier(') == 2
    shader += body
(out / 'kernels.metal').write_text(shader)

base = root / 'tests/.iq4_unroll'
hooks = (base / 'hooks.inc').read_text().replace('iq4_unroll', 'gqa_barrier')
hooks = hooks.replace('IQ4Unroll', 'GQABarrier').replace('iq4Unroll', 'gqaBarrier')
hooks = hooks.replace('IQ4 unroll', 'GQA barrier').replace('unroll_variant', 'barrier_variant')
hooks = hooks.replace('gqa_barrier_pipeline[3][3]', 'gqa_barrier_pipeline[3]', 1)
hooks = hooks.replace('static uint64_t gqa_barrier_calls,gqa_barrier_calls_by_kind[3];',
                      'static uint64_t gqa_barrier_calls,gqa_barrier_fallbacks;')
hooks = hooks.replace('static int barrier_variant=1;', '''static int barrier_variant=1,gqa_barrier_mode;
static _Thread_local void *gqa_barrier_encoder;
static _Thread_local int gqa_barrier_eligible;
''')
hooks = hooks.replace('- (void)gqaBarrierSetPipeline:',
                      '- (void)gqaBarrierSetBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index;\n- (void)gqaBarrierSetPipeline:', 1)
a = hooks.index('- (void)gqaBarrierSetPipeline:', hooks.index('@implementation'))
b = hooks.index('\n@end', a)
hooks = hooks[:a] + '''- (void)gqaBarrierSetPipeline:(id<MTLComputePipelineState>)pipeline {
    gqa_barrier_encoder=(__bridge void *)self;
    gqa_barrier_eligible=(pipeline==gqa_barrier_pipeline[0]);
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
- (void)gqaBarrierSetBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index {
    if (gqa_barrier_mode && gqa_barrier_encoder==(__bridge void *)self &&
        gqa_barrier_eligible && index==0 && length==5u*sizeof(uint32_t)) {
        const uint32_t *a=bytes;
        if (a[2] && a[1]%a[2]==0 && a[1]/a[2]>0 && a[1]/a[2]<=7u) {
            [(id<MTLComputeCommandEncoder>)self setComputePipelineState:gqa_barrier_pipeline[barrier_variant]];
            gqa_barrier_calls++;
        } else gqa_barrier_fallbacks++;
        gqa_barrier_eligible=0;
    }
    [(id<MTLComputeCommandEncoder>)self setBytes:bytes length:length atIndex:index];
}
''' + hooks[b:]
hooks = hooks.replace('#define setComputePipelineState gqaBarrierSetPipeline',
                      '#define setBytes gqaBarrierSetBytes\n#define setComputePipelineState gqaBarrierSetPipeline')
hooks = hooks.replace('#undef setComputePipelineState', '#undef setBytes\n#undef setComputePipelineState')
a = hooks.index('static int gqa_barrier_compile(void) {')
hooks = hooks[:a] + '''static int gqa_barrier_compile(void) {
    if (gqa_barrier_pipeline[2]) return 1;
    if (!qw3_metal_init() || !qw3_metal_compile_kernels()) return 0;
    gqa_barrier_pipeline[0]=g_gqa_attend_n_inner_pipeline;
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) options.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/.gqa_barrier/kernels.metal" encoding:NSUTF8StringEncoding error:&error];
    NSString *original=qw3_metal_full_kernel_source();
    NSString *source=extra && original ? [original stringByAppendingString:extra] : nil;
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    for (int v=1;v<=2;v++) {
        NSString *name=[NSString stringWithFormat:@"private_gqa_barrier%d",v];
        id<MTLFunction> fn=[lib newFunctionWithName:name];
        id<MTLComputePipelineState> p=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!p || p.threadExecutionWidth!=32 || p.maxTotalThreadsPerThreadgroup<256 ||
            p.staticThreadgroupMemoryLength!=gqa_barrier_pipeline[0].staticThreadgroupMemoryLength) {
            fprintf(stderr,"GQA barrier compile variant=%d: %s\\n",v,error.localizedDescription.UTF8String);return 0;
        }
        gqa_barrier_pipeline[v]=p;
    }
    for (int v=0;v<=2;v++) fprintf(stderr,
        "gqa_barrier_metadata variant=%d thread_width=%lu max_threads=%lu static_threadgroup_bytes=%lu dynamic_threadgroup_bytes=256 barriers_per_context=%d\\n",
        v,(unsigned long)gqa_barrier_pipeline[v].threadExecutionWidth,
        (unsigned long)gqa_barrier_pipeline[v].maxTotalThreadsPerThreadgroup,
        (unsigned long)gqa_barrier_pipeline[v].staticThreadgroupMemoryLength,v ? 2 : 3);
    return 1;
}
static __attribute__((unused)) void scratch_select(int mode) {
    gqa_barrier_mode=mode!=0;gqa_barrier_eligible=0;gqa_barrier_encoder=NULL;
}
static __attribute__((unused)) void gqa_barrier_print_counts(void) {
    fprintf(stderr,"gqa_barrier_dispatch_counts calls=%llu fallbacks=%llu\\n",
        (unsigned long long)gqa_barrier_calls,(unsigned long long)gqa_barrier_fallbacks);
}
'''
(out / 'hooks.inc').write_text(hooks)
for filename in ('model.inc', 'native.inc'):
    text = (base / filename).read_text().replace('iq4_unroll', 'gqa_barrier')
    text = text.replace('IQ4 unroll', 'GQA barrier').replace('unroll_variant', 'barrier_variant')
    text = text.replace('IQ4_UNROLL', 'GQA_BARRIER')
    text = text.replace('baseline/unroll2/unroll4', 'baseline/disjoint-dots/disjoint-dots-active-heads')
    text = text.replace('memset(gqa_barrier_calls_by_kind,0,sizeof(gqa_barrier_calls_by_kind));', 'gqa_barrier_fallbacks=0;')
    text = text.replace('32u*142u', '32u*16u')
    text = text.replace('            if (phase && (gqa_barrier_calls_by_kind[0]!=3072u || gqa_barrier_calls_by_kind[1]!=448u || gqa_barrier_calls_by_kind[2]!=1024u)) goto done;',
                        '            if (gqa_barrier_fallbacks) goto done;')
    text = text.replace('if (rc) fprintf(stderr,', 'if (rc) gqa_barrier_print_counts();\n        if (rc) fprintf(stderr,')
    (out / filename).write_text(text)
