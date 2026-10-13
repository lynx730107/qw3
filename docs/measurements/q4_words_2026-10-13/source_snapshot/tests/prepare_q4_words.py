"""Private aligned Q4_K payload words; retain scalar FP32 evaluation order."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / 'tests/.q4_words'
out.mkdir(exist_ok=True)
source = (root / 'metal/qw3_core_linear.metal').read_text()
name = 'qw3_matvec_q4_k'
a = source.index('kernel void ' + name + '(')
original = source[a:source.index('\n}\n', a)+3]
old = '''            for (uint l = 0; l < 32u; l++) {
                uint packed = uint(qs[qbase + l]);
                uint q = (g & 1u) == 0u ? (packed & 15u) : (packed >> 4u);
                sum += (dl * float(q) - ml) * xx[xbase + l];
            }'''
shader = ''
for variant, step, dtype in ((1, 2, 'ushort'), (2, 4, 'uint')):
    new = f'''            for (uint l = 0; l < 32u; l += {step}u) {{
                uint word = uint(*((device const {dtype} *)(qs + qbase + l)));
                for (uint j = 0; j < {step}u; j++) {{
                    uint packed = (word >> (8u * j)) & 255u;
                    uint q = (g & 1u) == 0u ? (packed & 15u) : (packed >> 4u);
                    sum += (dl * float(q) - ml) * xx[xbase + l + j];
                }}
            }}'''
    private = f'private_q4_words_fast{variant}'
    assert original.count(old) == 1
    body = original.replace(name, private).replace(old, new)
    assert body.replace(private, name).replace(new, old) == original
    shader += body
(out / 'kernels.metal').write_text(shader)

base = root / 'tests/.q4_rows'
s = (base / 'hooks.inc').read_text()
s = (s.replace('q4_rows', 'q4_words').replace('Q4Rows', 'Q4Words').replace('q4Rows', 'q4Words'))
s = s.replace('Two Q4_K rows per workgroup; native per-row arithmetic and data buffers.',
              'Aligned Q4_K payload words; native geometry, scalar FP32 arithmetic and buffers.')
s = s.replace('q4_words_pipeline[2]', 'q4_words_pipeline[3]', 1)
s = s.replace('static int q4_words_mode;', 'static int q4_words_mode,words_variant=1;')
s = s.replace('static _Thread_local uint32_t q4_words_out;',
              'static _Thread_local int q4_words_header,q4_words_weight_aligned;')
s = s.replace('@protocol QW3Q4WordsHooks\n', '@protocol QW3Q4WordsHooks\n- (void)q4WordsSetBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index;\n')
a = s.index('- (void)q4WordsSetPipeline:', s.index('@implementation'))
z = s.index('\n@end', a)
s = s[:a] + '''- (void)q4WordsSetPipeline:(id<MTLComputePipelineState>)pipeline {
    q4_words_encoder=(__bridge void *)self;
    q4_words_seen=(pipeline==q4_words_pipeline[0]);
    q4_words_header=0;q4_words_weight_aligned=0;q4_words_eligible=0;
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
- (void)q4WordsSetBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index {
    if (q4_words_mode && q4_words_encoder==(__bridge void *)self &&
        q4_words_seen && index==0 && length==3u*sizeof(uint32_t)) {
        const uint32_t *a=bytes;
        uint32_t alignment=words_variant==1 ? 2u : 4u;
        q4_words_header=a[0]>0u && a[0]%256u==0u && a[1]>0u && a[2]%alignment==0u;
    }
    [(id<MTLComputeCommandEncoder>)self setBytes:bytes length:length atIndex:index];
}
- (void)q4WordsSetBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index {
    if (q4_words_mode && q4_words_encoder==(__bridge void *)self && q4_words_seen && index==1u) {
        NSUInteger alignment=words_variant==1 ? 2u : 4u;
        q4_words_weight_aligned=buffer!=nil && offset%alignment==0u;
    }
    [(id<MTLComputeCommandEncoder>)self setBuffer:buffer offset:offset atIndex:index];
}
- (void)q4WordsDispatch:(MTLSize)groups threadsPerThreadgroup:(MTLSize)threads {
    if (q4_words_mode && q4_words_encoder==(__bridge void *)self && q4_words_seen) {
        if (q4_words_header && q4_words_weight_aligned) {
            [(id<MTLComputeCommandEncoder>)self setComputePipelineState:q4_words_pipeline[words_variant]];
            q4_words_calls++;
        } else q4_words_fallbacks++;
        q4_words_seen=0;
    }
    [(id<MTLComputeCommandEncoder>)self dispatchThreadgroups:groups threadsPerThreadgroup:threads];
}
''' + s[z:]
s = s.replace('#define setBytes q4WordsSetBytes', '#define setBuffer q4WordsSetBuffer\n#define setBytes q4WordsSetBytes')
s = s.replace('#undef setBytes', '#undef setBuffer\n#undef setBytes')
a = s.index('static int q4_words_compile(void) {')
s = s[:a] + '''static int q4_words_compile(void) {
    if (q4_words_pipeline[2]) return 1;
    if (!qw3_metal_init() || !qw3_metal_compile_kernels()) return 0;
    q4_words_pipeline[0]=g_matvec_q4_k_pipeline;
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) options.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/.q4_words/kernels.metal" encoding:NSUTF8StringEncoding error:&error];
    NSString *original=qw3_metal_full_kernel_source();
    NSString *source=extra && original ? [original stringByAppendingString:extra] : nil;
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    for (int v=1;v<=2;v++) {
        id<MTLFunction> fn=[lib newFunctionWithName:[NSString stringWithFormat:@"private_q4_words_fast%d",v]];
        id<MTLComputePipelineState> p=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!p || p.threadExecutionWidth!=32u || p.maxTotalThreadsPerThreadgroup<256u ||
            p.staticThreadgroupMemoryLength!=q4_words_pipeline[0].staticThreadgroupMemoryLength) {
            fprintf(stderr,"Q4 words compile variant=%d: %s\\n",v,error.localizedDescription.UTF8String);return 0;
        }
        q4_words_pipeline[v]=p;
    }
    for (int v=0;v<=2;v++) fprintf(stderr,"q4_words_metadata variant=%d width=%lu max_threads=%lu static_threadgroup_bytes=%lu dynamic_threadgroup_bytes=128\\n",v,(unsigned long)q4_words_pipeline[v].threadExecutionWidth,(unsigned long)q4_words_pipeline[v].maxTotalThreadsPerThreadgroup,(unsigned long)q4_words_pipeline[v].staticThreadgroupMemoryLength);
    return 1;
}
static void scratch_select(int mode) {
    q4_words_mode=mode!=0;q4_words_seen=0;q4_words_header=0;q4_words_weight_aligned=0;q4_words_encoder=NULL;
}
static void q4_words_print_counts(void) {
    fprintf(stderr,"q4_words_dispatch_counts calls=%llu fallbacks=%llu\\n",(unsigned long long)q4_words_calls,(unsigned long long)q4_words_fallbacks);
}
'''
(out / 'hooks.inc').write_text(s)
s = (base / 'model.inc').read_text().replace('q4_rows', 'q4_words').replace('Q4 rows', 'Q4 words')
s = s.replace('baseline/two-row64', 'baseline/ushort/uint').replace('phase<=1', 'phase<=2')
s = s.replace('            scratch_select(phase!=0);', '            words_variant=phase ? phase : 1;scratch_select(phase!=0);')
s = s.replace('32u*34u', '32u*36u').replace('32u*2u', '0u')
(out / 'model.inc').write_text(s)
s = (base / 'native.inc').read_text().replace('q4_rows', 'q4_words').replace('Q4_ROWS', 'Q4_WORDS')
s = s.replace('(strcmp(mode,"0") && strcmp(mode,"1"))', '(strcmp(mode,"0") && strcmp(mode,"1") && strcmp(mode,"2"))')
s = s.replace('int variant=atoi(mode);scratch_select(variant);', 'int variant=atoi(mode);words_variant=variant ? variant : 1;scratch_select(variant);')
(out / 'native.inc').write_text(s)
