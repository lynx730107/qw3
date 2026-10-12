"""Private IQ4_XS function constants for K count and original row stride."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / 'tests/.iq4_shape'
out.mkdir(exist_ok=True)
base = root / 'tests/.iq4_unroll'
source = (root / 'metal/qw3_core_linear.metal').read_text()
names = ['qw3_matvec_iq4_xs_fast', 'qw3_matvec_iq4_xs_pair_fast',
         'qw3_matvec_iq4_xs_add_x0_fast']
shader = '''/* Original helpers; function constants alter integer geometry only. */
constant uint private_iq4_shape_blocks [[function_constant(100)]];
constant uint private_iq4_shape_stride [[function_constant(101)]];
'''
for variant in (1, 2):
    for kind, name in enumerate(names):
        first = source.index('kernel void ' + name + '(')
        last = source.index('\n}\n', first) + 3
        original = source[first:last]
        private_name = f'private_iq4_shape_fast{variant}_{kind}'
        body = original.replace(name, private_name)
        assert body.count('uint n_blocks = args.n_in / 256u;') == 1
        body = body.replace('uint n_blocks = args.n_in / 256u;',
                            'uint n_blocks = private_iq4_shape_blocks;')
        if variant == 2:
            body = body.replace('args.row_bytes', 'private_iq4_shape_stride')
        assert body.replace(private_name, name).replace(
            'private_iq4_shape_blocks', 'args.n_in / 256u').replace(
            'private_iq4_shape_stride', 'args.row_bytes') == original
        shader += body
(out / 'kernels.metal').write_text(shader)


def rename(text):
    return (text.replace('iq4_unroll', 'iq4_shape').replace('IQ4Unroll', 'IQ4Shape')
            .replace('iq4Unroll', 'iq4Shape').replace('IQ4 unroll', 'IQ4 shape')
            .replace('unroll_variant', 'shape_variant')
            .replace('QW3_PRIVATE_IQ4_UNROLL', 'QW3_PRIVATE_IQ4_SHAPE'))


hooks = rename((base / 'hooks.inc').read_text())
hooks = hooks.replace('iq4_shape_pipeline[3][3]', 'iq4_shape_pipeline[3][4][3]', 1)
hooks = hooks.replace('static int shape_variant=1;', '''static int shape_variant=1, iq4_shape_mode;
static uint64_t iq4_shape_fallbacks;
static const uint32_t iq4_shape_inputs[4]={256u,4352u,5120u,17408u};
static _Thread_local void *iq4_shape_encoder;
static _Thread_local int iq4_shape_kind=-1;
''')
hooks = hooks.replace('- (void)iq4ShapeSetPipeline:',
                      '- (void)iq4ShapeSetBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index;\n- (void)iq4ShapeSetPipeline:', 1)
first = hooks.index('- (void)iq4ShapeSetPipeline:', hooks.index('@implementation'))
last = hooks.index('\n@end', first)
hooks = hooks[:first] + '''- (void)iq4ShapeSetPipeline:(id<MTLComputePipelineState>)pipeline {
    iq4_shape_encoder=(__bridge void *)self;
    iq4_shape_kind=-1;
    if (iq4_shape_mode) for (int kind=0;kind<3;kind++)
        if (pipeline==iq4_shape_pipeline[0][0][kind]) iq4_shape_kind=kind;
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
- (void)iq4ShapeSetBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index {
    if (iq4_shape_mode && iq4_shape_encoder==(__bridge void *)self &&
        iq4_shape_kind>=0 && index==0 && length==3u*sizeof(uint32_t)) {
        const uint32_t *args=bytes;
        int shape=-1;
        for (int i=0;i<4;i++) if (args[0]==iq4_shape_inputs[i] && args[2]==args[0]/256u*136u) shape=i;
        if (shape>=0) {
            id<MTLComputePipelineState> pipeline=iq4_shape_pipeline[shape_variant][shape][iq4_shape_kind];
            [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
            iq4_shape_calls++;iq4_shape_calls_by_kind[iq4_shape_kind]++;
        } else iq4_shape_fallbacks++;
        iq4_shape_kind=-1;
    }
    [(id<MTLComputeCommandEncoder>)self setBytes:bytes length:length atIndex:index];
}
''' + hooks[last:]
hooks = hooks.replace('#define setComputePipelineState iq4ShapeSetPipeline',
                      '#define setBytes iq4ShapeSetBytes\n#define setComputePipelineState iq4ShapeSetPipeline')
hooks = hooks.replace('#undef setComputePipelineState', '#undef setBytes\n#undef setComputePipelineState')
first = hooks.index('static int iq4_shape_compile(void) {')
hooks = hooks[:first] + '''static int iq4_shape_compile(void) {
    if (iq4_shape_pipeline[2][3][2]) return 1;
    if (!qw3_metal_compile_kernels()) return 0;
    for (int i=0;i<4;i++) {
        iq4_shape_pipeline[0][i][0]=g_matvec_iq4_xs_pipeline;
        iq4_shape_pipeline[0][i][1]=g_matvec_iq4_xs_pair_pipeline;
        iq4_shape_pipeline[0][i][2]=g_matvec_iq4_xs_add_x0_pipeline;
    }
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) options.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/.iq4_shape/kernels.metal" encoding:NSUTF8StringEncoding error:&error];
    NSString *original=qw3_metal_full_kernel_source();
    NSString *source=extra && original ? [original stringByAppendingString:extra] : nil;
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    for (int v=1;v<=2;v++) for (int i=0;i<4;i++) for (int kind=0;kind<3;kind++) {
        uint32_t blocks=iq4_shape_inputs[i]/256u,stride=blocks*136u;
        MTLFunctionConstantValues *constants=[MTLFunctionConstantValues new];
        [constants setConstantValue:&blocks type:MTLDataTypeUInt atIndex:100];
        if (v==2) [constants setConstantValue:&stride type:MTLDataTypeUInt atIndex:101];
        NSString *name=[NSString stringWithFormat:@"private_iq4_shape_fast%d_%d",v,kind];
        id<MTLFunction> fn=lib ? [lib newFunctionWithName:name constantValues:constants error:&error] : nil;
        id<MTLComputePipelineState> p=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!p || p.maxTotalThreadsPerThreadgroup<64u || p.threadExecutionWidth!=32 ||
            p.staticThreadgroupMemoryLength!=iq4_shape_pipeline[0][0][kind].staticThreadgroupMemoryLength) {
            fprintf(stderr,"IQ4 shape compile variant=%d n_in=%u kind=%d: %s\\n",v,iq4_shape_inputs[i],kind,error.localizedDescription.UTF8String);return 0;
        }
        iq4_shape_pipeline[v][i][kind]=p;
        fprintf(stderr,"iq4_shape_metadata variant=%d n_in=%u blocks=%u stride_constant=%u kind=%d thread_width=%lu max_threads=%lu static_threadgroup_bytes=%lu threads=64 dynamic_threadgroup_bytes=128\\n",
            v,iq4_shape_inputs[i],blocks,v==2 ? stride : 0u,kind,(unsigned long)p.threadExecutionWidth,
            (unsigned long)p.maxTotalThreadsPerThreadgroup,(unsigned long)p.staticThreadgroupMemoryLength);
    }
    return 1;
}
static __attribute__((unused)) void scratch_select(int mode) {
    iq4_shape_mode=mode!=0;iq4_shape_kind=-1;iq4_shape_encoder=NULL;
}
static __attribute__((unused)) void iq4_shape_print_counts(void) {
    fprintf(stderr,"iq4_shape_dispatch_counts single=%llu pair=%llu add=%llu fallbacks=%llu\\n",
        (unsigned long long)iq4_shape_calls_by_kind[0],(unsigned long long)iq4_shape_calls_by_kind[1],
        (unsigned long long)iq4_shape_calls_by_kind[2],(unsigned long long)iq4_shape_fallbacks);
}
'''
(out / 'hooks.inc').write_text(hooks)
for name in ('matrix.inc', 'model.inc', 'native.inc'):
    text = rename((base / name).read_text())
    text = text.replace('baseline/unroll2/unroll4', 'baseline/K-constant/K-and-stride-constants')
    if name == 'matrix.inc':
        text = text.replace('[enc setComputePipelineState:iq4_shape_pipeline[v ? shape_variant : 0][kind]];',
                            'scratch_select(v);\n                    [(id<QW3IQ4ShapeHooks>)enc iq4ShapeSetPipeline:iq4_shape_pipeline[0][0][kind]];')
        text = text.replace('[enc setBytes:&args length:sizeof(args) atIndex:0];',
                            '[(id<QW3IQ4ShapeHooks>)enc iq4ShapeSetBytes:&args length:sizeof(args) atIndex:0];')
        text = text.replace('        printf("PASS iq4_shapes=',
                            '        if (iq4_shape_calls!=54u || iq4_shape_fallbacks) goto done;\n        iq4_shape_print_counts();\n        printf("PASS iq4_shapes=')
    if name == 'model.inc':
        text = text.replace('iq4_shape_calls=0;', 'iq4_shape_calls=0;iq4_shape_fallbacks=0;')
        text = text.replace('!grid_oracle_state(oracle,phase==0,s) ||', '!grid_oracle_state(oracle,phase==0,s) || iq4_shape_fallbacks ||')
    (out / name).write_text(text)
