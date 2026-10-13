"""Private two-row Q4_K workgroups; keep each row's FP32 reductions."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / 'tests/.q4_rows'
out.mkdir(exist_ok=True)
source = (root / 'metal/qw3_core_linear.metal').read_text()
name = 'qw3_matvec_q4_k'
a = source.index('kernel void ' + name + '(')
original = source[a:source.index('\n}\n', a)+3]
changes = {
    name: 'private_q4_rows_fast',
    'uint row [[threadgroup_position_in_grid]]': 'uint group [[threadgroup_position_in_grid]]',
    '    if (row >= args.n_out) return;':
        '    if (group * 2u >= args.n_out) return;\n    uint row = group * 2u + uint(simd_idx);',
    '    for (uint b = tid; b < args.n_in / 256u; b += nt) {':
        '    for (uint b = uint(lane); row < args.n_out && b < args.n_in / 256u; b += nt) {',
    '    sum = lane < ((uint(nt) + 31u) / 32u) ? sh[lane] : 0.0f;':
        '    sum = lane == 0u ? sh[simd_idx] : 0.0f;',
    '    if (tid == 0) out[row] = sum;':
        '    if (lane == 0u && row < args.n_out) out[row] = sum;',
}
body = original
for before, after in changes.items():
    assert body.count(before) == 1
    body = body.replace(before, after)
restored = body
for before, after in reversed(list(changes.items())):
    restored = restored.replace(after, before)
assert restored == original
(out / 'kernels.metal').write_text(body)

base = root / 'tests/.gqa_barrier'
s = (base / 'hooks.inc').read_text()
s = (s.replace('gqa_barrier', 'q4_rows').replace('GQABarrier', 'Q4Rows')
       .replace('gqaBarrier', 'q4Rows').replace('barrier_variant', 'rows_variant'))
s = s.replace('Three IQ4_XS pipelines: K-loop unroll hints only; native arithmetic and buffers.',
              'Two Q4_K rows per workgroup; native per-row arithmetic and data buffers.')
s = s.replace('q4_rows_pipeline[3]', 'q4_rows_pipeline[2]', 1)
s = s.replace('static int rows_variant=1,q4_rows_mode;', 'static int q4_rows_mode;')
s = s.replace('static _Thread_local int q4_rows_eligible;',
              'static _Thread_local int q4_rows_seen,q4_rows_eligible;\nstatic _Thread_local uint32_t q4_rows_out;')
s = s.replace('@protocol QW3Q4RowsHooks\n', '@protocol QW3Q4RowsHooks\n- (void)q4RowsDispatch:(MTLSize)groups threadsPerThreadgroup:(MTLSize)threads;\n')
a = s.index('- (void)q4RowsSetPipeline:', s.index('@implementation'))
z = s.index('\n@end', a)
s = s[:a] + '''- (void)q4RowsSetPipeline:(id<MTLComputePipelineState>)pipeline {
    q4_rows_encoder=(__bridge void *)self;
    q4_rows_seen=(pipeline==q4_rows_pipeline[0]);
    q4_rows_eligible=0;q4_rows_out=0;
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
- (void)q4RowsSetBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index {
    if (q4_rows_mode && q4_rows_encoder==(__bridge void *)self &&
        q4_rows_seen && index==0 && length==3u*sizeof(uint32_t)) {
        const uint32_t *a=bytes;
        q4_rows_out=a[1];
        q4_rows_eligible=a[0]>0u && a[0]%256u==0u && a[0]<=8192u && a[1]>0u;
    }
    [(id<MTLComputeCommandEncoder>)self setBytes:bytes length:length atIndex:index];
}
- (void)q4RowsDispatch:(MTLSize)groups threadsPerThreadgroup:(MTLSize)threads {
    if (q4_rows_mode && q4_rows_encoder==(__bridge void *)self && q4_rows_seen) {
        if (q4_rows_eligible && groups.width>=q4_rows_out && groups.height==1u && groups.depth==1u &&
            threads.width==64u && threads.height==1u && threads.depth==1u) {
            [(id<MTLComputeCommandEncoder>)self setComputePipelineState:q4_rows_pipeline[1]];
            groups.width=(groups.width+1u)/2u;
            q4_rows_calls++;
        } else q4_rows_fallbacks++;
        q4_rows_seen=0;
    }
    [(id<MTLComputeCommandEncoder>)self dispatchThreadgroups:groups threadsPerThreadgroup:threads];
}
''' + s[z:]
s = s.replace('#define setBytes q4RowsSetBytes', '#define dispatchThreadgroups q4RowsDispatch\n#define setBytes q4RowsSetBytes')
s = s.replace('#undef setBytes', '#undef dispatchThreadgroups\n#undef setBytes')
a = s.index('static int q4_rows_compile(void) {')
s = s[:a] + '''static int q4_rows_compile(void) {
    if (q4_rows_pipeline[1]) return 1;
    if (!qw3_metal_init() || !qw3_metal_compile_kernels()) return 0;
    q4_rows_pipeline[0]=g_matvec_q4_k_pipeline;
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) options.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/.q4_rows/kernels.metal" encoding:NSUTF8StringEncoding error:&error];
    NSString *original=qw3_metal_full_kernel_source();
    NSString *source=extra && original ? [original stringByAppendingString:extra] : nil;
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    id<MTLFunction> fn=[lib newFunctionWithName:@"private_q4_rows_fast"];
    id<MTLComputePipelineState> p=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
    if (!p || p.threadExecutionWidth!=32u || p.maxTotalThreadsPerThreadgroup<64u ||
        p.staticThreadgroupMemoryLength!=q4_rows_pipeline[0].staticThreadgroupMemoryLength) {
        fprintf(stderr,"Q4 rows compile: %s\\n",error.localizedDescription.UTF8String);return 0;
    }
    q4_rows_pipeline[1]=p;
    for (int v=0;v<=1;v++) fprintf(stderr,"q4_rows_metadata variant=%d width=%lu max_threads=%lu static_threadgroup_bytes=%lu dynamic_threadgroup_bytes=128 threads=64\\n",v,(unsigned long)q4_rows_pipeline[v].threadExecutionWidth,(unsigned long)q4_rows_pipeline[v].maxTotalThreadsPerThreadgroup,(unsigned long)q4_rows_pipeline[v].staticThreadgroupMemoryLength);
    return 1;
}
static void scratch_select(int mode) {
    q4_rows_mode=mode!=0;q4_rows_seen=0;q4_rows_eligible=0;q4_rows_encoder=NULL;
}
static void q4_rows_print_counts(void) {
    fprintf(stderr,"q4_rows_dispatch_counts calls=%llu fallbacks=%llu\\n",(unsigned long long)q4_rows_calls,(unsigned long long)q4_rows_fallbacks);
}
'''
(out / 'hooks.inc').write_text(s)
for filename in ('model.inc', 'native.inc'):
    text = (root / 'tests/.q4_narrow' / filename).read_text()
    text = text.replace('q4_narrow', 'q4_rows').replace('Q4 narrow', 'Q4 rows').replace('Q4_NARROW', 'Q4_ROWS')
    text = text.replace('baseline/narrow32', 'baseline/two-row64')
    (out / filename).write_text(text)
