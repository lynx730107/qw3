"""Private IQ3_S buffers with original byte layout and compiled functions."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / 'tests/.iq3_private'
out.mkdir(exist_ok=True)
base = root / 'tests/.iq3_layout'

def rename(text):
    return text.replace('iq3_layout', 'iq3_private').replace('IQ3Layout', 'IQ3Private').replace('iq3Layout', 'iq3Private').replace('IQ3 layout', 'IQ3 private')

hooks = rename((base / 'hooks.inc').read_text()).replace('iq3_private_pipeline[3][2]', 'iq3_private_pipeline[2][2]').replace('variant<=2', 'variant<=1')
first = hooks.index('static int iq3_private_compile(')
last = hooks.index('static id<MTLBuffer> iq3_private_pack(', first)
hooks = hooks[:first] + '''static int iq3_private_compile(void) {
    if (iq3_private_pipeline[1][1]) return 1;
    if (!qw3_metal_compile_kernels()) return 0;
    iq3_private_pipeline[0][0]=g_matvec_iq3_s_pipeline;
    iq3_private_pipeline[0][1]=g_matvec_iq3_s_pair_pipeline;
    const char *names[]={"qw3_matvec_iq3_s_fast","qw3_matvec_iq3_s_pair_fast"};
    NSError *error=nil;
    for (int kind=0;kind<2;kind++) {
        /* Same already-compiled library and function as the baseline. */
        id<MTLFunction> fn=[g_library newFunctionWithName:[NSString stringWithUTF8String:names[kind]]];
        iq3_private_pipeline[1][kind]=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!iq3_private_pipeline[1][kind] || iq3_private_pipeline[1][kind]==iq3_private_pipeline[0][kind] || iq3_private_pipeline[1][kind].maxTotalThreadsPerThreadgroup<64) {
            fprintf(stderr,"IQ3 private pipeline kind=%d: %s\\n",kind,error.localizedDescription.UTF8String);return 0;
        }
    }
    return 1;
}
''' + hooks[last:]
first = hooks.index('static id<MTLBuffer> iq3_private_pack(')
last = hooks.index('static int iq3_private_prepare(', first)
hooks = hooks[:first] + '''static id<MTLBuffer> iq3_private_pack(id<MTLBuffer> native,uint64_t offset,uint32_t ni,uint32_t no) {
    @autoreleasepool {
        if (!ni || ni%256u || !no) return nil;
        const uint64_t bytes=(uint64_t)ni/256u*no*110u;
        if (!native || offset>native.length || bytes>native.length-offset || bytes>NSUIntegerMax-80u) return nil;
        const NSUInteger length=(NSUInteger)((bytes+64u+15u)&~15ull);
        id<MTLBuffer> staging=[g_device newBufferWithLength:length options:MTLResourceStorageModeShared];
        id<MTLBuffer> private=[g_device newBufferWithLength:length options:MTLResourceStorageModePrivate];
        if (!staging || !private) return nil;
        const unsigned char *src=(const unsigned char *)native.contents+offset;
        unsigned char *dst=staging.contents;memset(dst,0xa5,length);memcpy(dst+32u,src,bytes);
        id<MTLCommandBuffer> cb=[g_queue commandBuffer];id<MTLBlitCommandEncoder> blit=[cb blitCommandEncoder];
        [blit copyFromBuffer:staging sourceOffset:0 toBuffer:private destinationOffset:0 size:length];
        [blit endEncoding];[cb commit];[cb waitUntilCompleted];
        if (cb.status!=MTLCommandBufferStatusCompleted) return nil;
        /* Verify actual GPU copy bytes through a GPU-to-CPU readback. */
        memset(dst,0,length);
        cb=[g_queue commandBuffer];blit=[cb blitCommandEncoder];
        [blit copyFromBuffer:private sourceOffset:0 toBuffer:staging destinationOffset:0 size:length];
        [blit endEncoding];[cb commit];[cb waitUntilCompleted];
        if (cb.status!=MTLCommandBufferStatusCompleted || memcmp(dst+32u,src,bytes)) return nil;
        for (NSUInteger i=0;i<32u;i++) if (dst[i]!=0xa5) return nil;
        for (NSUInteger i=32u+bytes;i<length;i++) if (dst[i]!=0xa5) return nil;
        if (private.storageMode!=MTLStorageModePrivate) return nil;
        return private;
    }
}
static int iq3_private_guards(id<MTLBuffer> buffer) {
    @autoreleasepool {
        if (!buffer || buffer.length<64u) return 0;
        id<MTLBuffer> result=[g_device newBufferWithLength:64u options:MTLResourceStorageModeShared];
        if (!result) return 0;
        id<MTLCommandBuffer> cb=[g_queue commandBuffer];id<MTLBlitCommandEncoder> blit=[cb blitCommandEncoder];
        [blit copyFromBuffer:buffer sourceOffset:0 toBuffer:result destinationOffset:0 size:32u];
        [blit copyFromBuffer:buffer sourceOffset:buffer.length-32u toBuffer:result destinationOffset:32u size:32u];
        [blit endEncoding];[cb commit];[cb waitUntilCompleted];
        if (cb.status!=MTLCommandBufferStatusCompleted) return 0;
        const unsigned char *p=result.contents;
        for (unsigned i=0;i<64u;i++) if (p[i]!=0xa5) return 0;
        return 1;
    }
}
''' + hooks[last:]
hooks = hooks.replace('const double start=[NSDate timeIntervalSinceReferenceDate];', 'const double start=[NSDate timeIntervalSinceReferenceDate];\n    const NSUInteger before=g_device.currentAllocatedSize;')
anchor = '    return count>0;'
assert hooks.count(anchor) == 1
hooks = hooks.replace(anchor, '    fprintf(stderr,"IQ3 private GPU allocated before=%llu after=%llu\\n",(unsigned long long)before,(unsigned long long)g_device.currentAllocatedSize);\n' + anchor)
(out / 'hooks.inc').write_text(hooks)
for name in ['quality.inc', 'bench.inc']:
    text = rename((base / name).read_text()).replace('layout_variant>2', 'layout_variant>1')
    (out / name).write_text(text)
