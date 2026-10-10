static id<MTLComputePipelineState> sum8_head_pipeline,sum8_prepare_pipeline;
static id<MTLBuffer> sum8_buffer;
static unsigned long long sum8_head_calls;
static int sum8_compile(void) {
    if (sum8_head_pipeline && sum8_prepare_pipeline) return 1;
    if (!combo_compile()) return 0;
    NSError *error=nil;
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/.sum8_candidates/kernels.metal" encoding:NSUTF8StringEncoding error:&error];
    if (!extra) return 0;
    NSMutableString *source=[qw3_metal_kernel_source() mutableCopy]; [source appendString:extra];
    MTLCompileOptions *opt=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) opt.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    id<MTLLibrary> lib=[g_device newLibraryWithSource:source options:opt error:&error];
    if (lib) {
        sum8_head_pipeline=[g_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"sum8_q5_head"] error:&error];
        sum8_prepare_pipeline=[g_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"sum8_prepare"] error:&error];
    }
    if (!sum8_head_pipeline || !sum8_prepare_pipeline) fprintf(stderr,"sum8 compile: %s\n",error.localizedDescription.UTF8String);
    return sum8_head_pipeline && sum8_prepare_pipeline;
}
#include ".sum8_candidates/head.inc"
