#define qw3_metal_session_matvec_iq3_s_pair_x1_to_scratch groups_original_pair
#include "../qw3_metal.m"
#undef qw3_metal_session_matvec_iq3_s_pair_x1_to_scratch
static uint32_t groups_selected=2,groups_width=4;
static uint64_t groups_calls;
static id<MTLComputePipelineState> groups_pipeline[2];
#include ".iq3_groups/wrapper.inc"
#define qw3_metal_env_n_gpu_layers groups_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled groups_core_llama_split_enabled
#define qw3_metal_llama_split_start groups_core_llama_split_start
#include "../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
static int groups_compile(void) {
    if (groups_pipeline[0] && groups_pipeline[1]) return 1;
    if (!qw3_metal_compile_kernels()) return 0;
    NSString *all=qw3_metal_kernel_source();
    NSRange start=[all rangeOfString:@"kernel void qw3_matvec_iq3_s_pair_fast("];
    if (start.location==NSNotFound) return 0;
    NSRange end=[all rangeOfString:@"\n}\n" options:0
        range:NSMakeRange(start.location,all.length-start.location)];
    if (end.location==NSNotFound) return 0;
    NSString *body=[all substringWithRange:NSMakeRange(start.location,end.location+end.length-start.location)];
    for (int i=0;i<2;i++) {
        uint32_t groups=i ? 8u : 4u;
        NSMutableString *candidate=[body mutableCopy];
        NSString *name=[NSString stringWithFormat:@"private_iq3_s_groups%u",groups];
        NSArray *old=@[@"qw3_matvec_iq3_s_pair_fast",@"i += 64u",@"const uint nsg = 2u;"];
        NSArray *replacement=@[name,[NSString stringWithFormat:@"i += %uu",groups*32u],
            [NSString stringWithFormat:@"const uint nsg = %uu;",groups]];
        for (int j=0;j<3;j++) if ([candidate replaceOccurrencesOfString:old[j] withString:replacement[j]
                options:0 range:NSMakeRange(0,candidate.length)]!=1) return 0;
        NSMutableString *source=[all mutableCopy];[source appendString:candidate];
        MTLCompileOptions *options=[MTLCompileOptions new];
        if (g_metal4_tensor_api_enabled) options.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
        NSError *error=nil;
        id<MTLLibrary> lib=[g_device newLibraryWithSource:source options:options error:&error];
        id<MTLFunction> fn=[lib newFunctionWithName:name];
        groups_pipeline[i]=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!groups_pipeline[i] || groups_pipeline[i].maxTotalThreadsPerThreadgroup<groups*32u) {
            fprintf(stderr,"IQ3 groups compile: %s\n",error.localizedDescription.UTF8String);return 0;
        }
    }
    return 1;
}
static void scratch_select(int candidate) {
    setenv("QW3_METAL_DENSE_PREFILL_FAST","1",1);
    groups_selected=candidate ? groups_width : 2u;
}
