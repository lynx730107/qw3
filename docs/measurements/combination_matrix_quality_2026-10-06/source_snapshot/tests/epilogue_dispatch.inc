static id<MTLComputePipelineState> epilogue_iq3_pipeline,epilogue_iq4_pipeline;
static int epilogue_compile(void) {
    if (epilogue_iq3_pipeline && epilogue_iq4_pipeline) return 1;
    if (!combo_compile()) return 0;
    NSError *error=nil;
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/.epilogue_candidates/kernels.metal" encoding:NSUTF8StringEncoding error:&error];
    if (!extra) return 0;
    NSMutableString *source=[qw3_metal_kernel_source() mutableCopy]; [source appendString:extra];
    MTLCompileOptions *opt=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) opt.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    id<MTLLibrary> lib=[g_device newLibraryWithSource:source options:opt error:&error];
    if (lib) {
        epilogue_iq3_pipeline=[g_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"epilogue_iq3_s"] error:&error];
        epilogue_iq4_pipeline=[g_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"epilogue_iq4_xs"] error:&error];
    }
    if (!epilogue_iq3_pipeline || !epilogue_iq4_pipeline) fprintf(stderr,"epilogue compile: %s\n",error.localizedDescription.UTF8String);
    return epilogue_iq3_pipeline && epilogue_iq4_pipeline;
}
static int epilogue_down(QW3MetalSessionObj *obj,id<MTLCommandBuffer> cb,uint64_t offset,
    uint32_t type,uint32_t nt,uint32_t ni,uint32_t no,id<MTLBuffer> src,id<MTLBuffer> dst) {
    epilogue_saved=0;
    if (!(combo_mode&8) || nt%128u || (type!=21u && type!=23u) || !qw3_metal_staged128_enabled() ||
        (type==21u && !g_matmul_iq3_s_mpp_staged128_pipeline) ||
        (type==23u && !g_matmul_iq4_xs_mpp_staged128_pipeline))
        return qw3_metal_encode_dense_batch_matmul(obj,cb,offset,type,nt,ni,no,src,dst);
    if (!nt || !ni || !no || ni%256u || !epilogue_compile()) return 0;
    uint64_t row=(uint64_t)(ni/256u)*(type==21u ? 110ull : 136ull),inner=0;
    if (row>UINT32_MAX || src.length<(uint64_t)nt*ni*4ull || obj.prefillX0.length<(uint64_t)nt*no*4ull) return 0;
    id<MTLBuffer> weights=qw3_metal_model_view_for(offset,row*no,&inner);
    id<MTLBuffer> grid=type==21u ? qw3_metal_iq3s_expanded_kgrid_buffer() : nil;
    if (!weights || (type==21u && !grid)) return 0;
    struct {uint32_t ni,no,row,nt,io,is,oo,os;} a={ni,no,(uint32_t)row,nt,0,ni,0,no};
    id<MTLComputeCommandEncoder> enc=qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:type==21u ? epilogue_iq3_pipeline : epilogue_iq4_pipeline];
    [enc setBytes:&a length:sizeof(a) atIndex:0]; [enc setBuffer:weights offset:inner atIndex:1];
    [enc setBuffer:src offset:0 atIndex:2]; [enc setBuffer:obj.prefillX0 offset:0 atIndex:3];
    if (type==21u) [enc setBuffer:grid offset:0 atIndex:4];
    [enc setBuffer:obj.prefillX0 offset:0 atIndex:(type==21u ? 5 : 4)];
    [enc setThreadgroupMemoryLength:8192u atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(nt/128u,(no+63u)/64u,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    qw3_metal_end_compute_encoder(cb,enc);epilogue_saved=1;epilogue_calls++;return 1;
}
