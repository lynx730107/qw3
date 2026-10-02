/* Private bit-preserving GPU copy probe, substituting only three staging calls. */
static int prefill_copy_enabled;
static id<MTLComputePipelineState> prefill_copy_pipeline;
static NSString *const prefill_copy_source=@"\n"
    "struct prefill_copy_args { uint n, rows, si, so, oi, oo; };\n"
    "kernel void prefill_copy_rows(constant prefill_copy_args &a [[buffer(0)]],\n"
    "    device const uint *src [[buffer(1)]], device uint *dst [[buffer(2)]],\n"
    "    uint2 p [[thread_position_in_grid]]) {\n"
    "    if (p.x<a.n && p.y<a.rows) dst[ulong(p.y)*a.so+a.oo+p.x]=src[ulong(p.y)*a.si+a.oi+p.x];\n"
    "}\n";
static int prefill_encode_copy(id<MTLCommandBuffer> cb,id<MTLBuffer> src,id<MTLBuffer> dst,
    uint32_t rows,uint32_t n,uint32_t si,uint32_t so,uint32_t oi,uint32_t oo) {
    if (!cb || !src || !dst || !rows || !n || oi>si || n>si-oi || oo>so || n>so-oo ||
        ((uint64_t)(rows-1)*si+oi+n)*4>src.length ||
        ((uint64_t)(rows-1)*so+oo+n)*4>dst.length) return 0;
    struct {uint32_t n,rows,si,so,oi,oo;} args={n,rows,si,so,oi,oo};
    id<MTLComputeCommandEncoder> enc=qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:prefill_copy_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:0 atIndex:1];[enc setBuffer:dst offset:0 atIndex:2];
    NSUInteger threads=MIN((NSUInteger)256,prefill_copy_pipeline.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake(n,rows,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
    qw3_metal_end_compute_encoder(cb,enc);
    return 1;
}
static int prefill_copy_selftest(int production) {
    const uint32_t shapes[][2]={{1,1},{3,7},{31,33},{128,4097},{129,17}};
    for (int shape=0;shape<5;shape++) {
        const uint32_t rows=shapes[shape][0],n=shapes[shape][1],si=n+7,so=n+11,oi=3,oo=5;
        const size_t ni=(size_t)rows*si+8,no=(size_t)rows*so+8;
        id<MTLBuffer> src=[g_device newBufferWithLength:ni*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> dst=[g_device newBufferWithLength:no*4 options:MTLResourceStorageModeShared];
        if (!src || !dst) return 0;
        uint32_t *input=src.contents,*output=dst.contents;
        for (size_t i=0;i<ni;i++) input[i]=(uint32_t)(i*2654435761u)^0xffc00000u;
        for (size_t i=0;i<no;i++) output[i]=0xdeadbeefu;
        /* Explicit negative zero and quiet NaN payloads must survive bitwise. */
        input[oi]=0x80000000u;
        if (n>1) input[oi+1]=0x7fc12345u;
        int owned=0;id<MTLCommandBuffer> cb=qw3_metal_command_buffer(&owned);
        const int copied=production ?
            qw3_metal_encode_copy_rows(cb,src,dst,rows,n,si,so,oi,oo) :
            prefill_encode_copy(cb,src,dst,rows,n,si,so,oi,oo);
        if (!copied ||
            !qw3_metal_finish_command_buffer(cb,owned,"copy tail/guard selftest")) return 0;
        for (size_t i=0;i<no;i++) {
            const size_t row=i/so,col=i%so;
            const uint32_t expected=row<rows && col>=oo && col<oo+n ?
                input[row*si+oi+col-oo] : 0xdeadbeefu;
            if (output[i]!=expected) {
                fprintf(stderr,"copy selftest failed shape=%d index=%zu\n",shape,i);return 0;
            }
        }
        printf("copy_quant production=%d rows=%u columns=%u offsets=nonzero guards=ok bits_exact=1\n",production,rows,n);
    }
    return 1;
}
int qw3_metal_session_stage_dense_gqa_projections(qw3_metal_session *s,uint32_t nt,
    uint32_t qg,uint32_t kv,uint32_t qo,uint32_t ko,uint32_t vo,uint32_t stride) {
    if (!prefill_copy_enabled) return prefill_blit_gqa(s,nt,qg,kv,qo,ko,vo,stride);
    if (!s || !s->obj || !nt || !stride) return 0;
    QW3MetalSessionObj *obj=(__bridge QW3MetalSessionObj *)s->obj;
    if (!qw3_metal_session_ensure_prefill_scratch(obj,nt,(uint64_t)nt*stride*4)) return 0;
    int owned=0;id<MTLCommandBuffer> cb=qw3_metal_command_buffer(&owned);
    return prefill_encode_copy(cb,obj.denseBatchGqaQ,obj.prefillScratch,nt,qg,qg,stride,0,qo) &&
           prefill_encode_copy(cb,obj.denseBatchGqaK,obj.prefillScratch,nt,kv,kv,stride,0,ko) &&
           prefill_encode_copy(cb,obj.denseBatchGqaV,obj.prefillScratch,nt,kv,kv,stride,0,vo) &&
           qw3_metal_finish_command_buffer(cb,owned,"GPU GQA staging probe");
}
int qw3_metal_session_stage_dense_deltanet_projections(qw3_metal_session *s,uint32_t nt,
    uint32_t qkv,uint32_t z,uint32_t qo,uint32_t zo,uint32_t stride) {
    if (!prefill_copy_enabled) return prefill_blit_dn(s,nt,qkv,z,qo,zo,stride);
    if (!s || !s->obj || !nt || !stride) return 0;
    QW3MetalSessionObj *obj=(__bridge QW3MetalSessionObj *)s->obj;
    if (!qw3_metal_session_ensure_prefill_scratch(obj,nt,(uint64_t)nt*stride*4)) return 0;
    int owned=0;id<MTLCommandBuffer> cb=qw3_metal_command_buffer(&owned);
    return prefill_encode_copy(cb,obj.denseBatchQkv,obj.prefillScratch,nt,qkv,qkv,stride,0,qo) &&
           prefill_encode_copy(cb,obj.denseBatchZ,obj.prefillScratch,nt,z,z,stride,0,zo) &&
           qw3_metal_finish_command_buffer(cb,owned,"GPU DeltaNet staging probe");
}
int qw3_metal_session_copy_batch_scratch_to_x1(qw3_metal_session *s,uint32_t nt,
    uint32_t n,uint32_t offset,uint32_t stride) {
    if (!prefill_copy_enabled) return prefill_blit_compact(s,nt,n,offset,stride);
    if (!s || !s->obj || !nt || !stride) return 0;
    QW3MetalSessionObj *obj=(__bridge QW3MetalSessionObj *)s->obj;
    if (!qw3_metal_session_ensure_prefill_buffers(obj,nt,0,(uint64_t)nt*n*4)) return 0;
    int owned=0;id<MTLCommandBuffer> cb=qw3_metal_command_buffer(&owned);
    return prefill_encode_copy(cb,obj.prefillScratch,obj.prefillX1,nt,n,stride,n,offset,0) &&
           qw3_metal_finish_command_buffer(cb,owned,"GPU compaction probe");
}
