#include ".sum8_candidates/candidates.inc"

static int sum8_matrix_case(const qw3_tensor *w,uint32_t ni,uint32_t no) {
    const uint32_t guard=8,ns=ni/8u,row=(uint32_t)(w->dim[0]/256u)*176u;
    id<MTLBuffer> input=[g_device newBufferWithLength:(size_t)ni*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> partial=[g_device newBufferWithLength:(size_t)(ns+2*guard)*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> candidate=[g_device newBufferWithLength:(size_t)(no+2*guard)*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> reference=[g_device newBufferWithLength:(size_t)no*4 options:MTLResourceStorageModeShared];
    if (!input || !partial || !candidate || !reference) return 0;
    memset(partial.contents,0x5a,partial.length);memset(candidate.contents,0x5a,candidate.length);
    float *x=input.contents;uint32_t rng=1;
    for (uint32_t i=0;i<ni;i++) {rng=rng*1664525u+1013904223u;x[i]=(float)((int)(rng>>16u)-32768)/8192.0f;}
    NSData *before=[NSData dataWithBytes:input.contents length:input.length];
    uint64_t inner=0;id<MTLBuffer> weights=qw3_metal_model_view_for(w->offset,(uint64_t)row*no,&inner);
    if (!weights) return 0;
    id<MTLCommandBuffer> cb=[g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc=qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:sum8_prepare_pipeline];[enc setBytes:&ni length:sizeof(ni) atIndex:0];
    [enc setBuffer:input offset:0 atIndex:1];[enc setBuffer:partial offset:guard*4 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(ns,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
    qw3_metal_end_compute_encoder(cb,enc);
    struct {uint32_t ni,no,row;} a={ni,no,row};
    for (int v=0;v<2;v++) {
        enc=qw3_metal_compute_encoder(cb);
        [enc setComputePipelineState:v ? sum8_head_pipeline : g_matvec_q5_k_pipeline];
        [enc setBytes:&a length:sizeof(a) atIndex:0];[enc setBuffer:weights offset:inner atIndex:1];
        [enc setBuffer:input offset:0 atIndex:2];[enc setBuffer:v ? candidate : reference offset:v ? guard*4 : 0 atIndex:3];
        if (v) [enc setBuffer:partial offset:guard*4 atIndex:4];
        [enc setThreadgroupMemoryLength:64*sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((no+1u)/2u,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        qw3_metal_end_compute_encoder(cb,enc);
    }
    [cb commit];[cb waitUntilCompleted];
    if (cb.status!=MTLCommandBufferStatusCompleted || memcmp(before.bytes,input.contents,input.length)) return 0;
    const float *p=partial.contents,*y=candidate.contents,*r=reference.contents;
    for (uint32_t i=0;i<ns;i++) {
        volatile float sum=0.0f;
        for (uint32_t l=0;l<8;l++) sum=sum+x[i*8u+l];
        float want=sum;
        if (!isfinite(p[guard+i]) || memcmp(&want,p+guard+i,4)) {fprintf(stderr,"sum8 partial mismatch index=%u\n",i);return 0;}
    }
    for (uint32_t i=0;i<no;i++) if (!isfinite(r[i]) || !isfinite(y[guard+i]) || memcmp(r+i,y+guard+i,4)) {
        fprintf(stderr,"sum8 row mismatch n_in=%u rows=%u index=%u\n",ni,no,i);return 0;
    }
    const uint32_t *pb=partial.contents,*yb=candidate.contents;
    for (uint32_t i=0;i<guard;i++) if (pb[i]!=0x5a5a5a5au || pb[guard+ns+i]!=0x5a5a5a5au ||
        yb[i]!=0x5a5a5a5au || yb[guard+no+i]!=0x5a5a5a5au) return 0;
    printf("sum8_matrix_exact n_in=%u rows=%u partials=%u bit_exact guards=exact input=unchanged\n",ni,no,ns);fflush(stdout);return 1;
}

int main(int argc,char **argv) {
    @autoreleasepool {
        if (argc!=2) return 2;
        qw3_engine *engine=NULL;int rc=1,cases=0;
        qw3_engine_options options={.model_path=argv[1],.backend=QW3_BACKEND_METAL,.n_threads=4};
        scratch_select(0);
        if (qw3_engine_open(&engine,&options) || !qw3_metal_compile_kernels() || !sum8_compile()) goto done;
        const qw3_tensor *w=engine->weights.output;
        if (!w || w->type!=13u) goto done;
        const uint32_t inputs[]={256,(uint32_t)w->dim[0]},rows[]={1,3,65,(uint32_t)w->dim[1]};
        for (int n=0;n<2;n++) for (int k=0;k<4;k++) {
            if (!sum8_matrix_case(w,inputs[n],rows[k])) goto done;
            cases++;
        }
        printf("PASS sum8_matrix cases=%d\n",cases);rc=0;
done:
        if (rc) fprintf(stderr,"sum8 matrix quality failed after %d cases\n",cases);
        if (engine) qw3_engine_close(engine);return rc;
    }
}
