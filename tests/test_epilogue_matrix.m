#include ".epilogue_candidates/candidates.inc"

/* Compare the native GEMM followed by its native residual kernel against the
 * epilogue, including partial output tiles and nonzero in-place residuals. */
static int epilogue_matrix_case(QW3MetalSessionObj *obj,const qw3_tensor *w,
                                uint32_t nt,uint32_t no) {
    const uint32_t ni=(uint32_t)w->dim[0],io=3,is=ni+9,os=no+13,guard=8;
    const size_t in_bytes=(size_t)nt*is*4,out_bytes=((size_t)nt*os+2*guard)*4;
    id<MTLBuffer> input=[g_device newBufferWithLength:in_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> compact=[g_device newBufferWithLength:(size_t)nt*ni*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> output=[g_device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> reference=[g_device newBufferWithLength:(size_t)nt*no*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> product=[g_device newBufferWithLength:(size_t)nt*no*4 options:MTLResourceStorageModeShared];
    if (!input || !compact || !output || !reference || !product) return 0;
    memset(input.contents,0x5a,in_bytes);memset(output.contents,0x5a,out_bytes);
    float *x=input.contents,*c=compact.contents,*y=output.contents,*r=reference.contents;
    uint32_t rng=13;
    for (uint32_t t=0;t<nt;t++) {
        for (uint32_t i=0;i<ni;i++) {
            rng=rng*1664525u+1013904223u;
            c[(size_t)t*ni+i]=x[(size_t)t*is+io+i]=(float)((int)(rng>>16u)-32768)/32768.0f;
        }
        for (uint32_t i=0;i<no;i++) {
            rng=rng*1664525u+1013904223u;
            r[(size_t)t*no+i]=y[guard+(size_t)t*os+i]=(float)((int)(rng>>16u)-32768)/8192.0f;
        }
    }
    NSData *before=[NSData dataWithBytes:input.contents length:in_bytes];
    if (!qw3_metal_dense_batch_prepare_map(obj,nt)) return 0;
    id<MTLCommandBuffer> cb=[g_queue commandBuffer];
    if (!qw3_metal_encode_dense_batch_matmul(obj,cb,w->offset,w->type,nt,ni,no,compact,product)) return 0;
    struct {uint32_t n;float scale;} unary={nt*no,1.0f};
    id<MTLComputeCommandEncoder> enc=qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_scale_x1_add_x0_pipeline];
    [enc setBytes:&unary length:sizeof(unary) atIndex:0];
    [enc setBuffer:reference offset:0 atIndex:1];[enc setBuffer:product offset:0 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(unary.n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    qw3_metal_end_compute_encoder(cb,enc);
    uint64_t row=(uint64_t)(ni/256u)*(w->type==21u ? 110ull : 136ull),inner=0;
    id<MTLBuffer> weights=qw3_metal_model_view_for(w->offset,row*no,&inner);
    id<MTLBuffer> grid=w->type==21u ? qw3_metal_iq3s_expanded_kgrid_buffer() : nil;
    if (!weights || (w->type==21u && !grid)) return 0;
    struct {uint32_t ni,no,row,nt,io,is,oo,os;} a={ni,no,(uint32_t)row,nt,io,is,0,os};
    enc=qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:w->type==21u ? epilogue_iq3_pipeline : epilogue_iq4_pipeline];
    [enc setBytes:&a length:sizeof(a) atIndex:0];[enc setBuffer:weights offset:inner atIndex:1];
    [enc setBuffer:input offset:0 atIndex:2];[enc setBuffer:output offset:guard*4 atIndex:3];
    if (w->type==21u) [enc setBuffer:grid offset:0 atIndex:4];
    [enc setBuffer:output offset:guard*4 atIndex:w->type==21u ? 5 : 4];
    [enc setThreadgroupMemoryLength:8192 atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(nt/128u,(no+63u)/64u,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    qw3_metal_end_compute_encoder(cb,enc);[cb commit];[cb waitUntilCompleted];
    if (cb.status!=MTLCommandBufferStatusCompleted || memcmp(before.bytes,input.contents,in_bytes)) return 0;
    for (size_t j=0;j<out_bytes/4;j++) {
        const size_t p=j>=guard ? j-guard : SIZE_MAX;
        if (p<(size_t)nt*os && p%os<no) {
            const float *want=r+(p/os)*no+p%os;
            if (!isfinite(*want) || !isfinite(y[j]) || memcmp(want,y+j,4)) {
                fprintf(stderr,"epilogue matrix mismatch type=%u tokens=%u rows=%u index=%zu reference=%.9g candidate=%.9g\n",w->type,nt,no,j,*want,y[j]);return 0;
            }
        } else {
            uint32_t bits;memcpy(&bits,y+j,4);
            if (bits!=0x5a5a5a5au) {fprintf(stderr,"epilogue guard overwritten index=%zu\n",j);return 0;}
        }
    }
    printf("epilogue_matrix_exact type=%u tokens=%u n_in=%u rows=%u residual=nonzero guards=exact input=unchanged\n",w->type,nt,ni,no);fflush(stdout);return 1;
}

int main(int argc,char **argv) {
    @autoreleasepool {
        if (argc!=2) return 2;
        qw3_engine *engine=NULL;qw3_session *session=NULL;int rc=1,cases=0;
        QW3MetalSessionObj *obj=nil;
        qw3_engine_options options={.model_path=argv[1],.backend=QW3_BACKEND_METAL,.n_threads=4};
        scratch_select(0);
        if (qw3_engine_open(&engine,&options) || !qw3_metal_compile_kernels() ||
            !epilogue_compile() || qw3_session_create(&session,engine,512)) goto done;
        obj=(__bridge QW3MetalSessionObj *)session->metal->obj;
        for (uint32_t type=21;type<=23;type+=2) {
            const qw3_tensor *weight=NULL;
            for (int l=0;l<qw3_engine_layer_count(engine);l++) {
                const qw3_tensor *w=engine->weights.layer[l].ffn_down;
                if (w && w->type==type) {weight=w;break;}
            }
            if (!weight) goto done;
            const uint32_t tokens[]={128,256},rows[]={1,33,65,(uint32_t)weight->dim[1]};
            for (int n=0;n<2;n++) for (int k=0;k<4;k++) {
                if (!epilogue_matrix_case(obj,weight,tokens[n],rows[k])) goto done;
                cases++;
            }
        }
        printf("PASS epilogue_matrix cases=%d\n",cases);rc=0;
done:
        if (rc) fprintf(stderr,"epilogue matrix quality failed after %d cases\n",cases);
        if (session) qw3_session_free(session);if (engine) qw3_engine_close(engine);return rc;
    }
}
