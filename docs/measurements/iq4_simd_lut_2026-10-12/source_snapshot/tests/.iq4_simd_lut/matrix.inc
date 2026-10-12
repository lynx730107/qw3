int main(int argc,char **argv) {
    @autoreleasepool {
        if (argc!=3) return 2;
        lut_variant=atoi(argv[2]);if (lut_variant<1 || lut_variant>2) return 2;
        qw3_engine *engine=NULL;int rc=1,cases=0;
        const qw3_tensor *w[2]={NULL,NULL};
        qw3_engine_options opt={.model_path=argv[1],.backend=QW3_BACKEND_METAL};
        if (qw3_engine_open(&engine,&opt) || !iq4_simd_lut_compile()) goto done;
        for (int l=0;l<qw3_engine_layer_count(engine);l++) {
            const qw3_layer_weights *lw=&engine->weights.layer[l];
            if (lw->ffn_gate->type==23u && lw->ffn_up->type==23u) {
                w[0]=lw->ffn_gate;w[1]=lw->ffn_up;break;
            }
        }
        if (!w[0]) goto done;
        const uint32_t shapes[][2]={{256,1},{4352,33},{5120,4097},{5120,4111},{5120,17408},{17408,5120}};
        for (int shape=0;shape<6;shape++) for (int pattern=0;pattern<3;pattern++) {
            uint32_t ni=shapes[shape][0],no=shapes[shape][1];
            uint64_t offsets[2];id<MTLBuffer> wb[2];
            id<MTLBuffer> input=[g_device newBufferWithLength:(ni+8u)*4u options:MTLResourceStorageModeShared];
            if (!input || (uint64_t)ni*no>w[0]->elements || (uint64_t)ni*no>w[1]->elements) goto done;
            float *x=input.contents;
            for (uint32_t i=0;i<ni+8u;i++) x[i]=pattern==1 ? (i&1u ? -0.0f : 0.0f) :
                pattern==2 ? (float)((int)(i%7u)-3)*(i%5u==0u ? 0x1p8f : 0x1p-12f) : (float)((int)((i*37u)%101u)-50)/31.0f;
            NSData *before=[NSData dataWithBytes:input.contents length:input.length];
            for (int v=0;v<2;v++) {
                wb[v]=qw3_metal_model_view_for(w[v]->offset,(uint64_t)ni/256u*136u*no,&offsets[v]);
                if (!wb[v] || (offsets[v]&7u)) goto done;
            }
            for (int kind=0;kind<3;kind++) {
                const uint32_t n=kind==1 ? 2u*no : no;
                id<MTLBuffer> output[2];
                for (int v=0;v<2;v++) {
                    output[v]=[g_device newBufferWithLength:(n+12u)*4u options:MTLResourceStorageModeShared];
                    if (!output[v]) goto done;
                    float *y=output[v].contents;
                    for (uint32_t i=0;i<n+12u;i++) y[i]=kind==2 && i>=5u && i<no+5u ? (float)(i%17u)/13.0f : 123456.0f;
                    struct {uint32_t ni,no,rb;} args={ni,no,ni/256u*136u};const float scale=pattern==1 ? -0.0f : pattern==2 ? 0.75f : -0.25f;
                    id<MTLCommandBuffer> cb=[g_queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
                    [enc setComputePipelineState:iq4_simd_lut_pipeline[v ? lut_variant : 0][kind]];
                    [enc setBytes:&args length:sizeof(args) atIndex:0];
                    if (kind==2) [enc setBytes:&scale length:sizeof(scale) atIndex:1];
                    [enc setBuffer:wb[0] offset:offsets[0] atIndex:kind==2 ? 2 : 1];
                    if (kind==1) [enc setBuffer:wb[1] offset:offsets[1] atIndex:2];
                    [enc setBuffer:input offset:16 atIndex:kind ? 3 : 2];
                    [enc setBuffer:output[v] offset:20 atIndex:kind ? 4 : 3];
                    [enc setThreadgroupMemoryLength:128 atIndex:0];
                    NSUInteger rows=4u;
                    NSUInteger threads=64u;
                    [enc dispatchThreadgroups:MTLSizeMake((no+rows-1u)/rows+1u,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
                    [enc endEncoding];[cb commit];[cb waitUntilCompleted];
                    if (cb.status!=MTLCommandBufferStatusCompleted) goto done;
                }
                if (memcmp(before.bytes,input.contents,input.length) || memcmp(output[0].contents,output[1].contents,output[0].length)) {
                    const float *a=output[0].contents,*b=output[1].contents;
                    for (uint32_t i=0;i<n+12u;i++) if (memcmp(a+i,b+i,4)) {
                        uint32_t bits[2];memcpy(bits,a+i,4);memcpy(bits+1,b+i,4);
                        fprintf(stderr,"IQ4 SIMD LUT differs variant=%d ni=%u no=%u kind=%d pattern=%d index=%u reference=%.9g candidate=%.9g bits=%08x/%08x\n",lut_variant,ni,no,kind,pattern,i,a[i],b[i],bits[0],bits[1]);break;
                    }
                    fprintf(stderr,"IQ4 SIMD LUT bit mismatch shape=%d kind=%d pattern=%d\n",shape,kind,pattern);goto done;
                }
                const float *y=output[1].contents;
                for (uint32_t i=0;i<n+12u;i++) if (!isfinite(y[i]) ||
                    ((i<5u || i>=5u+n) && y[i]!=123456.0f)) goto done;
                cases++;printf("matrix_exact variant=%d n_in=%u n_out=%u kind=%d pattern=%d guards=exact input=unchanged\n",lut_variant,ni,no,kind,pattern);
            }
        }
        printf("PASS iq4_simd_luts=%d matrix_cases=%d finite_bit_exact=all\n",lut_variant,cases);rc=0;
done:
        if (rc) fprintf(stderr,"IQ4 SIMD LUT matrix failed after %d cases\n",cases);
        if (engine) qw3_engine_close(engine);return rc;
    }
}
