int main(int argc,char **argv) {
    @autoreleasepool {
        if (argc!=3) return 2;
        shape_variant=atoi(argv[2]);if (shape_variant<1 || shape_variant>2) return 2;
        qw3_engine *engine=NULL;int rc=1,cases=0;const qw3_tensor *w=NULL;id<MTLBuffer> grid=nil;
        qw3_engine_options opt={.model_path=argv[1],.backend=QW3_BACKEND_METAL};
        if (qw3_engine_open(&engine,&opt) || !iq3_shape_compile()) goto done;
        grid=qw3_metal_iq3s_kgrid_buffer();if (!grid) goto done;
        for (int l=0;l<qw3_engine_layer_count(engine);l++) {
            const qw3_layer_weights *lw=&engine->weights.layer[l];
            if (lw->ffn_gate->type==21u) {w=lw->ffn_gate;break;}
        }
        if (!w) goto done;
        const uint32_t shapes[][2]={{256,1},{4352,33},{5120,4097},{5120,4111},{5120,17408},{17408,5120},{6144,5120},{8192,33}};
        for (int shape=0;shape<8;shape++) for (int pattern=0;pattern<3;pattern++) for (int alignment=0;alignment<2;alignment++) {
            uint32_t ni=shapes[shape][0],no=shapes[shape][1];uint64_t offset=0;
            if ((uint64_t)ni*no>w->elements) goto done;
            id<MTLBuffer> weight=qw3_metal_model_view_for(w->offset,(uint64_t)ni/256u*110u*no,&offset);
            id<MTLBuffer> input=[g_device newBufferWithLength:(ni+8u)*4u options:MTLResourceStorageModeShared];
            if (!weight || !input) goto done;
            float *x=input.contents;
            for (uint32_t i=0;i<ni+8u;i++) x[i]=pattern==1 ? (i&1u ? -0.0f : 0.0f) :
                pattern==2 ? (float)((int)(i%7u)-3)*(i%5u==0u ? 0x1p8f : 0x1p-12f) : (float)((int)((i*37u)%101u)-50)/31.0f;
            NSData *before=[NSData dataWithBytes:input.contents length:input.length];
            id<MTLBuffer> output[2];
            for (int v=0;v<2;v++) {
                output[v]=[g_device newBufferWithLength:(no+12u)*4u options:MTLResourceStorageModeShared];
                if (!output[v]) goto done;
                float *y=output[v].contents;for (uint32_t i=0;i<no+12u;i++) y[i]=123456.0f;
                struct {uint32_t ni,no,rb;} args={ni,no,ni/256u*110u};
                id<MTLCommandBuffer> cb=[g_queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
                scratch_select(v);
                [(id<QW3IQ3ShapeHooks>)enc iq3ShapeSetPipeline:iq3_shape_pipeline[0][0]];
                [(id<QW3IQ3ShapeHooks>)enc iq3ShapeSetBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:weight offset:offset atIndex:1];[enc setBuffer:input offset:(alignment ? 4 : 16) atIndex:2];
                [enc setBuffer:output[v] offset:20 atIndex:3];[enc setBuffer:grid offset:0 atIndex:4];
                [enc setThreadgroupMemoryLength:256 atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake((no+7u)/8u+1u,1,1) threadsPerThreadgroup:MTLSizeMake(64u,1,1)];
                [enc endEncoding];[cb commit];[cb waitUntilCompleted];
                if (cb.status!=MTLCommandBufferStatusCompleted) goto done;
            }
            if (memcmp(before.bytes,input.contents,input.length)) goto done;
            const float *a=output[0].contents,*b=output[1].contents;
            for (uint32_t i=0;i<no+12u;i++) {
                if (!isfinite(a[i]) || !isfinite(b[i]) || memcmp(a+i,b+i,4)) {
                    uint32_t bits[2];memcpy(bits,a+i,4);memcpy(bits+1,b+i,4);
                    fprintf(stderr,"IQ3 shape differs variant=%d ni=%u no=%u pattern=%d index=%u reference=%.9g candidate=%.9g bits=%08x/%08x\n",shape_variant,ni,no,pattern,i,a[i],b[i],bits[0],bits[1]);goto done;
                }
                if ((i<5u || i>=5u+no) && b[i]!=123456.0f) goto done;
            }
            cases++;printf("matrix_exact variant=%d n_in=%u n_out=%u pattern=%d input_offset=%d guards=exact input=unchanged\n",shape_variant,ni,no,pattern,alignment ? 4 : 16);
        }
        if (iq3_shape_calls!=42u || iq3_shape_fallbacks!=6u) goto done;
        printf("PASS iq3_shape=%d matrix_cases=%d finite_bit_exact=all\n",shape_variant,cases);rc=0;
done:
        if (rc) fprintf(stderr,"IQ3 shape matrix failed after %d cases\n",cases);
        if (engine) qw3_engine_close(engine);return rc;
    }
}
