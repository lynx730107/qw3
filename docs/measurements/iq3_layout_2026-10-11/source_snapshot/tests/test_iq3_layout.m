#define main iq3_layout_full_quality_main
#include ".iq3_layout/quality.inc"
#undef main
int main(int argc,char **argv) {
    if (argc==4) return iq3_layout_full_quality_main(argc,argv);
    @autoreleasepool {
        if (argc!=3) return 2;
        layout_variant=atoi(argv[2]);if (layout_variant<1 || layout_variant>2) return 2;
        qw3_engine *engine=NULL;int rc=1,cases=0;
        const qw3_tensor *w[2]={NULL,NULL};id<MTLBuffer> grid=nil;
        qw3_engine_options opt={.model_path=argv[1],.backend=QW3_BACKEND_METAL};
        if (qw3_engine_open(&engine,&opt) || !iq3_layout_compile()) goto done;
        grid=qw3_metal_iq3s_kgrid_buffer();if (!grid) goto done;
        for (int l=0;l<qw3_engine_layer_count(engine);l++) {
            const qw3_layer_weights *lw=&engine->weights.layer[l];
            if (lw->ffn_gate->type==21u && lw->ffn_up->type==21u) {
                w[0]=lw->ffn_gate;w[1]=lw->ffn_up;break;
            }
        }
        if (!w[0]) goto done;
        const uint32_t shapes[][2]={{256,1},{4352,33},{5120,4097},{5120,4111},{5120,17408},{17408,5120}};
        for (int shape=0;shape<6;shape++) {
            uint32_t ni=shapes[shape][0],no=shapes[shape][1];
            uint64_t offsets[2];id<MTLBuffer> native[2],packed[2];
            for (int i=0;i<2;i++) {
                if ((uint64_t)ni*no>w[i]->elements) goto done;
                native[i]=qw3_metal_model_view_for(w[i]->offset,(uint64_t)ni/256u*110u*no,&offsets[i]);
                packed[i]=iq3_layout_pack(native[i],offsets[i],ni,no);
                if (!packed[i] || !iq3_layout_guards(packed[i])) goto done;
            }
            for (int pattern=0;pattern<3;pattern++) for (int kind=0;kind<2;kind++) {
                const uint32_t n=kind ? 2u*no : no;
                id<MTLBuffer> input=[g_device newBufferWithLength:(ni+8u)*4u options:MTLResourceStorageModeShared];
                if (!input) goto done;
                float *x=input.contents;
                for (uint32_t i=0;i<ni+8u;i++) x[i]=pattern==1 ? (i&1u ? -0.0f : 0.0f) :
                    pattern==2 ? (float)((int)(i%7u)-3)*(i%5u==0u ? 0x1p8f : 0x1p-12f) : (float)((int)((i*37u)%101u)-50)/31.0f;
                NSData *before=[NSData dataWithBytes:input.contents length:input.length];
                id<MTLBuffer> output[2];
                for (int v=0;v<2;v++) {
                    output[v]=[g_device newBufferWithLength:(n+12u)*4u options:MTLResourceStorageModeShared];
                    if (!output[v]) goto done;
                    float *y=output[v].contents;for (uint32_t i=0;i<n+12u;i++) y[i]=123456.0f;
                    struct {uint32_t ni,no,rb;} args={ni,no,ni/256u*(v ? 112u : 110u)};
                    id<MTLCommandBuffer> cb=[g_queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
                    [enc setComputePipelineState:iq3_layout_pipeline[v ? layout_variant : 0][kind]];
                    [enc setBytes:&args length:sizeof(args) atIndex:0];
                    [enc setBuffer:v ? packed[0] : native[0] offset:v ? 32u : offsets[0] atIndex:1];
                    if (kind) [enc setBuffer:v ? packed[1] : native[1] offset:v ? 32u : offsets[1] atIndex:2];
                    [enc setBuffer:input offset:16 atIndex:kind ? 3 : 2];
                    [enc setBuffer:output[v] offset:20 atIndex:kind ? 4 : 3];
                    [enc setBuffer:grid offset:0 atIndex:kind ? 5 : 4];
                    [enc setThreadgroupMemoryLength:256 atIndex:0];
                    [enc dispatchThreadgroups:MTLSizeMake((no+7u)/8u,1,1) threadsPerThreadgroup:MTLSizeMake(64u,1,1)];
                    [enc endEncoding];[cb commit];[cb waitUntilCompleted];
                    if (cb.status!=MTLCommandBufferStatusCompleted) goto done;
                }
                if (memcmp(before.bytes,input.contents,input.length)) goto done;
                const float *a=output[0].contents,*b=output[1].contents;
                for (uint32_t i=0;i<n+12u;i++) {
                    if (!isfinite(a[i]) || !isfinite(b[i]) || memcmp(a+i,b+i,4)) {
                        uint32_t bits[2];memcpy(bits,a+i,4);memcpy(bits+1,b+i,4);
                        fprintf(stderr,"IQ3 layout differs variant=%d ni=%u no=%u kind=%d pattern=%d index=%u reference=%.9g candidate=%.9g bits=%08x/%08x\n",layout_variant,ni,no,kind,pattern,i,a[i],b[i],bits[0],bits[1]);goto done;
                    }
                    if ((i<5u || i>=5u+n) && b[i]!=123456.0f) goto done;
                }
                if (!iq3_layout_guards(packed[0]) || !iq3_layout_guards(packed[1])) goto done;
                cases++;printf("matrix_exact variant=%d n_in=%u n_out=%u kind=%d pattern=%d guards=exact packed_guards=exact input=unchanged\n",layout_variant,ni,no,kind,pattern);
            }
        }
        printf("PASS iq3_layout=%d matrix_cases=%d finite_bit_exact=all\n",layout_variant,cases);rc=0;
done:
        if (rc) fprintf(stderr,"IQ3 layout matrix failed after %d cases\n",cases);
        if (engine) qw3_engine_close(engine);return rc;
    }
}
