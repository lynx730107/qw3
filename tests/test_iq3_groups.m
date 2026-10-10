#define main groups_full_quality_main
#include ".iq3_groups/quality.inc"
#undef main
int main(int argc,char **argv) {
    if (argc==4) return groups_full_quality_main(argc,argv);
    @autoreleasepool {
        if (argc!=3) return 2;
        groups_width=atoi(argv[2]);if (groups_width!=4 && groups_width!=8) return 2;
        qw3_engine *e=NULL;int rc=1,cases=0;
        const qw3_tensor *w[2]={NULL,NULL};
        qw3_engine_options opt={.model_path=argv[1],.backend=QW3_BACKEND_METAL,.n_threads=4};
        if (qw3_engine_open(&e,&opt) || !groups_compile()) goto done;
        for (int l=0;l<qw3_engine_layer_count(e);l++) {
            const qw3_layer_weights *lw=&e->weights.layer[l];
            if (lw->ffn_gate->type==21u && lw->ffn_up->type==21u) {
                w[0]=lw->ffn_gate;w[1]=lw->ffn_up;break;
            }
        }
        if (!w[0]) goto done;
        const uint32_t shapes[][2]={{256,1},{4352,33},{5120,4097},{5120,4111},{5120,17408},{17408,5120}};
        for (int k=0;k<6;k++) {
            uint32_t ni=shapes[k][0],no=shapes[k][1];
            uint64_t offsets[2];id<MTLBuffer> wb[2],dst[2];
            id<MTLBuffer> src=[g_device newBufferWithLength:(ni+8u)*4u options:MTLResourceStorageModeShared];
            id<MTLBuffer> grid=qw3_metal_iq3s_kgrid_buffer();
            if (!src || !grid || (uint64_t)ni*no>w[0]->elements || (uint64_t)ni*no>w[1]->elements) goto done;
            float *x=src.contents;
            for (uint32_t i=0;i<ni+8u;i++) x[i]=(float)((int)((i*37u)%101u)-50)/31.0f;
            NSData *before=[NSData dataWithBytes:src.contents length:src.length];
            for (int v=0;v<2;v++) {
                wb[v]=qw3_metal_model_view_for(w[v]->offset,(uint64_t)ni/256u*110u*no,&offsets[v]);
                dst[v]=[g_device newBufferWithLength:(2u*no+12u)*4u options:MTLResourceStorageModeShared];
                if (!wb[v] || !dst[v]) goto done;
                float *y=dst[v].contents;for (uint32_t i=0;i<2u*no+12u;i++) y[i]=123456.0f;
            }
            for (int v=0;v<2;v++) {
                const uint32_t groups=v ? groups_width : 2u;
                struct {uint32_t ni,no,rb;} args={ni,no,ni/256u*110u};
                id<MTLCommandBuffer> cb=[g_queue commandBuffer];
                id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
                [enc setComputePipelineState:v ? groups_pipeline[groups_width==8u] : g_matvec_iq3_s_pair_pipeline];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:wb[0] offset:offsets[0] atIndex:1];
                [enc setBuffer:wb[1] offset:offsets[1] atIndex:2];
                [enc setBuffer:src offset:16 atIndex:3];
                [enc setBuffer:dst[v] offset:20 atIndex:4];
                [enc setBuffer:grid offset:0 atIndex:5];
                [enc setThreadgroupMemoryLength:256 atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake((no+groups*4u-1u)/(groups*4u),1,1)
                    threadsPerThreadgroup:MTLSizeMake(groups*32u,1,1)];
                [enc endEncoding];[cb commit];[cb waitUntilCompleted];
                if (cb.status!=MTLCommandBufferStatusCompleted) goto done;
            }
            if (memcmp(before.bytes,src.contents,src.length) || memcmp(dst[0].contents,dst[1].contents,dst[0].length)) goto done;
            const float *y=dst[1].contents;
            for (uint32_t i=0;i<2u*no+12u;i++) if (!isfinite(y[i]) ||
                ((i<5u || i>=5u+2u*no) && y[i]!=123456.0f)) goto done;
            cases++;printf("matrix_exact groups=%u n_in=%u n_out=%u guards=ok input=unchanged\n",groups_width,ni,no);
        }
        printf("PASS groups=%u matrix_cases=%d\n",groups_width,cases);rc=0;
done:
        if (rc) fprintf(stderr,"IQ3 groups matrix failed after %d cases\n",cases);
        if (e) qw3_engine_close(e);return rc;
    }
}
