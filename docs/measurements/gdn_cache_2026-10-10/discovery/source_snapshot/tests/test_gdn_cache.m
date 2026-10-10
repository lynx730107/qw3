#define main cache_full_quality_main
#include ".gdn_cache/quality.inc"
#undef main
int main(int argc,char **argv) {
    if (argc==4) return cache_full_quality_main(argc,argv);
    @autoreleasepool {
        if (argc!=2) return 2;
        qw3_engine *e=NULL;int rc=1,cases=0,poison_cases=0;
        qw3_engine_options opt={.model_path=argv[1],.backend=QW3_BACKEND_METAL,.n_threads=4};
        if (qw3_engine_open(&e,&opt) || !cache_compile()) goto done;
        for (uint32_t hd=32;hd<=128;hd*=2) for (uint32_t vh=3;vh<=7;vh+=4) {
            const uint32_t qh=2,state_n=vh*hd*hd,inner_n=vh*hd;
            struct {uint32_t qh,vh,hd,alpha,beta,z;float eps;} args={qh,vh,hd,1,1+vh+3,1+2*vh+7,1e-6f};
            uint32_t sizes[]={qh*hd,qh*hd,inner_n,args.z+inner_n,vh,vh,hd};
            id<MTLBuffer> inputs[7],state[6],inner[6];NSData *input_copy[7];
            uint32_t rng=hd*37u+vh;
            for (int b=0;b<7;b++) {
                inputs[b]=[g_device newBufferWithLength:(sizes[b]+8u)*4u options:MTLResourceStorageModeShared];
                if (!inputs[b]) goto done;
                float *x=inputs[b].contents;
                for (uint32_t i=0;i<sizes[b]+8u;i++) {
                    rng=rng*1664525u+1013904223u;
                    x[i]=(float)((int)(rng>>16u)-32768)/65536.0f;
                    if (b==5) x[i]=-fabsf(x[i])-0.05f;
                    if (b==6) x[i]=fabsf(x[i])+0.25f;
                }
                input_copy[b]=[NSData dataWithBytes:inputs[b].contents length:inputs[b].length];
            }
            for (int v=0;v<6;v++) {
                state[v]=[g_device newBufferWithLength:(state_n+12u)*4u options:MTLResourceStorageModeShared];
                inner[v]=[g_device newBufferWithLength:(inner_n+12u)*4u options:MTLResourceStorageModeShared];
                if (!state[v] || !inner[v]) goto done;
                float *st=state[v].contents,*out=inner[v].contents;
                for (uint32_t i=0;i<state_n+12u;i++) st[i]=i<5u || i>=state_n+5u ? 123456.0f : (float)((int)((i*37u)%101u)-50)/1024.0f;
                for (uint32_t i=0;i<inner_n+12u;i++) out[i]=123456.0f;
            }
            for (int v=0;v<6;v++) {
                if (hd!=128u && v>=2 && v<=4) continue;
                id<MTLCommandBuffer> cb=[g_queue commandBuffer];
                id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
                [enc setComputePipelineState:v ? cache_pipeline[v-1] : cache_base];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:state[v] offset:20 atIndex:1];
                for (int b=0;b<7;b++) [enc setBuffer:inputs[b] offset:16 atIndex:b+2];
                [enc setBuffer:state[v] offset:20 atIndex:9];
                [enc setBuffer:inner[v] offset:20 atIndex:10];
                [enc setThreadgroupMemoryLength:128 atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake(vh+1u,1,1) threadsPerThreadgroup:MTLSizeMake(hd,1,1)];
                [enc endEncoding];[cb commit];[cb waitUntilCompleted];
                if (cb.status!=MTLCommandBufferStatusCompleted) goto done;
                for (int b=0;b<7;b++) if (memcmp(input_copy[b].bytes,inputs[b].contents,inputs[b].length)) goto done;
                const float *st=state[v].contents,*out=inner[v].contents;
                for (uint32_t i=0;i<state_n+12u;i++) if (!isfinite(st[i]) ||
                    ((i<5u || i>=state_n+5u) && st[i]!=123456.0f)) goto done;
                for (uint32_t i=0;i<inner_n+12u;i++) if (!isfinite(out[i]) ||
                    ((i<5u || i>=inner_n+5u) && out[i]!=123456.0f)) goto done;
                if (!v) continue;
                if (memcmp(state[0].contents,state[v].contents,state[0].length)) goto done;
                int equal=!memcmp(inner[0].contents,inner[v].contents,inner[0].length);
                if (v==5) {
                    if (equal) goto done;
                    poison_cases++;printf("poison_sensitive head_dim=%u v_heads=%u state=exact inner=different\n",hd,vh);
                } else {
                    if (!equal) {fprintf(stderr,"GDN cache output mismatch head_dim=%u variant=%d\n",hd,v);goto done;}
                    cases++;printf("matrix_exact head_dim=%u v_heads=%u variant=%d state=exact inner=bit_exact guards=ok inputs=unchanged\n",hd,vh,v);
                }
            }
        }
        printf("PASS matrix_cases=%d poison_cases=%d\n",cases,poison_cases);rc=0;
done:
        if (rc) fprintf(stderr,"GDN cache matrix failed after %d cases\n",cases);
        if (e) qw3_engine_close(e);return rc;
    }
}
