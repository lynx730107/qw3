#define main window_full_quality_main
#include "quality.inc"
#undef main
/* Independent double RMSNorm/gate reference from the updated GPU state. */
static double window_cpu_inner_error(id<MTLBuffer> __strong *inputs,id<MTLBuffer> state,
    id<MTLBuffer> inner,uint32_t qh,uint32_t vh,uint32_t hd,uint32_t z,float eps) {
    const float *q=(const float *)inputs[0].contents+4;
    const float *scratch=(const float *)inputs[3].contents+4;
    const float *w=(const float *)inputs[6].contents+4;
    const float *st=(const float *)state.contents+5,*out=(const float *)inner.contents+5;
    double worst=0,core[128];
    for (uint32_t h=0;h<vh;h++) {
        double ss=0;
        for (uint32_t j=0;j<hd;j++) {
            double dot=0;
            for (uint32_t i=0;i<hd;i++) dot+=(double)st[h*hd*hd+i*hd+j]*q[(h%qh)*hd+i];
            core[j]=dot/sqrt((double)hd);ss+=core[j]*core[j];
        }
        double scale=1/sqrt(ss/hd+eps);
        for (uint32_t j=0;j<hd;j++) {
            double zi=scratch[z+h*hd+j];
            double expected=core[j]*scale*w[j]*(zi/(1+exp(-zi)));
            if (!isfinite(expected) || !isfinite(out[h*hd+j])) return INFINITY;
            worst=fmax(worst,fabs(expected-out[h*hd+j]));
        }
    }
    return worst;
}
int main(int argc,char **argv) {
    if (argc==4) return window_full_quality_main(argc,argv);
    @autoreleasepool {
        if (argc!=2) return 2;
        qw3_engine *e=NULL;int rc=1,cases=0;
        qw3_engine_options opt={.model_path=argv[1],.backend=QW3_BACKEND_METAL,.n_threads=4};
        if (qw3_engine_open(&e,&opt) || !window_compile()) goto done;
        for (uint32_t hd=128;hd<=128;hd*=2) for (uint32_t vh=3;vh<=7;vh+=4) {
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
                id<MTLCommandBuffer> cb=[g_queue commandBuffer];
                id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
                [enc setComputePipelineState:v ? window_pipeline[v-1] : window_base];
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
                double cpu_error=window_cpu_inner_error(inputs,state[v],inner[v],qh,vh,hd,args.z,args.eps);
                if (!equal || cpu_error>0.00002) {
                    fprintf(stderr,"GDN windows mismatch head_dim=%u variant=%d max=%.9g\n",hd,v,cpu_error);goto done;
                }
                cases++;printf("matrix_exact head_dim=%u v_heads=%u variant=%d state=exact inner=native_cache64_exact cpu_norm_max=%.9g guards=ok inputs=unchanged\n",hd,vh,v,cpu_error);
            }
        }
        if (cases!=10) goto done;
        printf("PASS matrix_cases=%d\n",cases);rc=0;
done:
        if (rc) fprintf(stderr,"GDN windows matrix failed after %d cases\n",cases);
        if (e) qw3_engine_close(e);return rc;
    }
}
