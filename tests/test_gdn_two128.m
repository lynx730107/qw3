#include "gdn_two128_candidates.inc"
/* Double RMS/gate oracle, independently calculated from the updated state. */
static double two128_cpu_error(id<MTLBuffer> __strong *inputs,id<MTLBuffer> state,
    id<MTLBuffer> inner,uint32_t qh,uint32_t vh,uint32_t z,float eps) {
    const float *q=(const float *)inputs[0].contents+4;
    const float *scratch=(const float *)inputs[3].contents+4;
    const float *w=(const float *)inputs[6].contents+4;
    const float *st=(const float *)state.contents+5,*out=(const float *)inner.contents+5;
    double worst=0,core[128];
    for (uint32_t h=0;h<vh;h++) {
        double ss=0;
        for (uint32_t j=0;j<128;j++) {
            double dot=0;
            for (uint32_t i=0;i<128;i++) dot+=(double)st[h*128*128+i*128+j]*q[(h%qh)*128+i];
            core[j]=dot/sqrt(128.0);ss+=core[j]*core[j];
        }
        double scale=1/sqrt(ss/128+eps);
        for (uint32_t j=0;j<128;j++) {
            double zi=scratch[z+h*128+j];
            double expected=core[j]*scale*w[j]*(zi/(1+exp(-zi)));
            if (!isfinite(expected) || !isfinite(out[h*128+j])) return INFINITY;
            worst=fmax(worst,fabs(expected-out[h*128+j]));
        }
    }
    return worst;
}
int main(int argc,char **argv) {
    @autoreleasepool {
        if (argc!=3) return 2;
        two128_variant=atoi(argv[2]);if (two128_variant<1 || two128_variant>2) return 2;
        qw3_engine *engine=NULL;int rc=1,cases=0;
        qw3_engine_options opt={.model_path=argv[1],.backend=QW3_BACKEND_METAL,.n_threads=4};
        if (qw3_engine_open(&engine,&opt) || !two128_compile()) goto done;
        const uint32_t heads[]={3,7,48};
        for (int shape=0;shape<3;shape++) for (int pattern=0;pattern<3;pattern++) {
            uint32_t qh=2,vh=heads[shape],state_n=vh*128*128,inner_n=vh*128;
            struct {uint32_t qh,vh,hd,alpha,beta,z;float eps;} args={qh,vh,128,1,1+vh+3,1+2*vh+7,1e-6f};
            uint32_t sizes[]={qh*128,qh*128,inner_n,args.z+inner_n,vh,vh,128};
            id<MTLBuffer> inputs[7],state[2],inner[2];NSData *before[7];
            uint32_t rng=128u*37u+vh;
            for (int b=0;b<7;b++) {
                inputs[b]=[g_device newBufferWithLength:(sizes[b]+8u)*4u options:MTLResourceStorageModeShared];
                if (!inputs[b]) goto done;
                float *x=inputs[b].contents;
                for (uint32_t i=0;i<sizes[b]+8u;i++) {
                    rng=rng*1664525u+1013904223u;
                    x[i]=(float)((int)(rng>>16u)-32768)/65536.0f;
                    if (pattern==1) x[i]=i&1u ? -0.0f : 0.0f;
                    if (b==5) x[i]=-fabsf(x[i])-0.05f;
                    if (b==6) x[i]=fabsf(x[i])+0.25f;
                }
                if (pattern==2 && b==3) for (uint32_t h=0;h<vh;h++)
                    x[4+args.alpha+h]=h%3u==0u ? -25.0f : h%3u==1u ? 0.0f : 25.0f;
                before[b]=[NSData dataWithBytes:inputs[b].contents length:inputs[b].length];
            }
            for (int v=0;v<2;v++) {
                state[v]=[g_device newBufferWithLength:(state_n+12u)*4u options:MTLResourceStorageModeShared];
                inner[v]=[g_device newBufferWithLength:(inner_n+12u)*4u options:MTLResourceStorageModeShared];
                if (!state[v] || !inner[v]) goto done;
                float *st=state[v].contents,*y=inner[v].contents;
                for (uint32_t i=0;i<state_n+12u;i++) st[i]=i<5u || i>=state_n+5u ? 123456.0f :
                    pattern==1 ? (i&1u ? -0.0f : 0.0f) : (float)((int)((i*37u)%101u)-50)/1024.0f;
                for (uint32_t i=0;i<inner_n+12u;i++) y[i]=123456.0f;
                id<MTLCommandBuffer> cb=[g_queue commandBuffer];
                id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
                [enc setComputePipelineState:v ? two128_pipeline[two128_variant-1] : two128_base];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:state[v] offset:20 atIndex:1];
                for (int b=0;b<7;b++) [enc setBuffer:inputs[b] offset:16 atIndex:b+2];
                [enc setBuffer:state[v] offset:20 atIndex:9];
                [enc setBuffer:inner[v] offset:20 atIndex:10];
                [enc setThreadgroupMemoryLength:128 atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake(vh+1u,1,1) threadsPerThreadgroup:MTLSizeMake(128u,1,1)];
                [enc endEncoding];[cb commit];[cb waitUntilCompleted];
                if (cb.status!=MTLCommandBufferStatusCompleted) {
                    fprintf(stderr,"GDN two128 GPU error: %s\n",cb.error.localizedDescription.UTF8String);goto done;
                }
                for (int b=0;b<7;b++) if (memcmp(before[b].bytes,inputs[b].contents,inputs[b].length)) goto done;
                for (int kind=0;kind<2;kind++) {
                    const float *values=kind ? inner[v].contents : state[v].contents;
                    uint32_t n=kind ? inner_n : state_n;
                    for (uint32_t i=0;i<n+12u;i++) if (!isfinite(values[i]) ||
                        ((i<5u || i>=n+5u) && values[i]!=123456.0f)) goto done;
                }
            }
            for (int kind=0;kind<2;kind++) {
                id<MTLBuffer> a=kind ? inner[0] : state[0],b=kind ? inner[1] : state[1];
                if (memcmp(a.contents,b.contents,a.length)) {
                    const float *x=a.contents,*y=b.contents;
                    for (uint32_t i=0;i<a.length/4u;i++) if (memcmp(x+i,y+i,4)) {
                        fprintf(stderr,"GDN two128 mismatch variant=%d heads=%u pattern=%d kind=%s index=%u baseline=%a candidate=%a\n",
                            two128_variant,vh,pattern,kind ? "inner" : "state",i,(double)x[i],(double)y[i]);break;
                    }
                    goto done;
                }
            }
            double error=two128_cpu_error(inputs,state[1],inner[1],qh,vh,args.z,args.eps);
            if (error>0.00002) {fprintf(stderr,"GDN two128 CPU oracle error=%g\n",error);goto done;}
            cases++;printf("matrix_exact variant=%d heads=%u pattern=%d state=exact inner=exact cpu_norm_max=%.9g guards=ok inputs=unchanged\n",
                two128_variant,vh,pattern,error);fflush(stdout);
        }
        if (cases!=9) goto done;
        printf("PASS gdn_two128_matrix variant=%d cases=%d\n",two128_variant,cases);rc=0;
done:
        if (rc) fprintf(stderr,"GDN two128 matrix failed after %d cases\n",cases);
        if (engine) qw3_engine_close(engine);return rc;
    }
}
