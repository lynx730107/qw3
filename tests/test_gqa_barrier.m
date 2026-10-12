#include ".gqa_barrier/hooks.inc"

static float fixture_value(uint32_t i,int salt,int pattern) {
    if (pattern==1) return i&1u ? -0.0f : 0.0f;
    float v=(float)((int)((i*37u+(uint32_t)salt*17u)%101u)-50)/193.0f;
    return pattern==2 ? v*(i%5u==0u ? 16.0f : 0x1p-8f) : v;
}

int main(int argc,char **argv) {
    @autoreleasepool {
        if (argc!=2) return 2;
        barrier_variant=atoi(argv[1]);if (barrier_variant<1 || barrier_variant>2) return 2;
        if (!gqa_barrier_compile()) return 1;
        const uint32_t groups[]={1,6,7,8},dims[]={32,128,256},contexts[]={1,2,31,128};
        int cases=0;
        for (int gh=0;gh<4;gh++) for (int hd=0;hd<3;hd++)
        for (int ct=0;ct<4;ct++) for (int kv=0;kv<2;kv++) for (int pattern=0;pattern<3;pattern++) {
            @autoreleasepool {
                uint32_t nk=4,nh=nk*groups[gh],dim=dims[hd],ctx=contexts[ct];
                uint32_t qn=nh*dim,kn=ctx*nk*dim;
                id<MTLBuffer> input[4];
                input[0]=[g_device newBufferWithLength:qn*4u options:MTLResourceStorageModeShared];
                input[1]=[g_device newBufferWithLength:qn*4u options:MTLResourceStorageModeShared];
                input[2]=[g_device newBufferWithLength:kn*(kv ? 2u : 4u) options:MTLResourceStorageModeShared];
                input[3]=[g_device newBufferWithLength:kn*(kv ? 2u : 4u) options:MTLResourceStorageModeShared];
                id<MTLBuffer> out=[g_device newBufferWithLength:(qn+10u)*4u options:MTLResourceStorageModeShared];
                NSData *before[4];
                for (int b=0;b<4;b++) {
                    if (!input[b]) return 1;
                    uint32_t count=b<2 ? qn : kn;
                    for (uint32_t i=0;i<count;i++) {
                        float v=fixture_value(i,b+1,pattern);
                        if (b>=2 && kv) ((__fp16 *)input[b].contents)[i]=(__fp16)v;
                        else ((float *)input[b].contents)[i]=v;
                    }
                    before[b]=[NSData dataWithBytes:input[b].contents length:input[b].length];
                }
                if (!out) return 1;
                NSData *reference=nil;
                uint64_t calls=gqa_barrier_calls,fallbacks=gqa_barrier_fallbacks;
                for (int phase=0;phase<2;phase++) {
                    float *y=out.contents;
                    for (uint32_t i=0;i<qn+10u;i++) y[i]=123456.0f;
                    scratch_select(phase);
                    uint32_t args[5]={ctx,nh,nk,dim,(uint32_t)kv};
                    id<MTLCommandBuffer> cb=[g_queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
                    [(id<QW3GQABarrierHooks>)enc gqaBarrierSetPipeline:gqa_barrier_pipeline[0]];
                    [(id<QW3GQABarrierHooks>)enc gqaBarrierSetBytes:args length:sizeof(args) atIndex:0];
                    for (int b=0;b<4;b++) [enc setBuffer:input[b] offset:0 atIndex:b+1];
                    [enc setBuffer:out offset:20 atIndex:5];
                    [enc setBuffer:input[2] offset:0 atIndex:6];
                    [enc setBuffer:input[3] offset:0 atIndex:7];
                    [enc setThreadgroupMemoryLength:256u atIndex:0];
                    [enc dispatchThreadgroups:MTLSizeMake(nk+1u,1,1) threadsPerThreadgroup:MTLSizeMake(dim,1,1)];
                    [enc endEncoding];[cb commit];[cb waitUntilCompleted];
                    if (cb.status!=MTLCommandBufferStatusCompleted) return 1;
                    for (int b=0;b<4;b++) if (memcmp(before[b].bytes,input[b].contents,input[b].length)) return 1;
                    for (uint32_t i=0;i<qn+10u;i++) if (!isfinite(y[i]) ||
                        ((i<5u || i>=qn+5u) && y[i]!=123456.0f)) return 1;
                    if (!phase) reference=[NSData dataWithBytes:out.contents length:out.length];
                    else if (memcmp(reference.bytes,out.contents,out.length)) {
                        const float *a=reference.bytes;
                        for (uint32_t i=0;i<qn+10u;i++) if (memcmp(a+i,y+i,4)) {
                            uint32_t bits[2];memcpy(bits,a+i,4);memcpy(bits+1,y+i,4);
                            fprintf(stderr,"GQA barrier differs variant=%d group=%u dim=%u ctx=%u kv=%d pattern=%d index=%u reference=%.9g candidate=%.9g bits=%08x/%08x\n",
                                barrier_variant,groups[gh],dim,ctx,kv,pattern,i,a[i],y[i],bits[0],bits[1]);break;
                        }
                        return 1;
                    }
                }
                if (gqa_barrier_calls-calls!=(groups[gh]<8 ? 1u : 0u) ||
                    gqa_barrier_fallbacks-fallbacks!=(groups[gh]==8 ? 1u : 0u)) return 1;
                cases++;
                printf("matrix_exact variant=%d group=%u head_dim=%u n_ctx=%u kv=%d pattern=%d guards=exact input=unchanged\n",
                       barrier_variant,groups[gh],dim,ctx,kv,pattern);
            }
        }
        gqa_barrier_print_counts();
        printf("PASS gqa_barrier=%d matrix_cases=%d finite_bit_exact=all\n",barrier_variant,cases);
        return cases==288 ? 0 : 1;
    }
}
