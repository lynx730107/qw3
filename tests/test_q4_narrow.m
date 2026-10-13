#include ".q4_narrow/hooks.inc"

int main(void) {
    @autoreleasepool {
        if (!q4_narrow_compile()) return 1;
        const uint32_t shapes[][2]={{256,1},{4352,33},{5120,65},{6144,33},
                                  {8192,65},{8448,33},{17408,65}};
        int cases=0;
        for (int shape=0;shape<7;shape++) for (int pattern=0;pattern<3;pattern++) {
            @autoreleasepool {
                uint32_t ni=shapes[shape][0],no=shapes[shape][1],rb=ni/256u*144u;
                id<MTLBuffer> weights=[g_device newBufferWithLength:(uint64_t)rb*no options:MTLResourceStorageModeShared];
                id<MTLBuffer> input=[g_device newBufferWithLength:(ni+8u)*4u options:MTLResourceStorageModeShared];
                id<MTLBuffer> output=[g_device newBufferWithLength:(no+10u)*4u options:MTLResourceStorageModeShared];
                if (!weights || !input || !output) return 1;
                uint8_t *w=weights.contents;
                for (uint64_t i=0;i<weights.length;i++) w[i]=(uint8_t)(i*37u+19u);
                for (uint64_t off=0;off<weights.length;off+=144u) {
                    uint16_t d=0x3000u,dm=0x2800u;
                    memcpy(w+off,&d,2);memcpy(w+off+2u,&dm,2);
                }
                float *x=input.contents;
                for (uint32_t i=0;i<ni+8u;i++) x[i]=pattern==1 ? (i&1u ? -0.0f : 0.0f) :
                    pattern==2 ? (float)((int)(i%7u)-3)*(i%5u==0u ? 256.0f : 0x1p-12f) :
                    (float)((int)((i*37u)%101u)-50)/193.0f;
                NSData *wi=[NSData dataWithBytes:w length:weights.length];
                NSData *xi=[NSData dataWithBytes:x length:input.length],*reference=nil;
                uint64_t calls=q4_narrow_calls,fallbacks=q4_narrow_fallbacks;
                NSUInteger sizes[2];
                for (int phase=0;phase<2;phase++) {
                    scratch_select(phase);
                    sizes[phase]=qw3_metal_matvec_threads(g_matvec_q4_k_pipeline,ni/256u);
                    struct {uint32_t ni,no,rb;} args={ni,no,rb};
                    float *y=output.contents;
                    for (uint32_t i=0;i<no+10u;i++) y[i]=123456.0f;
                    id<MTLCommandBuffer> cb=[g_queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
                    [enc setComputePipelineState:g_matvec_q4_k_pipeline];
                    [enc setBytes:&args length:sizeof(args) atIndex:0];
                    [enc setBuffer:weights offset:0 atIndex:1];
                    [enc setBuffer:input offset:16 atIndex:2];
                    [enc setBuffer:output offset:20 atIndex:3];
                    [enc setThreadgroupMemoryLength:128u atIndex:0];
                    [enc dispatchThreadgroups:MTLSizeMake(no+1u,1,1) threadsPerThreadgroup:MTLSizeMake(sizes[phase],1,1)];
                    [enc endEncoding];[cb commit];[cb waitUntilCompleted];
                    if (cb.status!=MTLCommandBufferStatusCompleted ||
                        memcmp(wi.bytes,w,weights.length) || memcmp(xi.bytes,x,input.length)) return 1;
                    for (uint32_t i=0;i<no+10u;i++) if (!isfinite(y[i]) ||
                        ((i<5u || i>=no+5u) && y[i]!=123456.0f)) return 1;
                    if (!phase) reference=[NSData dataWithBytes:y length:output.length];
                    else if (memcmp(reference.bytes,y,output.length)) {
                        const float *a=reference.bytes;
                        for (uint32_t i=0;i<no+10u;i++) if (memcmp(a+i,y+i,4)) {
                            uint32_t bits[2];memcpy(bits,a+i,4);memcpy(bits+1,y+i,4);
                            fprintf(stderr,"Q4 narrow mismatch n_in=%u n_out=%u pattern=%d index=%u bits=%08x/%08x\n",ni,no,pattern,i,bits[0],bits[1]);break;
                        }
                        return 1;
                    }
                }
                int eligible=ni<=8192u;
                if (sizes[1]!=(eligible ? 32u : sizes[0]) ||
                    q4_narrow_calls-calls!=(eligible ? 1u : 0u) ||
                    q4_narrow_fallbacks-fallbacks!=(eligible ? 0u : 1u)) return 1;
                printf("matrix_exact n_in=%u n_out=%u pattern=%d native_threads=%lu candidate_threads=%lu guards=exact input=unchanged\n",ni,no,pattern,(unsigned long)sizes[0],(unsigned long)sizes[1]);cases++;
            }
        }
        q4_narrow_print_counts();
        printf("PASS q4_narrow matrix_cases=%d finite_bit_exact=all\n",cases);
        return cases==21 && q4_narrow_calls==15u && q4_narrow_fallbacks==6u ? 0 : 1;
    }
}
