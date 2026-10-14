#include ".iq3xxs_sign/hooks.inc"

int main(int argc,char **argv) {
    @autoreleasepool {
        if (argc!=2) return 2;
        sign_variant=atoi(argv[1]);if (sign_variant<1 || sign_variant>2) return 2;
        if (!iq3xxs_sign_compile()) return 1;
        id<MTLBuffer> grid=nil,signs=nil;
        if (!qw3_metal_dense_iq_buffers(&signs,NULL,NULL,&grid)) return 1;
        const uint32_t shapes[][2]={{256,1},{4352,33},{5120,65},{6144,33},
                                  {8192,65},{8448,33},{17408,65},{5120,64}};
        int cases=0;
        for (int shape=0;shape<8;shape++) for (int pattern=0;pattern<3;pattern++) for (int alignment=0;alignment<4;alignment++) for (int padding=0;padding<2;padding++) {
            @autoreleasepool {
                const uint32_t offsets[]={0,2,4,8};
                uint32_t ni=shapes[shape][0],no=shapes[shape][1],rb=ni/256u*98u+2u*padding,weight_offset=offsets[alignment];
                id<MTLBuffer> weights=[g_device newBufferWithLength:(uint64_t)rb*no+8u options:MTLResourceStorageModeShared];
                id<MTLBuffer> input=[g_device newBufferWithLength:(ni+8u)*4u options:MTLResourceStorageModeShared];
                id<MTLBuffer> output=[g_device newBufferWithLength:(no+10u)*4u options:MTLResourceStorageModeShared];
                if (!weights || !input || !output) return 1;
                uint8_t *w=weights.contents;
                for (uint64_t i=0;i<weights.length;i++) w[i]=(uint8_t)(i*37u+19u);
                for (uint32_t row=0;row<no;row++) for (uint32_t block=0;block<ni/256u;block++) {
                    uint64_t off=weight_offset+(uint64_t)row*rb+(uint64_t)block*98u;
                    uint16_t d=0x3000u;
                    memcpy(w+off,&d,2);
                }
                float *x=input.contents;
                for (uint32_t i=0;i<ni+8u;i++) x[i]=pattern==1 ? (i&1u ? -0.0f : 0.0f) :
                    pattern==2 ? (float)((int)(i%7u)-3)*(i%5u==0u ? 256.0f : 0x1p-12f) :
                    (float)((int)((i*37u)%101u)-50)/193.0f;
                NSData *wi=[NSData dataWithBytes:w length:weights.length];
                NSData *xi=[NSData dataWithBytes:x length:input.length],*reference=nil;
                uint64_t calls=iq3xxs_sign_calls;
                NSUInteger sizes[2];
                for (int phase=0;phase<2;phase++) {
                    scratch_select(phase);
                    sizes[phase]=64u;
                    struct {uint32_t ni,no,rb;} args={ni,no,rb};
                    float *y=output.contents;
                    for (uint32_t i=0;i<no+10u;i++) y[i]=123456.0f;
                    id<MTLCommandBuffer> cb=[g_queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
                    [(id<QW3IQ3XXSSignHooks>)enc iq3xxsSignSetPipeline:g_matvec_iq3_xxs_pipeline];
                    [enc setBytes:&args length:sizeof(args) atIndex:0];
                    [enc setBuffer:weights offset:weight_offset atIndex:1];
                    [enc setBuffer:input offset:16 atIndex:2];
                    [enc setBuffer:output offset:20 atIndex:3];
                    [enc setBuffer:grid offset:0 atIndex:4];[enc setBuffer:signs offset:0 atIndex:5];
                    [enc setThreadgroupMemoryLength:128u atIndex:0];
                    [enc dispatchThreadgroups:MTLSizeMake((no+3u)/4u+1u,1,1) threadsPerThreadgroup:MTLSizeMake(sizes[phase],1,1)];
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
                            fprintf(stderr,"IQ3XXS sign mismatch n_in=%u n_out=%u pattern=%d index=%u bits=%08x/%08x\n",ni,no,pattern,i,bits[0],bits[1]);break;
                        }
                        return 1;
                    }
                }
                if (sizes[1]!=sizes[0] || iq3xxs_sign_calls-calls!=1u) return 1;
                printf("matrix_exact n_in=%u n_out=%u pattern=%d weight_offset=%u row_padding=%d native_threads=%lu candidate_threads=%lu guards=exact input=unchanged\n",ni,no,pattern,weight_offset,padding ? 2 : 0,(unsigned long)sizes[0],(unsigned long)sizes[1]);cases++;
            }
        }
        printf("PASS iq3xxs_sign=%d matrix_cases=%d finite_bit_exact=all\n",sign_variant,cases);
        return cases==192 && iq3xxs_sign_calls==192u ? 0 : 1;
    }
}
