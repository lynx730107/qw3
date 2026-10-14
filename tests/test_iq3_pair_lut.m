#include ".iq3_pair_lut/hooks.inc"

int main(int argc,char **argv) {
    @autoreleasepool {
        if (argc!=2) return 2;
        pair_lut_variant=atoi(argv[1]);if (pair_lut_variant<1 || pair_lut_variant>2) return 2;
        if (!iq3_pair_lut_compile()) return 1;
        id<MTLBuffer> grid=qw3_metal_iq3s_kgrid_buffer();if (!grid) return 1;
        const uint32_t shapes[][2]={{256,1},{4352,33},{5120,65},{6144,33},
                                  {8192,65},{8448,33},{17408,65},{5120,64}};
        int cases=0;
        for (int shape=0;shape<8;shape++) for (int pattern=0;pattern<3;pattern++)
        for (int alignment=0;alignment<2;alignment++) for (int padding=0;padding<2;padding++) {
            @autoreleasepool {
                uint32_t ni=shapes[shape][0],no=shapes[shape][1],rb=ni/256u*110u+2u*padding;
                NSUInteger input_offset=alignment ? 32u : 16u;
                id<MTLBuffer> weights[2];NSData *wi[2];
                for (int k=0;k<2;k++) {
                    weights[k]=[g_device newBufferWithLength:(uint64_t)rb*no+8u options:MTLResourceStorageModeShared];
                    if (!weights[k]) return 1;
                    uint8_t *w=weights[k].contents;
                    for (uint64_t i=0;i<weights[k].length;i++) w[i]=(uint8_t)(i*(k ? 53u : 37u)+19u+k);
                    for (uint32_t row=0;row<no;row++) for (uint32_t b=0;b<ni/256u;b++) {
                        uint16_t d=(row+b+k)%3u==0u ? 0xb000u : 0x3000u;
                        memcpy(w+2u+(uint64_t)row*rb+b*110u,&d,2);
                    }
                    wi[k]=[NSData dataWithBytes:w length:weights[k].length];
                }
                id<MTLBuffer> input=[g_device newBufferWithLength:(ni+12u)*4u options:MTLResourceStorageModeShared];
                id<MTLBuffer> output=[g_device newBufferWithLength:(2u*no+10u)*4u options:MTLResourceStorageModeShared];
                if (!input || !output) return 1;
                float *x=input.contents;
                for (uint32_t i=0;i<ni+12u;i++) x[i]=pattern==1 ? (i&1u ? -0.0f : 0.0f) :
                    pattern==2 ? (float)((int)(i%7u)-3)*(i%5u==0u ? 256.0f : 0x1p-12f) :
                    (float)((int)((i*37u)%101u)-50)/193.0f;
                NSData *xi=[NSData dataWithBytes:x length:input.length],*reference=nil;
                for (int phase=0;phase<2;phase++) {
                    float *y=output.contents;
                    for (uint32_t i=0;i<2u*no+10u;i++) y[i]=123456.0f;
                    struct {uint32_t ni,no,rb;} args={ni,no,rb};
                    id<MTLCommandBuffer> cb=[g_queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
                    [(id<QW3IQ3PairLutHooks>)enc iq3PairLutSetPipeline:iq3_pair_lut_pipeline[phase ? pair_lut_variant : 0]];
                    [enc setBytes:&args length:sizeof(args) atIndex:0];
                    [enc setBuffer:weights[0] offset:2 atIndex:1];[enc setBuffer:weights[1] offset:2 atIndex:2];
                    [enc setBuffer:input offset:input_offset atIndex:3];[enc setBuffer:output offset:20 atIndex:4];
                    [enc setBuffer:grid offset:0 atIndex:5];[enc setThreadgroupMemoryLength:128u atIndex:0];
                    [enc dispatchThreadgroups:MTLSizeMake((no+7u)/8u+1u,1,1) threadsPerThreadgroup:MTLSizeMake(64u,1,1)];
                    [enc endEncoding];[cb commit];[cb waitUntilCompleted];
                    if (cb.status!=MTLCommandBufferStatusCompleted || memcmp(xi.bytes,x,input.length) ||
                        memcmp(wi[0].bytes,weights[0].contents,weights[0].length) ||
                        memcmp(wi[1].bytes,weights[1].contents,weights[1].length)) return 1;
                    for (uint32_t i=0;i<2u*no+10u;i++) if (!isfinite(y[i]) ||
                        ((i<5u || i>=2u*no+5u) && y[i]!=123456.0f)) return 1;
                    if (!phase) reference=[NSData dataWithBytes:y length:output.length];
                    else if (memcmp(reference.bytes,y,output.length)) {
                        const float *a=reference.bytes;
                        for (uint32_t i=0;i<2u*no+10u;i++) if (memcmp(a+i,y+i,4)) {
                            uint32_t bits[2];memcpy(bits,a+i,4);memcpy(bits+1,y+i,4);
                            fprintf(stderr,"IQ3 pair lut mismatch variant=%d n_in=%u n_out=%u pattern=%d input_offset=%lu padding=%d index=%u bits=%08x/%08x\n",pair_lut_variant,ni,no,pattern,(unsigned long)input_offset,padding,i,bits[0],bits[1]);break;
                        }
                        return 1;
                    }
                }
                cases++;printf("matrix_exact n_in=%u n_out=%u pattern=%d input_offset=%lu padding=%d guards=exact input=unchanged weights=unchanged\n",ni,no,pattern,(unsigned long)input_offset,padding);
            }
        }
        if (iq3_pair_lut_calls!=96u) return 1;
        printf("PASS iq3_pair_lut=%d matrix_cases=%d finite_bit_exact=all\n",pair_lut_variant,cases);
        return cases==96 ? 0 : 1;
    }
}
