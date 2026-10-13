#include "iq3_add_hooks.inc"

int main(int argc,char **argv) {
    @autoreleasepool {
        if (argc!=2) return 2;
        add_variant=atoi(argv[1]);if (add_variant<1 || add_variant>2) return 2;
        if (!iq3_add_compile()) return 1;
        id<MTLBuffer> grid=qw3_metal_iq3s_kgrid_buffer();if (!grid) return 1;
        const uint32_t shapes[][2]={{256,1},{4352,33},{5120,65},{6144,33},
                                  {8192,65},{8448,33},{17408,65},{5120,64}};
        const float scales[]={1.0f,0.5f,-1.0f};int cases=0;
        for (int shape=0;shape<8;shape++) for (int pattern=0;pattern<3;pattern++)
        for (int alignment=0;alignment<2;alignment++) for (int si=0;si<3;si++) {
            @autoreleasepool {
                uint32_t ni=shapes[shape][0],no=shapes[shape][1],rb=ni/256u*110u;
                NSUInteger input_offset=alignment ? 4u : 16u;
                float scale=scales[si];
                id<MTLBuffer> weights=[g_device newBufferWithLength:(uint64_t)rb*no options:MTLResourceStorageModeShared];
                id<MTLBuffer> input=[g_device newBufferWithLength:(ni+8u)*4u options:MTLResourceStorageModeShared];
                id<MTLBuffer> output=[g_device newBufferWithLength:(no+10u)*4u options:MTLResourceStorageModeShared];
                id<MTLBuffer> scratch=[g_device newBufferWithLength:(no+10u)*4u options:MTLResourceStorageModeShared];
                if (!weights || !input || !output || !scratch) return 1;
                uint8_t *w=weights.contents;
                for (uint64_t i=0;i<weights.length;i++) w[i]=(uint8_t)(i*37u+19u);
                for (uint32_t row=0;row<no;row++) for (uint32_t block=0;block<ni/256u;block++) {
                    uint16_t d=0x3000u;memcpy(w+(uint64_t)row*rb+block*110u,&d,2);
                }
                float *x=input.contents;
                for (uint32_t i=0;i<ni+8u;i++) x[i]=pattern==1 ? (i&1u ? -0.0f : 0.0f) :
                    pattern==2 ? (float)((int)(i%7u)-3)*(i%5u==0u ? 256.0f : 0x1p-12f) :
                    (float)((int)((i*37u)%101u)-50)/193.0f;
                NSData *wi=[NSData dataWithBytes:w length:weights.length];
                NSData *xi=[NSData dataWithBytes:x length:input.length],*reference=nil;
                for (int phase=0;phase<2;phase++) {
                    float *y=output.contents,*z=scratch.contents;
                    for (uint32_t i=0;i<no+10u;i++) {
                        z[i]=123456.0f;
                        y[i]=(i<5u || i>=no+5u) ? 123456.0f :
                            pattern==1 ? (i&1u ? -0.0f : 0.0f) : (float)((int)(i%13u)-6)/17.0f;
                    }
                    struct {uint32_t ni,no,rb;} args={ni,no,rb};
                    id<MTLCommandBuffer> cb=[g_queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
                    [enc setComputePipelineState:iq3_add_pipeline[phase ? add_variant : 0]];
                    [enc setBytes:&args length:sizeof(args) atIndex:0];
                    [enc setBuffer:weights offset:0 atIndex:1];[enc setBuffer:input offset:input_offset atIndex:2];
                    [enc setBuffer:(phase ? output : scratch) offset:20 atIndex:3];[enc setBuffer:grid offset:0 atIndex:4];
                    if (phase) [enc setBytes:&scale length:sizeof(scale) atIndex:5];
                    [enc setThreadgroupMemoryLength:128u atIndex:0];
                    [enc dispatchThreadgroups:MTLSizeMake((no+7u)/8u+1u,1,1) threadsPerThreadgroup:MTLSizeMake(64u,1,1)];
                    [enc endEncoding];
                    if (!phase) {
                        struct {uint32_t n;float scale;} add_args={no,scale};
                        enc=[cb computeCommandEncoder];[enc setComputePipelineState:g_scale_x1_add_x0_pipeline];
                        [enc setBytes:&add_args length:sizeof(add_args) atIndex:0];
                        [enc setBuffer:output offset:20 atIndex:1];[enc setBuffer:scratch offset:20 atIndex:2];
                        [enc dispatchThreads:MTLSizeMake(no,1,1) threadsPerThreadgroup:MTLSizeMake(64u,1,1)];
                        [enc endEncoding];
                    }
                    [cb commit];[cb waitUntilCompleted];
                    if (cb.status!=MTLCommandBufferStatusCompleted || memcmp(wi.bytes,w,weights.length) ||
                        memcmp(xi.bytes,x,input.length)) return 1;
                    for (uint32_t i=0;i<no+10u;i++) {
                        if (!isfinite(y[i]) || ((i<5u || i>=no+5u) && y[i]!=123456.0f)) return 1;
                        if (!phase && (!isfinite(z[i]) || ((i<5u || i>=no+5u) && z[i]!=123456.0f))) return 1;
                        if (phase && z[i]!=123456.0f) return 1;
                    }
                    if (!phase) reference=[NSData dataWithBytes:y length:output.length];
                    else if (memcmp(reference.bytes,y,output.length)) {
                        const float *a=reference.bytes;
                        for (uint32_t i=0;i<no+10u;i++) if (memcmp(a+i,y+i,4)) {
                            uint32_t bits[2];memcpy(bits,a+i,4);memcpy(bits+1,y+i,4);
                            fprintf(stderr,"IQ3 add mismatch variant=%d n_in=%u n_out=%u pattern=%d scale=%.1f input_offset=%lu index=%u bits=%08x/%08x\n",add_variant,ni,no,pattern,scale,(unsigned long)input_offset,i,bits[0],bits[1]);break;
                        }
                        return 1;
                    }
                }
                cases++;printf("matrix_exact n_in=%u n_out=%u pattern=%d scale=%.1f input_offset=%lu guards=exact input=unchanged weights=unchanged\n",ni,no,pattern,scale,(unsigned long)input_offset);
            }
        }
        printf("PASS iq3_add=%d matrix_cases=%d finite_bit_exact=all\n",add_variant,cases);
        return cases==144 ? 0 : 1;
    }
}
