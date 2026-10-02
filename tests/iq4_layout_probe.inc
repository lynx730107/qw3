/* Private IQ4_XS storage experiments. Included after the native Metal bridge.
 * Preserve all 136 bytes/block and the original lane/dot/reduction order.
 * Layouts use compact final row tiles: no padding or format expansion. */
static NSString *lut_extract(NSString *, NSString *, NSString *);
static int lut_replace(NSMutableString *, NSString *, NSString *, NSUInteger);
static id<MTLComputePipelineState> layout_pipeline[5][3];
static NSMutableDictionary<NSNumber *, id<MTLBuffer>> *layout_weights;
static NSMutableDictionary<NSNumber *, NSArray<NSNumber *> *> *layout_shapes;
static int layout_selected = -1;
static uint64_t layout_bytes, layout_calls;
static const uint32_t layout_rows[5] = {4,16,4,16,1};
static const char *layout_names[5] = {"blocks4","blocks16","planes4","planes16","native_copy"};

static int layout_initialize(void) {
    NSString *all=qw3_metal_kernel_source();
    NSString *table=lut_extract(all,@"constant float qw3_iq4nl_table[16]",@"\n};");
    NSString *value=lut_extract(all,@"inline float qw3_iq4nl_val(",@"\n}\n");
    NSString *helper=lut_extract(all,@"inline float qw3_iq4xs_dot8_vector(",@"\n}\n");
    const char *names[]={"qw3_matvec_iq4_xs_fast","qw3_matvec_iq4_xs_pair_fast","qw3_matvec_iq4_xs_add_x0_fast"};
    if (!table || !value || !helper) return 0;
    for (int variant=0;variant<4;variant++) {
        NSMutableString *source=[@"#include <metal_stdlib>\nusing namespace metal;\nstruct qw3_matvec_q8_0_args {uint n_in,n_out,row_bytes;};\n" mutableCopy];
        [source appendString:table];[source appendString:@"\n"];[source appendString:value];
        NSMutableString *dot=[helper mutableCopy];
        NSString *signature=@"device const uchar *blk,";
        NSString *replacement=@"device const uchar *weights, constant qw3_matvec_q8_0_args &args, uint row, uint b,";
        NSString *address=[NSString stringWithFormat:
            @"\n    const uint tile = row / %uu * %uu;\n    const uint count = min(%uu, args.n_out - tile);\n    const uint local = row - tile;\n    device const uchar *base = weights + uint64_t(tile) * args.row_bytes;\n    device const uchar *blk = base + (uint64_t(b) * count + local) * %uull;\n",
            layout_rows[variant],layout_rows[variant],layout_rows[variant],variant<2 ? 136 : 8];
        if (!lut_replace(dot,signature,replacement,1) || !lut_replace(dot,@"uint ib, uint il) {",[@"uint ib, uint il) {" stringByAppendingString:address],1)) return 0;
        if (variant>=2 && !lut_replace(dot,
            @"(device const uint *)(blk + 8 + ib * 16u + il * 8u)",
            @"(device const uint *)(base + uint64_t(args.n_in / 256u) * count * 8ull + ((uint64_t(b) * 16ull + ib * 2u + il) * count + local) * 8ull)",1)) return 0;
        [source appendString:dot];
        for (int kind=0;kind<3;kind++) {
            NSString *body=lut_extract(all,[@"kernel void " stringByAppendingString:[NSString stringWithUTF8String:names[kind]]],@"\n}\n");
            if (!body) return 0;
            NSMutableString *kernel=[body mutableCopy];
            if (kind==0) {
                if (!lut_replace(kernel,@"weights +\n                uint64_t(row) * uint64_t(args.row_bytes) + block_off,",@"weights, args, row, b,",2)) return 0;
            } else if (kind==1) {
                if (!lut_replace(kernel,@"gate_weights + off,",@"gate_weights, args, row, b,",2) ||
                    !lut_replace(kernel,@"up_weights + off,",@"up_weights, args, row, b,",2)) return 0;
            } else if (!lut_replace(kernel,@"qw3_iq4xs_dot8_vector(blk,",@"qw3_iq4xs_dot8_vector(weights, args, row, b,",2)) return 0;
            [source appendString:kernel];
        }
        NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
        id<MTLLibrary> library=[g_device newLibraryWithSource:source options:options error:&error];
        for (int kind=0;kind<3;kind++) {
            id<MTLFunction> fn=[library newFunctionWithName:[NSString stringWithUTF8String:names[kind]]];
            layout_pipeline[variant][kind]=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
            if (!layout_pipeline[variant][kind] || layout_pipeline[variant][kind].maxTotalThreadsPerThreadgroup<64) {
                fprintf(stderr,"layout pipeline failed %s/%d: %s\n",layout_names[variant],kind,error ? error.localizedDescription.UTF8String : "");return 0;
            }
        }
    }
    /* Control for mapped-file versus allocated-buffer memory: same native
     * byte layout and original kernels, redirected through the same wrappers. */
    layout_pipeline[4][0]=g_matvec_iq4_xs_pipeline;
    layout_pipeline[4][1]=g_matvec_iq4_xs_pair_pipeline;
    layout_pipeline[4][2]=g_matvec_iq4_xs_add_x0_pipeline;
    return 1;
}

static id<MTLBuffer> layout_pack(uint64_t offset,uint32_t ni,uint32_t no,int variant) {
    if (variant<0 || variant>=5 || !ni || ni%256 || !no) return nil;
    const uint64_t rb=(uint64_t)ni/256*136,bytes=rb*no;
    uint64_t inner=0;
    id<MTLBuffer> native=qw3_metal_model_view_for(offset,bytes,&inner);
    if (!native || bytes>NSUIntegerMax-64) return nil;
    id<MTLBuffer> packed=[g_device newBufferWithLength:bytes+64 options:MTLResourceStorageModeShared];
    if (!packed) return nil;
    const unsigned char *src=(const unsigned char *)native.contents+inner;
    unsigned char *allocation=packed.contents,*dst=allocation+32;
    memset(allocation,0xa5,32);memset(dst+bytes,0xa5,32);
    if (variant==4) {
        memcpy(dst,src,bytes);
        return memcmp(dst,src,bytes) ? nil : packed;
    }
    const uint32_t rows=layout_rows[variant],blocks=ni/256;
    for (uint32_t tile=0;tile<no;tile+=rows) {
        const uint32_t count=MIN(rows,no-tile);
        for (uint32_t b=0;b<blocks;b++) for (uint32_t local=0;local<count;local++) {
            const unsigned char *block=src+(uint64_t)(tile+local)*rb+b*136ull;
            if (variant<2) memcpy(dst+(uint64_t)tile*rb+((uint64_t)b*count+local)*136,block,136);
            else {
                memcpy(dst+(uint64_t)tile*rb+((uint64_t)b*count+local)*8,block,8);
                for (uint32_t it=0;it<16;it++) memcpy(dst+(uint64_t)tile*rb+(uint64_t)blocks*count*8+
                    (((uint64_t)b*16+it)*count+local)*8,block+8+it*8,8);
            }
        }
    }
    /* Check every prepared byte against its source, not just selected rows. */
    for (uint32_t tile=0;tile<no;tile+=rows) {
        const uint32_t count=MIN(rows,no-tile);
        for (uint32_t b=0;b<blocks;b++) for (uint32_t local=0;local<count;local++) {
            const unsigned char *block=src+(uint64_t)(tile+local)*rb+b*136ull;
            if (variant<2) {
                if (memcmp(dst+(uint64_t)tile*rb+((uint64_t)b*count+local)*136,block,136)) return nil;
            } else {
                if (memcmp(dst+(uint64_t)tile*rb+((uint64_t)b*count+local)*8,block,8)) return nil;
                for (uint32_t it=0;it<16;it++) if (memcmp(dst+(uint64_t)tile*rb+(uint64_t)blocks*count*8+
                    (((uint64_t)b*16+it)*count+local)*8,block+8+it*8,8)) return nil;
            }
        }
    }
    return packed;
}
static id<MTLBuffer> layout_weight(uint64_t offset,uint32_t ni,uint32_t no) {
    NSArray<NSNumber *> *shape=layout_shapes[@(offset)];
    id<MTLBuffer> buffer=layout_weights[@(offset)];
    if (!buffer || shape.count!=2 || shape[0].unsignedIntValue!=ni || shape[1].unsignedIntValue!=no) {
        fprintf(stderr,"unprepared layout weight offset=%llu shape=%ux%u\n",(unsigned long long)offset,ni,no);return nil;
    }
    return buffer;
}
static int layout_dispatch(int variant,int kind,id<MTLBuffer> gate,id<MTLBuffer> up,
    uint64_t go,uint64_t uo,id<MTLBuffer> input,uint64_t io,id<MTLBuffer> output,uint64_t oo,
    uint32_t ni,uint32_t no,float scale,double *gpu) {
    if (variant<0 || variant>=5 || kind<0 || kind>=3 || !gate || !input || !output ||
        !ni || ni%256 || !no || (kind==1 && !up)) return 0;
    struct {uint32_t ni,no,rb;} args={ni,no,ni/256*136};
    int owned=0;id<MTLCommandBuffer> cb=qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc=qw3_metal_compute_encoder(cb);
    if (!cb || !enc) return 0;
    [enc setComputePipelineState:layout_pipeline[variant][kind]];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    if (kind==2) [enc setBytes:&scale length:sizeof(scale) atIndex:1];
    [enc setBuffer:gate offset:go atIndex:kind==2 ? 2 : 1];
    if (kind==1) [enc setBuffer:up offset:uo atIndex:2];
    [enc setBuffer:input offset:io atIndex:kind ? 3 : 2];
    [enc setBuffer:output offset:oo atIndex:kind ? 4 : 3];
    [enc setThreadgroupMemoryLength:128 atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((no+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
    qw3_metal_end_compute_encoder(cb,enc);
    if (!qw3_metal_finish_command_buffer(cb,owned,"IQ4 layout") || cb.status==MTLCommandBufferStatusError) return 0;
    if (gpu) *gpu=cb.GPUEndTime-cb.GPUStartTime;
    return 1;
}
static int layout_x1(qw3_metal_session *s,uint64_t offset,uint32_t type,uint32_t ni,uint32_t no,uint32_t oo,float *out) {
    /* Only prepared dense FFN tensors participate; DeltaNet projections
     * sharing this entry point remain on the original mapped layout. */
    if (layout_selected<0 || type!=23 || !layout_shapes[@(offset)]) return qw3_metal_session_matvec_quant_x1_to_scratch(s,offset,type,ni,no,oo,out);
    if (!s || !s->obj || out) return 0; /* full benchmark requests GPU logits */
    QW3MetalSessionObj *obj=(__bridge QW3MetalSessionObj *)s->obj;
    if (obj.x1.length<(uint64_t)ni*4 || obj.scratch.length<((uint64_t)oo+no)*4) return 0;
    layout_calls++;
    return layout_dispatch(layout_selected,0,layout_weight(offset,ni,no),nil,32,0,obj.x1,0,obj.scratch,(uint64_t)oo*4,ni,no,1,NULL);
}
static int layout_pair(qw3_metal_session *s,uint64_t gate,uint64_t up,uint32_t ni,uint32_t no) {
    if (layout_selected<0) return qw3_metal_session_matvec_iq4_xs_pair_x1_to_scratch(s,gate,up,ni,no);
    if (!s || !s->obj) return 0;
    QW3MetalSessionObj *obj=(__bridge QW3MetalSessionObj *)s->obj;
    if (obj.x1.length<(uint64_t)ni*4 || obj.scratch.length<(uint64_t)no*8) return 0;
    layout_calls++;
    return layout_dispatch(layout_selected,1,layout_weight(gate,ni,no),layout_weight(up,ni,no),32,32,obj.x1,0,obj.scratch,0,ni,no,1,NULL);
}
static int layout_down(qw3_metal_session *s,uint64_t offset,uint32_t ni,uint32_t no) {
    if (layout_selected<0) return qw3_metal_session_matvec_iq4_xs_inner_add_x0(s,offset,ni,no);
    if (!s || !s->obj) return 0;
    QW3MetalSessionObj *obj=(__bridge QW3MetalSessionObj *)s->obj;
    if (obj.inner.length<(uint64_t)ni*4 || obj.x0.length<(uint64_t)no*4) return 0;
    layout_calls++;
    return layout_dispatch(layout_selected,2,layout_weight(offset,ni,no),nil,32,0,obj.inner,0,obj.x0,0,ni,no,1,NULL);
}
/* Redirect only core call sites; the native bridge remains unchanged and
 * prefill matrix dispatches still read their original mapped GGUF tensors. */
#define qw3_metal_session_matvec_quant_x1_to_scratch layout_x1
#define qw3_metal_session_matvec_iq4_xs_pair_x1_to_scratch layout_pair
#define qw3_metal_session_matvec_iq4_xs_inner_add_x0 layout_down
