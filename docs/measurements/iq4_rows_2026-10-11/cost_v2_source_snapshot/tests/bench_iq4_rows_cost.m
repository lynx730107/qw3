/* Diagnostic real decode shapes; GPU replay times are not model throughput. */
#define main iq4_rows_unused_matrix_main
#include ".iq4_rows/quality.inc"
#undef main
static int rows_cost_encode(id<MTLCommandBuffer> cb,int variant,int kind,
    id<MTLBuffer> __strong *weights,uint64_t *offset,id<MTLBuffer> input,id<MTLBuffer> output,
    uint32_t ni,uint32_t no) {
    struct {uint32_t ni,no,rb;} args={ni,no,ni/256u*136u};const float scale=-0.25f;
    id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
    if (!enc) return 0;
    [enc setComputePipelineState:iq4_rows_pipeline[variant][kind]];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    if (kind==2) [enc setBytes:&scale length:sizeof(scale) atIndex:1];
    [enc setBuffer:weights[0] offset:offset[0] atIndex:kind==2 ? 2 : 1];
    if (kind==1) [enc setBuffer:weights[1] offset:offset[1] atIndex:2];
    [enc setBuffer:input offset:0 atIndex:kind ? 3 : 2];
    [enc setBuffer:output offset:0 atIndex:kind ? 4 : 3];
    [enc setThreadgroupMemoryLength:128 atIndex:0];
    uint32_t columns=variant==0 ? 4u : variant==1 ? 8u : 16u;
    for (int replay=0;replay<8;replay++) {
        [enc dispatchThreadgroups:MTLSizeMake((no+columns-1u)/columns,1,1) threadsPerThreadgroup:MTLSizeMake(64u,1,1)];
        if (replay!=7) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    [enc endEncoding];return 1;
}
int main(int argc,char **argv) {
    @autoreleasepool {
        if (argc!=2) return 2;
        qw3_engine *engine=NULL;int rc=1;NSMutableArray *results=nil;NSData *json=nil;
        qw3_engine_options opt={.model_path=argv[1],.backend=QW3_BACKEND_METAL};
        if (qw3_engine_open(&engine,&opt) || !iq4_rows_compile()) goto done;
        const uint32_t cases[][4]={{5120,10240,0,18},{5120,12288,0,3},{5120,17408,0,9},
            {5120,6144,0,26},{6144,5120,0,40},{5120,17408,1,14},{17408,5120,2,32}};
        results=[NSMutableArray new];
        for (int c=0;c<7;c++) {
            uint32_t ni=cases[c][0],no=cases[c][1],kind=cases[c][2];
            const qw3_tensor *w[2]={NULL,NULL};
            if (kind==1) {
                for (int l=0;l<qw3_engine_layer_count(engine);l++) {
                    const qw3_layer_weights *lw=&engine->weights.layer[l];
                    if (lw->ffn_gate->type==23u && lw->ffn_up->type==23u) {w[0]=lw->ffn_gate;w[1]=lw->ffn_up;break;}
                }
            } else for (uint32_t t=0;t<engine->model.n_tensors;t++) {
                const qw3_tensor *candidate=&engine->model.tensors[t];
                if (candidate->type==23u && candidate->ndim==2u && candidate->dim[0]==ni && candidate->dim[1]==no) {
                    w[0]=w[1]=candidate;break;
                }
            }
            if (!w[0] || !w[1]) goto done;
            uint64_t offsets[2];id<MTLBuffer> weights[2];
            for (int i=0;i<2;i++) {weights[i]=qw3_metal_model_view_for(w[i]->offset,(uint64_t)ni/256u*136u*no,&offsets[i]);if (!weights[i]) goto done;}
            id<MTLBuffer> input=[g_device newBufferWithLength:ni*4u options:MTLResourceStorageModeShared];
            size_t bytes=(kind==1 ? 2u : 1u)*no*4u;
            id<MTLBuffer> output=[g_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            if (!input || !output) goto done;
            float *x=input.contents;for (uint32_t i=0;i<ni;i++) x[i]=(float)((int)((i*37u)%101u)-50)/31.0f;
            NSData *before=[NSData dataWithBytes:input.contents length:input.length],*oracle=nil;
            NSMutableArray *blocks=[NSMutableArray new];int order[]={0,1,0,2,0};
            for (int block=0;block<5;block++) {
                int variant=order[block];NSMutableArray *samples=[NSMutableArray new];
                for (int rep=0;rep<21;rep++) {
                    memset(output.contents,0,bytes);id<MTLCommandBuffer> cb=[g_queue commandBuffer];
                    if (!rows_cost_encode(cb,variant,kind,weights,offsets,input,output,ni,no)) goto done;
                    [cb commit];[cb waitUntilCompleted];
                    if (cb.status!=MTLCommandBufferStatusCompleted || memcmp(before.bytes,input.contents,input.length)) goto done;
                    const float *y=output.contents;for (uint32_t i=0;i<bytes/4u;i++) if (!isfinite(y[i])) goto done;
                    if (!oracle) oracle=[NSData dataWithBytes:output.contents length:bytes];
                    if (memcmp(oracle.bytes,output.contents,bytes)) goto done;
                    if (rep>=16) {
                        double ms=(cb.GPUEndTime-cb.GPUStartTime)*1000/8;
                        if (!isfinite(ms) || ms<=0) goto done;[samples addObject:@(ms)];
                    }
                }
                [blocks addObject:@{@"variant":@(variant),@"gpu_ms":samples}];
            }
            NSString *name=[[NSString alloc] initWithBytes:w[0]->name length:w[0]->name_len encoding:NSUTF8StringEncoding];
            [results addObject:@{@"n_in":@(ni),@"n_out":@(no),@"kind":@(kind),@"calls_per_decode_step":@(cases[c][3]),@"tensor":name,@"blocks":blocks}];
        }
        json=[NSJSONSerialization dataWithJSONObject:@{@"scope":@"Isolated actual decode shapes, eight ordered GPU replays per command. Sixteen warmup commands then five timed samples per block. Includes replay barriers; not model throughput or native stage percentages.",@"matrices":results} options:NSJSONWritingPrettyPrinted error:nil];
        printf("%s\n",[[NSString alloc] initWithData:json encoding:NSUTF8StringEncoding].UTF8String);
        fprintf(stderr,"PASS iq4_rows_cost matrices=7 every_replay_buffer=finite_bit_exact input=unchanged\n");rc=0;
done:
        if (rc) fprintf(stderr,"IQ4 row cost failed\n");
        if (engine) qw3_engine_close(engine);return rc;
    }
}
