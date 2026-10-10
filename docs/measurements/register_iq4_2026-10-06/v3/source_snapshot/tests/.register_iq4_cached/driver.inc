#include "adapter.inc"

static int register_quality(QW3MetalSessionObj *obj,const qw3_tensor *w,uint32_t nt,uint32_t no) {
    const uint32_t ni=(uint32_t)w->dim[0],io=3,is=ni+9,oo=5,os=no+13;
    size_t in_bytes=(size_t)nt*is*4,out_bytes=((size_t)nt*os+8)*4;
    id<MTLBuffer> input=[g_device newBufferWithLength:in_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> out[2]={[g_device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared],
        [g_device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared]};
    if (!input || !out[0] || !out[1]) return 0;
    memset(input.contents,0x5a,in_bytes);uint32_t rng=1;float *x=input.contents;
    for (uint32_t t=0;t<nt;t++) for (uint32_t i=0;i<ni;i++) {
        rng=rng*1664525u+1013904223u;
        x[(size_t)t*is+io+i]=(float)((int)(rng>>16u)-32768)/8192.0f;
    }
    NSData *before=[NSData dataWithBytes:input.contents length:in_bytes];
    for (int v=0;v<2;v++) {
        memset(out[v].contents,0x5a,out_bytes);
        id<MTLCommandBuffer> cb=[g_queue commandBuffer];
        if (!register_iq4_encode(obj,cb,v,w->offset,nt,ni,no,input,out[v],io,is,oo,os)) return 0;
        [cb commit];[cb waitUntilCompleted];
        if (cb.status!=MTLCommandBufferStatusCompleted || memcmp(before.bytes,input.contents,in_bytes)) return 0;
    }
    const float *a=out[0].contents,*b=out[1].contents;
    for (size_t j=0;j<out_bytes/4;j++) {
        if (j<(size_t)nt*os && j%os>=oo && j%os<oo+no) {
            if (!isfinite(a[j]) || !isfinite(b[j]) || memcmp(a+j,b+j,4)) {
                fprintf(stderr,"register mismatch tokens=%u n_in=%u n_out=%u index=%zu reference=%.9g candidate=%.9g\n",nt,ni,no,j,a[j],b[j]);return 0;
            }
        } else {
            uint32_t bits[2];memcpy(bits,a+j,4);memcpy(bits+1,b+j,4);
            if (bits[0]!=0x5a5a5a5au || bits[1]!=0x5a5a5a5au) {
                fprintf(stderr,"register output guard overwritten index=%zu\n",j);return 0;
            }
        }
    }
    id<MTLCommandBuffer> invalid=[g_queue commandBuffer];
    if (register_iq4_encode(obj,invalid,1,w->offset,nt,ni,no,input,out[1],io,ni+io-1,oo,os) ||
        register_iq4_encode(obj,invalid,1,w->offset,nt,ni,no,input,out[1],io,is,os+1,os) ||
        register_iq4_encode(obj,invalid,1,w->offset,nt+1,ni,no,input,out[1],io,is,oo,os)) return 0;
    printf("register_matrix_exact tokens=%u n_in=%u n_out=%u finite=all guards=exact input=unchanged invalid_layout=rejected\n",nt,ni,no);
    fflush(stdout);return 1;
}

static int register_bench(QW3MetalSessionObj *obj,const qw3_tensor *w,int cooldown) {
    const uint32_t nt=128,ni=(uint32_t)w->dim[0],no=(uint32_t)w->dim[1];
    const int replays=8,reps=5,order[]={0,1,0,1,0};
    id<MTLBuffer> input=[g_device newBufferWithLength:(size_t)nt*ni*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> output=[g_device newBufferWithLength:(size_t)nt*no*4 options:MTLResourceStorageModeShared];
    if (!input || !output) return 0;
    uint32_t rng=1;float *x=input.contents;
    for (size_t i=0;i<(size_t)nt*ni;i++) {rng=rng*1664525u+1013904223u;x[i]=(float)((int)(rng>>16u)-32768)/8192.0f;}
    NSData *before=[NSData dataWithBytes:input.contents length:input.length];
    float *oracle=malloc((size_t)nt*no*4);if (!oracle) return 0;
    NSMutableArray *blocks=[NSMutableArray new];int ok=1;
    for (int block=0;ok && block<5;block++) {
        int mode=order[block];NSMutableArray *samples=[NSMutableArray new];
        fprintf(stderr,"REGISTER_BEGIN block=%d mode=%s cooldown=%d\n",block,mode ? "register" : "off",cooldown);fflush(stderr);
        sleep((unsigned)cooldown);
        for (int rep=0;ok && rep<=reps;rep++) {
            id<MTLCommandBuffer> cb=[g_queue commandBuffer];
            for (int replay=0;replay<replays;replay++)
                if (!register_iq4_encode(obj,cb,mode,w->offset,nt,ni,no,input,output,0,ni,0,no)) {ok=0;break;}
            if (!ok) break;
            [cb commit];[cb waitUntilCompleted];
            if (cb.status!=MTLCommandBufferStatusCompleted || memcmp(before.bytes,input.contents,input.length)) {ok=0;break;}
            if (!block && !rep) memcpy(oracle,output.contents,(size_t)nt*no*4);
            if (memcmp(oracle,output.contents,(size_t)nt*no*4)) {fprintf(stderr,"register replay output mismatch\n");ok=0;break;}
            if (rep) {
                double ms=(cb.GPUEndTime-cb.GPUStartTime)*1000/replays;
                if (!(ms>0) || !isfinite(ms)) {ok=0;break;}
                [samples addObject:@(ms)];fprintf(stderr,"REGISTER_SAMPLE block=%d mode=%s repetition=%d gpu_ms=%.9f\n",block,mode ? "register" : "off",rep,ms);fflush(stderr);
            }
        }
        [blocks addObject:@{@"block":@(block),@"mode":mode ? @"register" : @"off",@"gpu_ms":samples}];
    }
    if (ok) {
        NSString *name=[[NSString alloc] initWithBytes:w->name length:w->name_len encoding:NSUTF8StringEncoding];
        NSDictionary *data=@{@"scope":@"isolated IQ4_XS matrix; GPU completed work; not end-to-end model throughput",@"tensor":name,
            @"offset":@(w->offset),@"n_in":@(ni),@"n_out":@(no),@"tokens":@(nt),@"replays_per_command":@(replays),
            @"warmup_commands_per_block":@1,@"repetitions":@(reps),@"cooldown_seconds":@(cooldown),@"blocks":blocks};
        NSData *j=[NSJSONSerialization dataWithJSONObject:data options:NSJSONWritingPrettyPrinted error:nil];
        printf("REGISTER_JSON\n%s\n",[[NSString alloc] initWithData:j encoding:NSUTF8StringEncoding].UTF8String);fflush(stdout);
    }
    free(oracle);return ok;
}

int main(int argc,char **argv) {
    @autoreleasepool {
        if (argc<3 || argc>4 || (strcmp(argv[2],"quality") && strcmp(argv[2],"bench"))) return 2;
        const int bench=!strcmp(argv[2],"bench"),cooldown=argc==4 ? atoi(argv[3]) : 60;
        if (cooldown<0 || cooldown>60) return 2;
        qw3_engine *engine=NULL;qw3_session *session=NULL;int rc=1,cases=0;
        QW3MetalSessionObj *obj=nil;
        const qw3_tensor *down=NULL,*gate=NULL;
        qw3_engine_options options={.model_path=argv[1],.backend=QW3_BACKEND_METAL,.n_threads=4};
        setenv("QW3_METAL_DENSE_PREFILL_FAST","1",1);setenv("QW3_METAL_NGL","64",1);
        if (qw3_engine_open(&engine,&options) || !register_iq4_compile() || qw3_session_create(&session,engine,512)) goto done;
        obj=(__bridge QW3MetalSessionObj *)session->metal->obj;
        if (!qw3_metal_dense_batch_prepare_map(obj,256)) goto done;
        for (int l=0;l<qw3_engine_layer_count(engine);l++) {
            qw3_layer_weights *w=&engine->weights.layer[l];
            if (!down && w->ffn_down && w->ffn_down->type==23u) down=w->ffn_down;
            if (!gate && w->ffn_gate && w->ffn_gate->type==23u) gate=w->ffn_gate;
        }
        if (!down || !gate) goto done;
        if (bench) {if (!register_bench(obj,down,cooldown)) goto done;}
        else {
            const qw3_tensor *weights[]={down,gate};const uint32_t tokens[]={128,256};
            for (int w=0;w<2;w++) {
                const uint32_t rows[]={1,33,65,(uint32_t)weights[w]->dim[1]};
                for (int n=0;n<2;n++) for (int k=0;k<4;k++) {
                    if (!register_quality(obj,weights[w],tokens[n],rows[k])) goto done;
                    cases++;
                }
            }
            printf("PASS register_iq4 quality_cases=%d threads=4 ctx512 kv=f16 ngl64\n",cases);
        }
        rc=0;
done:
        if (rc) fprintf(stderr,"register IQ4 diagnostic failed mode=%s cases=%d\n",argv[2],cases);
        if (session) qw3_session_free(session);if (engine) qw3_engine_close(engine);return rc;
    }
}
