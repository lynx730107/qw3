#include "kstage_candidates.inc"
static int stage_encode(QW3MetalSessionObj *obj,uint32_t type,id<MTLCommandBuffer> cb,int candidate,
    uint64_t offset,uint32_t nt,uint32_t ni,uint32_t no,id<MTLBuffer> src,id<MTLBuffer> dst,
    uint32_t io,uint32_t is,uint32_t oo,uint32_t os) {
    if (candidate && (!nt || nt%128u)) return 0;
    scratch_select(candidate ? (type==21u ? 1 : 2) : 0);
    return qw3_metal_encode_dense_batch_matmul_strided(obj,cb,offset,type,nt,ni,no,src,dst,io,is,oo,os);
}
#include ".kstage_candidates/matrix.inc"
static int stage_micro(QW3MetalSessionObj *obj,const qw3_tensor *w) {
    const uint32_t nt=128,ni=w->dim[0],no=w->dim[1];
    const int reps=5,replays=8,order[]={0,1,0,1,0};
    id<MTLBuffer> src=[g_device newBufferWithLength:(size_t)nt*ni*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> dst=[g_device newBufferWithLength:(size_t)nt*no*4 options:MTLResourceStorageModeShared];
    if (!src || !dst) return 0;
    float *x=src.contents,*oracle=malloc((size_t)nt*no*4);if (!oracle) return 0;
    uint32_t rng=1;
    for (size_t i=0;i<(size_t)nt*ni;i++) {rng=rng*1664525u+1013904223u;x[i]=(float)((int)(rng>>16u)-32768)/8192.0f;}
    NSData *before=[NSData dataWithBytes:src.contents length:src.length];int ok=1;
    for (int block=0;ok && block<5;block++) for (int rep=0;ok && rep<=reps;rep++) {
        id<MTLCommandBuffer> cb=[g_queue commandBuffer];
        for (int replay=0;ok && replay<replays;replay++)
            ok=stage_encode(obj,w->type,cb,order[block],w->offset,nt,ni,no,src,dst,0,ni,0,no);
        if (!ok) break;
        [cb commit];[cb waitUntilCompleted];
        if (cb.status!=MTLCommandBufferStatusCompleted || memcmp(before.bytes,src.contents,src.length)) {ok=0;break;}
        if (!block && !rep) memcpy(oracle,dst.contents,(size_t)nt*no*4);
        const float *result=dst.contents;
        for (size_t i=0;ok && i<(size_t)nt*no;i++) if (!isfinite(result[i])) ok=0;
        if (memcmp(oracle,dst.contents,(size_t)nt*no*4)) {ok=0;break;}
        double ms=(cb.GPUEndTime-cb.GPUStartTime)*1000/replays;
        if (!(ms>0) || !isfinite(ms)) {ok=0;break;}
        if (rep) printf("MICRO stage_k=%d type=%u n_in=%u n_out=%u block=%d candidate=%d rep=%d gpu_ms=%.9f\n",
            stage_width,w->type,ni,no,block,order[block],rep,ms);
    }
    free(oracle);return ok;
}
int main(int argc,char **argv) {
    @autoreleasepool {
        if (argc!=4 || (strcmp(argv[3],"quality") && strcmp(argv[3],"micro"))) return 2;
        stage_width=atoi(argv[2]);if (stage_width!=64 && stage_width!=128) return 2;
        qw3_engine *e=NULL;qw3_session *s=NULL;int rc=1,cases=0;
        QW3MetalSessionObj *obj=nil;
        const qw3_tensor *weights[4]={NULL,NULL,NULL,NULL};
        qw3_engine_options opt={.model_path=argv[1],.backend=QW3_BACKEND_METAL,.n_threads=4};
        setenv("QW3_METAL_NGL","64",1);setenv("QW3_METAL_DENSE_PREFILL_FAST","1",1);
        if (qw3_engine_open(&e,&opt) || !stage_compile() || qw3_session_create(&s,e,512)) goto done;
        obj=(__bridge QW3MetalSessionObj *)s->metal->obj;
        if (!qw3_metal_dense_batch_prepare_map(obj,256)) goto done;
        for (int l=0;l<qw3_engine_layer_count(e);l++) {
            const qw3_tensor *w[]={e->weights.layer[l].ffn_gate,e->weights.layer[l].ffn_down};
            for (int j=0;j<2;j++) if (w[j] && (w[j]->type==21u || w[j]->type==23u)) {
                int i=(w[j]->type==23u ? 2 : 0)+j;
                if (!weights[i]) weights[i]=w[j];
            }
        }
        for (int i=0;i<4;i++) {
            const qw3_tensor *w=weights[i];if (!w) goto done;
            if (!strcmp(argv[3],"micro")) { if (!stage_micro(obj,w)) goto done;cases++; }
            else {
                uint32_t nt[]={128,256},no[]={1,33,65,(uint32_t)w->dim[1]};
                for (int n=0;n<2;n++) for (int r=0;r<4;r++) {
                    if (!stage_quality(obj,w,nt[n],no[r])) goto done;cases++;
                }
            }
        }
        printf("PASS stage_k=%d mode=%s cases=%d iq3=%llu iq4=%llu\n",stage_width,argv[3],cases,
            (unsigned long long)stage_iq3_calls,(unsigned long long)stage_iq4_calls);rc=0;
done:
        if (rc) fprintf(stderr,"K-stage failed after %d cases\n",cases);
        if (s) qw3_session_free(s);if (e) qw3_engine_close(e);return rc;
    }
}
