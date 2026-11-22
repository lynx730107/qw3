/* Greedy generation diagnostic for the 27B full-Metal path. */
#include "qw3.c"

int probe_tail_copy(qw3_metal_session *, uint32_t, uint32_t, int);
enum { KEY = 12, WIDTH = 8, LIMIT = 96 };

static int lookup(qw3_session *s, int token, int draft[WIDTH]) {
    const int n=s->tokens.len, *h=s->tokens.v;
    if (n < KEY) return 0;
    draft[0]=token;
    for (int j=n-KEY-(WIDTH-1); j>=0; j--) {
        if (h[j+KEY-1]!=token || memcmp(h+j,h+n-KEY+1,(KEY-1)*sizeof(int))) continue;
        memcpy(draft+1,h+j+KEY,(WIDTH-1)*sizeof(int));
        for (int i=1;i<WIDTH;i++) if (draft[i]==qw3_token_eos(s->engine)) return 0;
        return 1;
    }
    return 0;
}

static int project(qw3_session *s, int row) {
    qw3_engine *e=s->engine;
    const uint32_t n=e->model.spec->n_embd;
    return qw3_metal_session_copy_batch_x0_to_x0(s->metal,row,n) &&
           qw3_metal_session_rmsnorm_weight_f32(s->metal,e->weights.output_norm->offset,n,QW3_RMS_EPS,NULL) &&
           qw3_metal_session_matvec_q5_k_x1_to_logits(s->metal,e->weights.output->offset,n,e->model.spec->n_vocab,NULL);
}

static int restore(qw3_session *s, int pos) {
    if (!probe_tail_copy(s->metal,pos,WIDTH,1)) return 0;
    s->tokens.len=pos;
    s->kv.pos=pos;
    s->valid=true;
    s->cpu_logits_valid=false;
    return 1;
}

static int same_payload(FILE *a, FILE *b) {
    unsigned char left[16384],right[16384];
    rewind(a); rewind(b);
    for (;;) {
        size_t n=fread(left,1,sizeof(left),a);
        if (fread(right,1,sizeof(right),b)!=n || memcmp(left,right,n)) return 0;
        if (n==0) return feof(a) && feof(b);
    }
}

int main(int argc,char **argv) {
    if (argc!=3) return 2;
    int limit=atoi(argv[2]),ids[8192],n=0;
    if (limit<WIDTH || limit>LIMIT) return 2;
    while(n<8192 && scanf("%d",&ids[n])==1) n++;
    if (n<KEY || n+limit>=8192) return 2;
    qw3_engine_options opt={.model_path=argv[1],.backend=QW3_BACKEND_METAL,.n_threads=4};
    qw3_engine *e;
    qw3_session *s;
    char err[256]={0};
    if (qw3_engine_open(&e,&opt) || qw3_session_create(&s,e,8192)) return 1;
    qw3_tokens prompt={.v=ids,.len=n,.cap=n};
    if(qw3_session_sync(s,&prompt,err,sizeof(err))) goto failed;
    FILE *checkpoint=tmpfile(), *restored=tmpfile();
    uint64_t bytes=qw3_session_payload_bytes(s);
    if (!checkpoint || !restored || qw3_session_save_payload(s,checkpoint,err,sizeof(err))) goto failed;
    if (!probe_tail_copy(s->metal,n,WIDTH,0)) goto failed;
    int wrong[WIDTH]={42,73,12,44,48,11,62,27};
    if (qw3_metal_session_eval_dense_prefill_batch(s,wrong,WIDTH,err,sizeof(err),QW3_METAL_LOGITS_GPU) ||
        !restore(s,n) || qw3_session_save_payload(s,restored,err,sizeof(err)) ||
        !same_payload(checkpoint,restored)) goto failed;
    if (probe_tail_copy(s->metal,8192-WIDTH,WIDTH,0)) goto failed;
    fclose(restored);
    printf("rollback_payload_equal=1 bytes=%llu high_context_declined=1\n",(unsigned long long)bytes);
    int oracle[LIMIT],oracle_len=0;
    for (int trial=0;trial<4;trial++) {
        rewind(checkpoint);
        if(qw3_session_load_payload(s,checkpoint,bytes,err,sizeof(err))) goto failed;
        int on=trial==1 || trial==2;
        int count=0,attempts=0,accepted=0,failed_blocks=0,output[LIMIT];
        double saves=0,loads=0,start=qw3_now_sec();
        while(count<limit) {
            int anchor=qw3_session_argmax(s),draft[WIDTH],pos=s->tokens.len;
            if(anchor<0) goto failed;
            if(anchor==qw3_token_eos(e)) break;
            if(!on || count+WIDTH>limit || !lookup(s,anchor,draft)) {
                if(qw3_session_eval_gpu_logits(s,anchor,err,sizeof(err))) goto failed;
                output[count++]=anchor;
                continue;
            }
            double t=qw3_now_sec();
            if(!probe_tail_copy(s->metal,pos,WIDTH,0)) goto failed;
            saves+=qw3_now_sec()-t;
            attempts++;
            if(qw3_metal_session_eval_dense_prefill_batch(s,draft,WIDTH,err,sizeof(err),QW3_METAL_LOGITS_DEFER)) goto failed;
            int keep=1;
            while(keep<WIDTH) {
                if(!project(s,keep-1)) goto failed;
                int actual=qw3_session_argmax(s);
                if(actual<0) goto failed;
                if(actual!=draft[keep]) break;
                keep++;
            }
            accepted+=keep-1;
            if(keep==WIDTH) {
                if(!project(s,WIDTH-1)) goto failed;
                s->cpu_logits_valid=false;
            } else {
                failed_blocks++;
                t=qw3_now_sec();
                if(!restore(s,pos)) goto failed;
                loads+=qw3_now_sec()-t;
                for(int j=0;j<keep;j++)
                    if(qw3_session_eval_gpu_logits(s,draft[j],err,sizeof(err))) goto failed;
            }
            memcpy(output+count,draft,keep*sizeof(int));
            count+=keep;
        }
        double elapsed=qw3_now_sec()-start;
        if(!trial){oracle_len=count;memcpy(oracle,output,count*sizeof(int));}
        int same=count==oracle_len && !memcmp(oracle,output,count*sizeof(int));
        printf("trial=%d on=%d output=%d same=%d attempts=%d accepted=%d partial=%d save_ms=%.2f restore_ms=%.2f ms=%.2f tps=%.3f\n",
               trial,on,count,same,attempts,accepted,failed_blocks,1000*saves,1000*loads,1000*elapsed,count/elapsed);
        fflush(stdout);
    }
    fclose(checkpoint);
    qw3_session_free(s); qw3_engine_close(e);
    return 0;
failed:
    fprintf(stderr,"spec probe failed: %s\n",err);
    return 1;
}
