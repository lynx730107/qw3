/* Isolated experiment. Include the local core to inspect each dense batch row. */
#include "qw3.c"

static int read_batch_heads(qw3_session *s, float *rows, int width, int vocab) {
    qw3_engine *engine = s->engine;
    uint32_t embd = engine->model.spec->n_embd;
    for (int t = 0; t < width; t++) {
        if (!qw3_metal_session_copy_batch_x0_to_x0(s->metal, t, embd) ||
            !qw3_metal_session_rmsnorm_weight_f32(s->metal, engine->weights.output_norm->offset,
                                                  embd, QW3_RMS_EPS, NULL) ||
            !qw3_metal_session_matvec_q5_k_x1_to_logits(s->metal, engine->weights.output->offset,
                                                       embd, vocab, rows + (size_t)t * vocab))
            return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    int n, before;
    enum {width = 8};
    if (scanf("%d%d", &n, &before) != 2 || n < 1 || before < 0 || n+before+width >= 8192) return 2;
    int ids[8192];
    for (int i = 0; i < n+before+width; i++) if (scanf("%d",&ids[i]) != 1) return 2;
    qw3_tokens prompt = {.v=ids, .len=n, .cap=n};
    qw3_engine_options opt = {.model_path=argv[1], .backend=QW3_BACKEND_METAL, .n_threads=4};
    qw3_engine *engine;
    qw3_session *s;
    if (qw3_engine_open(&engine,&opt) || qw3_session_create(&s,engine,8192)) return 1;
    char err[512] = {0};
    if (qw3_session_sync(s,&prompt,err,sizeof(err))) goto failed;
    for (int i=0; i<before; i++) if (qw3_session_eval(s,ids[n+i],err,sizeof(err))) goto failed;
    const int *suffix = ids+n+before;
    const int vocab = qw3_vocab_size(engine);
    float *rows[2];
    for (int v=0; v<2; v++) {
        rows[v]=calloc((size_t)vocab*width,sizeof(float));
        if (!rows[v]) return 1;
    }
    FILE *fp=tmpfile();
    uint64_t payload=qw3_session_payload_bytes(s);
    if (!fp || qw3_session_save_payload(s,fp,err,sizeof(err)) || fflush(fp)) goto failed;
    for (int rep=-1; rep<3; rep++) {
        double elapsed[2];
        for (int order=0; order<2; order++) {
            int batch=order ^ ((rep+1)&1);
            rewind(fp);
            if (qw3_session_load_payload(s,fp,payload,err,sizeof(err))) goto failed;
            double start=qw3_now_sec();
            if (batch) {
                if (qw3_metal_session_eval_dense_prefill_batch(s,suffix,width,err,sizeof(err),
                                                               QW3_METAL_LOGITS_DEFER) ||
                    !read_batch_heads(s,rows[1],width,vocab)) goto failed;
            } else {
                for (int t=0;t<width;t++) {
                    if (qw3_session_eval(s,suffix[t],err,sizeof(err)) ||
                        qw3_session_copy_logits(s,rows[0]+(size_t)t*vocab,vocab)!=vocab) goto failed;
                }
            }
            elapsed[batch]=(qw3_now_sec()-start)*1000;
        }
        int unequal=0, same_top=0;
        double maxerr=0, worst_rms=0;
        for (int t=0;t<width;t++) {
            float *a=rows[0]+(size_t)t*vocab, *b=rows[1]+(size_t)t*vocab;
            int top_a=0,top_b=0;
            double ss=0;
            for (int j=0;j<vocab;j++) {
                uint32_t left_bits, right_bits;
                memcpy(&left_bits, a+j, sizeof(left_bits));
                memcpy(&right_bits, b+j, sizeof(right_bits));
                if ((left_bits & 0x7f800000u) == 0x7f800000u ||
                    (right_bits & 0x7f800000u) == 0x7f800000u) return 1;
                unequal+=memcmp(a+j,b+j,sizeof(float))!=0;
                double delta=(double)a[j]-b[j];
                ss+=delta*delta;
                maxerr=fmax(maxerr,fabs(delta));
                if (a[j]>a[top_a]) top_a=j;
                if (b[j]>b[top_b]) top_b=j;
            }
            same_top+=top_a==top_b;
            worst_rms=fmax(worst_rms,sqrt(ss/vocab));
        }
        printf("rep=%d pos=%d serial_ms=%.2f batch_ms=%.2f top1_equal=%d/8 diff_float=%d max_abs=%.7g worst_row_rmse=%.7g\n",
               rep,n+before,elapsed[0],elapsed[1],same_top,unequal,maxerr,worst_rms);
    }
    fclose(fp);
    for (int i=0;i<2;i++) free(rows[i]);
    qw3_session_free(s);
    qw3_engine_close(engine);
    return 0;
failed:
    fprintf(stderr,"block probe failed: %s\n",err);
    return 1;
}
