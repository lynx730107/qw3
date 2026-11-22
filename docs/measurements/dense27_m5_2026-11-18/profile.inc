/* Collect isolated encoder timestamps for identical token offsets at three depths. */
int main(int argc,char **argv) {
    if (argc!=3) return 2;
    const int depth=atoi(argv[2]);
    if (depth!=1024 && depth!=4096 && depth!=8192) return 2;
    qw3_engine *engine=NULL;
    qw3_session *session=NULL;
    qw3_tokens prefix={0}, tokens={0};
    qw3_engine_options opt={.model_path=argv[1],.backend=QW3_BACKEND_METAL,.n_threads=4};
    char error[512]={0};
    int rc=1;
    FILE *checkpoint=NULL, *oracle=NULL;
    float *values=NULL;
    if (qw3_engine_open(&engine,&opt) || !qw3_metal_compile_kernels() || !probe_profile_initialize() ||
        fill_synthetic_tokens(&prefix,depth,qw3_vocab_size(engine),qw3_token_eos(engine),1u) ||
        fill_synthetic_tokens(&tokens,16,qw3_vocab_size(engine),qw3_token_eos(engine),2u) ||
        qw3_session_create(&session,engine,depth+256) || session->metal_n_gpu_layers!=64 ||
        qw3_session_sync(session,&prefix,error,sizeof(error))) goto done;
    checkpoint=tmpfile();
    oracle=tmpfile();
    const int vocab=qw3_vocab_size(engine);
    values=malloc((size_t)vocab*sizeof(float));
    const uint64_t bytes=qw3_session_payload_bytes(session);
    if (!checkpoint || !oracle || !values ||
        qw3_session_save_payload(session,checkpoint,error,sizeof(error))) goto done;
    for (int phase=0;phase<2;phase++) {
        if (phase) {
            qw3_session_invalidate(session);
            if (fseek(checkpoint,0,SEEK_SET) || fseek(oracle,0,SEEK_SET) ||
                qw3_session_load_payload(session,checkpoint,bytes,error,sizeof(error))) goto done;
        }
        for (int i=0;i<16;i++) {
            const int capture=phase && (i==1 || i==9);
            if (capture) probe_profile_start();
            double start=bench_now_sec();
            if (qw3_session_eval_gpu_logits(session,tokens.v[i],error,sizeof(error))) goto done;
            if (capture) probe_profile_end(i,bench_now_sec()-start);
            if (qw3_session_copy_logits(session,values,vocab)!=vocab) goto done;
            for (int j=0;j<vocab;j++) if (!isfinite(values[j])) goto done;
            if (!phase) {
                if (fwrite(values,sizeof(float),vocab,oracle)!=(size_t)vocab) goto done;
            } else {
                float expected[4096];
                for (int j=0;j<vocab;j+=4096) {
                    size_t count=(size_t)MIN(vocab-j,4096);
                    if (fread(expected,sizeof(float),count,oracle)!=count ||
                        memcmp(expected,values+j,count*sizeof(float))) goto done;
                }
            }
        }
    }
    fprintf(stderr,"PASS depth=%d 16 full vocabulary comparisons bit-exact\n",depth);
    rc=0;
done:
    if (checkpoint) fclose(checkpoint);
    if (oracle) fclose(oracle);
    free(values);
    if (rc) fprintf(stderr,"profile failed %s\n",error);
    if (session) qw3_session_free(session);
    if (engine) qw3_engine_close(engine);
    qw3_tokens_free(&tokens);qw3_tokens_free(&prefix);
    return rc;
}
