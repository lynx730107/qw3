/* Private full-vocabulary head checks and alternating paired cost probe.
 * Compare every value with ordinary Q5 GEMV, plus independent CPU boundary
 * rows. Include odd strides/offsets, partial output rows and token groups. */
static int decode_head_scalar(qw3_engine *e, id<MTLBuffer> src, id<MTLBuffer> dst,
    uint32_t ni, uint32_t no, uint32_t nt, uint32_t io, uint32_t is,
    uint32_t oo, uint32_t os) {
    uint64_t offset = 0;
    const uint32_t rb = ni/256u*176u;
    id<MTLBuffer> wb = qw3_metal_model_view_for(e->weights.output->offset,(uint64_t)rb*no,&offset);
    if (!wb || !qw3_metal_begin_commands()) return 0;
    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    const uint32_t args[] = {ni,no,rb};
    [enc setComputePipelineState:g_matvec_q5_k_pipeline];
    [enc setBytes:args length:sizeof(args) atIndex:0];
    [enc setBuffer:wb offset:(NSUInteger)offset atIndex:1];
    [enc setThreadgroupMemoryLength:32*sizeof(float) atIndex:0];
    for (uint32_t t = 0; t < nt; t++) {
        [enc setBuffer:src offset:((NSUInteger)t*is+io)*sizeof(float) atIndex:2];
        [enc setBuffer:dst offset:((NSUInteger)t*os+oo)*sizeof(float) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((no+1u)/2u,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
    }
    qw3_metal_end_compute_encoder(cb,enc);
    return qw3_metal_synchronize();
}

static int decode_head_checks(qw3_engine *e) {
    if (!qw3_metal_compile_kernels() || e->weights.output->type != QW3_TENSOR_Q5_K) return 0;
    const uint32_t shapes[][2] = {{256,1},{4352,33},{5120,4097},{5120,4111},{5120,248319},{5120,248320}};
    const uint32_t counts[] = {1,2,3,4,5,8,15};
    const struct ggml_type_traits *traits = ggml_get_type_traits(GGML_TYPE_Q5_K);
    if (!traits || !traits->to_float) return 0;
    printf("mtp_head_config group=%u production_enabled=0\n",decode_head_group);
    int checks = 0;
    double worst = 0, worst_rms = 0, worst_cpu = 0;
    for (int shape = 0; shape < 6; shape++) for (int count = 0; count < 7; count++) {
        const uint32_t ni=shapes[shape][0], no=shapes[shape][1], nt=counts[count];
        const uint32_t is=ni+13u, os=no+11u, rb=ni/256u*176u;
        id<MTLBuffer> x = [g_device newBufferWithLength:(NSUInteger)nt*is*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> y = [g_device newBufferWithLength:(NSUInteger)nt*os*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> ref = [g_device newBufferWithLength:(NSUInteger)nt*no*sizeof(float) options:MTLResourceStorageModeShared];
        if (!x || !y || !ref) return 0;
        float *input=x.contents, *output=y.contents;
        for (uint64_t i=0; i<(uint64_t)nt*is; i++) input[i]=((int)((i*37u+count*19u)%101u)-50)/31.0f;
        for (uint64_t i=0; i<(uint64_t)nt*os; i++) output[i]=123456.0f;
        if (!decode_head_encode(e,x,y,ni,no,nt,7,is,5,os) ||
            !decode_head_scalar(e,x,ref,ni,no,nt,7,is,0,no)) return 0;
        const float *r=ref.contents;
        double max=0, ss=0, cpu_max=0;
        for (uint32_t t=0; t<nt; t++) for (uint32_t j=0; j<os; j++) {
            const float value=output[(size_t)t*os+j];
            if (j<5 || j>=5+no) {if (value!=123456.0f) return 0; continue;}
            const double diff=fabs((double)value-r[(size_t)t*no+j-5]);
            if (!isfinite(value) || !isfinite(diff)) return 0;
            if (diff>max) max=diff;
            ss+=diff*diff;
        }
        float *row=malloc(ni*sizeof(float));
        if (!row) return 0;
        const uint32_t boundary[] = {0,no/2,no-1};
        for (int b=0; b<3; b++) {
            traits->to_float(tensor_data(&e->model,e->weights.output)+(uint64_t)boundary[b]*rb,row,ni);
            for (uint32_t t=0; t<nt; t++) {
                double sum=0;
                for (uint32_t i=0; i<ni; i++) sum+=(double)row[i]*input[(size_t)t*is+7+i];
                const double diff=fabs(sum-output[(size_t)t*os+5+boundary[b]]);
                if (!isfinite(diff)) {free(row);return 0;}
                if (diff>cpu_max) cpu_max=diff;
            }
        }
        free(row);
        const double rms=sqrt(ss/((uint64_t)nt*no));
        printf("mtp_head_check ni=%u no=%u nt=%u max=%.9g rms=%.9g cpu_max=%.9g guards=ok\n",ni,no,nt,max,rms,cpu_max);
        fflush(stdout);
        if (max>0.0001 || rms>0.00001 || cpu_max>0.0001) return 0;
        if (max>worst) worst=max;
        if (rms>worst_rms) worst_rms=rms;
        if (cpu_max>worst_cpu) worst_cpu=cpu_max;
        checks++;
    }
    printf("mtp_head_checks=ok checks=%d max=%.9g rms=%.9g cpu_max=%.9g\n",checks,worst,worst_rms,worst_cpu);
    const uint32_t ni=e->model.spec->n_embd, no=e->model.spec->n_vocab, nt=8;
    id<MTLBuffer> x=[g_device newBufferWithLength:(NSUInteger)nt*ni*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> y=[g_device newBufferWithLength:(NSUInteger)nt*no*sizeof(float) options:MTLResourceStorageModeShared];
    if (!x || !y) return 0;
    float *input=x.contents;
    for (uint32_t i=0; i<nt*ni; i++) input[i]=((int)(i%101)-50)/31.0f;
    double elapsed[2]={0};
    for (int trial=0; trial<12; trial++) {
        double pair[2]={0};
        for (int order=0; order<2; order++) {
            const int variant=order^(trial&1);
            const double start=qw3_now_sec();
            const int ok=variant ? decode_head_encode(e,x,y,ni,no,nt,0,ni,0,no) :
                decode_head_scalar(e,x,y,ni,no,nt,0,ni,0,no);
            if (!ok) return 0;
            pair[variant]=qw3_now_sec()-start;
            if (trial>=2) elapsed[variant]+=pair[variant];
        }
        printf("mtp_head_pair trial=%d warmup=%d scalar_ms=%.3f shared_ms=%.3f\n",trial,trial<2,pair[0]*1000,pair[1]*1000);
    }
    printf("mtp_head_cost nt=%u pairs=10 scalar_ms=%.3f shared_ms=%.3f ratio=%.3f component_only=1\n",
        nt,elapsed[0]*100,elapsed[1]*100,elapsed[0]/elapsed[1]);
    return checks==42;
}
