/* Included by the private MTP diagnostic. No production kernel promotion. */
static int decode_short_quant_checks(qw3_engine *e) {
    if (!qw3_metal_compile_kernels()) {fprintf(stderr,"MTP quant base kernels unavailable\n");return 0;}
    const uint32_t shapes[][2] = {{256,1},{4352,33},{5120,4097},{5120,4111},{5120,17408},{17408,5120}};
    const uint32_t counts[] = {1,2,3,4,5,8,15};
    int checks = 0;
    for (int kind = 0; kind < 2; kind++) {
        const uint32_t type = kind ? QW3_TENSOR_IQ4_XS : QW3_TENSOR_IQ3_S;
        const qw3_tensor *weight = NULL;
        for (int layer = 0; layer < e->model.spec->n_layer && !weight; layer++) {
            const qw3_layer_weights *w = &e->weights.layer[layer];
            const qw3_tensor *choices[] = {w->ffn_gate,w->ffn_up,w->ffn_down};
            for (int i = 0; i < 3; i++) if (choices[i] && choices[i]->type == type) { weight = choices[i]; break; }
        }
        if (!weight) {fprintf(stderr,"MTP quant tensor missing type=%u\n",type);return 0;}
        id<MTLBuffer> grid = kind ? nil : qw3_metal_iq3s_kgrid_buffer();
        for (int shape = 0; shape < 6; shape++) for (int count = 0; count < 7; count++) {
            const uint32_t ni = shapes[shape][0], no = shapes[shape][1], nt = counts[count];
            const uint32_t rb = ni/256u*(kind ? 136u : 110u), is = ni+13u, os = no+11u;
            if ((uint64_t)ni*no > weight->elements) return 0;
            uint64_t offset = 0;
            id<MTLBuffer> wb = qw3_metal_model_view_for(weight->offset,(uint64_t)no*rb,&offset);
            id<MTLBuffer> x = [g_device newBufferWithLength:(NSUInteger)nt*is*sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> packed = [g_device newBufferWithLength:(NSUInteger)nt*ni*sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> out = [g_device newBufferWithLength:(NSUInteger)nt*os*sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> ref = [g_device newBufferWithLength:(NSUInteger)nt*no*sizeof(float) options:MTLResourceStorageModeShared];
            if (!wb || !x || !packed || !out || !ref || (!kind && !grid)) {fprintf(stderr,"MTP quant allocation/view failed type=%u ni=%u no=%u nt=%u\n",type,ni,no,nt);return 0;}
            float *input = x.contents, *clean = packed.contents, *y = out.contents;
            for (uint64_t i = 0; i < (uint64_t)nt*is; i++) input[i] = ((int)((i*37u+count*19u)%101u)-50)/31.0f;
            for (uint32_t t = 0; t < nt; t++) memcpy(clean+(size_t)t*ni,input+(size_t)t*is+7u,ni*sizeof(float));
            for (uint64_t i = 0; i < (uint64_t)nt*os; i++) y[i] = 123456.0f;
            struct {uint32_t ni,no,rb,nt,io,is,oo,os;} args = {ni,no,rb,nt,7,is,5,os};
            if (!qw3_metal_begin_commands()) {fprintf(stderr,"MTP quant command begin failed\n");return 0;}
            int owned = 0;
            id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
            id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
            [enc setComputePipelineState:kind ? decode_iq4xs_short : decode_iq3s_short];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:wb offset:(NSUInteger)offset atIndex:1];
            [enc setBuffer:x offset:0 atIndex:2];
            [enc setBuffer:out offset:0 atIndex:3];
            if (grid) [enc setBuffer:grid offset:0 atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake((no+3u)/4u,(nt+3u)/4u,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
            const uint32_t scalar_args[] = {ni,no,rb};
            [enc setComputePipelineState:kind ? g_matvec_iq4_xs_pipeline : g_matvec_iq3_s_pipeline];
            [enc setBytes:scalar_args length:sizeof(scalar_args) atIndex:0];
            [enc setThreadgroupMemoryLength:32*sizeof(float) atIndex:0];
            for (uint32_t t = 0; t < nt; t++) {
                [enc setBuffer:packed offset:(NSUInteger)t*ni*sizeof(float) atIndex:2];
                [enc setBuffer:ref offset:(NSUInteger)t*no*sizeof(float) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((no+(kind ? 3u : 7u))/(kind ? 4u : 8u),1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            }
            qw3_metal_end_compute_encoder(cb,enc);
            if (!qw3_metal_synchronize()) {fprintf(stderr,"MTP quant command failed\n");return 0;}
            const float *r = ref.contents;
            double max = 0, ss = 0, cpu_max = 0;
            for (uint32_t t = 0; t < nt; t++) for (uint32_t j = 0; j < os; j++) {
                const float value = y[(size_t)t*os+j];
                if (j < 5u || j >= 5u+no) {if (value != 123456.0f) {fprintf(stderr,"MTP quant guard failed type=%u ni=%u no=%u nt=%u row=%u col=%u value=%g\n",type,ni,no,nt,t,j,value);return 0;} continue;}
                const double diff = fabs((double)value-r[(size_t)t*no+j-5u]);
                if (!isfinite(value) || !isfinite(diff)) return 0;
                if (diff > max) max = diff;
                ss += diff*diff;
            }
            const struct ggml_type_traits *traits = ggml_get_type_traits((enum ggml_type)type);
            if (!traits || !traits->to_float) return 0;
            float *row = malloc(ni*sizeof(float));
            if (!row) return 0;
            const uint32_t rows[] = {0,no/2u,no-1u};
            for (int j = 0; j < 3; j++) {
                traits->to_float(tensor_data(&e->model,weight)+(uint64_t)rows[j]*rb,row,ni);
                for (uint32_t t = 0; t < nt; t++) {
                    double sum = 0;
                    for (uint32_t i = 0; i < ni; i++) sum += (double)row[i]*clean[(size_t)t*ni+i];
                    const double diff = fabs(sum-y[(size_t)t*os+5u+rows[j]]);
                    if (!isfinite(diff)) {free(row);return 0;}
                    if (diff > cpu_max) cpu_max = diff;
                }
            }
            free(row);
            const double rms = sqrt(ss/((uint64_t)nt*no));
            printf("mtp_short_quant type=%u ni=%u no=%u nt=%u max=%.9g rms=%.9g cpu_max=%.9g guards=ok\n",type,ni,no,nt,max,rms,cpu_max);
            if (max > 0.0001 || rms > 0.00001 || cpu_max > 0.0001) return 0;
            checks++;
        }
    }
    printf("mtp_short_quant_checks=ok checks=%d\n",checks);
    return checks == 84;
}
