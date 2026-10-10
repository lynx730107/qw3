#include "projection_candidates.inc"
/* Diagnostic replay: GPU times of representative real GGUF matrices. These
 * are NOT additive estimates of normal model wall time. Inputs are synthetic. */
static uint64_t probe_tensor_bytes(const qw3_tensor *t) {
    if (!t) return 0;
    if (t->type==QW3_TENSOR_F32) return t->elements*4;
    if (t->type==QW3_TENSOR_F16 || t->type==QW3_TENSOR_BF16) return t->elements*2;
    if (t->type==QW3_TENSOR_Q8_0) return t->elements/32*34;
    if (t->type==QW3_TENSOR_Q6_K) return t->elements/256*210;
    uint64_t n=cpu_quant_k_block_size(t->type); if (n) return t->elements/256*n;
    if (t->type==QW3_TENSOR_IQ3_S) return t->elements/256*110;
    if (t->type==QW3_TENSOR_IQ4_XS) return t->elements/256*136;
    if (t->type==QW3_TENSOR_IQ4_NL) return t->elements/32*18;
    n=cpu_quant_iq_block_size(t->type);if(n) return t->elements/cpu_quant_iq_block_elems(t->type)*n;
    fprintf(stderr,"unknown tensor bytes type=%u\n",t->type);abort();
}
static void probe_add_tensor(NSMutableArray *all,qw3_tensor *t,NSString *role) {
    if (!t) return;
    NSString *name=[[NSString alloc] initWithBytes:t->name length:t->name_len encoding:NSUTF8StringEncoding];
    [all addObject:@{@"name":name,@"role":role,@"type":@(t->type),@"n_in":@(t->dim[0]),
        @"n_out":@(t->ndim>=2 ? t->dim[1] : 1),@"bytes":@(probe_tensor_bytes(t)),@"offset":@(t->offset)}];
}
static void probe_emit(id obj) {
    NSData *j=[NSJSONSerialization dataWithJSONObject:obj options:NSJSONWritingPrettyPrinted error:nil];
    printf("%s\n",[[NSString alloc] initWithData:j encoding:NSUTF8StringEncoding].UTF8String);fflush(stdout);
}
int main(int argc,char **argv) {
    @autoreleasepool {
        if (argc<3 || (strcmp(argv[2],"inventory") && strcmp(argv[2],"matrices") && strcmp(argv[2],"bandwidth"))) return 2;
        int inventory=!strcmp(argv[2],"inventory");
        qw3_engine *e=NULL;qw3_session *s=NULL;
        qw3_engine_options o={.model_path=argv[1],.backend=inventory ? QW3_BACKEND_CPU : QW3_BACKEND_METAL,.n_threads=4};
        // Inventory requires no GPU allocation or private pipeline compilation.
        if ((inventory ? probe_engine_open_base(&e,&o) : qw3_engine_open(&e,&o))) return 1;
        NSMutableArray *all=[NSMutableArray new];
        for (int l=0;l<qw3_engine_layer_count(e);l++) {
            qw3_layer_weights *w=&e->weights.layer[l];
#define ADD(field,role) probe_add_tensor(all,w->field,@role)
            ADD(attn_norm,"norm");ADD(ffn_norm,"norm");
            ADD(attn_q_proj,"projection");ADD(attn_k_proj,"projection");ADD(attn_v_proj,"projection");ADD(attn_o_proj,"projection");
            ADD(attn_q_norm,"norm");ADD(attn_k_norm,"norm");
            ADD(linear_qkv_proj,"projection");ADD(linear_gate_proj,"projection");ADD(linear_ssm_out,"projection");
            ADD(linear_ssm_alpha,"small_projection");ADD(linear_ssm_beta,"small_projection");
            ADD(linear_ssm_a,"recurrent");ADD(linear_ssm_dt_bias,"recurrent");ADD(linear_ssm_norm,"norm");ADD(linear_conv_weight,"conv");
            ADD(ffn_gate,"projection");ADD(ffn_up,"projection");ADD(ffn_down,"projection");
#undef ADD
        }
        probe_add_tensor(all,e->weights.output_norm,@"norm");
        probe_add_tensor(all,e->weights.output,@"lm_head");
        // The full embedding is resident, but only ONE row is required per token.
        probe_add_tensor(all,e->weights.token_embd,@"embedding");
        if (inventory) {
            uint64_t active=0,resident=0;
            NSMutableSet *offsets=[NSMutableSet new];
            for (NSDictionary *t in all) {
                NSNumber *offset=t[@"offset"];uint64_t b=[t[@"bytes"] unsignedLongLongValue];
                if (![offsets containsObject:offset]) {
                    [offsets addObject:offset];resident+=b;
                    active+=[t[@"role"] isEqualToString:@"embedding"] ? b/[t[@"n_out"] unsignedLongLongValue] : b;
                }
            }
            probe_emit(@{@"gguf_bytes":@(e->model.map_size),@"layers":@(qw3_engine_layer_count(e)),
                @"active_weight_bytes_per_token_one_read":@(active),@"bound_resident_bytes_unique":@(resident),@"tensors":all});
        } else if (!strcmp(argv[2],"matrices")) {
            if (qw3_session_create(&s,e,512)) return 1;
            QW3MetalSessionObj *obj=(__bridge QW3MetalSessionObj *)s->metal->obj;
            if (!qw3_metal_dense_batch_prepare_map(obj,128)) return 1;
            NSMutableDictionary *shapes=[NSMutableDictionary new];
            for (NSDictionary *t in all) if ([t[@"role"] isEqualToString:@"projection"]) {
                NSString *key=[NSString stringWithFormat:@"%@-%@-%@",t[@"type"],t[@"n_in"],t[@"n_out"]];
                NSMutableDictionary *r=shapes[key];
                if (!r) { r=[t mutableCopy];r[@"count"]=@0;shapes[key]=r; }
                r[@"count"]=@([r[@"count"] intValue]+1);
            }
            NSMutableArray *results=[NSMutableArray new];
            for (NSString *key in [[shapes allKeys] sortedArrayUsingSelector:@selector(compare:)]) {
                NSMutableDictionary *r=shapes[key];uint32_t ni=[r[@"n_in"] unsignedIntValue],no=[r[@"n_out"] unsignedIntValue],type=[r[@"type"] unsignedIntValue];
                id<MTLBuffer> src=[g_device newBufferWithLength:(NSUInteger)128*ni*4 options:MTLResourceStorageModeShared];
                id<MTLBuffer> dst=[g_device newBufferWithLength:(NSUInteger)128*no*4 options:MTLResourceStorageModeShared];
                if (!src || !dst) return 1;
                float *x=src.contents;for(uint64_t i=0;i<128ull*ni;i++) x[i]=((int)(i%257)-128)/128.f;
                NSMutableArray *times=[NSMutableArray new];
                int modes=(type==QW3_TENSOR_Q3_K || type==QW3_TENSOR_IQ3_S || type==QW3_TENSOR_IQ4_XS) ? 5 : 1;
                float *oracle=malloc((size_t)128*no*4);
                for (int mode=0;mode<modes;mode++) {
                    probe_select(mode%2 ? (type==QW3_TENSOR_Q3_K ? 1 : 2) : 0);
                    NSMutableArray *samples=[NSMutableArray new];
                    for (int rep=0;rep<6;rep++) {
                        id<MTLCommandBuffer> cb=[g_queue commandBuffer];
                        // Each replay has identical original encoder boundaries.
                        for (int repeat=0;repeat<8;repeat++) if (!qw3_metal_encode_dense_batch_matmul(obj,(id)cb,
                            [r[@"offset"] unsignedLongLongValue],type,128,ni,no,src,dst)) return 1;
                        [cb commit];[cb waitUntilCompleted];
                        if (cb.status!=MTLCommandBufferStatusCompleted) return 1;
                        if (rep) [samples addObject:@((cb.GPUEndTime-cb.GPUStartTime)*1000/8)];
                    }
                    if (!mode) memcpy(oracle,dst.contents,(size_t)128*no*4);
                    if (memcmp(oracle,dst.contents,(size_t)128*no*4)) { fprintf(stderr,"matrix mismatch %s\n",key.UTF8String);return 1; }
                    for(uint64_t i=0;i<128ull*no;i++) if (!isfinite(oracle[i])) return 1;
                    [times addObject:@{@"candidate":@(mode%2),@"gpu_ms":samples}];
                }
                free(oracle);r[@"replays"]=times;[results addObject:r];probe_select(0);
            }
            probe_emit(@{@"scope":@"isolated real matrices, synthetic FP32 inputs, 128 rows; not model stage percentages",@"matrices":results});
        } else {
            NSString *source=@"#include <metal_stdlib>\nusing namespace metal;\nkernel void init(device uint4 *p [[buffer(0)]],uint i [[thread_position_in_grid]]) {p[i]=uint4(i,i+1,i+2,i+3)^uint4(0x9e3779b9u);}\nkernel void stream(device const uint4 *p [[buffer(0)]],device uint *o [[buffer(1)]],constant uint &n [[buffer(2)]],uint i [[thread_position_in_grid]],uint stride [[threads_per_grid]]) {uint4 s=uint4(0);for(uint j=i;j<n;j+=stride) s+=p[j];o[i]=s.x^s.y^s.z^s.w;}";
            NSError *err=nil;id<MTLLibrary> lib=[g_device newLibraryWithSource:source options:nil error:&err];
            id<MTLComputePipelineState> init=[g_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"init"] error:&err];
            id<MTLComputePipelineState> stream=[g_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"stream"] error:&err];
            if (!init || !stream) { fprintf(stderr,"stream compile %s\n",err.localizedDescription.UTF8String);return 1; }
            // Mapped model pages stay untouched in this process. No inference overlap.
            NSMutableArray *runs=[NSMutableArray new];
            for (int mb=128;mb<=1024;mb*=2) {
                uint64_t bytes=(uint64_t)mb*1024*1024;uint32_t n=(uint32_t)(bytes/16);
                uint32_t threads=1024*256;
                id<MTLBuffer> buf=[g_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                id<MTLBuffer> out=[g_device newBufferWithLength:threads*4 options:MTLResourceStorageModeShared];
                if (!buf || !out) return 1;
                id<MTLCommandBuffer> cb=[g_queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
                [enc setComputePipelineState:init];[enc setBuffer:buf offset:0 atIndex:0];
                [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[enc endEncoding];[cb commit];[cb waitUntilCompleted];
                NSMutableArray *samples=[NSMutableArray new];uint32_t checksum=0;
                for (int rep=0;rep<7;rep++) {
                    cb=[g_queue commandBuffer];enc=[cb computeCommandEncoder];
                    [enc setComputePipelineState:stream];[enc setBuffer:buf offset:0 atIndex:0];[enc setBuffer:out offset:0 atIndex:1];[enc setBytes:&n length:4 atIndex:2];
                    [enc dispatchThreads:MTLSizeMake(threads,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[enc endEncoding];[cb commit];[cb waitUntilCompleted];
                    if (cb.status!=MTLCommandBufferStatusCompleted) return 1;
                    if (rep) [samples addObject:@((cb.GPUEndTime-cb.GPUStartTime)*1000)];
                    uint32_t *p=out.contents;
                    const uint32_t lanes[]={0u,1u,17u,123u,threads/2,threads-1};
                    for (unsigned lane=0;lane<sizeof(lanes)/sizeof(lanes[0]);lane++) {
                        uint32_t sum[4]={0};
                        for (uint32_t j=lanes[lane];j<n;j+=threads)
                            for (uint32_t k=0;k<4;k++) sum[k]+=(j+k)^0x9e3779b9u;
                        if (p[lanes[lane]]!=(sum[0]^sum[1]^sum[2]^sum[3])) {
                            fprintf(stderr,"stream oracle mismatch size=%d lane=%u\n",mb,lanes[lane]);return 1;
                        }
                    }
                    uint32_t check=0;for(uint32_t i=0;i<threads;i++) check^=p[i];
                    if (rep && check!=checksum) return 1;checksum=check;
                }
                [runs addObject:@{@"bytes":@(bytes),@"gpu_ms":samples,@"checksum":@(checksum)}];
            }
            probe_emit(@{@"scope":@"read-only GPU streaming, one warmup, six timed samples; NOT inference DRAM traffic",@"runs":runs});
        }
        if (s) qw3_session_free(s);qw3_engine_close(e);return 0;
    }
}
