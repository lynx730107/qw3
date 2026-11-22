#import "qw3_metal.m"

/* Diagnostic for the full-offload dense F16 layout. Save mutable state in
 * already allocated KV rows beyond the block and attention padding. */
int probe_tail_copy(qw3_metal_session *s, uint32_t pos, uint32_t width, int restore) {
    @autoreleasepool {
        qw3_metal_state_info info;
        if (!qw3_metal_session_get_state_info(s,&info) || info.kv_type != 1 ||
            width != 8 || pos > info.ctx_size || info.ctx_size-pos < width) return 0;
        uint64_t first = (((uint64_t)pos + width + 31) / 32 + 1) * 32;
        if (first >= info.ctx_size) return 0;
        uint64_t capacity = (info.ctx_size-first) * info.kv_row_bytes;
        uint64_t required = info.deltanet_bytes + info.conv_bytes + info.logits_bytes +
                            info.n_full_layers * info.flash_tail_layer_bytes;
        if (capacity * 2 * info.n_full_layers < required) return 0;
        if (!qw3_metal_synchronize()) return 0;

        QW3MetalSessionObj *obj=(__bridge QW3MetalSessionObj *)s->obj;
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        if (!blit) return 0;
        uint64_t written=0;
        for (uint32_t i=0; i<3+info.n_full_layers; i++) {
            qw3_metal_state_buffer state_kind = i==0 ? QW3_METAL_STATE_DELTANET :
                i==1 ? QW3_METAL_STATE_CONV : i==2 ? QW3_METAL_STATE_LOGITS : QW3_METAL_STATE_GQA_FLASH_TAIL;
            id<MTLBuffer> state=nil;
            uint64_t base,length;
            if (!qw3_metal_session_state_span(obj,state_kind,i<3?0:i-3,&state,&base,&length)) {
                [blit endEncoding]; return 0;
            }
            for (uint64_t copied=0; copied<length;) {
                uint32_t slot=(uint32_t)(written/capacity);
                uint64_t offset=written%capacity;
                uint64_t part=MIN(length-copied,capacity-offset);
                uint64_t kvbase,kvlen;
                id<MTLBuffer> kv=nil;
                qw3_metal_state_buffer kind=slot%2 ? QW3_METAL_STATE_GQA_V : QW3_METAL_STATE_GQA_K;
                if (!qw3_metal_session_state_span(obj,kind,slot/2,&kv,&kvbase,&kvlen) ||
                    first*info.kv_row_bytes+offset+part > kvlen) {
                    [blit endEncoding]; return 0;
                }
                uint64_t at=kvbase+first*info.kv_row_bytes+offset;
                if (restore) {
                    [blit copyFromBuffer:kv sourceOffset:at toBuffer:state destinationOffset:base+copied size:part];
                } else {
                    [blit copyFromBuffer:state sourceOffset:base+copied toBuffer:kv destinationOffset:at size:part];
                }
                copied+=part; written+=part;
            }
        }
        [blit endEncoding];
        [cb commit]; [cb waitUntilCompleted];
        return cb.status==MTLCommandBufferStatusCompleted && written==required;
    }
}
