static int stage_quality(QW3MetalSessionObj *obj,const qw3_tensor *w,uint32_t nt,uint32_t no) {
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
        if (!stage_encode(obj,w->type,cb,v,w->offset,nt,ni,no,input,out[v],io,is,oo,os)) return 0;
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
    if (stage_encode(obj,w->type,invalid,1,w->offset,nt,ni,no,input,out[1],io,ni+io-1,oo,os) ||
        stage_encode(obj,w->type,invalid,1,w->offset,nt,ni,no,input,out[1],io,is,os+1,os) ||
        stage_encode(obj,w->type,invalid,1,w->offset,nt+1,ni,no,input,out[1],io,is,oo,os)) return 0;
    printf("kstage_matrix_exact tokens=%u n_in=%u n_out=%u finite=all guards=exact input=unchanged invalid_layout=rejected\n",nt,ni,no);
    fflush(stdout);return 1;
}

