#include "micro_candidates.inc"
int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc!=2) return 2;
        qw3_engine *e=NULL; qw3_session *s=NULL;
        qw3_engine_options opt={.model_path=argv[1],.backend=QW3_BACKEND_METAL,.n_threads=4};
        int rc=1, cases=0;
        QW3MetalSessionObj *obj=nil;
        if (qw3_engine_open(&e,&opt) || qw3_session_create(&s,e,512) || !micro_compile()) goto done;
        obj=(__bridge QW3MetalSessionObj *)s->metal->obj;
        for (int l=0;l<64;l++) {
            qw3_layer_weights *w=&e->weights.layer[l];
            if (!w->linear_ssm_alpha) continue;
            uint32_t ni=(uint32_t)w->linear_ssm_alpha->dim[0],no=(uint32_t)w->linear_ssm_alpha->dim[1];
            uint32_t oa=5,ob=oa+no+7,stride=ob+no+9;
            obj.x1=[g_device newBufferWithLength:ni*4u options:MTLResourceStorageModeShared];
            obj.scratch=[g_device newBufferWithLength:stride*4u options:MTLResourceStorageModeShared];
            if (!obj.x1 || !obj.scratch) goto done;
            const uint32_t lengths[]={1,15,16,31,127,128,129};
            for (int k=0;k<7;k++) {
                uint32_t nt=lengths[k];
                obj.prefillX1=[g_device newBufferWithLength:(NSUInteger)nt*ni*4u options:MTLResourceStorageModeShared];
                obj.prefillScratch=[g_device newBufferWithLength:(NSUInteger)nt*stride*4u options:MTLResourceStorageModeShared];
                if (!obj.prefillX1 || !obj.prefillScratch) goto done;
                float *x=obj.prefillX1.contents; uint32_t rng=13u+l*17u;
                for (uint32_t i=0;i<nt*ni;i++) { rng=rng*1664525u+1013904223u; x[i]=(float)((int)(rng>>16u)-32768)/32768.0f; }
                NSData *before=[NSData dataWithBytes:x length:(NSUInteger)nt*ni*4u];
                memset(obj.prefillScratch.contents,0x5a,(size_t)nt*stride*4u);
                scratch_select(0);
                if (!micro_batch_pair(s->metal,w->linear_ssm_alpha->offset,w->linear_ssm_beta->offset,nt,ni,no,oa,ob,stride)) goto done;
                NSData *ref=[NSData dataWithBytes:obj.prefillScratch.contents length:(NSUInteger)nt*stride*4u];
                memset(obj.prefillScratch.contents,0x5a,(size_t)nt*stride*4u);
                scratch_select(1);
                if (!micro_batch_pair(s->metal,w->linear_ssm_alpha->offset,w->linear_ssm_beta->offset,nt,ni,no,oa,ob,stride)) goto done;
                if (memcmp(ref.bytes,obj.prefillScratch.contents,ref.length) || memcmp(before.bytes,x,before.length)) {
                    fprintf(stderr,"batch mismatch layer=%d tokens=%u\n",l,nt);goto done;
                }
                cases++;
            }
            memcpy(obj.x1.contents,obj.prefillX1.contents,ni*4u);
            memset(obj.scratch.contents,0x5a,obj.scratch.length);
            scratch_select(0);
            if (!micro_decode_pair(s->metal,w->linear_ssm_alpha->offset,w->linear_ssm_beta->offset,8,8,ni,no,oa,ob)) goto done;
            NSData *ref=[NSData dataWithBytes:obj.scratch.contents length:obj.scratch.length];
            memset(obj.scratch.contents,0x5a,obj.scratch.length);
            scratch_select(2);
            if (!micro_decode_pair(s->metal,w->linear_ssm_alpha->offset,w->linear_ssm_beta->offset,8,8,ni,no,oa,ob)) goto done;
            float *a=(float *)ref.bytes,*b=obj.scratch.contents;
            int mismatches=0;
            for (uint32_t i=0;i<obj.scratch.length/4u;i++) if (memcmp(a+i,b+i,4)) {
                if (mismatches<3) fprintf(stderr,"decode mismatch layer=%d i=%u want=%.9g got=%.9g\n",l,i,a[i],b[i]);
                mismatches++;
            }
            if (mismatches) { fprintf(stderr,"decode mismatches=%d\n",mismatches);goto done; }
            cases++;printf("q8_exact layer=%d cases=%d\n",l,cases);fflush(stdout);
        }
        printf("PASS q8_cases=%d\n",cases);rc=0;
done:
        if (s) qw3_session_free(s);if (e) qw3_engine_close(e);return rc;
    }
}
