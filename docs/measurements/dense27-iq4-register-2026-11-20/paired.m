#import "qw3_metal.m"
#include "qw3.h"
#include <time.h>

static int iq3;
static id<MTLComputePipelineState> states[3][3];

static int compile_variant(int v, const char *path) {
    setenv("QW3_METAL_KERNEL_DIR", path, 1);
    NSString *source = qw3_metal_full_kernel_source();
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.preprocessorMacros = @{@"QW3_METAL_HAS_TENSOR":@"1", @"QW3_ENABLE_PARTIAL_TILES":@"1",
                                   @"QW3_ENABLE_IQ3_NIBBLE":@"1"};
    NSError *error = nil;
    id<MTLLibrary> library = [g_device newLibraryWithSource:source options:options error:&error];
    NSArray *names = iq3 ? @[@"qw3_matvec_iq3_s_fast"] : @[@"qw3_matvec_iq4_xs_fast", @"qw3_matvec_iq4_xs_pair_fast", @"qw3_matvec_iq4_xs_add_x0_fast"];
    for (unsigned i=0;i<names.count;i++) {
        id<MTLFunction> fn = [library newFunctionWithName:names[i]];
        if (fn) states[v][i] = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!states[v][i]) {
            fprintf(stderr,"Metal: %s\n",error.localizedDescription.UTF8String);
            return 0;
        }
    }
    return 1;
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + 1e-9*t.tv_nsec;
}

int main(int argc,char **argv) {
    @autoreleasepool {
        if (argc!=4) return 2;
        iq3 = getenv("QW3_PROBE_IQ3") != NULL;
        qw3_engine_options opt={.model_path=argv[1],.backend=QW3_BACKEND_METAL,.n_threads=4};
        qw3_engine *e;
        qw3_session *s;
        if(qw3_engine_open(&e,&opt) || qw3_session_create(&s,e,1152)) return 1;
        states[0][0]=iq3 ? g_matvec_iq3_s_pipeline : g_matvec_iq4_xs_pipeline;
        states[0][1]=g_matvec_iq4_xs_pair_pipeline;
        states[0][2]=g_matvec_iq4_xs_add_x0_pipeline;
        if (!states[0][0] || !compile_variant(1,argv[2]) || !compile_variant(2,argv[3])) return 1;
        setenv("QW3_METAL_KERNEL_DIR","metal",1);
        int ids[1056], vocab=qw3_vocab_size(e);
        uint32_t rng=1;
        for(int i=0;i<1056;i++) {
            rng=rng*1664525u+1013904223u;
            ids[i]=rng%(uint32_t)vocab;
            if(ids[i]==qw3_token_eos(e)) ids[i]=(ids[i]+1)%vocab;
        }
        qw3_tokens prefix={.v=ids,.len=1024,.cap=1024};
        char err[256]={0};
        FILE *fp=tmpfile(), *oracle=tmpfile();
        if (!fp || !oracle || qw3_session_sync(s,&prefix,err,sizeof(err)) ||
            qw3_session_save_payload(s,fp,err,sizeof(err))) return 1;
        uint64_t bytes=qw3_session_payload_bytes(s);
        float *row=malloc(vocab*sizeof(float)),*ref=malloc(vocab*sizeof(float));
        if (!row || !ref) return 1;
        // Quality pass also warms all variants. Six timing blocks rotate the order.
        const int order[]={0,1,2, 0,1,2, 1,2,0, 2,0,1, 2,1,0, 0,2,1, 1,0,2};
        for (unsigned k=0;k<sizeof(order)/sizeof(order[0]);k++) {
            int v=order[k];
            rewind(fp); rewind(oracle);
            if(qw3_session_load_payload(s,fp,bytes,err,sizeof(err))) return 1;
            if (iq3) {
                g_matvec_iq3_s_pipeline=states[v][0];
            } else {
                g_matvec_iq4_xs_pipeline=states[v][0];
                g_matvec_iq4_xs_pair_pipeline=states[v][1];
                g_matvec_iq4_xs_add_x0_pipeline=states[v][2];
            }
            double maxerr=0,worst_rms=0;
            int top_equal=0;
            double start=now();
            for(int i=0;i<32;i++) {
                if(qw3_session_eval(s,ids[1024+i],err,sizeof(err))) {
                    fprintf(stderr,"%s\n",err);return 1;
                }
                if(k>=3) continue;
                if(qw3_session_copy_logits(s,row,vocab)!=vocab) return 1;
                if(k==0 && fwrite(row,sizeof(float),vocab,oracle)!=(size_t)vocab) return 1;
                if(k && fread(ref,sizeof(float),vocab,oracle)!=(size_t)vocab) return 1;
                double ss=0;
                int a=0,b=0;
                for(int j=0;j<vocab;j++) {
                    if(!isfinite(row[j])) return 1;
                    if(!k) continue;
                    double dx=(double)row[j]-ref[j];
                    maxerr=fmax(maxerr,fabs(dx)); ss+=dx*dx;
                    if(row[j]>row[a]) a=j;
                    if(ref[j]>ref[b]) b=j;
                }
                top_equal+=a==b;
                worst_rms=fmax(worst_rms,sqrt(ss/vocab));
            }
            double elapsed=now()-start;
            if(k<3) printf("quality variant=%d max=%.9g worst_rmse=%.9g same_top=%d/32\n",v,maxerr,worst_rms,top_equal);
            else printf("time block=%u variant=%d ms=%.3f tps=%.4f\n",(k-3)/3,v,1000*elapsed,32/elapsed);
            fflush(stdout);
        }
        qw3_session_free(s);qw3_engine_close(e);
        free(row);free(ref);fclose(fp);fclose(oracle);
    }
    return 0;
}
