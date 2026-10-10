/* Reuse the exact production benchmark case with one engine across blocks.
 * Each case still creates one fresh session and performs its own warmup. */
#include "candidates.inc"
#define main campaign_original_bench_main
#include "../../qw3_bench.c"
#undef main
int main(int argc,char **argv) {
    @autoreleasepool {
        if (argc!=6) return 2;
        bench_config cfg={.model_path=argv[1],.backend=QW3_BACKEND_METAL,
            .threads=4,.ctx_alloc=512,.seed=1,.repetitions=atoi(argv[4])};
        const int pp=!strcmp(argv[3],"pp128");
        if (!pp && strcmp(argv[3],"tg32_d128")) return 2;
        cfg.depth=pp ? 0 : 128;
        int cooldown=atoi(argv[5]);
        if (cfg.repetitions<1 || cfg.repetitions>10 || cooldown<0 || cooldown>60) return 2;
        setbuf(stdout,NULL); setbuf(stderr,NULL);
        setenv("QW3_METAL_NGL","64",1);
        scratch_select(0);
        qw3_engine_options opt={.model_path=cfg.model_path,.backend=cfg.backend,.n_threads=4};
        qw3_engine *engine=NULL;
        if (qw3_engine_open(&engine,&opt) || !sum8_compile() || !epilogue_compile()) return 1;
        char *order=strdup(argv[2]),*save=NULL;
        int rc=0,block=0;
        for (char *name=strtok_r(order,",",&save); name; name=strtok_r(NULL,",",&save),block++) {
            int mode=!strcmp(name,"off") ? 0 : !strcmp(name,"residual") ? 1 :
                !strcmp(name,"round") ? 2 : !strcmp(name,"both") ? 3 :
                !strcmp(name,"logits") ? 4 : !strcmp(name,"all") ? 7 : !strcmp(name,"down") ? 8 : !strcmp(name,"residual_down") ? 9 : !strcmp(name,"both_down") ? 11 : !strcmp(name,"all_down") ? 15 : -1;
            if (mode<0) { rc=2; break; }
            printf("COMBO_BEGIN %d %s\n",block,name);
            fprintf(stderr,"COMBO_BEGIN %d %s\n",block,name);
            sleep((unsigned)cooldown);
            scratch_select(mode);
            combo_residual_calls=combo_round_calls=combo_round_saved=sum8_head_calls=epilogue_calls=0;
            rc=run_llama_style_case(engine,&cfg,pp ? 128 : 0,pp ? 0 : 32,true);
            fprintf(stderr,"private dispatches pair=0 add=0 gdn=0 residual_norm=%llu round_norm=%llu round_saved=%llu sum8_head=%llu down_epilogue=%llu\n",
                combo_residual_calls,combo_round_calls,combo_round_saved,sum8_head_calls,epilogue_calls);
            printf("COMBO_END %d %s %d\n",block,name,rc);
            fprintf(stderr,"COMBO_END %d %s %d\n",block,name,rc);
            if (rc) break;
        }
        free(order); qw3_engine_close(engine); return rc;
    }
}
