#include "kstage_candidates.inc"
#define main stage_original_bench_main
#include "../qw3_bench.c"
#undef main
int main(int argc,char **argv) {
    @autoreleasepool {
        if (argc!=7) return 2;
        stage_width=atoi(argv[2]);
        int mode=!strcmp(argv[3],"iq3") ? 1 : !strcmp(argv[3],"iq4") ? 2 : !strcmp(argv[3],"both") ? 3 : -1;
        int pp=!strcmp(argv[4],"pp128"),cooldown=atoi(argv[6]);
        bench_config cfg={.model_path=argv[1],.backend=QW3_BACKEND_METAL,.threads=4,
            .ctx_alloc=512,.seed=1,.repetitions=atoi(argv[5]),.depth=pp ? 0 : 128};
        if ((stage_width!=64 && stage_width!=128) || mode<1 || (!pp && strcmp(argv[4],"tg32_d128")) ||
            cooldown<0 || cooldown>60 || cfg.repetitions<1 || cfg.repetitions>10) return 2;
        setbuf(stdout,NULL);setbuf(stderr,NULL);
        setenv("QW3_METAL_NGL","64",1);setenv("QW3_METAL_DENSE_PREFILL_FAST","1",1);
        setenv("QW3_BENCH_PRINT_SAMPLES","1",1);
        qw3_engine *e=NULL;qw3_engine_options opt={.model_path=argv[1],.backend=cfg.backend,.n_threads=4};
        if (qw3_engine_open(&e,&opt) || !stage_compile()) return 1;
        int order[]={0,1,0,1,0,1,0},rc=0;
        for (int block=0;block<7;block++) {
            fprintf(stderr,"KSTAGE_BEGIN block=%d candidate=%d cooldown=%d\n",block,order[block],cooldown);
            sleep((unsigned)cooldown);scratch_select(order[block] ? mode : 0);
            stage_iq3_calls=stage_iq4_calls=0;
            printf("KSTAGE_BEGIN %d %d\n",block,order[block]);
            rc=run_llama_style_case(e,&cfg,pp ? 128 : 0,pp ? 0 : 32,true);
            fprintf(stderr,"KSTAGE_END block=%d candidate=%d rc=%d iq3=%llu iq4=%llu\n",block,order[block],rc,
                (unsigned long long)stage_iq3_calls,(unsigned long long)stage_iq4_calls);
            printf("KSTAGE_END %d %d %d\n",block,order[block],rc);
            if (rc || (!order[block] && (stage_iq3_calls || stage_iq4_calls)) ||
                (order[block] && (((mode&1) && !stage_iq3_calls) || ((mode&2) && !stage_iq4_calls)))) {rc=1;break;}
        }
        qw3_engine_close(e);return rc;
    }
}
