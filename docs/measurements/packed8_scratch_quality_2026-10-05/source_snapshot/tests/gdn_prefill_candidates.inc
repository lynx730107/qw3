/* Optional diagnostic binary only. Production engine/shaders are untouched. */
#define qw3_metal_session_batch_deltanet_fused_gdn_from_scratch probe_base_gdn
#include "../qw3_metal.m"
#undef qw3_metal_session_batch_deltanet_fused_gdn_from_scratch
#include "gdn_prefill_dispatch.inc"

#define qw3_metal_env_n_gpu_layers qw3_probe_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled qw3_probe_core_llama_split_enabled
#define qw3_metal_llama_split_start qw3_probe_core_llama_split_start
#define qw3_engine_open probe_engine_open_base
#include "../qw3.c"
#undef qw3_engine_open
static void probe_select(int mode) { probe_gdn=mode; }
int qw3_engine_open(qw3_engine **engine,const qw3_engine_options *options) {
    int rc=probe_engine_open_base(engine,options); if (rc) return rc;
    if (!probe_initialize()) { qw3_engine_close(*engine); *engine=NULL; return -1; }
    const char *mode=getenv("QW3_PRIVATE_GDN_LAYOUT");
    probe_select(!mode ? 0 : !strcmp(mode,"packed4") ? 1 : !strcmp(mode,"packed8") ? 2 : !strcmp(mode,"pipeline") ? 3 : !strcmp(mode,"packed8_pipeline") ? 4 : -1);
    if (probe_gdn<0) { qw3_engine_close(*engine); *engine=NULL; return -1; }
    fprintf(stderr,"private GDN layout=%s\n",mode ? mode : "off"); return 0;
}
