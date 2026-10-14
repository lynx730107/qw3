"""Private scheduling-only window for the two independent fallback FFN projections."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent.parent
out = root / 'tests/.decode_ffn_concurrent'
out.mkdir(exist_ok=True)
source = (root / 'qw3.c').read_text()
a = source.index('static int qw3_metal_session_eval_dense_token(')
b = source.index('\nint qw3_engine_metal_dense_logits_test(', a)
part = source[a:b]
original = '''            ok = qw3_metal_session_matvec_quant_x1_to_scratch(
                     s->metal, lw->ffn_gate->offset, lw->ffn_gate->type,
                     n_embd, n_ff, 0, NULL) &&
                 qw3_metal_session_matvec_quant_x1_to_scratch(
                     s->metal, lw->ffn_up->offset, lw->ffn_up->type,
                     n_embd, n_ff, n_ff, NULL);'''
changed = '''            ok = private_ffn_window_begin() &&
                 qw3_metal_session_matvec_quant_x1_to_scratch(
                     s->metal, lw->ffn_gate->offset, lw->ffn_gate->type,
                     n_embd, n_ff, 0, NULL) &&
                 qw3_metal_session_matvec_quant_x1_to_scratch(
                     s->metal, lw->ffn_up->offset, lw->ffn_up->type,
                     n_embd, n_ff, n_ff, NULL);
            private_ffn_window_end();'''
assert part.count(original) == 1
modified = source[:a] + part.replace(original, changed) + source[b:]
assert modified.replace(changed, original) == source
(out / 'qw3_private.c').write_text(modified)
hooks = '''/* Original shaders, weights, data buffers and command buffer; two FFN dispatches overlap. */
#include "../../qw3_metal.m"
static int private_ffn_enabled;
static int private_ffn_active;
static uint64_t private_ffn_calls;
static int private_ffn_window_begin(void) {
    if (!private_ffn_enabled) return 1;
    if (private_ffn_active || !g_batch_cb || g_batch_concurrent || g_batch_separate_encoders) return 0;
    qw3_metal_close_batch_encoder();
    g_batch_concurrent=1;
    private_ffn_active=1;
    private_ffn_calls++;
    return 1;
}
static void private_ffn_window_end(void) {
    if (!private_ffn_active) return;
    qw3_metal_close_batch_encoder();
    g_batch_concurrent=0;
    private_ffn_active=0;
}
#define qw3_metal_env_n_gpu_layers ffn_concurrent_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled ffn_concurrent_core_llama_split_enabled
#define qw3_metal_llama_split_start ffn_concurrent_core_llama_split_start
#include "qw3_private.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
'''
(out / 'hooks.inc').write_text(hooks)
base = root / 'docs/measurements/iq3xxs_sign_2026-10-14'
m = json.loads((base / 'manifest.json').read_text())


def frozen(name):
    key = 'tests/.iq3xxs_sign/' + name
    data = (base / m['source_snapshot_paths'][key]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == m['source_sha256'][key]
    return data.decode()


model = frozen('model.inc').replace('iq3xxs_sign', 'ffn_concurrent').replace('IQ3XXS sign', 'FFN concurrent')
model = model.replace(' || !ffn_concurrent_compile()', '')
model = model.replace('phase<=2', 'phase<=1')
model = model.replace('sign_variant=phase ? phase : 1;scratch_select(phase!=0);', 'private_ffn_enabled=phase!=0;')
model = model.replace('ffn_concurrent_calls', 'private_ffn_calls').replace('32u*34u', '32u*31u')
model = model.replace('private_ffn_calls!=(phase ? 32u*31u : 0u)', 'private_ffn_active || g_batch_concurrent || private_ffn_calls!=(phase ? 32u*31u : 0u)')
(out / 'model.inc').write_text(model)
native = '''/* Scheduling configured before the original warmup; original benchmark timers. */
#define main ffn_concurrent_original_bench_main
#include "../../qw3_bench.c"
#undef main
int main(int argc,char **argv) {
    @autoreleasepool {
        const char *mode=getenv("QW3_PRIVATE_FFN_CONCURRENT");
        if (!mode || (strcmp(mode,"0") && strcmp(mode,"1"))) return 2;
        private_ffn_enabled=atoi(mode);
        int rc=ffn_concurrent_original_bench_main(argc,argv);
        if (private_ffn_active || g_batch_concurrent) return 1;
        fprintf(stderr,"PASS ffn_concurrent_native variant=%d calls=%llu additional_weight_buffer_bytes=0\\n",
                private_ffn_enabled,(unsigned long long)private_ffn_calls);
        return rc;
    }
}
'''
(out / 'native.inc').write_text(native)
print('FFN scheduling window generated; original core restored exactly after removing two hooks')
