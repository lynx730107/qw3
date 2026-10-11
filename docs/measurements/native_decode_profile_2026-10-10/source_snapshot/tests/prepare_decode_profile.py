"""Adapt timestamp-only profiling to native decode without candidate kernels."""
from pathlib import Path
root=Path(__file__).resolve().parent.parent
out=root/'tests/.decode_profile';out.mkdir(exist_ok=True)
s=(root/'tests/projection_candidates.inc').read_text()
start=s.index('/* Private experiments:')
end=s.index('#define qw3_metal_env_n_gpu_layers')
hooks=s[start:end]
hooks=hooks.replace('#include "../qw3_metal.m"', '''#define qw3_metal_begin_commands profile_native_begin_commands
#include "../../qw3_metal.m"
#undef qw3_metal_begin_commands
int qw3_metal_begin_commands(void) {
    int ok=profile_native_begin_commands();
    if (ok && probe_capture) g_batch_separate_encoders=1;
    return ok;
}''')
end=s.index('int qw3_engine_open(qw3_engine **e,const qw3_engine_options *opt) {')
start=s.index('static __attribute__((unused)) int probe_profile_initialize(')
profile=s[start:end]
(out/'hooks.inc').write_text(hooks+'\n'+profile)

quality=(root/'tests/test_native_prefill.m').read_text()
start=quality.index('static int logits_equal(')
end=quality.index('int main(')
(out/'oracle.inc').write_text(quality[start:end])
