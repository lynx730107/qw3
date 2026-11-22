"""Build an isolated one-dispatch IQ3_S/IQ4_XS projection experiment."""
from pathlib import Path
import shutil
import subprocess
import tempfile

repo = Path.cwd()
archive = Path(__file__).resolve().parent
work = Path(tempfile.mkdtemp(prefix='qw3-dual-dispatch-'))
for name in ('qw3.c', 'qw3.h', 'qw3_metal.h', 'qw3_metal.m', 'qw3_bench.c',
             'qw3_runtime_fingerprint.h'):
    shutil.copy2(repo / name, work / name)
shutil.copytree(repo / 'metal', work / 'metal')
shutil.copy2(archive / 'adapter.inc', work)

metal = work / 'metal'
moe = metal / 'qw3_core_moe.metal'
linear = metal / 'qw3_core_linear.metal'
moe_source = moe.read_text()

def body(source, name):
    start = source.index('kernel void ' + name + '(')
    start = source.index(' {\n', start) + 3
    end = source.index('\n}\n', start)
    return source[start:end]

iq3_body = body(moe_source, 'qw3_matvec_iq3_s_fast')
assert iq3_body.count('    (void)sh;\n    threadgroup ushort grid[512];\n') == 1
iq3_body = iq3_body.replace('    (void)sh;\n    threadgroup ushort grid[512];\n', '')
iq4_body = body(linear.read_text(), 'qw3_matvec_iq4_xs_fast')
addition = '''
inline void dual_run_iq3(constant qw3_matvec_q8_0_args &args,
    device const uchar *weights, device const float *x, device float *out,
    device const ushort *kgrid, threadgroup ushort *grid, uint group, ushort simd_idx, ushort lane) {
''' + iq3_body + '''
}
inline void dual_run_iq4(constant qw3_matvec_q8_0_args &args,
    device const uchar *weights, device const float *x, device float *out,
    threadgroup float *sh, uint group, ushort simd_idx, ushort lane) {
''' + iq4_body + '''
}
struct dual_matvec_args {
    qw3_matvec_q8_0_args a, b;
    uint groups_a, type_a, type_b;
};
'''
for kind in ('iq3', 'iq4', 'mixed'):
    grid_arg = '\n    device const ushort *kgrid [[buffer(5)]],' if kind != 'iq4' else ''
    mem3 = '    threadgroup ushort grid[512];\n' if kind != 'iq4' else ''
    mem4 = '    threadgroup float sh[32];\n' if kind != 'iq3' else ''
    addition += f'''
kernel void dual_{kind}(constant dual_matvec_args &params [[buffer(0)]],
    device const uchar *wa [[buffer(1)]], device const uchar *wb [[buffer(2)]],
    device const float *x [[buffer(3)]], device float *out [[buffer(4)]],{grid_arg}
    uint group [[threadgroup_position_in_grid]], ushort simd_idx [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {{
{mem3}{mem4}    const bool first = group < params.groups_a;
    constant qw3_matvec_q8_0_args &args = first ? params.a : params.b;
    device const uchar *weights = first ? wa : wb;
    const uint local_group = first ? group : group - params.groups_a;
    device float *dst = first ? out : out + params.a.n_out;
'''
    call3 = 'dual_run_iq3(args, weights, x, dst, kgrid, grid, local_group, simd_idx, lane);'
    call4 = 'dual_run_iq4(args, weights, x, dst, sh, local_group, simd_idx, lane);'
    if kind == 'mixed':
        addition += '    const uint type = first ? params.type_a : params.type_b;\n'
        addition += f'    if (type == 21u) {call3}\n    else {call4}\n'
    else:
        addition += '    ' + (call3 if kind == 'iq3' else call4) + '\n'
    addition += '}\n'

moe.write_text(moe_source + addition)
(archive / 'dual-kernels.metal').write_text(addition)

core = work / 'qw3.c'
source = core.read_text()
start = source.index('            stage = "DeltaNet projections";', source.index('int ok = qw3_metal_begin_commands() &&\n        e->weights.token_embd->type', source.index('static int qw3_metal_session_eval_dense_token(qw3_session *s, int token,\n                                               char *err, size_t errlen,\n                                               int logits_mode) {')))
head = source[:start]
tail = source[start:]
old_start = tail.index('            ok = qw3_metal_session_matvec_quant_x1_to_scratch(')
old_end = tail.index('                 qw3_metal_session_matvec_quant_x1_to_scratch(\n                     s->metal, lw->linear_ssm_alpha')
original = tail[old_start:old_end]
tail = tail[:old_start] + '''            ok = (dual_enabled(lw->linear_qkv_proj->type, lw->linear_gate_proj->type)
                ? dual_matvec(s->metal, lw->linear_qkv_proj->offset, lw->linear_gate_proj->offset,
                    lw->linear_qkv_proj->type, lw->linear_gate_proj->type, n_embd, n_qkv, inner_n)
                : (''' + original.removeprefix('            ok = ').removesuffix(' &&\n') + ''')) &&
''' + tail[old_end:]
needle = '        } else {\n            ok = qw3_metal_session_matvec_quant_x1_to_scratch(\n                     s->metal, lw->ffn_gate->offset, lw->ffn_gate->type,'
assert tail.count(needle) >= 1
tail = tail.replace(needle, '''        } else if (dual_enabled(lw->ffn_gate->type, lw->ffn_up->type)) {
            ok = dual_matvec(s->metal, lw->ffn_gate->offset, lw->ffn_up->offset,
                lw->ffn_gate->type, lw->ffn_up->type, n_embd, n_ff, n_ff);
''' + needle, 1)
core.write_text(head + tail)

(work / 'bench.m').write_text(r'''
#include "adapter.inc"
#define main original_bench
#include "qw3_bench.c"
#undef main
int main(int argc, char **argv) {
    const char *mode = getenv("QW3_DUAL_DISPATCH");
    dual_mode = mode ? (unsigned)atoi(mode) : 0;
    if (dual_mode > 3) return 2;
    int rc = original_bench(argc, argv);
    fprintf(stderr, "dual_dispatch mode=%u same=%u mixed=%u\n", dual_mode, dual_same_calls, dual_mixed_calls);
    return rc;
}
''')
prior = repo / 'docs/measurements/staged128_partial_2026-11-17/quality.m'
q = prior.read_text()
q = '#include "adapter.inc"\n' + q[q.index('#include "gqa_heads_numeric_oracles.inc"'):]
q = q.replace('partial_mode', 'dual_mode').replace('partial staged quality failed', 'dual dispatch quality failed')
q = q.replace('                dual_mode = phase ? mode : 0;', '''                dual_mode = phase ? mode : 0;
                dual_same_calls = dual_mixed_calls = 0;''')
q = q.replace('                if (!grid_oracle_state(oracle, !phase, session) ||', '''                if (dual_same_calls != (phase && (mode & 1) ? 12u * (unsigned)steps : 0u) ||
                    dual_mixed_calls != (phase && (mode & 2) ? 25u * (unsigned)steps : 0u)) {
                    fprintf(stderr, "unexpected dispatch counts same=%u mixed=%u\\n", dual_same_calls, dual_mixed_calls);
                    goto done;
                }
                if (!grid_oracle_state(oracle, !phase, session) ||''')
(work / 'quality.m').write_text(q)
flags = ['-O3','-ffast-math','-mcpu=native','-Wall','-Wextra','-fobjc-arc']
for name, extra in [('bench', []), ('quality', ['-fno-finite-math-only'])]:
    subprocess.run(['cc', *flags, *extra, '-I'+str(repo/'tests'), f'{name}.m',
                    '-lm','-pthread','-framework','Foundation','-framework','Metal',
                    '-o',name], cwd=work, check=True)
print(work)
