"""Private IQ3_S single-matvec sign folding; accept only bit-exact compiled results."""
from pathlib import Path
root = Path(__file__).resolve().parent.parent
out = root / 'tests/.iq3_signed'
out.mkdir(exist_ok=True)
source = (root / 'metal/qw3_core_moe.metal').read_text()
def extract(marker):
    first = source.index(marker)
    last = source.index('\n}\n', first) + 3
    return source[first:last]
helper = extract('template<typename Grid>\ninline float qw3_iq3s_dot32_shared(')
kernel = extract('kernel void qw3_matvec_iq3_s_fast(')
old = '''            const float sign1 = (uint(s) & (1u << j)) ? -1.0f : 1.0f;
            const float sign2 = (uint(s) & (1u << (j + 4u))) ? -1.0f : 1.0f;
            const float v1 = qw3_iq3s_grid_value(g1, j);
            const float v2 = qw3_iq3s_grid_value(g2, j);
            sum += db * v1 * sign1 * xx[8u * l + j];
            sum += db * v2 * sign2 * xx[8u * l + j + 4u];'''
assert helper.count(old) == 1
replacement = '''            const int m1 = int(2u * ((uint(g1) >> (3u * j)) & 7u) + 1u);
            const int m2 = int(2u * ((uint(g2) >> (3u * j)) & 7u) + 1u);
            const float v1 = float((uint(s) & (1u << j)) ? -m1 : m1);
            const float v2 = float((uint(s) & (1u << (j + 4u))) ? -m2 : m2);
            sum += db * v1 * xx[8u * l + j];
            sum += db * v2 * xx[8u * l + j + 4u];'''
shader = '#include <metal_stdlib>\nusing namespace metal;\nstruct qw3_matvec_q8_0_args {uint n_in,n_out,row_bytes;};\n'
for variant in [1, 2]:
    dot = helper.replace(old, replacement).replace('qw3_iq3s_dot32_shared', f'private_iq3_signed_dot{variant}')
    if variant == 2:
        dot = dot.replace('sum += db * v1 * xx[8u * l + j];', 'sum = fma(db * v1, xx[8u * l + j], sum);')
        dot = dot.replace('sum += db * v2 * xx[8u * l + j + 4u];', 'sum = fma(db * v2, xx[8u * l + j + 4u], sum);')
    shader += dot + kernel.replace('qw3_matvec_iq3_s_fast', f'private_iq3_signed_fast{variant}').replace('qw3_iq3s_dot32_shared', f'private_iq3_signed_dot{variant}')
(out / 'kernels.metal').write_text(shader)

hooks = (root / 'tests/iq4_load_candidates.inc').read_text().replace('iq4_load', 'iq3_signed').replace('IQ4Load', 'IQ3Signed').replace('iq4Load', 'iq3Signed').replace('iq4_variant', 'signed_variant')
hooks = hooks.replace('/* Private integer-load variants. Preserve all native FP expressions. */', '/* Fold exact integer signs in a private IQ3_S shader; numerical gates remain strict. */')
hooks = hooks.replace('iq3_signed_pipeline[4][3]', 'iq3_signed_pipeline[3]').replace('iq3_signed_calls[3]', 'iq3_signed_calls')
first = hooks.index('- (void)iq3SignedSetPipeline:', hooks.index('@implementation'))
last = hooks.index('\n@end', first)
hooks = hooks[:first] + '''- (void)iq3SignedSetPipeline:(id<MTLComputePipelineState>)pipeline {
    if (pipeline==iq3_signed_pipeline[1] || pipeline==iq3_signed_pipeline[2]) iq3_signed_calls++;
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}''' + hooks[last:]
hooks = hooks.replace('#include "../qw3_metal.m"', '#include "../../qw3_metal.m"').replace('#include "../qw3.c"', '#include "../../qw3.c"')
first = hooks.index('static NSString *iq3_signed_extract(')
hooks = hooks[:first] + '''static int iq3_signed_compile(void) {
    if (iq3_signed_pipeline[2]) return 1;
    if (!qw3_metal_compile_kernels()) return 0;
    iq3_signed_pipeline[0]=g_matvec_iq3_s_pipeline;
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    NSString *source=[NSString stringWithContentsOfFile:@"tests/.iq3_signed/kernels.metal" encoding:NSUTF8StringEncoding error:&error];
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    for (int v=1;v<=2;v++) {
        NSString *name=[NSString stringWithFormat:@"private_iq3_signed_fast%d",v];
        id<MTLFunction> fn=[lib newFunctionWithName:name];
        iq3_signed_pipeline[v]=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!iq3_signed_pipeline[v] || iq3_signed_pipeline[v].maxTotalThreadsPerThreadgroup<64) {
            fprintf(stderr,"IQ3 signed compile variant=%d: %s\\n",v,error.localizedDescription.UTF8String);return 0;
        }
    }
    return 1;
}
static void scratch_select(int mode) {
    setenv("QW3_METAL_DENSE_PREFILL_FAST","1",1);
    g_matvec_iq3_s_pipeline=iq3_signed_pipeline[mode ? signed_variant : 0];
}
'''
(out / 'hooks.inc').write_text(hooks)
quality = (root / 'tests/test_native_prefill.m').read_text().replace('#include "native_prefill_validation.inc"', '#include "hooks.inc"')
quality = quality.replace('if (argc!=3) return 2;', 'if (argc!=4) return 2;\n        signed_variant=atoi(argv[3]);if (signed_variant<1 || signed_variant>2) return 2;')
quality = quality.replace('!qw3_metal_compile_kernels() ||', '!iq3_signed_compile() ||')
first = quality.index('        if (((mode & 1)');last = quality.index('        rc=0;', first)
quality = quality[:first] + '''        if (!iq3_signed_calls) goto done;
        printf("PASS IQ3_S signed variant=%d candidate_calls=%llu\\n",signed_variant,(unsigned long long)iq3_signed_calls);
''' + quality[last:]
(out / 'quality.inc').write_text(quality)
bench = (root / 'tests/bench_iq3_groups.m').read_text().replace('.iq3_groups/quality.inc', 'quality.inc').replace('../qw3_bench.c', '../../qw3_bench.c')
bench = bench.replace('groups_width=atoi(argv[2]);if (groups_width!=4 && groups_width!=8) return 2;', 'signed_variant=atoi(argv[2]);if (signed_variant<1 || signed_variant>2) return 2;')
bench = bench.replace('groups_compile()', 'iq3_signed_compile()').replace('groups_calls', 'iq3_signed_calls').replace('72u*19u', '72u*73u')
bench = bench.replace('groups_width', 'signed_variant').replace('PASS groups=%u', 'PASS iq3_signed=%u').replace('IQ3 groups failed:', 'IQ3 signed failed:')
(out / 'bench.inc').write_text(bench)
