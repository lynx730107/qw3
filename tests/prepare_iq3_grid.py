"""Read the existing IQ3_S packed grid directly; preserve native FP expressions."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / 'tests/.iq3_grid'
out.mkdir(exist_ok=True)
source = (root / 'metal/qw3_core_moe.metal').read_text()


def extract(marker):
    first = source.index(marker)
    last = source.index('\n}\n', first) + 3
    return source[first:last]


value = extract('inline float qw3_iq3s_grid_value(ushort packed, uint j)')
helper = extract('template<typename Grid>\ninline float qw3_iq3s_dot32_shared(')
kernel = extract('kernel void qw3_matvec_iq3_s_fast(')
copy = '''    threadgroup ushort grid[512];
    const uint tid = uint(simd_idx) * 32u + uint(lane);
    for (uint i = tid; i < 512u; i += 64u) grid[i] = kgrid[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);'''
assert kernel.count(copy) == 1
assert helper.count('threadgroup const Grid *grid') == 1
shader = '#include <metal_stdlib>\nusing namespace metal;\nstruct qw3_matvec_q8_0_args {uint n_in,n_out,row_bytes;};\n' + value
for variant, space in [(1, 'device'), (2, 'constant')]:
    name = f'private_iq3_grid_dot{variant}'
    dot = helper.replace('threadgroup const Grid *grid', f'{space} const Grid *grid')
    dot = dot.replace('qw3_iq3s_dot32_shared', name)
    body = kernel.replace(copy, f'    {space} const ushort *grid = kgrid;')
    body = body.replace('device const ushort *kgrid', f'{space} const ushort *kgrid')
    body = body.replace('qw3_matvec_iq3_s_fast', f'private_iq3_grid_fast{variant}')
    body = body.replace('qw3_iq3s_dot32_shared', name)
    shader += dot + body
(out / 'kernels.metal').write_text(shader)

# Reuse only the private dispatch-counting bridge, with its own namespace.
hooks = (root / 'tests/iq4_load_candidates.inc').read_text()
hooks = hooks.replace('iq4_load', 'iq3_grid').replace('IQ4Load', 'IQ3Grid').replace('iq4Load', 'iq3Grid').replace('iq4_variant', 'grid_variant')
hooks = hooks.replace('/* Private integer-load variants. Preserve all native FP expressions. */', '/* Direct reads of the existing packed grid; no extra weight/session buffers. */')
hooks = hooks.replace('iq3_grid_pipeline[4][3]', 'iq3_grid_pipeline[3]').replace('iq3_grid_calls[3]', 'iq3_grid_calls')
first = hooks.index('- (void)iq3GridSetPipeline:', hooks.index('@implementation'))
last = hooks.index('\n@end', first)
hooks = hooks[:first] + '''- (void)iq3GridSetPipeline:(id<MTLComputePipelineState>)pipeline {
    if (pipeline==iq3_grid_pipeline[1] || pipeline==iq3_grid_pipeline[2]) iq3_grid_calls++;
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}''' + hooks[last:]
hooks = hooks.replace('#include "../qw3_metal.m"', '#include "../../qw3_metal.m"').replace('#include "../qw3.c"', '#include "../../qw3.c"')
first = hooks.index('static NSString *iq3_grid_extract(')
hooks = hooks[:first] + '''static int iq3_grid_compile(void) {
    if (iq3_grid_pipeline[2]) return 1;
    if (!qw3_metal_compile_kernels()) return 0;
    iq3_grid_pipeline[0]=g_matvec_iq3_s_pipeline;
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    NSString *source=[NSString stringWithContentsOfFile:@"tests/.iq3_grid/kernels.metal" encoding:NSUTF8StringEncoding error:&error];
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    for (int v=1;v<=2;v++) {
        NSString *name=[NSString stringWithFormat:@"private_iq3_grid_fast%d",v];
        id<MTLFunction> fn=[lib newFunctionWithName:name];
        iq3_grid_pipeline[v]=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!iq3_grid_pipeline[v] || iq3_grid_pipeline[v].maxTotalThreadsPerThreadgroup<64) {
            fprintf(stderr,"IQ3 grid compile variant=%d: %s\\n",v,error.localizedDescription.UTF8String);return 0;
        }
    }
    return 1;
}
static __attribute__((unused)) void scratch_select(int mode) {
    g_matvec_iq3_s_pipeline=iq3_grid_pipeline[mode ? grid_variant : 0];
}
'''
(out / 'hooks.inc').write_text(hooks)

# Matrix checks run without any inference session or copied model weights.
matrix = (root / 'tests/test_iq3_signed.m').read_text()
matrix = matrix[matrix.index('int main(int argc,char **argv) {'):]
matrix = matrix.replace('    if (argc==4) return iq3_signed_full_quality_main(argc,argv);\n', '')
matrix = matrix.replace('iq3_signed', 'iq3_grid').replace('signed_variant', 'grid_variant').replace('IQ3 signed', 'IQ3 grid')
(out / 'matrix.inc').write_text(matrix)
