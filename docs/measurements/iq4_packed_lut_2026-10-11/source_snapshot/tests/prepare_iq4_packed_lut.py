"""Private packed integer IQ4_XS lookup; retain native FP32 expressions."""
import re
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / 'tests/.iq4_packed_lut'
out.mkdir(exist_ok=True)
linear = (root / 'metal/qw3_core_linear.metal').read_text()
common = (root / 'metal/qw3_core_common.metal').read_text()

def extract(source, marker, end='\n}\n'):
    first = source.index(marker)
    last = source.index(end, first) + len(end)
    return source[first:last]

table = extract(common, 'constant float qw3_iq4nl_table[16]', '\n};')
values = [int(float(v)) for v in re.findall(r'(-?\d+\.\d+)f', table)]
assert len(values) == 16 and all(-128 <= v <= 127 for v in values)
names = ['qw3_matvec_iq4_xs_fast', 'qw3_matvec_iq4_xs_pair_fast', 'qw3_matvec_iq4_xs_add_x0_fast']
shader = '#include <metal_stdlib>\nusing namespace metal;\nstruct qw3_matvec_q8_0_args {uint n_in,n_out,row_bytes;};\n'
for variant in [1, 2]:
    encoded = [(v & 255) if variant == 1 else v + 128 for v in values]
    words = [sum(encoded[i + j] << (8 * j) for j in range(4)) for i in range(0, 16, 4)]
    for q, expected in enumerate(values):
        b = (words[q // 4] >> ((q % 4) * 8)) & 255
        actual = (b if b < 128 else b - 256) if variant == 1 else b - 128
        assert actual == expected
    shader += f'''inline float private_iq4_lut{variant}(uint q) {{
    const uint word = (q & 8u) ? ((q & 4u) ? 0x{words[3]:08x}u : 0x{words[2]:08x}u)
                               : ((q & 4u) ? 0x{words[1]:08x}u : 0x{words[0]:08x}u);
'''
    if variant == 1:
        shader += '    const int value = int(word << ((3u - (q & 3u)) * 8u)) >> 24;\n'
    else:
        shader += '    const int value = int((word >> ((q & 3u) * 8u)) & 255u) - 128;\n'
    shader += '    return float(value);\n}\n'
    dot = extract(linear, 'inline float qw3_iq4xs_dot8_vector(')
    for q in ['qlo.x', 'qlo.y', 'qlo.z', 'qlo.w', 'qhi.x', 'qhi.y', 'qhi.z', 'qhi.w']:
        old = f'table[{q}]'
        assert dot.count(old) == 1
        dot = dot.replace(old, f'private_iq4_lut{variant}(uint({q}))')
    dot = dot.replace('qw3_iq4xs_dot8_vector', f'private_iq4_dot{variant}')
    shader += dot
    for kind, name in enumerate(names):
        body = extract(linear, 'kernel void ' + name + '(')
        init = 'if (simd_idx == 0u && lane < 16u) sh[lane] = qw3_iq4nl_val(uint(lane));\n    threadgroup_barrier(mem_flags::mem_threadgroup);'
        assert body.count(init) == 1
        body = body.replace(init, '(void)sh;').replace(name, f'private_iq4_packed{variant}_{kind}')
        shader += body.replace('qw3_iq4xs_dot8_vector', f'private_iq4_dot{variant}')
(out / 'kernels.metal').write_text(shader)

hooks = (root / 'tests/iq4_load_candidates.inc').read_text().replace('iq4_load', 'iq4_packed_lut').replace('IQ4Load', 'IQ4PackedLut').replace('iq4Load', 'iq4PackedLut')
hooks = hooks.replace('/* Private integer-load variants. Preserve all native FP expressions. */', '/* Private packed integer lookup; native FP32 expressions are retained. */')
hooks = hooks.replace('iq4_packed_lut_pipeline[4][3]', 'iq4_packed_lut_pipeline[3][3]').replace('variant<=3', 'variant<=2')
hooks = hooks.replace('#include "../qw3_metal.m"', '#include "../../qw3_metal.m"').replace('#include "../qw3.c"', '#include "../../qw3.c"')
first = hooks.index('static NSString *iq4_packed_lut_extract(')
last = hooks.index('static void scratch_select(', first)
hooks = hooks[:first] + '''static int iq4_packed_lut_compile(void) {
    if (iq4_packed_lut_pipeline[2][2]) return 1;
    if (!qw3_metal_compile_kernels()) return 0;
    iq4_packed_lut_pipeline[0][0]=g_matvec_iq4_xs_pipeline;
    iq4_packed_lut_pipeline[0][1]=g_matvec_iq4_xs_pair_pipeline;
    iq4_packed_lut_pipeline[0][2]=g_matvec_iq4_xs_add_x0_pipeline;
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    NSString *source=[NSString stringWithContentsOfFile:@"tests/.iq4_packed_lut/kernels.metal" encoding:NSUTF8StringEncoding error:&error];
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    for (int v=1;v<=2;v++) for (int kind=0;kind<3;kind++) {
        NSString *name=[NSString stringWithFormat:@"private_iq4_packed%d_%d",v,kind];
        id<MTLFunction> fn=[lib newFunctionWithName:name];
        iq4_packed_lut_pipeline[v][kind]=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!iq4_packed_lut_pipeline[v][kind] || iq4_packed_lut_pipeline[v][kind].maxTotalThreadsPerThreadgroup<64) {
            fprintf(stderr,"IQ4 packed LUT compile variant=%d kind=%d: %s\\n",v,kind,error.localizedDescription.UTF8String);return 0;
        }
    }
    return 1;
}
''' + hooks[last:]
(out / 'hooks.inc').write_text(hooks)

quality = (root / 'tests/test_native_prefill.m').read_text().replace('#include "native_prefill_validation.inc"', '#include "hooks.inc"')
quality = quality.replace('if (argc!=3) return 2;', 'if (argc!=4) return 2;\n        iq4_variant=atoi(argv[3]);if (iq4_variant<1 || iq4_variant>2) return 2;')
quality = quality.replace('!qw3_metal_compile_kernels() ||', '!iq4_packed_lut_compile() || !iq4_packed_lut_model_aligned(e) ||')
first = quality.index('        if (((mode & 1)');last = quality.index('        rc=0;', first)
quality = quality[:first] + '''        if (!iq4_packed_lut_calls[0] || !iq4_packed_lut_calls[1] || !iq4_packed_lut_calls[2]) goto done;
        printf("PASS IQ4_XS packed LUT variant=%d single=%llu pair=%llu add=%llu\\n",iq4_variant,
            (unsigned long long)iq4_packed_lut_calls[0],(unsigned long long)iq4_packed_lut_calls[1],(unsigned long long)iq4_packed_lut_calls[2]);
''' + quality[last:]
(out / 'quality.inc').write_text(quality)
matrix = (root / 'tests/test_iq4_loads.m').read_text().replace('.iq4_loads/quality.inc', 'quality.inc').replace('iq4_load', 'iq4_packed_lut').replace('IQ4 loads', 'IQ4 packed LUT').replace('iq4_variant>3', 'iq4_variant>2')
matrix = matrix.replace('iq4_packed_luts=', 'iq4_packed_lut=')
matrix = matrix.replace('pattern<2', 'pattern<3').replace('x[i]=pattern ?', 'x[i]=pattern==2 ? (float)((int)(i%7u)-3)*(i%5u==0u ? 0x1p8f : 0x1p-12f) : pattern==1 ?')
(out / 'matrix.inc').write_text(matrix)
bench = (root / 'tests/bench_iq3_groups.m').read_text().replace('.iq3_groups/quality.inc', 'quality.inc').replace('../qw3_bench.c', '../../qw3_bench.c')
bench = bench.replace('groups_width=atoi(argv[2]);if (groups_width!=4 && groups_width!=8) return 2;', 'iq4_variant=atoi(argv[2]);if (iq4_variant<1 || iq4_variant>2) return 2;')
bench = bench.replace('groups_compile()', 'iq4_packed_lut_compile() || !iq4_packed_lut_model_aligned(e)').replace('groups_calls=0;', 'memset(iq4_packed_lut_calls,0,sizeof(iq4_packed_lut_calls));')
bench = bench.replace('groups_calls!=72u*19u', '(iq4_packed_lut_calls[0]!=72u*96u || iq4_packed_lut_calls[1]!=72u*14u || iq4_packed_lut_calls[2]!=72u*32u)')
bench = bench.replace('PASS groups=%u', 'PASS iq4_packed_lut=%u').replace('groups_width,', 'iq4_variant,')
bench = bench.replace('(unsigned long long)groups_calls', '(unsigned long long)(iq4_packed_lut_calls[0]+iq4_packed_lut_calls[1]+iq4_packed_lut_calls[2])').replace('IQ3 groups failed:', 'IQ4 packed LUT failed:')
(out / 'bench.inc').write_text(bench)
