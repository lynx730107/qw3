"""Private IQ4_XS row-reuse probe with native per-row FP32 expressions."""
from pathlib import Path
root = Path(__file__).resolve().parent.parent
out = root / 'tests/.iq4_rows'
out.mkdir(exist_ok=True)
linear = (root / 'metal/qw3_core_linear.metal').read_text()
all_source = '\n'.join(p.read_text() for p in sorted((root / 'metal').glob('*.metal')))
def extract(s, marker, end='\n}\n'):
    first = s.index(marker)
    last = s.index(end, first) + len(end)
    return s[first:last]
names = ['qw3_matvec_iq4_xs_fast', 'qw3_matvec_iq4_xs_pair_fast', 'qw3_matvec_iq4_xs_add_x0_fast']
shader = '#include <metal_stdlib>\nusing namespace metal;\nstruct qw3_matvec_q8_0_args {uint n_in,n_out,row_bytes;};\n'
shader += extract(all_source, 'constant float qw3_iq4nl_table[16]', '\n};') + '\n'
shader += extract(all_source, 'inline float qw3_iq4nl_val(')
shader += extract(linear, 'inline float qw3_iq4xs_dot8_vector(')
for rows in [4,8]:
    for kind, name in enumerate(names):
        body = extract(linear, 'kernel void ' + name + '(')
        first = body.index('    float gate0' if kind == 1 else '    float sum0')
        nblocks = body.index('    uint n_blocks', first)
        prefix = body[:first].replace(name, f'private_iq4_rows{rows}_{kind}').replace('const uint nr0 = 2u;', f'const uint nr0 = {rows}u;')
        variables = ['gate','up'] if kind == 1 else ['sum']
        prefix += ''.join(f'    float {v}{i} = 0.0f;\n' for v in variables for i in range(rows))
        loop = body[nblocks:body.index('        uint row = first_row;', nblocks)]
        # Reuse the exact native activation load order and K-block loop.
        prefix += loop
        for i in range(rows):
            prefix += f'        {{\n            uint row = first_row + {i}u;\n            if (row < args.n_out) {{\n'
            if kind == 1:
                prefix += '                uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;\n'
                prefix += ''.join(f'                {side}{i} += qw3_iq4xs_dot8_vector({side}_weights + off, xl, sh, ib, il);\n' for side in ['gate','up'])
            else:
                prefix += '                device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;\n'
                prefix += f'                sum{i} += qw3_iq4xs_dot8_vector(blk, xl, sh, ib, il);\n'
            prefix += '            }\n        }\n'
        prefix += '    }\n'
        prefix += ''.join(f'    {v}{i} = simd_sum({v}{i});\n' for v in variables for i in range(rows))
        prefix += '    if (lane == 0) {\n'
        for i in range(rows):
            prefix += f'        {{ uint row = first_row + {i}u;\n            if (row < args.n_out) {{\n'
            if kind == 1:
                prefix += f'                out[row] = gate{i};\n                out[args.n_out + row] = up{i};\n'
            elif kind == 2:
                prefix += f'                x0[row] = x0[row] + sum{i} * scale;\n'
            else:
                prefix += f'                out[row] = sum{i};\n'
            prefix += '            }\n        }\n'
        shader += prefix + '    }\n}\n'
(out / 'kernels.metal').write_text(shader)

hooks = (root / 'tests/iq4_load_candidates.inc').read_text().replace('iq4_load', 'iq4_rows').replace('IQ4Load', 'IQ4Rows').replace('iq4Load', 'iq4Rows')
hooks = hooks.replace('/* Private integer-load variants. Preserve all native FP expressions. */', '/* Private row-reuse variants; preserve native FP32 expressions per row. */')
hooks = hooks.replace('#import <Metal/Metal.h>', '#import <Metal/Metal.h>\n#import <objc/runtime.h>')
hooks = hooks.replace('iq4_rows_pipeline[4][3]', 'iq4_rows_pipeline[3][3]')
hooks = hooks.replace('static int iq4_variant=1;', 'static int iq4_variant=1;\nstatic void *iq4_rows_factor_key=&iq4_rows_factor_key;')
hooks = hooks.replace('@protocol QW3IQ4RowsHooks', '@protocol QW3IQ4RowsHooks\n- (void)iq4RowsDispatch:(MTLSize)groups threadsPerThreadgroup:(MTLSize)threads;')
first = hooks.index('- (void)iq4RowsSetPipeline:', hooks.index('@implementation'))
last = hooks.index('\n@end', first)
hooks = hooks[:first] + '''- (void)iq4RowsSetPipeline:(id<MTLComputePipelineState>)pipeline {
    unsigned factor=1;
    for (int kind=0;kind<3;kind++) for (int variant=1;variant<=2;variant++)
        if (pipeline==iq4_rows_pipeline[variant][kind]) {iq4_rows_calls[kind]++;factor=variant==1 ? 2u : 4u;}
    objc_setAssociatedObject(self,iq4_rows_factor_key,@(factor),OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
- (void)iq4RowsDispatch:(MTLSize)groups threadsPerThreadgroup:(MTLSize)threads {
    unsigned factor=[objc_getAssociatedObject(self,iq4_rows_factor_key) unsignedIntValue];
    if (factor>1u) {
        if (groups.height!=1 || groups.depth!=1 || threads.width!=64) abort();
        groups.width=(groups.width+factor-1u)/factor;
    }
    [(id<MTLComputeCommandEncoder>)self dispatchThreadgroups:groups threadsPerThreadgroup:threads];
}''' + hooks[last:]
hooks = hooks.replace('#define setComputePipelineState iq4RowsSetPipeline', '#define dispatchThreadgroups iq4RowsDispatch\n#define setComputePipelineState iq4RowsSetPipeline')
hooks = hooks.replace('#undef setComputePipelineState', '#undef dispatchThreadgroups\n#undef setComputePipelineState')
hooks = hooks.replace('#include "../qw3_metal.m"', '#include "../../qw3_metal.m"').replace('#include "../qw3.c"', '#include "../../qw3.c"')
first = hooks.index('static NSString *iq4_rows_extract(')
last = hooks.index('static void scratch_select(', first)
hooks = hooks[:first] + '''static int iq4_rows_compile(void) {
    if (iq4_rows_pipeline[2][2]) return 1;
    if (!qw3_metal_compile_kernels()) return 0;
    iq4_rows_pipeline[0][0]=g_matvec_iq4_xs_pipeline;
    iq4_rows_pipeline[0][1]=g_matvec_iq4_xs_pair_pipeline;
    iq4_rows_pipeline[0][2]=g_matvec_iq4_xs_add_x0_pipeline;
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    NSString *source=[NSString stringWithContentsOfFile:@"tests/.iq4_rows/kernels.metal" encoding:NSUTF8StringEncoding error:&error];
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    for (int v=1;v<=2;v++) for (int kind=0;kind<3;kind++) {
        NSString *name=[NSString stringWithFormat:@"private_iq4_rows%u_%d",v==1 ? 4u : 8u,kind];
        id<MTLFunction> fn=[lib newFunctionWithName:name];
        iq4_rows_pipeline[v][kind]=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!iq4_rows_pipeline[v][kind] || iq4_rows_pipeline[v][kind].maxTotalThreadsPerThreadgroup<64) {
            fprintf(stderr,"IQ4 rows compile variant=%d kind=%d: %s\\n",v,kind,error.localizedDescription.UTF8String);return 0;
        }
    }
    return 1;
}
''' + hooks[last:]
(out / 'hooks.inc').write_text(hooks)

quality = (root / 'tests/test_native_prefill.m').read_text().replace('#include "native_prefill_validation.inc"', '#include "hooks.inc"')
quality = quality.replace('if (argc!=3) return 2;', 'if (argc!=4) return 2;\n        iq4_variant=atoi(argv[3]);if (iq4_variant<1 || iq4_variant>2) return 2;')
quality = quality.replace('!qw3_metal_compile_kernels() ||', '!iq4_rows_compile() || !iq4_rows_model_aligned(e) ||')
first = quality.index('        if (((mode & 1)');last = quality.index('        rc=0;', first)
quality = quality[:first] + '''        if (!iq4_rows_calls[0] || !iq4_rows_calls[1] || !iq4_rows_calls[2]) goto done;
        printf("PASS IQ4_XS rows variant=%d single=%llu pair=%llu add=%llu\\n",iq4_variant,
            (unsigned long long)iq4_rows_calls[0],(unsigned long long)iq4_rows_calls[1],(unsigned long long)iq4_rows_calls[2]);
''' + quality[last:]
(out / 'quality.inc').write_text(quality)
matrix = (root / 'tests/test_iq4_loads.m').read_text().replace('.iq4_loads/quality.inc', 'quality.inc').replace('iq4_load', 'iq4_rows').replace('IQ4 loads', 'IQ4 rows').replace('iq4_variant>3', 'iq4_variant>2')
matrix = matrix.replace('(no+3u)/4u', '(no+(v ? (iq4_variant==1 ? 8u : 16u) : 4u)-1u)/(v ? (iq4_variant==1 ? 8u : 16u) : 4u)')
(out / 'matrix.inc').write_text(matrix)
bench = (root / 'tests/bench_iq3_groups.m').read_text().replace('.iq3_groups/quality.inc', 'quality.inc').replace('../qw3_bench.c', '../../qw3_bench.c')
bench = bench.replace('groups_width=atoi(argv[2]);if (groups_width!=4 && groups_width!=8) return 2;', 'iq4_variant=atoi(argv[2]);if (iq4_variant<1 || iq4_variant>2) return 2;')
bench = bench.replace('groups_compile()', 'iq4_rows_compile() || !iq4_rows_model_aligned(e)').replace('groups_calls=0;', 'memset(iq4_rows_calls,0,sizeof(iq4_rows_calls));')
bench = bench.replace('groups_calls!=72u*19u', '(iq4_rows_calls[0]!=72u*96u || iq4_rows_calls[1]!=72u*14u || iq4_rows_calls[2]!=72u*32u)')
bench = bench.replace('PASS groups=%u', 'PASS iq4_rows=%u').replace('groups_width,', 'iq4_variant,')
bench = bench.replace('(unsigned long long)groups_calls', '(unsigned long long)(iq4_rows_calls[0]+iq4_rows_calls[1]+iq4_rows_calls[2])').replace('IQ3 groups failed:', 'IQ4 rows failed:')
(out / 'bench.inc').write_text(bench)
