"""Stage several K=32 weight tiles; keep native TensorOps accumulation order."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / 'tests/.kstage_candidates'
out.mkdir(exist_ok=True)
source = (root / 'metal/qw3_core_moe.metal').read_text()
kernels = []
for fmt in ('iq3_s', 'iq4_xs'):
    name = f'qw3_matmul_{fmt}_mpp_staged128'
    start = source.index('kernel void ' + name + '(')
    end = source.index('\n}\n', start) + 3
    original = source[start:end]
    for width in (64, 128):
        body = original.replace(name, f'kstage_{fmt}_{width}')
        body = body.replace('constexpr int NK = 32;', f'constexpr int NK = 32;\n    constexpr int STAGE_K = {width};')
        body = body.replace('tensor(sa, dextents<int32_t, 2>(NK, NR0))',
                            'tensor(sa, dextents<int32_t, 2>(STAGE_K, NR0))')
        anchor = '    for (uint loop_k = 0u; loop_k < args.n_in; loop_k += NK) {'
        assert body.count(anchor) == 1
        body = body.replace(anchor, '''    for (uint stage_k = 0u; stage_k < args.n_in; stage_k += STAGE_K) {
        for (uint sub_k = 0u; sub_k < STAGE_K; sub_k += NK) {
        const uint loop_k = stage_k + sub_k;''')
        body = body.replace('sa[row * NK + half_k * 16 + i]',
                            'sa[row * STAGE_K + int(sub_k) + half_k * 16 + i]')
        anchor = '''        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(int(loop_k),
                           r1 + int(simd_group) * ROWS_PER_SG);
        mm.run(mB, mA, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);'''
        assert body.count(anchor) == 1
        body = body.replace(anchor, '''        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint sub_k = 0u; sub_k < STAGE_K; sub_k += NK) {
            auto mA = tA.slice(int(sub_k), 0);
            auto mB = tB.slice(int(stage_k + sub_k),
                               r1 + int(simd_group) * ROWS_PER_SG);
            mm.run(mB, mA, cT);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);''')
        kernels.append(body)
(out / 'kernels.metal').write_text('#ifdef QW3_METAL_HAS_TENSOR\n' + '\n'.join(kernels) + '#endif\n')

bridge = (root / 'qw3_metal.m').read_text()
anchor = '[direct setThreadgroupMemoryLength:8192u atIndex:0];'
assert bridge.count(anchor) == 2
bridge = bridge.replace(anchor,
    '[direct setThreadgroupMemoryLength:stage_pipeline_memory(selected_pipeline) atIndex:0];')
(out / 'bridge.inc').write_text(bridge)

quality = (root / 'tests/test_native_prefill.m').read_text()
quality = quality.replace('#include "native_prefill_validation.inc"', '#include "../kstage_candidates.inc"')
quality = quality.replace('return name && !strcmp(name, "all") ? 3 : -1;',
    'return name && !strcmp(name,"iq3") ? 1 : name && !strcmp(name,"iq4") ? 2 : name && !strcmp(name,"both") ? 3 : -1;')
quality = quality.replace('if (argc!=3) return 2;', 'if (argc!=4) return 2;\n        stage_width=atoi(argv[3]);\n        if (stage_width!=64 && stage_width!=128) return 2;')
quality = quality.replace('!qw3_metal_compile_kernels() ||', '!stage_compile() ||')
start = quality.index('        if (((mode & 1)')
end = quality.index('        rc=0;', start)
quality = quality[:start] + '''        if (((mode & 1) && !stage_iq3_calls) || ((mode & 2) && !stage_iq4_calls)) goto done;
        printf("PASS mode=%s stage_k=%d iq3=%llu iq4=%llu\\n",argv[2],stage_width,
            (unsigned long long)stage_iq3_calls,(unsigned long long)stage_iq4_calls);
''' + quality[end:]
(out / 'quality.inc').write_text(quality)

matrix = (root / 'tests/test_register_iq4.m').read_text()
start = matrix.index('static int register_quality(')
end = matrix.index('static int register_bench(', start)
matrix = matrix[start:end].replace('register_quality', 'stage_quality')
matrix = matrix.replace('register_iq4_encode(obj,', 'stage_encode(obj,w->type,')
matrix = matrix.replace('register_matrix_exact', 'kstage_matrix_exact')
(out / 'matrix.inc').write_text(matrix)
