"""Private cache extent/position controls against the native M5 cache64."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / 'tests/.gdn_windows'
out.mkdir(exist_ok=True)
source = (root / 'metal/qw3_core_deltanet.metal').read_text()
name = 'qw3_deltanet_fused_gdn_cache64'
start = source.index('kernel void ' + name + '(')
end = source.index('\n}\n', start) + 3
native = source[start:end]
variants = [('first48', 48, 0), ('first80', 80, 0), ('first96', 96, 0),
            ('middle64', 64, 32), ('last64', 64, 64)]
kernels = []
for label, count, offset in variants:
    body = native.replace(name, 'private_gdn_' + label)
    anchor = '    constexpr uint CACHE_N = 64u;'
    assert body.count(anchor) == 1
    body = body.replace(anchor, f'    constexpr uint CACHE_N = {count}u;\n    constexpr uint CACHE_START = {offset}u;')
    anchor = 'if (i < CACHE_N) cached[i] = prior;'
    assert body.count(anchor) == 1
    body = body.replace(anchor, 'if (i >= CACHE_START && i < CACHE_START + CACHE_N) cached[i - CACHE_START] = prior;')
    anchor = '(i < CACHE_N ? cached[i] : sin[idx])'
    assert body.count(anchor) == 1
    body = body.replace(anchor, '(i >= CACHE_START && i < CACHE_START + CACHE_N ? cached[i - CACHE_START] : sin[idx])')
    kernels.append(body)
(out / 'kernels.metal').write_text('\n'.join(kernels))

quality = (root / 'tests/test_native_prefill.m').read_text()
quality = quality.replace('#include "native_prefill_validation.inc"', '#include "../gdn_windows_candidates.inc"')
quality = quality.replace('if (argc!=3) return 2;', '''if (argc!=4) return 2;
        window_variant=atoi(argv[3]);
        if (window_variant<1 || window_variant>5) return 2;''')
quality = quality.replace('!qw3_metal_compile_kernels() ||', '!window_compile() ||')
start = quality.index('        if (((mode & 1)')
end = quality.index('        rc=0;', start)
quality = quality[:start] + '''        if (!window_calls || !window_baseline_calls) goto done;
        printf("PASS gdn_window=%u candidate_calls=%llu baseline_cache64_calls=%llu\\n",window_variant,
            (unsigned long long)window_calls,(unsigned long long)window_baseline_calls);
''' + quality[end:]
(out / 'quality.inc').write_text(quality)

bench = (root / 'tests/bench_iq3_groups.m').read_text()
bench = bench.replace('".iq3_groups/quality.inc"', '"quality.inc"')
bench = bench.replace('"../qw3_bench.c"', '"../../qw3_bench.c"')
bench = bench.replace('groups_', 'window_').replace('IQ3 groups', 'GDN windows')
bench = bench.replace('window_width', 'window_variant')
bench = bench.replace('window_variant!=4 && window_variant!=8', 'window_variant<1 || window_variant>5')
bench = bench.replace('window_calls=0;', 'window_calls=0;window_baseline_calls=0;')
bench = bench.replace('window_calls!=72u*19u', 'window_calls!=72u*48u || window_baseline_calls!=72u*48u')
bench = bench.replace('PASS groups=', 'PASS window=')
(out / 'bench.inc').write_text(bench)

# Preserve the independent double RMSNorm/gate check, guarded inputs and
# in-place state from the cache test. Only HD128 is supported by these probes.
matrix = (root / 'tests/test_gdn_cache.m').read_text()
matrix = matrix.replace('cache_', 'window_').replace('GDN cache', 'GDN windows')
matrix = matrix.replace('".gdn_cache/quality.inc"', '"quality.inc"')
matrix = matrix.replace('int rc=1,cases=0,poison_cases=0,native_disagreements=0;', 'int rc=1,cases=0;')
matrix = matrix.replace('for (uint32_t hd=32;hd<=128;hd*=2)', 'for (uint32_t hd=128;hd<=128;hd*=2)')
matrix = matrix.replace('state[7],inner[7]', 'state[6],inner[6]')
matrix = matrix.replace('for (int v=0;v<7;v++)', 'for (int v=0;v<6;v++)')
matrix = matrix.replace('                if (hd!=128u && ((v>=2 && v<=4) || v==6)) continue;\n', '')
start = matrix.index('                if (v==5) {')
end = matrix.index('\n            }\n        }', start)
matrix = matrix[:start] + '''                double cpu_error=window_cpu_inner_error(inputs,state[v],inner[v],qh,vh,hd,args.z,args.eps);
                if (!equal || cpu_error>0.00002) {
                    fprintf(stderr,"GDN windows mismatch head_dim=%u variant=%d max=%.9g\\n",hd,v,cpu_error);goto done;
                }
                cases++;printf("matrix_exact head_dim=%u v_heads=%u variant=%d state=exact inner=native_cache64_exact cpu_norm_max=%.9g guards=ok inputs=unchanged\\n",hd,vh,v,cpu_error);''' + matrix[end:]
start = matrix.index('        if (native_disagreements)')
end = matrix.index('\ndone:', start)
matrix = matrix[:start] + '        if (cases!=10) goto done;\n        printf("PASS matrix_cases=%d\\n",cases);rc=0;' + matrix[end:]
(out / 'matrix.inc').write_text(matrix)
