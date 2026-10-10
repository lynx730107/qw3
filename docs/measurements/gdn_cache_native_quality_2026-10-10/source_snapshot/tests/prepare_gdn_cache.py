"""Private decode state-load reuse, preserving per-column FP32 sums."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / 'tests/.gdn_cache'
out.mkdir(exist_ok=True)
source = (root / 'metal/qw3_core_deltanet.metal').read_text()
name = 'qw3_deltanet_fused_gdn_scratch'
start = source.index('kernel void ' + name + '(')
end = source.index('\n}\n', start) + 3
native = source[start:end]
kernels = []
for width in (0, 32, 64, 128):
    body = native.replace(name, f'private_gdn_cache{width}')
    # Read only the SIMD partial sums actually written by this workgroup.
    anchor = 'ss = lane < 32 ? sh[lane] : 0.0f;'
    assert body.count(anchor) == 1 or 'ss = lane < ((args.head_dim + 31u) / 32u) ? sh[lane] : 0.0f;' in body
    body = body.replace(anchor, 'ss = lane < ((args.head_dim + 31u) / 32u) ? sh[lane] : 0.0f;')
    if width:
        body = body.replace('    if (hv >= args.v_heads || j >= args.head_dim) return;',
                            '    if (hv >= args.v_heads || j >= args.head_dim || args.head_dim != 128u) return;')
        anchor = '''    float sk = 0.0f;
    for (uint i = 0; i < args.head_dim; i++) sk += sin[i * args.head_dim + j] * kh[i];'''
        assert body.count(anchor) == 1
        body = body.replace(anchor, f'''    constexpr uint CACHE_N = {width}u;
    float cached[CACHE_N];
    float sk = 0.0f;
    #pragma unroll
    for (uint i = 0; i < 128u; i++) {{
        const float prior = sin[i * args.head_dim + j];
        if (i < CACHE_N) cached[i] = prior;
        sk += prior * kh[i];
    }}''')
        anchor = 'float sv = sin[idx] * g + kh[i] * d;'
        assert body.count(anchor) == 1
        body = body.replace(anchor, 'float sv = (i < CACHE_N ? cached[i] : sin[idx]) * g + kh[i] * d;')
        anchor = '    for (uint i = 0; i < args.head_dim; i++) {'
        assert body.count(anchor) == 1
        body = body.replace(anchor, '    #pragma unroll\n    for (uint i = 0; i < 128u; i++) {')
    kernels.append(body)
# Deliberate nonzero values in unwritten slots expose the old reduction's
# dependence on uninitialized threadgroup memory; not a performance variant.
poison = native.replace(name, 'private_gdn_poison')
poison = poison.replace('ss = lane < ((args.head_dim + 31u) / 32u) ? sh[lane] : 0.0f;',
                        'ss = lane < 32 ? sh[lane] : 0.0f;')
anchor = '    if (j == 0) {'
assert poison.count(anchor) == 1
poison = poison.replace(anchor, '''    if (j < 32u) sh[j] = 17.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (j == 0) {''')
kernels.append(poison)
unrolled = native.replace(name, 'private_gdn_unroll')
unrolled = unrolled.replace('    if (hv >= args.v_heads || j >= args.head_dim) return;',
                            '    if (hv >= args.v_heads || j >= args.head_dim || args.head_dim != 128u) return;')
unrolled = unrolled.replace('ss = lane < 32 ? sh[lane] : 0.0f;',
                           'ss = lane < ((args.head_dim + 31u) / 32u) ? sh[lane] : 0.0f;')
assert unrolled.count('for (uint i = 0; i < args.head_dim; i++)') == 2
unrolled = unrolled.replace('for (uint i = 0; i < args.head_dim; i++)',
                           '#pragma unroll\n    for (uint i = 0; i < 128u; i++)')
kernels.append(unrolled)
(out / 'kernels.metal').write_text('\n'.join(kernels))

quality = (root / 'tests/test_native_prefill.m').read_text()
quality = quality.replace('#include "native_prefill_validation.inc"', '#include "../gdn_cache_candidates.inc"')
quality = quality.replace('if (argc!=3) return 2;', '''if (argc!=4) return 2;
        cache_width=atoi(argv[3]);
        if (cache_width!=0 && cache_width!=16 && cache_width!=32 && cache_width!=64 && cache_width!=128) return 2;''')
quality = quality.replace('!qw3_metal_compile_kernels() ||', '!cache_compile() ||')
start = quality.index('        if (((mode & 1)')
end = quality.index('        rc=0;', start)
quality = quality[:start] + '''        if (!cache_calls) goto done;
        printf("PASS gdn_cache=%u calls=%llu\\n",cache_width,(unsigned long long)cache_calls);
''' + quality[end:]
(out / 'quality.inc').write_text(quality)

native_quality = (root / 'tests/test_native_prefill.m').read_text()
native_quality = native_quality.replace('#include "native_prefill_validation.inc"',
                                      '#include "../native_prefill_validation.inc"')
assert native_quality.count('        rc=0;') == 1
native_quality = native_quality.replace('        rc=0;', '''        if (!g_deltanet_fused_gdn_cache64_pipeline || !native_gdn_cache_calls) goto done;
        printf("PASS native_gdn_cache_calls=%llu\\n",(unsigned long long)native_gdn_cache_calls);
        rc=0;''')
(out / 'native_quality.inc').write_text(native_quality)

bench = (root / 'tests/bench_iq3_groups.m').read_text()
bench = bench.replace('.iq3_groups/quality.inc', '.gdn_cache/quality.inc')
bench = bench.replace('".gdn_cache/quality.inc"', '"quality.inc"')
bench = bench.replace('"../qw3_bench.c"', '"../../qw3_bench.c"')
bench = bench.replace('groups_', 'cache_').replace('IQ3 groups', 'GDN cache')
bench = bench.replace('cache_width!=4 && cache_width!=8',
                      'cache_width!=0 && cache_width!=16 && cache_width!=32 && cache_width!=64 && cache_width!=128')
bench = bench.replace('72u*19u', '72u*48u')
bench = bench.replace('PASS groups=', 'PASS cache=')
(out / 'bench.inc').write_text(bench)
