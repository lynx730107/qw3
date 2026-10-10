"""Private GDN gate scheduling: remove the first threadgroup barrier."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / 'tests/.gdn_local_gates'
out.mkdir(exist_ok=True)
source = (root / 'metal/qw3_core_deltanet.metal').read_text()
name = 'qw3_deltanet_fused_gdn_cache64'
start = source.index('kernel void ' + name + '(')
end = source.index('\n}\n', start) + 3
native = source[start:end]
gate = '''    float beta_raw = scratch[args.beta_offset + hv];
    float b = 1.0f / (1.0f + exp(-beta_raw));
    float alpha_raw = scratch[args.alpha_offset + hv] + dt_bias[hv];
    float sp = alpha_raw > 20.0f ? alpha_raw : (alpha_raw < -20.0f ? exp(alpha_raw) : log(1.0f + exp(alpha_raw)));
    float g = exp(sp * a[hv]);
'''
simd_gate = '''    float b = 0.0f, g = 0.0f;
    if (lane == 0) {
        float beta_raw = scratch[args.beta_offset + hv];
        b = 1.0f / (1.0f + exp(-beta_raw));
        float alpha_raw = scratch[args.alpha_offset + hv] + dt_bias[hv];
        float sp = alpha_raw > 20.0f ? alpha_raw : (alpha_raw < -20.0f ? exp(alpha_raw) : log(1.0f + exp(alpha_raw)));
        g = exp(sp * a[hv]);
    }
    b = simd_shuffle(b, 0u);
    g = simd_shuffle(g, 0u);
'''
kernels = []
for label, replacement in [('all', gate), ('simd', simd_gate)]:
    body = native.replace(name, 'private_gdn_local_' + label)
    start = body.index('    if (j == 0) {')
    end = body.index('    constexpr uint CACHE_N = 64u;', start)
    assert body[start:end].count('threadgroup_barrier') == 1
    assert body[start:end].endswith('    float g = sh[1];\n')
    body = body[:start] + replacement + body[end:]
    kernels.append(body)
(out / 'kernels.metal').write_text('\n'.join(kernels))

quality = (root / 'tests/test_native_prefill.m').read_text()
quality = quality.replace('#include "native_prefill_validation.inc"', '#include "../gdn_local_gates_candidates.inc"')
quality = quality.replace('if (argc!=3) return 2;', '''if (argc!=4) return 2;
        local_variant=atoi(argv[3]);
        if (local_variant<1 || local_variant>2) return 2;''')
quality = quality.replace('!qw3_metal_compile_kernels() ||', '!local_compile() ||')
start = quality.index('        if (((mode & 1)')
end = quality.index('        rc=0;', start)
quality = quality[:start] + '''        if (!local_calls || !local_baseline_calls) goto done;
        printf("PASS gdn_local_gates=%u candidate_calls=%llu baseline_cache64_calls=%llu\\n",local_variant,
            (unsigned long long)local_calls,(unsigned long long)local_baseline_calls);
''' + quality[end:]
(out / 'quality.inc').write_text(quality)

bench = (root / 'tests/bench_iq3_groups.m').read_text()
bench = bench.replace('".iq3_groups/quality.inc"', '"quality.inc"')
bench = bench.replace('"../qw3_bench.c"', '"../../qw3_bench.c"')
bench = bench.replace('groups_', 'local_').replace('IQ3 groups', 'GDN local gates')
bench = bench.replace('local_width', 'local_variant')
bench = bench.replace('local_variant!=4 && local_variant!=8', 'local_variant<1 || local_variant>2')
bench = bench.replace('local_calls=0;', 'local_calls=0;local_baseline_calls=0;')
bench = bench.replace('local_calls!=72u*19u', 'local_calls!=72u*48u || local_baseline_calls!=72u*48u')
bench = bench.replace('PASS groups=', 'PASS local_gates=')
(out / 'bench.inc').write_text(bench)

matrix = (root / 'tests/test_gdn_cache.m').read_text()
matrix = matrix.replace('cache_', 'local_').replace('GDN cache', 'GDN local gates')
matrix = matrix.replace('".gdn_cache/quality.inc"', '"quality.inc"')
matrix = matrix.replace('int rc=1,cases=0,poison_cases=0,native_disagreements=0;', 'int rc=1,cases=0;')
matrix = matrix.replace('for (uint32_t hd=32;hd<=128;hd*=2)', 'for (uint32_t hd=128;hd<=128;hd*=2)')
matrix = matrix.replace('state[7],inner[7]', 'state[3],inner[3]')
matrix = matrix.replace('for (int v=0;v<7;v++)', 'for (int v=0;v<3;v++)')
matrix = matrix.replace('                if (hd!=128u && ((v>=2 && v<=4) || v==6)) continue;\n', '')
anchor = '            for (int v=0;v<3;v++) {'
assert matrix.count(anchor) == 2
extreme_gates = '''            /* Exercise both softplus tails and saturated sigmoid inputs. */
            float *gate_scratch=(float *)inputs[3].contents+4;
            const float *bias=(const float *)inputs[4].contents+4;
            for (uint32_t h=0;h<vh;h++) {
                gate_scratch[args.alpha+h]=(h%3u==0u ? -21.0f : h%3u==1u ? 21.0f : 0.25f)-bias[h];
                gate_scratch[args.beta+h]=h%4u==0u ? -80.0f : h%4u==1u ? 80.0f : h%4u==2u ? 0.7f : -0.35f;
            }
            for (int b=0;b<7;b++) input_copy[b]=[NSData dataWithBytes:inputs[b].contents length:inputs[b].length];
'''
matrix = matrix.replace(anchor, extreme_gates + anchor, 1)
anchor = '                if (memcmp(state[0].contents,state[v].contents,state[0].length)) goto done;'
assert matrix.count(anchor) == 1
matrix = matrix.replace(anchor, '''                if (memcmp(state[0].contents,state[v].contents,state[0].length)) {
                    const float *baseline=state[0].contents,*candidate=state[v].contents;
                    for (uint32_t i=0;i<state_n+12u;i++) if (memcmp(baseline+i,candidate+i,sizeof(float))) {
                        fprintf(stderr,"GDN local gates state mismatch variant=%d head_dim=%u v_heads=%u index=%u baseline=%.9g candidate=%.9g\\n",
                            v,hd,vh,i,baseline[i],candidate[i]);break;
                    }
                    goto done;
                }''')
start = matrix.index('                if (v==5) {')
end = matrix.index('\n            }\n        }', start)
matrix = matrix[:start] + '''                double cpu_error=local_cpu_inner_error(inputs,state[v],inner[v],qh,vh,hd,args.z,args.eps);
                if (!equal || cpu_error>0.00002) {
                    fprintf(stderr,"GDN local gates mismatch head_dim=%u variant=%d max=%.9g\\n",hd,v,cpu_error);goto done;
                }
                cases++;printf("matrix_exact head_dim=%u v_heads=%u variant=%d state=exact inner=native_cache64_exact cpu_norm_max=%.9g guards=ok inputs=unchanged\\n",hd,vh,v,cpu_error);''' + matrix[end:]
start = matrix.index('        if (native_disagreements)')
end = matrix.index('\ndone:', start)
matrix = matrix[:start] + '        if (cases!=4) goto done;\n        printf("PASS matrix_cases=%d\\n",cases);rc=0;' + matrix[end:]
(out / 'matrix.inc').write_text(matrix)
