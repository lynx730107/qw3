"""Private two-column controls against the native M5 cache64."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / 'tests/.gdn_two_columns'
out.mkdir(exist_ok=True)
source = (root / 'metal/qw3_core_deltanet.metal').read_text()
name = 'qw3_deltanet_fused_gdn_cache64'
start = source.index('kernel void ' + name + '(')
end = source.index('\n}\n', start) + 3
native = source[start:end]
# Cache 32/48/64 rows for each of two columns handled by the same thread.
variants = [('cache32', 32), ('cache48', 48), ('cache64', 64)]
kernels = []
for label, count in variants:
    body = native.replace(name, 'private_gdn_two_' + label)
    edits = {
        'j >= args.head_dim || args.head_dim != 128u': 'j >= 64u || args.head_dim != 128u',
        '    uint hk = hv % args.q_heads;': '    uint j1 = uint(j) + 64u;\n    uint hk = hv % args.q_heads;',
        'constexpr uint CACHE_N = 64u;': f'constexpr uint CACHE_N = {count}u;',
        'float cached[CACHE_N];': 'float2 cached[CACHE_N];',
        'float sk = 0.0f;': 'float sk0 = 0.0f, sk1 = 0.0f;',
        'const float prior = sin[i * args.head_dim + j];\n        if (i < CACHE_N) cached[i] = prior;\n        sk += prior * kh[i];':
        'const float prior0 = sin[i * args.head_dim + j];\n        const float prior1 = sin[i * args.head_dim + j1];\n        if (i < CACHE_N) cached[i] = float2(prior0, prior1);\n        sk0 += prior0 * kh[i];\n        sk1 += prior1 * kh[i];',
        'float d = b * (vh[j] - sk * g);': 'float d0 = b * (vh[j] - sk0 * g);\n    float d1 = b * (vh[j1] - sk1 * g);',
        'float sum = 0.0f;': 'float sum0 = 0.0f, sum1 = 0.0f;',
        'float sv = (i < CACHE_N ? cached[i] : sin[idx]) * g + kh[i] * d;\n        sout[idx] = sv;\n        sum += sv * qh[i];':
        'float sv0 = (i < CACHE_N ? cached[i].x : sin[idx]) * g + kh[i] * d0;\n        float sv1 = (i < CACHE_N ? cached[i].y : sin[idx + 64u]) * g + kh[i] * d1;\n        sout[idx] = sv0;\n        sout[idx + 64u] = sv1;\n        sum0 += sv0 * qh[i];\n        sum1 += sv1 * qh[i];',
        'float core = sum * rsqrt(float(args.head_dim));\n    float ss = simd_sum(core * core);\n    if (lane == 0) sh[simd_idx] = ss;':
        'float core0 = sum0 * rsqrt(float(args.head_dim));\n    float core1 = sum1 * rsqrt(float(args.head_dim));\n    float ss = simd_sum(core0 * core0);\n    float ss1 = simd_sum(core1 * core1);\n    if (lane == 0) { sh[simd_idx] = ss; sh[simd_idx + 2u] = ss1; }',
        '    float zi = scratch[args.z_offset + uint64_t(hv) * args.head_dim + j];\n    float gate = zi / (1.0f + exp(-zi));\n    inner_out[uint64_t(hv) * args.head_dim + j] = core * scale * w[j] * gate;':
        '    float zi0 = scratch[args.z_offset + uint64_t(hv) * args.head_dim + j];\n    float zi1 = scratch[args.z_offset + uint64_t(hv) * args.head_dim + j1];\n    float gate0 = zi0 / (1.0f + exp(-zi0));\n    float gate1 = zi1 / (1.0f + exp(-zi1));\n    inner_out[uint64_t(hv) * args.head_dim + j] = core0 * scale * w[j] * gate0;\n    inner_out[uint64_t(hv) * args.head_dim + j1] = core1 * scale * w[j1] * gate1;'
    }
    for anchor, replacement in edits.items():
        assert body.count(anchor) == 1, anchor
        body = body.replace(anchor, replacement)
    kernels.append(body)
(out / 'kernels.metal').write_text('\n'.join(kernels))
core = (root / 'qw3_metal.m').read_text()
start = core.index('int qw3_metal_session_deltanet_fused_gdn_from_scratch(')
end = core.index('\n}\n', start) + 3
body = core[start:end]
anchor = 'threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)'
assert body.count(anchor) == 1
body = body.replace(anchor, 'threadsPerThreadgroup:MTLSizeMake(two_live ? head_dim / 2u : head_dim, 1, 1)')
(out / 'core.inc').write_text(core[:start] + body + core[end:])

quality = (root / 'tests/test_native_prefill.m').read_text()
quality = quality.replace('#include "native_prefill_validation.inc"', '#include "../gdn_two_columns_candidates.inc"')
quality = quality.replace('if (argc!=3) return 2;', '''if (argc!=4) return 2;
        two_variant=atoi(argv[3]);
        if (two_variant<1 || two_variant>3) return 2;''')
quality = quality.replace('!qw3_metal_compile_kernels() ||', '!two_compile() ||')
start = quality.index('        if (((mode & 1)')
end = quality.index('        rc=0;', start)
quality = quality[:start] + '''        if (!two_calls || !two_baseline_calls) goto done;
        printf("PASS gdn_two_columns=%u candidate_calls=%llu baseline_cache64_calls=%llu\\n",two_variant,
            (unsigned long long)two_calls,(unsigned long long)two_baseline_calls);
''' + quality[end:]
(out / 'quality.inc').write_text(quality)

bench = (root / 'tests/bench_iq3_groups.m').read_text()
bench = bench.replace('".iq3_groups/quality.inc"', '"quality.inc"')
bench = bench.replace('"../qw3_bench.c"', '"../../qw3_bench.c"')
bench = bench.replace('groups_', 'two_').replace('IQ3 groups', 'GDN two columns')
bench = bench.replace('two_width', 'two_variant')
bench = bench.replace('two_variant!=4 && two_variant!=8', 'two_variant<1 || two_variant>3')
bench = bench.replace('two_calls=0;', 'two_calls=0;two_baseline_calls=0;')
bench = bench.replace('two_calls!=72u*19u', 'two_calls!=72u*48u || two_baseline_calls!=72u*48u')
bench = bench.replace('PASS groups=', 'PASS two_columns=')
(out / 'bench.inc').write_text(bench)

# Preserve the independent double RMSNorm/gate check, guarded inputs and
# in-place state from the cache test. Only HD128 is supported by these probes.
matrix = (root / 'tests/test_gdn_cache.m').read_text()
matrix = matrix.replace('cache_', 'two_').replace('GDN cache', 'GDN two columns')
matrix = matrix.replace('".gdn_cache/quality.inc"', '"quality.inc"')
matrix = matrix.replace('int rc=1,cases=0,poison_cases=0,native_disagreements=0;', 'int rc=1,cases=0;')
matrix = matrix.replace('for (uint32_t hd=32;hd<=128;hd*=2)', 'for (uint32_t hd=128;hd<=128;hd*=2)')
matrix = matrix.replace('state[7],inner[7]', 'state[4],inner[4]')
matrix = matrix.replace('for (int v=0;v<7;v++)', 'for (int v=0;v<4;v++)')
matrix = matrix.replace('                if (hd!=128u && ((v>=2 && v<=4) || v==6)) continue;\n', '')
start = matrix.index('                if (v==5) {')
end = matrix.index('\n            }\n        }', start)
matrix = matrix[:start] + '''                double cpu_error=two_cpu_inner_error(inputs,state[v],inner[v],qh,vh,hd,args.z,args.eps);
                if (!equal || cpu_error>0.00002) {
                    fprintf(stderr,"GDN two columns mismatch head_dim=%u variant=%d max=%.9g\\n",hd,v,cpu_error);goto done;
                }
                cases++;printf("matrix_exact head_dim=%u v_heads=%u variant=%d state=exact inner=native_cache64_exact cpu_norm_max=%.9g guards=ok inputs=unchanged\\n",hd,vh,v,cpu_error);''' + matrix[end:]
start = matrix.index('        if (native_disagreements)')
end = matrix.index('\ndone:', start)
matrix = matrix[:start] + '        if (cases!=6) goto done;\n        printf("PASS matrix_cases=%d\\n",cases);rc=0;' + matrix[end:]
matrix = matrix.replace('threadsPerThreadgroup:MTLSizeMake(hd,1,1)',
                        'threadsPerThreadgroup:MTLSizeMake(v ? hd / 2u : hd,1,1)')
matrix = matrix.replace('if (argc!=2) return 2;', '''if (argc<2 || argc>3) return 2;
        int filter=argc==3 ? atoi(argv[2]) : 0;
        if (argc==3 && (filter<1 || filter>3)) return 2;''')
anchor = '            for (int v=0;v<4;v++) {'
assert matrix.count(anchor) == 2
start = matrix.index(anchor, matrix.index(anchor)+len(anchor))
matrix = matrix[:start] + matrix[start:].replace(anchor, anchor + '\n                if (v && filter && v!=filter) continue;', 1)
matrix = matrix.replace('cases!=6', 'cases!=(filter ? 2 : 6)')
anchor = '                if (memcmp(state[0].contents,state[v].contents,state[0].length)) goto done;'
assert matrix.count(anchor) == 1
matrix = matrix.replace(anchor, '''                if (memcmp(state[0].contents,state[v].contents,state[0].length)) {
                    const float *baseline=state[0].contents,*candidate=state[v].contents;
                    for (uint32_t i=0;i<state_n+12u;i++) if (memcmp(baseline+i,candidate+i,sizeof(float))) {
                        fprintf(stderr,"GDN two columns state mismatch variant=%d head_dim=%u v_heads=%u index=%u baseline=%a candidate=%a\\n",
                            v,hd,vh,i,(double)baseline[i],(double)candidate[i]);break;
                    }
                    goto done;
                }''')
matrix = matrix.replace('if (cb.status!=MTLCommandBufferStatusCompleted) goto done;',
    'if (cb.status!=MTLCommandBufferStatusCompleted) { fprintf(stderr,"GDN two columns GPU failure variant=%d error=%s\\n",v,cb.error.localizedDescription.UTF8String);goto done; }')
anchor = '                int equal=!memcmp(inner[0].contents,inner[v].contents,inner[0].length);'
assert matrix.count(anchor) == 1
matrix = matrix.replace(anchor, anchor + '''
                if (!equal) {
                    const float *baseline=inner[0].contents,*candidate=inner[v].contents;
                    for (uint32_t i=0;i<inner_n+12u;i++) if (memcmp(baseline+i,candidate+i,sizeof(float))) {
                        fprintf(stderr,"GDN two columns inner mismatch variant=%d index=%u baseline=%a candidate=%a\\n",
                            v,i,(double)baseline[i],(double)candidate[i]);break;
                    }
                }''')
(out / 'matrix.inc').write_text(matrix)
