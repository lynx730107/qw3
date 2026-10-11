"""Two-column DeltaNet with native 128-lane output RMS and existing output scratch."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / 'tests/.gdn_two128'
out.mkdir(exist_ok=True)
source = (root / 'metal/qw3_core_deltanet.metal').read_text()
name = 'qw3_deltanet_fused_gdn_cache64'
start = source.index('kernel void ' + name + '(')
end = source.index('\n}\n', start) + 3
native = source[start:end]
begin = native.index('    constexpr uint CACHE_N')
tail = native.index('    float ss = simd_sum(core * core);')
kernels = []
for count in (32, 64):
    core = f'''    if (j < 64u) {{
        uint j1 = uint(j) + 64u;
        constexpr uint CACHE_N = {count}u;
        float2 cached[CACHE_N];
        float sk0 = 0.0f, sk1 = 0.0f;
        #pragma unroll
        for (uint i = 0; i < 128u; i++) {{
            float p0 = sin[i * args.head_dim + j];
            float p1 = sin[i * args.head_dim + j1];
            if (i < CACHE_N) cached[i] = float2(p0, p1);
            sk0 = fma(p0, kh[i], sk0);
            sk1 = fma(p1, kh[i], sk1);
        }}
        float d0 = b * fma(-sk0, g, vh[j]);
        float d1 = b * fma(-sk1, g, vh[j1]);
        float sum0 = 0.0f, sum1 = 0.0f;
        #pragma unroll
        for (uint i = 0; i < 128u; i++) {{
            uint idx = i * args.head_dim + j;
            float sv0 = fma(kh[i], d0, (i < CACHE_N ? cached[i].x : sin[idx]) * g);
            float sv1 = fma(kh[i], d1, (i < CACHE_N ? cached[i].y : sin[idx + 64u]) * g);
            sout[idx] = sv0;
            sout[idx + 64u] = sv1;
            sum0 = fma(sv0, qh[i], sum0);
            sum1 = fma(sv1, qh[i], sum1);
        }}
        inner_out[uint64_t(hv) * args.head_dim + j] = sum0 * rsqrt(float(args.head_dim));
        inner_out[uint64_t(hv) * args.head_dim + j1] = sum1 * rsqrt(float(args.head_dim));
    }}
    // Every thread participates, including the two groups that skipped state work.
    threadgroup_barrier(mem_flags::mem_device);
    float core = inner_out[uint64_t(hv) * args.head_dim + j];
'''
    body = native[:begin] + core + native[tail:]
    body = body.replace(name, f'private_gdn_two128_cache{count}')
    assert 'j >= args.head_dim' in body and 'j >= 64u' not in body
    assert body.count('threadgroup_barrier(mem_flags::mem_device)') == 1
    kernels.append(body)
(out / 'kernels.metal').write_text('\n'.join(kernels))

# Reuse the low-memory complete-logit/live-state disk oracle. Only dispatch
# selection/counters differ; the model/session lifecycle remains one session.
model = (root / 'tests/.iq3_immutable/model.inc').read_text()
model = model.replace('iq3_immutable_compile', 'two128_compile')
model = model.replace('immutable_variant', 'two128_variant').replace('iq3_immutable_calls', 'two128_calls')
model = model.replace('IQ3 immutable', 'GDN two128').replace('iq3_immutable_model', 'gdn_two128_model')
model = model.replace('32u*73u', '32u*48u')
model = model.replace('baseline/immutable-weight/immutable-weight-and-grid', 'baseline/cache32/cache64')
model = model.replace('if (argc!=2) return 2;', '''if (argc<2 || argc>3) return 2;
        int filter=argc==3 ? atoi(argv[2]) : 0;
        if (argc==3 && (filter<1 || filter>2)) return 2;''')
model = model.replace('for (int phase=0;phase<=2;phase++) {',
                      'for (int phase=0;phase<=2;phase++) {\n            if (phase && filter && phase!=filter) continue;')
(out / 'model.inc').write_text(model)
native_runner = (root / 'tests/.iq3_immutable/native.inc').read_text()
native_runner = native_runner.replace('iq3_immutable_compile', 'two128_compile')
native_runner = native_runner.replace('immutable_variant', 'two128_variant').replace('iq3_immutable_calls', 'two128_calls')
native_runner = native_runner.replace('iq3_immutable_native', 'gdn_two128_native')
native_runner = native_runner.replace('QW3_PRIVATE_IQ3_IMMUTABLE', 'QW3_PRIVATE_GDN_TWO128')
(out / 'native.inc').write_text(native_runner)
