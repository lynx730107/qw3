"""Per-SIMD IQ4_XS LUT within the original 128-byte threadgroup allocation."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / 'tests/.iq4_simd_lut'
out.mkdir(exist_ok=True)
base = root / 'tests/.iq4_share'
source = (root / 'metal/qw3_core_linear.metal').read_text()
names = ['qw3_matvec_iq4_xs_fast', 'qw3_matvec_iq4_xs_pair_fast',
         'qw3_matvec_iq4_xs_add_x0_fast']
shader = '/* Native arithmetic; per-SIMD LUT uses the existing 128 bytes. */\n'
loop = '    for (uint b = ix; b < n_blocks; b += 2u) {'
old_init = '    if (simd_idx == 0u && lane < 16u) sh[lane] = qw3_iq4nl_val(uint(lane));\n    threadgroup_barrier(mem_flags::mem_threadgroup);'
new_init = '    threadgroup float *table = sh + uint(simd_idx) * 16u;\n    if (lane < 16u) table[lane] = qw3_iq4nl_val(uint(lane));\n    simdgroup_barrier(mem_flags::mem_threadgroup);'
for variant, factor in ((1, 0), (2, 4)):
    for kind, name in enumerate(names):
        first = source.index('kernel void ' + name + '(')
        last = source.index('\n}\n', first) + 3
        original = source[first:last]
        assert original.count(loop) == original.count(old_init) == 1
        private_name = f'private_iq4_simd_lut_fast{variant}_{kind}'
        body = original.replace(name, private_name).replace(old_init, new_init)
        assert body.count('xl, sh, ib, il)') == (4 if kind == 1 else 2)
        body = body.replace('xl, sh, ib, il)', 'xl, table, ib, il)')
        if factor:
            body = body.replace(loop, f'#pragma unroll({factor})\n' + loop)
        restored = body.replace(private_name, name).replace(new_init, old_init).replace('xl, table, ib, il)', 'xl, sh, ib, il)')
        if factor:
            restored = restored.replace(f'#pragma unroll({factor})\n', '')
        assert restored == original
        shader += body
(out / 'kernels.metal').write_text(shader)


def rename(text):
    return (text.replace('iq4_share', 'iq4_simd_lut')
            .replace('IQ4Share', 'IQ4SimdLut').replace('iq4Share', 'iq4SimdLut')
            .replace('IQ4 share', 'IQ4 SIMD LUT').replace('share_variant', 'lut_variant')
            .replace('QW3_PRIVATE_IQ4_SHARE', 'QW3_PRIVATE_IQ4_SIMD_LUT'))


hooks = rename((base / 'hooks.inc').read_text())
first = hooks.index('static NSUInteger iq4_simd_lut_rows_per_group=')
last = hooks.index('#include "core.inc"', first) + len('#include "core.inc"')
hooks = hooks[:first] + '#include "../../qw3_metal.m"' + hooks[last:]
hooks = hooks.replace('p.maxTotalThreadsPerThreadgroup<(v==1 ? 128u : 256u)',
                      'p.maxTotalThreadsPerThreadgroup<64u')
hooks = hooks.replace('simdgroups=%d threads=%d dynamic_threadgroup_bytes=128',
                      'simdgroups=2 threads=64 dynamic_threadgroup_bytes=128 lut_bytes=%d')
hooks = hooks.replace('v==1 ? 4 : v==2 ? 8 : 2,v==1 ? 128 : v==2 ? 256 : 64',
                      'v ? 128 : 64')
hooks = hooks.replace('    iq4_simd_lut_rows_per_group=mode ? (lut_variant==1 ? 8u : 16u) : 4u;\n', '')
hooks = hooks.replace('    iq4_simd_lut_thread_count=mode ? (lut_variant==1 ? 128u : 256u) : 64u;\n', '')
hooks = hooks.replace('Three IQ4_XS pipelines: native row arithmetic, LUT and data buffers.',
                      'Three IQ4_XS pipelines: Per-SIMD LUT and optional unroll4; native arithmetic and data buffers.')
assert 'core.inc' not in hooks and 'rows_per_group' not in hooks
(out / 'hooks.inc').write_text(hooks)
for filename in ('matrix.inc', 'model.inc', 'native.inc'):
    text = rename((base / filename).read_text())
    text = text.replace('baseline/row2-SG4/row2-SG8', 'baseline/SIMD-LUT/SIMD-LUT-and-unroll4')
    text = text.replace('NSUInteger rows=v ? (lut_variant==1 ? 8u : 16u) : 4u;', 'NSUInteger rows=4u;')
    text = text.replace('NSUInteger threads=v ? (lut_variant==1 ? 128u : 256u) : 64u;', 'NSUInteger threads=64u;')
    if filename == 'matrix.inc':
        failure = '                    fprintf(stderr,"IQ4 SIMD LUT bit mismatch shape=%d kind=%d pattern=%d\\n",shape,kind,pattern);goto done;'
        assert text.count(failure) == 1
        text = text.replace(failure, '''                    const float *a=output[0].contents,*b=output[1].contents;
                    for (uint32_t i=0;i<n+12u;i++) if (memcmp(a+i,b+i,4)) {
                        uint32_t bits[2];memcpy(bits,a+i,4);memcpy(bits+1,b+i,4);
                        fprintf(stderr,"IQ4 SIMD LUT differs variant=%d ni=%u no=%u kind=%d pattern=%d index=%u reference=%.9g candidate=%.9g bits=%08x/%08x\\n",lut_variant,ni,no,kind,pattern,i,a[i],b[i],bits[0],bits[1]);break;
                    }
''' + failure)
    (out / filename).write_text(text)
