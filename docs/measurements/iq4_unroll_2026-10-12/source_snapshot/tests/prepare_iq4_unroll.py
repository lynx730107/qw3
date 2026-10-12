"""Private IQ4_XS K-loop unrolling; original arithmetic, geometry and buffers."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / 'tests/.iq4_unroll'
out.mkdir(exist_ok=True)
base = root / 'tests/.iq4_share'
source = (root / 'metal/qw3_core_linear.metal').read_text()
names = ['qw3_matvec_iq4_xs_fast', 'qw3_matvec_iq4_xs_pair_fast',
         'qw3_matvec_iq4_xs_add_x0_fast']
shader = '/* Native helpers are supplied by the complete production Metal source. */\n'
loop = '    for (uint b = ix; b < n_blocks; b += 2u) {'
for variant, factor in ((1, 2), (2, 4)):
    for kind, name in enumerate(names):
        first = source.index('kernel void ' + name + '(')
        last = source.index('\n}\n', first) + 3
        original = source[first:last]
        assert original.count(loop) == 1
        private_name = f'private_iq4_unroll_fast{variant}_{kind}'
        body = original.replace(name, private_name)
        body = body.replace(loop, f'#pragma unroll({factor})\n' + loop)
        assert body.replace(private_name, name).replace(
            f'#pragma unroll({factor})\n', '') == original
        shader += body
(out / 'kernels.metal').write_text(shader)


def rename(text):
    return (text.replace('iq4_share', 'iq4_unroll')
            .replace('IQ4Share', 'IQ4Unroll').replace('iq4Share', 'iq4Unroll')
            .replace('IQ4 share', 'IQ4 unroll').replace('share_variant', 'unroll_variant')
            .replace('QW3_PRIVATE_IQ4_SHARE', 'QW3_PRIVATE_IQ4_UNROLL'))


hooks = rename((base / 'hooks.inc').read_text())
first = hooks.index('static NSUInteger iq4_unroll_rows_per_group=')
last = hooks.index('#include "core.inc"', first) + len('#include "core.inc"')
hooks = hooks[:first] + '#include "../../qw3_metal.m"' + hooks[last:]
hooks = hooks.replace('p.maxTotalThreadsPerThreadgroup<(v==1 ? 128u : 256u)',
                      'p.maxTotalThreadsPerThreadgroup<64u')
hooks = hooks.replace('simdgroups=%d threads=%d dynamic_threadgroup_bytes=128',
                      'simdgroups=2 threads=64 dynamic_threadgroup_bytes=128 loop_unroll=%d')
hooks = hooks.replace('v==1 ? 4 : v==2 ? 8 : 2,v==1 ? 128 : v==2 ? 256 : 64',
                      'v==1 ? 2 : v==2 ? 4 : 0')
hooks = hooks.replace('    iq4_unroll_rows_per_group=mode ? (unroll_variant==1 ? 8u : 16u) : 4u;\n', '')
hooks = hooks.replace('    iq4_unroll_thread_count=mode ? (unroll_variant==1 ? 128u : 256u) : 64u;\n', '')
hooks = hooks.replace('Three IQ4_XS pipelines: native row arithmetic, LUT and data buffers.',
                      'Three IQ4_XS pipelines: K-loop unroll hints only; native arithmetic and buffers.')
assert 'core.inc' not in hooks and 'rows_per_group' not in hooks
(out / 'hooks.inc').write_text(hooks)
for filename in ('matrix.inc', 'model.inc', 'native.inc'):
    text = rename((base / filename).read_text())
    text = text.replace('baseline/row2-SG4/row2-SG8', 'baseline/unroll2/unroll4')
    text = text.replace('NSUInteger rows=v ? (unroll_variant==1 ? 8u : 16u) : 4u;', 'NSUInteger rows=4u;')
    text = text.replace('NSUInteger threads=v ? (unroll_variant==1 ? 128u : 256u) : 64u;', 'NSUInteger threads=64u;')
    if filename == 'matrix.inc':
        failure = '                    fprintf(stderr,"IQ4 unroll bit mismatch shape=%d kind=%d pattern=%d\\n",shape,kind,pattern);goto done;'
        assert text.count(failure) == 1
        text = text.replace(failure, '''                    const float *a=output[0].contents,*b=output[1].contents;
                    for (uint32_t i=0;i<n+12u;i++) if (memcmp(a+i,b+i,4)) {
                        uint32_t bits[2];memcpy(bits,a+i,4);memcpy(bits+1,b+i,4);
                        fprintf(stderr,"IQ4 unroll differs variant=%d ni=%u no=%u kind=%d pattern=%d index=%u reference=%.9g candidate=%.9g bits=%08x/%08x\\n",unroll_variant,ni,no,kind,pattern,i,a[i],b[i],bits[0],bits[1]);break;
                    }
''' + failure)
    (out / filename).write_text(text)
