"""Private IQ3_S decode geometry probe; preserve each row's FP32 operations."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / 'tests/.iq3_groups'
out.mkdir(exist_ok=True)
source = (root / 'qw3_metal.m').read_text()
name = 'qw3_metal_session_matvec_iq3_s_pair_x1_to_scratch'
start = source.index('int ' + name + '(')
end = source.index('\n}\n', start) + 3
body = source[start:end]
anchor = '    if (!s ||'
assert body.count(anchor) == 1
body = body.replace(anchor, '''    if (groups_selected==2) return groups_original_pair(s,gate_offset,up_offset,n_in,n_out);
    groups_calls++;
    if (!s ||''')
body = body.replace('[enc setComputePipelineState:g_matvec_iq3_s_pair_pipeline];',
                    '[enc setComputePipelineState:groups_pipeline[groups_selected==8]];')
assert '(n_out + 7u) / 8u' in body and 'MTLSizeMake(64u, 1u, 1u)' in body
body = body.replace('(n_out + 7u) / 8u', '(n_out + groups_selected*4u - 1u)/(groups_selected*4u)')
body = body.replace('MTLSizeMake(64u, 1u, 1u)', 'MTLSizeMake(groups_selected*32u, 1u, 1u)')
(out / 'wrapper.inc').write_text(body)

native = (root / 'tests/test_native_prefill.m').read_text()
native = native.replace('#include "native_prefill_validation.inc"', '#include "../iq3_groups_candidates.inc"')
start = native.index('        if (((mode & 1)')
end = native.index('        rc=0;', start)
native = native[:start] + '''        if (!groups_calls) goto done;
        printf("PASS IQ3_S groups=%u candidate_calls=%llu\\n",groups_width,(unsigned long long)groups_calls);
''' + native[end:]
native = native.replace('if (argc!=3) return 2;', '''if (argc!=4) return 2;
        groups_width=atoi(argv[3]);
        if (groups_width!=4 && groups_width!=8) return 2;''')
native = native.replace('!qw3_metal_compile_kernels() ||', '!groups_compile() ||')
(out / 'quality.inc').write_text(native)
