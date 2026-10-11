"""Reuse full native correctness and paired decode checks for IQ4_XS loads."""
from pathlib import Path
root = Path(__file__).resolve().parent.parent
out = root / 'tests/.iq4_loads'
out.mkdir(exist_ok=True)
quality = (root / 'tests/test_native_prefill.m').read_text()
quality = quality.replace('#include "native_prefill_validation.inc"', '#include "../iq4_load_candidates.inc"')
quality = quality.replace('if (argc!=3) return 2;', 'if (argc!=4) return 2;\n        iq4_variant=atoi(argv[3]);if (iq4_variant<1 || iq4_variant>3) return 2;')
quality = quality.replace('!qw3_metal_compile_kernels() ||', '!iq4_load_compile() || !iq4_load_model_aligned(e) ||')
start = quality.index('        if (((mode & 1)')
end = quality.index('        rc=0;', start)
quality = quality[:start] + '''        if (!iq4_load_calls[0] || !iq4_load_calls[1] || !iq4_load_calls[2]) goto done;
        printf("PASS IQ4_XS loads variant=%d single=%llu pair=%llu add=%llu\\n",iq4_variant,
            (unsigned long long)iq4_load_calls[0],(unsigned long long)iq4_load_calls[1],(unsigned long long)iq4_load_calls[2]);
''' + quality[end:]
(out / 'quality.inc').write_text(quality)
bench = (root / 'tests/bench_iq3_groups.m').read_text()
bench = bench.replace('.iq3_groups/quality.inc', 'quality.inc').replace('../qw3_bench.c', '../../qw3_bench.c')
bench = bench.replace('groups_width=atoi(argv[2]);if (groups_width!=4 && groups_width!=8) return 2;', 'iq4_variant=atoi(argv[2]);if (iq4_variant<1 || iq4_variant>3) return 2;')
bench = bench.replace('groups_compile()', 'iq4_load_compile() || !iq4_load_model_aligned(e)').replace('groups_calls=0;', 'memset(iq4_load_calls,0,sizeof(iq4_load_calls));')
bench = bench.replace('groups_calls!=72u*19u', '(iq4_load_calls[0]!=72u*96u || iq4_load_calls[1]!=72u*14u || iq4_load_calls[2]!=72u*32u)')
bench = bench.replace('PASS groups=%u', 'PASS iq4_loads=%u').replace('groups_width,', 'iq4_variant,')
bench = bench.replace('(unsigned long long)groups_calls', '(unsigned long long)(iq4_load_calls[0]+iq4_load_calls[1]+iq4_load_calls[2])')
bench = bench.replace('IQ3 groups failed:', 'IQ4 loads failed:')
(out / 'bench.inc').write_text(bench)
