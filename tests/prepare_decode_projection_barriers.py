"""Extend the scheduling-only barrier exception to four independent DeltaNet projections."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent.parent
out = root / 'tests/.decode_projection_barriers'
out.mkdir(exist_ok=True)
base = root / 'docs/measurements/decode_ffn_barriers_2026-10-14'
m = json.loads((base / 'manifest.json').read_text())
assert m['all_passed']


def frozen(name):
    key = 'tests/.decode_ffn_barriers/' + name
    data = (base / m['source_snapshot_paths'][key]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == m['source_sha256'][key]
    return data.decode()


core = frozen('qw3_private.c')
core = core.replace('private_ffn_window_begin()', 'private_ffn_window_begin(2u)')
start = core.index('            stage = "DeltaNet projections";', core.index('int ok = private_decode_begin() &&'))
end = core.index('            if (!ok) break;', start)
part = core[start:end]
assert part.count('qw3_metal_session_matvec_quant_x1_to_scratch(') == 4
original = part
part = part.replace('            ok = qw3_metal_session_matvec_quant_x1_to_scratch(',
                    '            ok = private_ffn_window_begin(4u) &&\n                 qw3_metal_session_matvec_quant_x1_to_scratch(', 1)
part += '            private_ffn_window_end();\n'
assert part.replace('            ok = private_ffn_window_begin(4u) &&\n                 qw3_metal_session_matvec_quant_x1_to_scratch(',
                    '            ok = qw3_metal_session_matvec_quant_x1_to_scratch(', 1).removesuffix('            private_ffn_window_end();\n') == original
core = core[:start] + part + core[end:]
(out / 'qw3_private.c').write_text(core)
hooks = frozen('hooks.inc')
hooks = hooks.replace('static unsigned private_window_dispatches;',
                      'static unsigned private_window_dispatches,private_window_expected;\nstatic uint64_t private_two_windows,private_four_windows;')
hooks = hooks.replace('private_window_dispatches==1u', 'private_window_dispatches>0u && private_window_dispatches<private_window_expected')
hooks = hooks.replace('++private_window_dispatches>2u', '++private_window_dispatches>private_window_expected')
hooks = hooks.replace('private_ffn_window_begin(void)', 'private_ffn_window_begin(unsigned expected)')
hooks = hooks.replace('private_ffn_active=1;private_window_dispatches=0;private_ffn_calls++;',
                      '''if (expected!=2u && expected!=4u) {private_error=1;return 0;}
    private_window_expected=expected;
    if (expected==2u) private_two_windows++;else private_four_windows++;
    private_ffn_active=1;private_window_dispatches=0;private_ffn_calls++;''')
hooks = hooks.replace('private_window_dispatches!=2u', 'private_window_dispatches!=private_window_expected')
(out / 'hooks.inc').write_text(hooks)
(out / 'qw3_private_metal.m').write_text(frozen('qw3_private_metal.m'))
model = frozen('model.inc').replace('ffn_barriers','projection_barriers').replace('FFN barriers','Projection barriers')
model = model.replace('32u*31u','32u*79u').replace('private_skips!=992u','private_skips!=5600u')
model = model.replace('private_ffn_calls=0;', 'private_ffn_calls=0;private_two_windows=0;private_four_windows=0;')
model = model.replace('            if (phase && (private_skips',
                      '            if (phase && (private_two_windows!=992u || private_four_windows!=1536u)) goto done;\n            if (phase && (private_skips')
needle='            printf("dispatch_stats'
model = model.replace(needle, '            printf("scope_stats phase=%d two_windows=%llu four_windows=%llu\\n",phase,(unsigned long long)private_two_windows,(unsigned long long)private_four_windows);\n'+needle)
(out / 'model.inc').write_text(model)
native = frozen('native.inc').replace('ffn_barriers','projection_barriers').replace('FFN_BARRIERS','PROJECTION_BARRIERS')
needle='        fprintf(stderr,"dispatch_stats'
native = native.replace(needle,'        fprintf(stderr,"scope_stats two_windows=%llu four_windows=%llu\\n",(unsigned long long)private_two_windows,(unsigned long long)private_four_windows);\n'+needle)
(out / 'native.inc').write_text(native)
print('DeltaNet four-projection scopes prepared; not compiled or executed')
