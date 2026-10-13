"""Private IQ3_S down/residual fusion; original data buffers and reduction."""
import hashlib
import json
import difflib
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root/'tests/.iq3_add'
out.mkdir(exist_ok=True)
source = (root/'metal/qw3_core_moe.metal').read_text()
name = 'qw3_matvec_iq3_s_fast'
a = source.index('kernel void '+name+'(')
original = source[a:source.index('\n}\n',a)+3]
shader = ''
for variant in (1,2):
    private = f'private_iq3_add_fast{variant}'
    body = original.replace(name,private)
    old = '                                  threadgroup float *sh,'
    assert body.count(old)==1
    body = body.replace(old,'                                  constant float &scale,\n'+old)
    for i in range(4):
        old_store = f'out[row] = sum{i};'
        new_store = (f'out[row] = out[row] + sum{i} * scale;' if variant==1 else
                     f'out[row] = fma(sum{i}, scale, out[row]);')
        assert body.count(old_store)==1
        body = body.replace(old_store,new_store)
    restored = body.replace(private,name).replace('                                  constant float &scale,\n','')
    for i in range(4):
        restored = restored.replace(f'out[row] = out[row] + sum{i} * scale;',f'out[row] = sum{i};')
        restored = restored.replace(f'out[row] = fma(sum{i}, scale, out[row]);',f'out[row] = sum{i};')
    assert restored==original
    shader += body
(out/'kernels.metal').write_text(shader)

core = (root/'qw3.c').read_text()
old = '''        } else if (ok) {
            ok = qw3_metal_session_matvec_quant_inner_to_x1(
                     s->metal, lw->ffn_down->offset, lw->ffn_down->type,
                     n_ff, n_embd, NULL) &&
                 qw3_metal_session_scale_x1_add_x0(
                     s->metal, 1.0f, n_embd);
        }'''
new = '''        } else if (ok && iq3_add_mode &&
                   lw->ffn_down->type == QW3_TENSOR_IQ3_S &&
                   n_ff == 17408u && n_embd == 5120u) {
            ok = private_iq3_inner_add_x0(
                s->metal, lw->ffn_down->offset, n_ff, n_embd);
        } else if (ok) {
            ok = qw3_metal_session_matvec_quant_inner_to_x1(
                     s->metal, lw->ffn_down->offset, lw->ffn_down->type,
                     n_ff, n_embd, NULL) &&
                 qw3_metal_session_scale_x1_add_x0(
                     s->metal, 1.0f, n_embd);
        }'''
assert core.count(old)==1
changed = core.replace(old,new)
assert changed.replace(new,old)==core
(out/'core.inc').write_text(changed)
(out/'core.diff').write_text(''.join(difflib.unified_diff(
    core.splitlines(True),changed.splitlines(True),fromfile='qw3.c',tofile='private/core.inc')))

previous = root/'docs/measurements/iq3_input_2026-10-12'
manifest = json.loads((previous/'manifest.json').read_text())
def read_base(filename):
    key = 'tests/.iq3_input/'+filename
    content = (previous/manifest['source_snapshot_paths'][key]).read_bytes()
    assert hashlib.sha256(content).hexdigest()==manifest['source_sha256'][key]
    return content.decode()
def rename(s):
    return (s.replace('iq3_input','iq3_add').replace('input_variant','add_variant')
            .replace('IQ3 input','IQ3 add').replace('IQ3_INPUT','IQ3_ADD')
            .replace('baseline/packed-input/packed-input-unroll2','baseline/fused-expression/fused-explicit-fma'))
model = rename(read_base('model.inc')).replace('32u*73u','32u*17u')
(out/'model.inc').write_text(model)
native = rename(read_base('native.inc'))
(out/'native.inc').write_text(native)
