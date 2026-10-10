"""Generate private dense residual/norm/round candidates from exact anchors."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent.parent
out = root / 'tests/.combination_candidates'
out.mkdir(exist_ok=True)
bridge = (root / 'qw3_metal.m').read_text()
start = bridge.index('int qw3_metal_session_batch_dense_ffn_mixed(')
end = bridge.index('int qw3_metal_session_batch_dense_ffn(', start)
body = bridge[start:end]
anchor = 'if (iq3_pair_eligible && !paired_iq3_s) {'
assert body.count(anchor) == 1
body = body.replace(anchor, '''if (iq3_pair_eligible && !paired_iq3_s && combo_round_ready)
        combo_round_saved++;
    if (iq3_pair_eligible && !paired_iq3_s && !combo_round_ready) {''')
anchor = '    if ((paired_iq3_s && !qw3_metal_encode_dense_iq3_s_gate_up('
assert body.count(anchor) == 1
body = body.replace(anchor, '    combo_round_ready = 0;\n' + anchor)
(out / 'bridge.inc').write_text(bridge[:start] + body + bridge[end:])

core = (root / 'qw3.c').read_text()
start = core.index('    const uint32_t dense_gqa_qg_offset = 0u;', core.rindex('static int qw3_metal_session_eval_dense_prefill_batch'))
end = core.index('        if (ok && !layer_batch)', start)
body = core[start:end]
anchor = 'qw3_metal_session_batch_dense_gqa_output_add_layout('
assert body.count(anchor) == 2
body = body.replace(anchor, 'combo_output_add_layout(')
anchor = '''qw3_metal_session_batch_rmsnorm_weight_f32_x0_to_x1(
                s->metal, lw->ffn_norm->offset, ntok,
                n_embd, QW3_RMS_EPS)'''
assert body.count(anchor) == 2
body = body.replace(anchor, '''combo_ffn_norm(
                s->metal, lw->ffn_norm->offset, ntok,
                n_embd, QW3_RMS_EPS, lw->ffn_gate->type, lw->ffn_up->type)''')
(out / 'core.inc').write_text(core[:start] + body + core[end:])

quality = (root / 'tests/test_native_prefill.m').read_text()
quality = quality.replace('#include "native_prefill_validation.inc"', '#include "../combination_candidates.inc"')
quality = quality.replace('return name && !strcmp(name, "all") ? 3 : -1;',
    'return name && !strcmp(name,"residual") ? 1 : name && !strcmp(name,"round") ? 2 : name && !strcmp(name,"both") ? 3 : -1;')
start = quality.index('        if (((mode & 1)')
end = quality.index('        rc=0;', start)
quality = quality[:start] + '''        if (((mode & 1) && !combo_residual_calls) || ((mode & 2) && !combo_round_saved)) goto done;
        printf("PASS mode=%s residual_norm=%llu round_saved=%llu\\n",argv[2],
            (unsigned long long)combo_residual_calls,(unsigned long long)combo_round_saved);
''' + quality[end:]
(out / 'quality.inc').write_text(quality)
files = ['qw3.c', 'qw3_metal.m', 'tests/test_native_prefill.m', 'tests/prepare_combination_candidates.py']
(out / 'manifest.json').write_text(json.dumps({f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in files}, indent=2) + '\n')
