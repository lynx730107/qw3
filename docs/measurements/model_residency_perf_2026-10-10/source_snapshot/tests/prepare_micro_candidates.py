"""Build private graph variants from current production source, with exact anchors."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent.parent
out = root / 'tests/.micro_candidates'
out.mkdir(exist_ok=True)
source = (root / 'qw3.c').read_text()
old = '''qw3_metal_session_batch_matmul_q8_0_x1_to_scratch(
                s->metal, lw->linear_ssm_alpha->offset, ntok,
                n_embd, (uint32_t)spec->n_linear_v_heads,
                dense_dn_alpha_offset, dense_dn_stride) &&
                 qw3_metal_session_batch_matmul_q8_0_x1_to_scratch(
                s->metal, lw->linear_ssm_beta->offset, ntok,
                n_embd, (uint32_t)spec->n_linear_v_heads,
                dense_dn_beta_offset, dense_dn_stride)'''
new = '''micro_batch_pair(s->metal, lw->linear_ssm_alpha->offset,
                lw->linear_ssm_beta->offset, ntok, n_embd,
                (uint32_t)spec->n_linear_v_heads,
                dense_dn_alpha_offset, dense_dn_beta_offset, dense_dn_stride)'''
assert source.count(old) == 1
source = source.replace(old, new)
old = '''qw3_metal_session_batch_conv1d_step_from_scratch(
                s->metal, lw->linear_conv_weight->offset,
                (uint32_t)s->dn.linear_layer_map[il], ntok, n_qkv,
                dense_dn_qkv_offset, dense_dn_conv_offset,
                dense_dn_stride);
        }
        if (ok && full_batch_deltanet) {
            stage = "batch dense DeltaNet QK norm";
            ok = qw3_metal_session_batch_l2norm_qk_from_scratch(
                s->metal, ntok, dense_dn_conv_offset, dense_dn_stride,
                (uint32_t)spec->n_linear_qk_heads,
                (uint32_t)spec->n_linear_head_dim, QW3_RMS_EPS)'''
new = '''micro_conv_norm(s->metal, lw->linear_conv_weight->offset,
                (uint32_t)s->dn.linear_layer_map[il], ntok, n_qkv,
                dense_dn_qkv_offset, dense_dn_conv_offset, dense_dn_stride,
                (uint32_t)spec->n_linear_qk_heads,
                (uint32_t)spec->n_linear_head_dim, QW3_RMS_EPS)'''
assert source.count(old) == 1
source = source.replace(old,new)
old = '''qw3_metal_session_matvec_quant_x1_to_scratch(
                     s->metal, lw->linear_ssm_alpha->offset,
                     lw->linear_ssm_alpha->type, n_embd,
                     (uint32_t)spec->n_linear_v_heads,
                     alpha_offset, NULL) &&
                 qw3_metal_session_matvec_quant_x1_to_scratch(
                     s->metal, lw->linear_ssm_beta->offset,
                     lw->linear_ssm_beta->type, n_embd,
                     (uint32_t)spec->n_linear_v_heads,
                     beta_offset, NULL)'''
new = '''micro_decode_pair(s->metal, lw->linear_ssm_alpha->offset,
                     lw->linear_ssm_beta->offset, lw->linear_ssm_alpha->type,
                     lw->linear_ssm_beta->type, n_embd,
                     (uint32_t)spec->n_linear_v_heads, alpha_offset, beta_offset)'''
assert source.count(old) == 1
source = source.replace(old, new)
(out / 'core.inc').write_text(source)
quality = (root / 'tests/test_native_prefill.m').read_text()
quality = quality.replace('#include "native_prefill_validation.inc"', '#include "../micro_candidates.inc"')
quality = quality.replace('return name && !strcmp(name, "all") ? 3 : -1;',
    'return name && !strcmp(name, "prefill") ? 1 : name && !strcmp(name, "decode") ? 2 : name && !strcmp(name, "pair") ? 3 : name && !strcmp(name,"conv") ? 4 : name && !strcmp(name,"all") ? 7 : -1;')
start = quality.index('        if (((mode & 1)')
end = quality.index('        rc=0;', start)
quality = quality[:start] + '''        if (((mode & 1) && !micro_prefill_calls) || ((mode & 2) && !micro_decode_calls)) goto done;
        if ((mode & 4) && !micro_conv_calls) goto done;
        printf("PASS mode=%s prefill_pair=%llu decode_pair=%llu conv_norm=%llu\\n",argv[2],
            (unsigned long long)micro_prefill_calls,(unsigned long long)micro_decode_calls,
            (unsigned long long)micro_conv_calls);
''' + quality[end:]
(out / 'quality.inc').write_text(quality)
files = ['qw3.c', 'qw3_metal.m', 'tests/test_native_prefill.m', 'tests/prepare_micro_candidates.py']
(out / 'manifest.json').write_text(json.dumps({f: hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files}, indent=2)+'\n')
