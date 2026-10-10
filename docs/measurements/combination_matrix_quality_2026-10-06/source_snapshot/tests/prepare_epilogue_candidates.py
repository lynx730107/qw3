"""Private staged128 down/residual epilogues, with unchanged GEMM loop."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / 'tests/.epilogue_candidates'
out.mkdir(exist_ok=True)
source = (root / 'metal/qw3_core_moe.metal').read_text()
kernels = '#ifdef QW3_METAL_HAS_TENSOR\n'
for kind in ['iq3_s', 'iq4_xs']:
    name = f'qw3_matmul_{kind}_mpp_staged128'
    start = source.index('kernel void ' + name + '(')
    end = source.index('\n}\n', start) + 3
    body = source[start:end].replace(name, 'epilogue_' + kind)
    anchor = '    threadgroup char *shmem [[threadgroup(0)]],'
    assert body.count(anchor) == 1
    body = body.replace(anchor, '    device const float *residual,\n' + anchor)
    anchor = '    cT.store(tD.slice(r0, r1 + int(simd_group) * ROWS_PER_SG));'
    assert body.count(anchor) == 1
    body = body.replace(anchor, '''    auto tR = tensor((device float *)residual,
                    dextents<int32_t, 2>(int(args.n_out), int(args.n_tokens)),
                    array<int, 2>({1, int(args.out_stride)}));
    auto cR = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
    cR.load(tR.slice(r0, r1 + int(simd_group) * ROWS_PER_SG));
    for (uint16_t i=0; i<cT.get_capacity(); ++i)
        if (cT.is_valid_element(i)) cT[i] = cR[i] + cT[i];
''' + anchor)
    kernels += body
kernels += '#endif\n'
(out / 'kernels.metal').write_text(kernels)

bridge = (root / 'tests/.combination_candidates/bridge.inc').read_text()
start = bridge.index('int qw3_metal_session_batch_dense_ffn_mixed(')
end = bridge.index('int qw3_metal_session_batch_dense_ffn(', start)
body = bridge[start:end]
anchor = '''        if (!qw3_metal_encode_dense_batch_matmul(
                obj, cb, down_offset, down_type, n_tokens, n_ff, n_embd,
                obj.denseBatchGate, obj.denseBatchDown)) return 0;'''
assert body.count(anchor) == 1
body = body.replace(anchor, '''        if (!epilogue_down(obj,cb,down_offset,down_type,n_tokens,n_ff,n_embd,
                obj.denseBatchGate,obj.denseBatchDown)) return 0;
        if (!epilogue_saved) {''')
anchor = '''        qw3_metal_end_compute_encoder(cb, enc);
    }
    return qw3_metal_finish_command_buffer'''
assert body.count(anchor) == 1
body = body.replace(anchor, '''        qw3_metal_end_compute_encoder(cb, enc);
        }
    }
    return qw3_metal_finish_command_buffer''')
(out / 'bridge.inc').write_text(bridge[:start] + body + bridge[end:])
adapter = (root / 'tests/.sum8_candidates/candidates.inc').read_text()
adapter = '''#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
@class QW3MetalSessionObj;
static int epilogue_saved;
static unsigned long long epilogue_calls;
static int epilogue_down(QW3MetalSessionObj *, id<MTLCommandBuffer>, uint64_t,
    uint32_t,uint32_t,uint32_t,uint32_t,id<MTLBuffer>,id<MTLBuffer>);
''' + adapter
adapter = adapter.replace('"../.combination_candidates/bridge.inc"', '"bridge.inc"')
anchor = '#include "../sum8_dispatch.inc"'
assert adapter.count(anchor) == 1
adapter = adapter.replace(anchor, anchor + '\n#include "../epilogue_dispatch.inc"')
(out / 'candidates.inc').write_text(adapter)
quality = (root / 'tests/.sum8_candidates/quality.inc').read_text()
quality = quality.replace('!g_deltanet_batch_recur_packed8_pipeline)', '!g_deltanet_batch_recur_packed8_pipeline || !epilogue_compile())')
quality = quality.replace('return name && !strcmp(name,"logits") ? 4 : name && !strcmp(name,"all") ? 7 : -1;',
    'return name && !strcmp(name,"down") ? 8 : name && !strcmp(name,"residual_down") ? 9 : name && !strcmp(name,"both_down") ? 11 : name && !strcmp(name,"all_down") ? 15 : -1;')
quality = quality.replace('if (!sum8_head_calls) goto done;', 'if ((mode & 4) && !sum8_head_calls) goto done;\n        if (!epilogue_calls) goto done;\n        printf("down_epilogue=%llu\\n",epilogue_calls);')
(out / 'quality.inc').write_text(quality)
bench = (root / 'tests/bench_combination_campaign.m').read_text()
bench = bench.replace('".sum8_candidates/candidates.inc"', '"candidates.inc"')
bench = bench.replace('"../qw3_bench.c"', '"../../qw3_bench.c"')
bench = bench.replace('|| !sum8_compile()', '|| !sum8_compile() || !epilogue_compile()')
bench = bench.replace('!strcmp(name,"all") ? 7 : -1;', '!strcmp(name,"all") ? 7 : !strcmp(name,"down") ? 8 : !strcmp(name,"residual_down") ? 9 : !strcmp(name,"both_down") ? 11 : !strcmp(name,"all_down") ? 15 : -1;')
bench = bench.replace('            combo_residual_calls=combo_round_calls=combo_round_saved=sum8_head_calls=0;', '            combo_residual_calls=combo_round_calls=combo_round_saved=sum8_head_calls=epilogue_calls=0;')
bench = bench.replace('sum8_head=%llu\\n",', 'sum8_head=%llu down_epilogue=%llu\\n",')
bench = bench.replace('combo_residual_calls,combo_round_calls,combo_round_saved,sum8_head_calls);', 'combo_residual_calls,combo_round_calls,combo_round_saved,sum8_head_calls,epilogue_calls);')
(out / 'bench.m').write_text(bench)
