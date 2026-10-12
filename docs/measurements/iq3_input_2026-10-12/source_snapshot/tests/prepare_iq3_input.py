"""Private IQ3_S packed activation reads; identical scalar arithmetic."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / 'tests/.iq3_input'
out.mkdir(exist_ok=True)
source = (root / 'metal/qw3_core_moe.metal').read_text()
name = 'qw3_matvec_iq3_s_fast'
a = source.index('kernel void ' + name + '(')
original = source[a:source.index('\n}\n', a)+3]
old = '        for (uint j = 0; j < 32u; j++) xx[j] = x[uint64_t(ib32) * 32ull + j];'
new = '        device const packed_float4 *x4 = (device const packed_float4 *)(x + uint64_t(ib32) * 32ull);\n        for (uint j = 0; j < 8u; j++) {\n            const packed_float4 v = x4[j];\n            xx[4u*j] = v[0]; xx[4u*j+1u] = v[1];\n            xx[4u*j+2u] = v[2]; xx[4u*j+3u] = v[3];\n        }'
assert original.count(old) == 1
loop = '    for (uint ib32 = uint(lane); ib32 < nb32; ib32 += 32u) {'
shader = '/* Helpers supplied by production source; packed input requires only float alignment. */\n'
shader += 'static_assert(sizeof(packed_float4)==16 && alignof(packed_float4)==4, "packed activation ABI");\n'
for variant in (1, 2):
    private = f'private_iq3_input_fast{variant}'
    body = original.replace(name, private).replace(old, new)
    if variant == 2:
        assert body.count(loop) == 1
        body = body.replace(loop, '#pragma unroll(2)\n'+loop)
    assert body.replace(private, name).replace(new, old).replace('#pragma unroll(2)\n', '') == original
    shader += body
(out / 'kernels.metal').write_text(shader)

base = root / 'tests/.iq3_words'
def rename(s):
    return (s.replace('iq3_words','iq3_input').replace('IQ3Words','IQ3Input')
            .replace('iq3Words','iq3Input').replace('words_variant','input_variant')
            .replace('IQ3 words','IQ3 input').replace('IQ3_WORDS','IQ3_INPUT')
            .replace('baseline/QS-word/QS-and-sign-word','baseline/packed-input/packed-input-unroll2'))
hooks = rename((base/'hooks.inc').read_text())
hooks = hooks.replace('Aligned integer loads from original IQ3_S weights; native grid and FP32 math.',
                      'Packed input loads; native weights, grid, FP32 arithmetic and data buffers.')
old_compile = '    NSString *source=[NSString stringWithContentsOfFile:@"tests/.iq3_input/kernels.metal" encoding:NSUTF8StringEncoding error:&error];'
new_compile = '    if (g_metal4_tensor_api_enabled) options.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};\n    NSString *extra=[NSString stringWithContentsOfFile:@"tests/.iq3_input/kernels.metal" encoding:NSUTF8StringEncoding error:&error];\n    NSString *original=qw3_metal_full_kernel_source();\n    NSString *source=extra && original ? [original stringByAppendingString:extra] : nil;'
assert hooks.count(old_compile) == 1
hooks = hooks.replace(old_compile, new_compile)
hooks = hooks.replace('iq3_input_pipeline[v].maxTotalThreadsPerThreadgroup<64',
                      'iq3_input_pipeline[v].maxTotalThreadsPerThreadgroup<64 || iq3_input_pipeline[v].threadExecutionWidth!=32 || iq3_input_pipeline[v].staticThreadgroupMemoryLength!=iq3_input_pipeline[0].staticThreadgroupMemoryLength')
hooks = hooks.replace('    return 1;\n}\nstatic __attribute__', '    for (int v=0;v<=2;v++) fprintf(stderr,"iq3_input_metadata variant=%d width=%lu max_threads=%lu static_threadgroup_bytes=%lu dynamic_threadgroup_bytes=256 threads=64\\n",v,(unsigned long)iq3_input_pipeline[v].threadExecutionWidth,(unsigned long)iq3_input_pipeline[v].maxTotalThreadsPerThreadgroup,(unsigned long)iq3_input_pipeline[v].staticThreadgroupMemoryLength);\n    return 1;\n}\nstatic __attribute__')
(out/'hooks.inc').write_text(hooks)
for filename in ('model.inc','native.inc'):
    (out/filename).write_text(rename((base/filename).read_text()))
matrix = rename((base/'matrix.inc').read_text())
matrix = matrix.replace('[enc setComputePipelineState:iq3_input_pipeline[v ? input_variant : 0]];',
                        '[(id<QW3IQ3InputHooks>)enc iq3InputSetPipeline:iq3_input_pipeline[v ? input_variant : 0]];')
matrix = matrix.replace('for (int pattern=0;pattern<3;pattern++) {', 'for (int pattern=0;pattern<3;pattern++) for (int alignment=0;alignment<2;alignment++) {')
matrix = matrix.replace('offset:16 atIndex:2', 'offset:(alignment ? 4 : 16) atIndex:2')
matrix = matrix.replace('n_out=%u pattern=%d guards=', 'n_out=%u pattern=%d input_offset=%d guards=')
matrix = matrix.replace('input_variant,ni,no,pattern);', 'input_variant,ni,no,pattern,alignment ? 4 : 16);')
matrix = matrix.replace('        printf("PASS iq3_input=', '        if (iq3_input_calls!=36u) goto done;\n        printf("PASS iq3_input=')
(out/'matrix.inc').write_text(matrix)
