"""Private aligned Q5_K payload words; retain native FP32 expressions and order."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / 'tests/.q5_words'
out.mkdir(exist_ok=True)
source = (root / 'metal/qw3_core_linear.metal').read_text()
name = 'qw3_matvec_q5_k_fast'
first = source.index('kernel void '+name+'(')
original = source[first:source.index('\n}\n', first)+3]
marker = '        float4 acc1 = float4(0.0f), acc2 = float4(0.0f);'
assert original.count(marker) == 1
shader = ''
for variant in (1, 2):
    private = f'private_q5_words_fast{variant}'
    words = '''        const uint2 q1_words = *((device const uint2 *)q1);
        const uint2 q2_words = *((device const uint2 *)q2);
'''
    if variant == 2:
        words += '        const uint2 qh_words = *((device const uint2 *)qh);\n'
    body = original.replace(name, private).replace(marker, words+marker)
    body = body.replace('            const uchar h = qh[l];', '''            const uchar q1_byte = uchar((q1_words[l >> 2u] >> (8u * (l & 3u))) & 255u);
            const uchar q2_byte = uchar((q2_words[l >> 2u] >> (8u * (l & 3u))) & 255u);
            const uchar h = qh[l];''')
    if variant == 2:
        body = body.replace('const uchar h = qh[l];',
                            'const uchar h = uchar((qh_words[l >> 2u] >> (8u * (l & 3u))) & 255u);')
    body = body.replace('float(q1[l] &', 'float(q1_byte &').replace('float(q2[l] &', 'float(q2_byte &')
    restored = body.replace(private, name).replace(words, '')
    restored = restored.replace('            const uchar q1_byte = uchar((q1_words[l >> 2u] >> (8u * (l & 3u))) & 255u);\n', '')
    restored = restored.replace('            const uchar q2_byte = uchar((q2_words[l >> 2u] >> (8u * (l & 3u))) & 255u);\n', '')
    restored = restored.replace('const uchar h = uchar((qh_words[l >> 2u] >> (8u * (l & 3u))) & 255u);', 'const uchar h = qh[l];')
    restored = restored.replace('float(q1_byte &', 'float(q1[l] &').replace('float(q2_byte &', 'float(q2[l] &')
    assert restored == original
    shader += body
(out/'kernels.metal').write_text(shader)

previous = root/'docs/measurements/q4_words_2026-10-13'
manifest = json.loads((previous/'manifest.json').read_text())
def read_base(filename):
    name = 'tests/.q4_words/'+filename
    content = (previous/manifest['source_snapshot_paths'][name]).read_bytes()
    assert hashlib.sha256(content).hexdigest() == manifest['source_sha256'][name], name
    return content.decode()

def rename(text):
    return (text.replace('q4_words', 'q5_words').replace('Q4Words', 'Q5Words')
            .replace('q4Words', 'q5Words').replace('Q4 words', 'Q5 words')
            .replace('Q4_WORDS', 'Q5_WORDS').replace('qw3-q4-words', 'qw3-q5-words'))

hooks = rename(read_base('hooks.inc'))
hooks = hooks.replace('Aligned Q4_K payload words; native geometry, scalar FP32 arithmetic and buffers.',
                      'Aligned Q5_K payload words; native geometry, FP32 arithmetic and buffers.')
hooks = hooks.replace('g_matvec_q4_k_pipeline', 'g_matvec_q5_k_pipeline')
hooks = hooks.replace('words_variant==1 ? 2u : 4u', '8u')
hooks = hooks.replace('p.maxTotalThreadsPerThreadgroup<256u', 'p.maxTotalThreadsPerThreadgroup<64u')
(out/'hooks.inc').write_text(hooks)
model = rename(read_base('model.inc'))
model = model.replace('baseline/ushort/uint', 'baseline/payload-words/payload-and-highbit-words')
model = model.replace('32u*36u', '32u*26u')
(out/'model.inc').write_text(model)
(out/'native.inc').write_text(rename(read_base('native.inc')))
