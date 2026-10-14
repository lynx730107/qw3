"""Partition independent query heads; preserve each head's scalar/SIMD/time order."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent.parent
out = root / 'tests/.gqa_groups'
out.mkdir(exist_ok=True)
source = (root / 'metal/qw3_core_sequence.metal').read_text()
a = source.index('kernel void qw3_gqa_attend_n_inner(')
original = source[a:source.index('\n}\n', a) + 3]
shader = ''
for variant, capacity in ((1, 1), (2, 3)):
    name = f'private_gqa_groups{variant}'
    body = original.replace('qw3_gqa_attend_n_inner', name)
    opening = '    if (h >= args.n_kv_heads'
    prelude = f'''    if (args.n_kv_heads == 0u) return;
    const uint full_heads = args.n_heads / args.n_kv_heads;
    if (full_heads == 0u || full_heads > 8u) return;
    const uint chunks = (full_heads + {capacity-1}u) / {capacity}u;
    const uint kvh = h / chunks;
    const uint head_chunk = h % chunks;
'''
    body = body.replace(opening, prelude + '    if (kvh >= args.n_kv_heads')
    body = body.replace('    uint kvh = h;\n', '')
    old_group = '    uint group_heads = args.n_heads / args.n_kv_heads;'
    new_group = f'    uint group_heads = min({capacity}u, full_heads - head_chunk * {capacity}u);'
    body = body.replace(old_group, new_group)
    old_first = '    uint first_qh = kvh * group_heads;'
    new_first = f'    uint first_qh = kvh * full_heads + head_chunk * {capacity}u;'
    body = body.replace(old_first, new_first)
    assert body.count('gh < 8u') == 5
    body = body.replace('gh < 8u', f'gh < {capacity}u')
    for array in ('qv', 'max_score', 'denom', 'acc'):
        body = body.replace(f'float {array}[8];', f'float {array}[{capacity}];')
    restored = body.replace(name, 'qw3_gqa_attend_n_inner')
    restored = restored.replace(prelude + '    if (kvh >= args.n_kv_heads', opening)
    restored = restored.replace('    uint i = uint(tid);\n', '    uint i = uint(tid);\n    uint kvh = h;\n')
    restored = restored.replace(new_group, old_group).replace(new_first, old_first)
    restored = restored.replace(f'gh < {capacity}u', 'gh < 8u')
    for array in ('qv', 'max_score', 'denom', 'acc'):
        restored = restored.replace(f'float {array}[{capacity}];', f'float {array}[8];')
    assert restored == original
    assert body.count('threadgroup_barrier(') == 3
    shader += body
(out / 'kernels.metal').write_text(shader)
base = root / 'docs/measurements/gqa_barrier_2026-10-12'
m = json.loads((base / 'manifest.json').read_text())


def frozen(name):
    key = 'tests/.gqa_barrier/' + name
    data = (base / m['source_snapshot_paths'][key]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == m['source_sha256'][key]
    return data.decode()


def rename(text):
    return (text.replace('gqa_barrier', 'gqa_groups').replace('GQABarrier', 'GQAGroups')
            .replace('gqaBarrier', 'gqaGroups').replace('barrier_variant', 'groups_variant')
            .replace('GQA barrier', 'GQA groups').replace('GQA_BARRIER', 'GQA_GROUPS'))


hooks = rename(frozen('hooks.inc'))
hooks = hooks.replace('static _Thread_local int gqa_groups_eligible;',
                      'static _Thread_local int gqa_groups_eligible;\nstatic _Thread_local uint32_t gqa_groups_multiplier=1;')
marker='- (void)gqaGroupsSetBytes:'
first=hooks.index(marker)
hooks=hooks[:first]+'- (void)gqaGroupsDispatchThreadgroups:(MTLSize)groups threadsPerThreadgroup:(MTLSize)threads;\n'+hooks[first:]
hooks=hooks.replace('    gqa_groups_encoder=(__bridge void *)self;', '    gqa_groups_multiplier=1;\n    gqa_groups_encoder=(__bridge void *)self;')
hooks=hooks.replace('a[1]/a[2]<=7u', 'a[1]/a[2]<=8u')
hooks=hooks.replace('            gqa_groups_calls++;', '            const uint32_t cap=groups_variant==1 ? 1u : 3u;\n            gqa_groups_multiplier=(a[1]/a[2]+cap-1u)/cap;\n            gqa_groups_calls++;')
start=hooks.index('@implementation')
end=hooks.index('\n@end',start)
hooks=hooks[:end]+'''
- (void)gqaGroupsDispatchThreadgroups:(MTLSize)groups threadsPerThreadgroup:(MTLSize)threads {
    if (gqa_groups_encoder==(__bridge void *)self) groups.width*=gqa_groups_multiplier;
    [(id<MTLComputeCommandEncoder>)self dispatchThreadgroups:groups threadsPerThreadgroup:threads];
}
'''+hooks[end:]
hooks=hooks.replace('#define setBytes gqaGroupsSetBytes', '#define dispatchThreadgroups gqaGroupsDispatchThreadgroups\n#define setBytes gqaGroupsSetBytes')
hooks=hooks.replace('#undef setBytes', '#undef dispatchThreadgroups\n#undef setBytes')
hooks=hooks.replace('barriers_per_context=%d', 'original_barriers_per_context=3 array_head_capacity=%d')
hooks=hooks.replace('v ? 2 : 3', 'v==1 ? 1 : v==2 ? 3 : 8')
(out / 'hooks.inc').write_text(hooks)
for name in ('model.inc', 'native.inc'):
    (out / name).write_text(rename(frozen(name)))
print('GQA query-head partition candidates generated; original per-head expressions restored exactly')
