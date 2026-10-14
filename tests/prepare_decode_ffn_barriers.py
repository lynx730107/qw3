"""One concurrent decode encoder; barriers before dispatches except independent FFN up."""
from pathlib import Path
import hashlib
import json
import re

root = Path(__file__).resolve().parent.parent
out = root / 'tests/.decode_ffn_barriers'
out.mkdir(exist_ok=True)
source = (root / 'qw3.c').read_text()
match = re.search(r'static int qw3_metal_session_eval_dense_token\([^;{}]*\)\s*\{', source)
assert match
a = match.start()
b = source.index('\nint qw3_engine_metal_dense_logits_test(', a)
part = source[a:b]
original = '''            ok = qw3_metal_session_matvec_quant_x1_to_scratch(
                     s->metal, lw->ffn_gate->offset, lw->ffn_gate->type,
                     n_embd, n_ff, 0, NULL) &&
                 qw3_metal_session_matvec_quant_x1_to_scratch(
                     s->metal, lw->ffn_up->offset, lw->ffn_up->type,
                     n_embd, n_ff, n_ff, NULL);'''
changed = original.replace('            ok = ', '            ok = private_ffn_window_begin() &&\n                 ', 1) + '\n            private_ffn_window_end();'
assert part.count(original) == 1
assert part.count('int ok = qw3_metal_begin_commands() &&') == 1
part = part.replace(original, changed).replace('int ok = qw3_metal_begin_commands() &&', 'int ok = private_decode_begin() &&')
part = part.replace('return -1;', '{private_decode_end(); return -1;}').replace('    return 0;', '    private_decode_end(); return 0;')
modified = source[:a] + part + source[b:]
restored = modified.replace(changed, original).replace('int ok = private_decode_begin() &&', 'int ok = qw3_metal_begin_commands() &&')
restored = restored.replace('{private_decode_end(); return -1;}', 'return -1;').replace('private_decode_end(); return 0;', 'return 0;')
assert restored == source
(out / 'qw3_private.c').write_text(modified)
hooks = '''#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
static int private_ffn_enabled,private_decode_active,private_ffn_active,private_error;
static uint64_t private_ffn_calls,private_dispatches,private_barriers,private_skips;
static unsigned private_window_dispatches;
static void private_before_dispatch(id<MTLComputeCommandEncoder> enc) {
    if (!private_decode_active) return;
    private_dispatches++;
    if (private_ffn_active && private_window_dispatches==1u) private_skips++;
    else { [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];private_barriers++; }
    if (private_ffn_active && ++private_window_dispatches>2u) private_error=1;
}
@protocol QW3FFNBarrierHooks
- (void)privateDispatchThreadgroups:(MTLSize)groups threadsPerThreadgroup:(MTLSize)threads;
- (void)privateDispatchThreads:(MTLSize)threads threadsPerThreadgroup:(MTLSize)group;
@end
@interface NSObject (QW3FFNBarrierProbe) <QW3FFNBarrierHooks>
@end
@implementation NSObject (QW3FFNBarrierProbe)
- (void)privateDispatchThreadgroups:(MTLSize)groups threadsPerThreadgroup:(MTLSize)threads {
    private_before_dispatch((id<MTLComputeCommandEncoder>)self);
    [(id<MTLComputeCommandEncoder>)self dispatchThreadgroups:groups threadsPerThreadgroup:threads];
}
- (void)privateDispatchThreads:(MTLSize)threads threadsPerThreadgroup:(MTLSize)group {
    private_before_dispatch((id<MTLComputeCommandEncoder>)self);
    [(id<MTLComputeCommandEncoder>)self dispatchThreads:threads threadsPerThreadgroup:group];
}
@end
#include "qw3_private_metal.m"
static int private_decode_begin(void) {
    private_decode_active=private_ffn_enabled;
    return private_ffn_enabled ? qw3_metal_begin_commands_concurrent() : qw3_metal_begin_commands();
}
static void private_decode_end(void) {
    if (private_ffn_active) private_error=1;
    private_decode_active=0;private_ffn_active=0;
}
static int private_ffn_window_begin(void) {
    if (!private_decode_active) return 1;
    if (private_ffn_active || !g_batch_cb || !g_batch_concurrent || g_batch_separate_encoders) {private_error=1;return 0;}
    private_ffn_active=1;private_window_dispatches=0;private_ffn_calls++;
    return 1;
}
static void private_ffn_window_end(void) {
    if (!private_ffn_active) return;
    if (private_window_dispatches!=2u) private_error=1;
    private_ffn_active=0;
}
#define qw3_metal_env_n_gpu_layers ffn_barriers_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled ffn_barriers_core_llama_split_enabled
#define qw3_metal_llama_split_start ffn_barriers_core_llama_split_start
#include "qw3_private.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
'''
(out / 'hooks.inc').write_text(hooks)

bridge = (root / 'qw3_metal.m').read_text()
pattern = r'\[([A-Za-z_]\w*) dispatch(Threadgroups|Threads):'
modified_bridge, count = re.subn(pattern, lambda m: '[(id<QW3FFNBarrierHooks>)' + m[1] + ' privateDispatch' + m[2] + ':', bridge)
assert count == len(re.findall(r'\bdispatch(?:Threadgroups|Threads):', bridge))
restored_bridge = re.sub(r'\[\(id<QW3FFNBarrierHooks>\)([A-Za-z_]\w*) privateDispatch(Threadgroups|Threads):', lambda m: '[' + m[1] + ' dispatch' + m[2] + ':', modified_bridge)
assert restored_bridge == bridge
(out / 'qw3_private_metal.m').write_text(modified_bridge)

base = root / 'docs/measurements/decode_ffn_concurrent_2026-10-14'
m = json.loads((base / 'manifest.json').read_text())


def frozen(name):
    key = 'tests/.decode_ffn_concurrent/' + name
    data = (base / m['source_snapshot_paths'][key]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == m['source_sha256'][key]
    return data.decode()


model = frozen('model.inc').replace('ffn_concurrent', 'ffn_barriers').replace('FFN concurrent', 'FFN barriers')
model = model.replace('private_ffn_calls=0;', 'private_ffn_calls=0;private_dispatches=0;private_barriers=0;private_skips=0;')
model = model.replace('private_ffn_active || g_batch_concurrent ||', 'private_error || private_decode_active || private_ffn_active || g_batch_concurrent ||')
needle='            printf("PASS ffn_barriers_model'
assert model.count(needle) == 1
model = model.replace(needle, '''            if (phase && (private_skips!=992u || private_barriers+private_skips!=private_dispatches)) goto done;
            printf("dispatch_stats phase=%d dispatches=%llu barriers=%llu independent_skips=%llu\\n",phase,(unsigned long long)private_dispatches,(unsigned long long)private_barriers,(unsigned long long)private_skips);
'''+needle)
(out / 'model.inc').write_text(model)
native = frozen('native.inc').replace('ffn_concurrent','ffn_barriers').replace('FFN_CONCURRENT','FFN_BARRIERS')
native = native.replace('if (private_ffn_active || g_batch_concurrent)', 'if (private_error || private_decode_active || private_ffn_active || g_batch_concurrent)')
native = native.replace('        fprintf(stderr,"PASS ffn_barriers_native', '''        fprintf(stderr,"dispatch_stats dispatches=%llu barriers=%llu independent_skips=%llu\\n",(unsigned long long)private_dispatches,(unsigned long long)private_barriers,(unsigned long long)private_skips);
        fprintf(stderr,"PASS ffn_barriers_native''')
(out / 'native.inc').write_text(native)
print('single-encoder FFN barrier scheduling generated; original core restoration exact')
