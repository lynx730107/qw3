"""Private tracked/untracked probe with explicit encoder and dispatch ordering."""
from pathlib import Path
import re
import runpy
root=Path(__file__).resolve().parent.parent
runpy.run_path(str(root/'tests/prepare_hazard_tracking.py'))
old=root/'tests/.hazard_tracking'
out=root/'tests/.hazard_fences';out.mkdir(exist_ok=True)
core=(old/'core.inc').read_text()
# Variant 0 compares the old shared-buffer call with the managed-only guard.
start=core.index('static id<MTLBuffer> qw3_metal_new_private_buffer(')
end=core.index('static NSArray<id<MTLBuffer>> *qw3_metal_new_private_buffer_layers',start)
factory=core[start:end].replace('hazard_untracked', '(hazard_untracked && hazard_variant!=0)')
core=core[:start]+factory+core[end:]
mask='if (obj.flashAttnMask.storageMode == MTLStorageModeManaged) {'
assert core.count(mask)==1
core=core.replace(mask, '''hazard_mask_calls[hazard_untracked!=0]++;
    if (obj.flashAttnMask.storageMode == MTLStorageModeManaged ||
        (hazard_variant==0 && !hazard_untracked)) {''')
anchor='static id<MTLComputeCommandEncoder> qw3_metal_compute_encoder('
i=core.index(anchor)
helpers='''/* Wait only before encoding work; update immediately before endEncoding. */
static id<MTLFence> hazard_fence;
static int hazard_fence_written;
static void hazard_encoder_wait(id encoder) {
    if (!hazard_untracked || !(hazard_variant & 2) || !encoder) return;
    if (!hazard_fence) hazard_fence=[g_device newFence];
    if (!hazard_fence) abort();
    if (hazard_fence_written) {
        [encoder waitForFence:hazard_fence];hazard_waits++;
    }
}
static void hazard_encoder_update(id encoder) {
    if (!hazard_untracked || !(hazard_variant & 2) || !encoder) return;
    if (!hazard_fence) hazard_fence=[g_device newFence];
    if (!hazard_fence) abort();
    [encoder updateFence:hazard_fence];hazard_updates++;hazard_fence_written=1;
}
'''
core=core[:i]+helpers+core[i:]
old='''                [cb computeCommandEncoder];
        }
        return g_batch_enc;
    }
    return [cb computeCommandEncoder];'''
new='''                [cb computeCommandEncoder];
            hazard_encoder_wait(g_batch_enc);
        }
        return g_batch_enc;
    }
    id<MTLComputeCommandEncoder> encoder=[cb computeCommandEncoder];
    hazard_encoder_wait(encoder);
    return encoder;'''
assert core.count(old)==1
core=core.replace(old,new)
core=core.replace('[enc endEncoding];','hazard_encoder_update(enc);\n    [enc endEncoding];')
core=core.replace('[g_batch_enc endEncoding];','hazard_encoder_update(g_batch_enc);\n    [g_batch_enc endEncoding];')
assert core.count('id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];')==47
core=core.replace('id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];','id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];\n    hazard_encoder_wait(blit);')
assert core.count('[blit endEncoding];')==47
core=core.replace('[blit endEncoding];','hazard_encoder_update(blit);\n    [blit endEncoding];')
pattern=r'(?m)^([ \t]*)\[(enc|direct|round_enc) (dispatch\w+):'
def barrier(m):
    indent,encoder,method=m.groups()
    return f'{indent}if (hazard_untracked && (hazard_variant & 1)) {{\n{indent}    [{encoder} memoryBarrierWithScope:MTLBarrierScopeBuffers];hazard_barriers++;\n{indent}}}\n{indent}[{encoder} {method}:'
core,n=re.subn(pattern,barrier,core)
assert n>100
(out/'core.inc').write_text(core)
validation=(root/'tests/.hazard_tracking/validation.inc').read_text()
validation=validation.replace('static int hazard_untracked;', 'static int hazard_untracked,hazard_variant;\nstatic uint64_t hazard_waits,hazard_updates,hazard_barriers,hazard_mask_calls[2];')
(out/'validation.inc').write_text(validation)
quality=(root/'tests/.hazard_tracking/quality.inc').read_text()
quality=quality.replace('if (argc!=3) return 2;', '''if (argc!=4) return 2;
        hazard_variant=atoi(argv[3]);
        if (hazard_variant<0 || hazard_variant>3) return 2;
        int one=!strcmp(argv[2],"one");''')
quality=quality.replace('int mode=scratch_parse_mode(argv[2]),rc=1;', 'int mode=one ? 3 : scratch_parse_mode(argv[2]),rc=1;')
quality=quality.replace('trial<8;', 'trial<(one ? 1 : 8);')
quality=quality.replace('v ? MTLHazardTrackingModeUntracked', '(v && hazard_variant!=0) ? MTLHazardTrackingModeUntracked')
quality=quality.replace('if (!hazard_allocations[0] || !hazard_allocations[1]) goto done;', '''if (!hazard_allocations[0] || (hazard_variant!=0 && !hazard_allocations[1]) ||
            (hazard_variant==0 && (hazard_allocations[1] || !hazard_mask_calls[0] || !hazard_mask_calls[1]))) goto done;''')

quality=quality.replace('        printf("PASS hazard_tracking', '''        if (((hazard_variant & 1) && !hazard_barriers) ||
            ((hazard_variant & 2) && (!hazard_waits || !hazard_updates))) goto done;
        printf("mask_write_calls baseline=%llu candidate=%llu\\n",(unsigned long long)hazard_mask_calls[0],(unsigned long long)hazard_mask_calls[1]);
        printf("explicit_ordering variant=%d waits=%llu updates=%llu barriers=%llu\\n",hazard_variant,
            (unsigned long long)hazard_waits,(unsigned long long)hazard_updates,(unsigned long long)hazard_barriers);
        printf("PASS hazard_tracking''')
(out/'quality.inc').write_text(quality)
print(f'hazard_fences: dispatches={n}, blit_wait_sites=47, blit_update_sites=47')
