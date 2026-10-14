"""Prepare narrower partitions for the already validated M5 27B F16 route.

Only the actual public three-head pipeline is intercepted. Other shapes keep
their public fallback. This does not reconsider the rejected generic cap-one
kernel: candidate numerical claims are explicitly limited to this model route.
No compilation or model execution occurs here.
"""
from pathlib import Path
import hashlib
import json

root=Path(__file__).resolve().parent.parent
out=root/'tests/.gqa_heads_probe'
out.mkdir(exist_ok=True)
source=(root/'metal/qw3_core_sequence.metal').read_text()
start=source.index('kernel void qw3_gqa_attend_n_inner_heads3(')
original=source[start:source.index('\n}\n',start)+3]
shaders=[]
for variant,capacity in ((1,2),(2,1)):
    body=original.replace('qw3_gqa_attend_n_inner_heads3',f'private_gqa_heads_probe{variant}')
    body=body.replace('(full_heads + 2u) / 3u',f'(full_heads + {capacity-1}u) / {capacity}u')
    body=body.replace('min(3u, full_heads - head_chunk * 3u)',f'min({capacity}u, full_heads - head_chunk * {capacity}u)')
    body=body.replace('head_chunk * 3u',f'head_chunk * {capacity}u')
    assert body.count('gh < 3u')==5
    body=body.replace('gh < 3u',f'gh < {capacity}u')
    for name in ('qv','max_score','denom','acc'):
        body=body.replace(f'float {name}[3];',f'float {name}[{capacity}];')
    restored=body.replace(f'private_gqa_heads_probe{variant}','qw3_gqa_attend_n_inner_heads3')
    restored=restored.replace(f'(full_heads + {capacity-1}u) / {capacity}u','(full_heads + 2u) / 3u')
    restored=restored.replace(f'min({capacity}u, full_heads - head_chunk * {capacity}u)','min(3u, full_heads - head_chunk * 3u)')
    restored=restored.replace(f'head_chunk * {capacity}u','head_chunk * 3u')
    restored=restored.replace(f'gh < {capacity}u','gh < 3u')
    for name in ('qv','max_score','denom','acc'):
        restored=restored.replace(f'float {name}[{capacity}];',f'float {name}[3];')
    assert restored==original
    assert body.count('threadgroup_barrier(')==3
    shaders.append(body)
(out/'kernels.metal').write_text(''.join(shaders))

hooks=(root/'tests/gqa_heads3_public/hooks.inc').read_text()
hooks=hooks.replace('/* Observe the actual public pipeline; never replace a kernel or change a dispatch. */',
    '/* Private cap-two/cap-one probes, restricted to the actual public 27B F16 route. */')
hooks=hooks.replace('static int groups_variant=2;',
    'static int groups_variant=1, probe_mode;\nstatic _Thread_local uint32_t probe_grid,probe_base_grid;\nstatic _Thread_local int probe_eligible;')
a=hooks.index('- (void)gqaGroupsSetPipeline:',hooks.index('@implementation'))
b=hooks.index('\n@end',a)
hooks=hooks[:a]+'''- (void)gqaGroupsSetPipeline:(id<MTLComputePipelineState>)pipeline {
    probe_grid=0;probe_base_grid=0;
    probe_eligible=probe_mode && pipeline==gqa_groups_pipeline[0];
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
- (void)gqaGroupsSetBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index {
    if (probe_eligible && index==0 && length==5u*sizeof(uint32_t)) {
        const uint32_t *a=bytes;
        if (a[1]==24u && a[2]==4u && a[3]==256u && a[4]==1u) {
            const uint32_t cap=groups_variant==1 ? 2u : 1u;
            probe_base_grid=8u;probe_grid=4u*((6u+cap-1u)/cap);
            [(id<MTLComputeCommandEncoder>)self setComputePipelineState:gqa_groups_pipeline[groups_variant]];
            gqa_groups_calls++;
        } else gqa_groups_fallbacks++;
        probe_eligible=0;
    }
    [(id<MTLComputeCommandEncoder>)self setBytes:bytes length:length atIndex:index];
}
- (void)gqaGroupsDispatchThreadgroups:(MTLSize)groups threadsPerThreadgroup:(MTLSize)threads {
    if (probe_grid) {
        if (groups.width<probe_base_grid) gqa_groups_fallbacks++;
        else groups.width=probe_grid+(groups.width-probe_base_grid);
    }
    [(id<MTLComputeCommandEncoder>)self dispatchThreadgroups:groups threadsPerThreadgroup:threads];
}
'''+hooks[b:]
a=hooks.index('static int gqa_groups_compile(void)')
hooks=hooks[:a]+'''static int gqa_groups_compile(void) {
    @autoreleasepool {
    if (gqa_groups_pipeline[2]) return 1;
    if (!qw3_metal_init() || !qw3_metal_compile_kernels()) return 0;
    gqa_groups_pipeline[0]=g_gqa_attend_n_inner_heads3_pipeline;
    if (!gqa_groups_pipeline[0]) return 0;
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) options.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/.gqa_heads_probe/kernels.metal" encoding:NSUTF8StringEncoding error:&error];
    NSString *original=qw3_metal_full_kernel_source();
    NSString *source=extra && original ? [original stringByAppendingString:extra] : nil;
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    for (int v=1;v<=2;v++) {
        NSString *name=[NSString stringWithFormat:@"private_gqa_heads_probe%d",v];
        id<MTLFunction> fn=[lib newFunctionWithName:name];
        id<MTLComputePipelineState> p=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!p || p.threadExecutionWidth!=32 || p.maxTotalThreadsPerThreadgroup<256 ||
            p.staticThreadgroupMemoryLength!=gqa_groups_pipeline[0].staticThreadgroupMemoryLength) return 0;
        gqa_groups_pipeline[v]=p;
    }
    for (int v=0;v<=2;v++) fprintf(stderr,
        "gqa_heads_probe_metadata variant=%d capacity=%d width=%lu max_threads=%lu static_bytes=%lu dynamic_bytes=256\\n",
        v,v==0 ? 3 : v==1 ? 2 : 1,
        (unsigned long)gqa_groups_pipeline[v].threadExecutionWidth,
        (unsigned long)gqa_groups_pipeline[v].maxTotalThreadsPerThreadgroup,
        (unsigned long)gqa_groups_pipeline[v].staticThreadgroupMemoryLength);
    return 1;
    }
}
static void scratch_select(int mode) {
    probe_mode=mode!=0;probe_eligible=0;probe_grid=0;probe_base_grid=0;
}
static void gqa_groups_print_counts(void) {
    fprintf(stderr,"gqa_heads_probe_dispatch_counts calls=%llu fallbacks=%llu\\n",
       (unsigned long long)gqa_groups_calls,(unsigned long long)gqa_groups_fallbacks);
}
'''
(out/'hooks.inc').write_text(hooks)

fixture=(root/'tests/test_public_gqa_heads3.m').read_text()
fixture=fixture.replace('gqa_heads3_public/hooks.inc','.gqa_heads_probe/hooks.inc')
fixture=fixture.replace('int eligible=phase && kv==1','int eligible=kv==1')
fixture=fixture.replace('(pipeline==gqa_groups_pipeline[2])','(pipeline==gqa_groups_pipeline[0])')
fixture=fixture.replace('PASS public_gqa_heads3=%d','PASS gqa_heads_probe=%d')
fixture=fixture.replace('static float fixture_value', '#include "../optimization_acceptance.h"\n\nstatic float fixture_value')
a=fixture.index('else if (memcmp(reference.bytes,out.contents,out.length))')
z=fixture.index('\n                if (gqa_groups_calls-calls',a)
fixture=fixture[:a]+'''else {
                        const float *expected=reference.bytes;qw3_opt_error error={0};
                        for (uint32_t i=5u;i<qn+5u;i++) qw3_opt_error_add(&error,expected[i],y[i]);
                        printf("matrix_numeric variant=%d nmse=%.9g rmse=%.9g max_abs=%.9g different_bits=%llu\\n",
                            groups_variant,qw3_opt_error_nmse(&error),qw3_opt_error_rmse(&error),error.max_abs,
                            (unsigned long long)error.different_bits);
                        if (!qw3_opt_kernel_valid(&error)) return 1;
                    }
                }
'''+fixture[z:]
fixture=fixture.replace('finite_bit_exact=all','finite_numerically_valid=all policy=2')
fixture=fixture.replace('matrix_exact variant=','matrix_numeric_valid variant=')
(out/'matrix.inc').write_text(fixture.replace('#include ".gqa_heads_probe/hooks.inc"\n',''))

model=(root/'tests/gqa_heads3_public/model.inc').read_text()
model='#include "../gqa_heads_numeric_oracles.inc"\n\n'+model[model.index('int main('):]
model=model.replace('if (argc!=2) return 2;',
    'if (argc!=3) return 2;\n        groups_variant=atoi(argv[2]);if (groups_variant<1 || groups_variant>2) return 2;')
model=model.replace('        puts("selected_gqa_variant=2");',
    '        printf("gqa_heads_probe_variant=%d\\n",groups_variant);')
model=model.replace('groups_variant=phase ? 2 : 1;scratch_select(phase!=0);','scratch_select(phase!=0);')
model=model.replace('qw3_engine_open(&e,&opt) || !gqa_groups_compile()',
    '!gqa_groups_compile() || qw3_engine_open(&e,&opt)')
model=model.replace('GQA groups','GQA heads probe').replace('PASS gqa_groups_model','PASS gqa_heads_probe_model')
forced='} else { if (fread(&expected,sizeof(expected),1,oracle)!=1) goto done; token=expected; }'
model=model.replace('} else if (fread(&expected,sizeof(expected),1,oracle)!=1 || expected!=token) goto done;',forced)
model=model.replace('finite_logits=bit_exact canonical_state=exact',
    'finite_logits=numerically_valid canonical_state=numerically_valid input_tokens=teacher_forced')
model=model.replace('        rc=0;','        grid_numeric_report();rc=0;')
(out/'model.inc').write_text(model)

quality=(root/'tests/test_public_gqa_heads3_quality.m').read_text()
quality=quality.replace('gqa_heads3_public/hooks.inc','.gqa_heads_probe/hooks.inc')
quality=quality.replace('gqa_heads3_public/model.inc','model.inc')
quality=quality.replace('if (groups_variant!=2) return 2;',
    'if (groups_variant<1 || groups_variant>2) return 2;')
quality=quality.replace('public_gqa_heads3_quality','gqa_heads_probe_quality')
quality=quality.replace('qw3_engine_open(&e,&options) || !gqa_groups_compile()',
    '!gqa_groups_compile() || qw3_engine_open(&e,&options)')
quality=quality.replace('#include ".gqa_heads_probe/hooks.inc"\n','')
quality=quality.replace('static int checkpoint_exact(qw3_session *s,int vocab,char *err,size_t errlen) {',
    'static int checkpoint_exact(qw3_session *s,int vocab,char *err,size_t errlen) {\n    int previous=grid_require_exact_state;grid_require_exact_state=1;')
quality=quality.replace('    return ok;','    grid_require_exact_state=previous;return ok;')
quality=quality.replace('} else if (fread(&expected,sizeof(expected),1,oracle)!=1 || expected!=token) goto done;',forced)
quality=quality.replace('finite_logits=bit_exact canonical_state=exact',
    'finite_logits=numerically_valid canonical_state=numerically_valid input_tokens=teacher_forced')
quality=quality.replace('eight_prefixes=exact','eight_prefixes=numerical')
quality=quality.replace('        puts("PASS gqa_heads_probe_quality',
    '        grid_numeric_report();puts("PASS gqa_heads_probe_quality')
(out/'quality.inc').write_text(quality)

archive=root/'docs/measurements/gqa_groups_2026-10-14'
manifest=json.loads((archive/'manifest.json').read_text())
native_name='tests/.gqa_groups/native.inc'
native_path=archive/manifest['source_snapshot_paths'][native_name]
assert hashlib.sha256(native_path.read_bytes()).hexdigest()==manifest['source_sha256'][native_name]
native=native_path.read_text()
native=native.replace('    int rc=qw3_engine_open(e,o);if (rc) return rc;\n    if (!gqa_groups_compile()) {qw3_engine_close(*e);*e=NULL;return 1;}',
    '    if (!gqa_groups_compile()) return 1;\n    int rc=qw3_engine_open(e,o);if (rc) return rc;')
native=native.replace('QW3_PRIVATE_GQA_GROUPS','QW3_PRIVATE_GQA_HEADS_PROBE')
native=native.replace('PASS gqa_groups_native','PASS gqa_heads_probe_native')
(out/'native.inc').write_text(native)
print('Prepared cap-two/cap-one probes against public cap-three. No numerical or performance claims yet.')
