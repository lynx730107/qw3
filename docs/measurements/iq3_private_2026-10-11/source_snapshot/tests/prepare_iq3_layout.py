"""Private 112-byte IQ3_S copy: preserve native floating-point expressions."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / 'tests/.iq3_layout'
out.mkdir(exist_ok=True)
source = (root / 'metal/qw3_core_moe.metal').read_text()

def extract(marker):
    first = source.index(marker)
    last = source.index('\n}\n', first) + 3
    return source[first:last]

shader = '#include <metal_stdlib>\nusing namespace metal;\nstruct qw3_matvec_q8_0_args {uint n_in,n_out,row_bytes;};\n'
for marker in ['inline float4 qw3_iq3s_float_grid(', 'inline float qw3_iq3s_grid_value(ushort', 'inline float qw3_iq3s_grid_value(float4']:
    shader += extract(marker)
names = ['qw3_matvec_iq3_s_fast', 'qw3_matvec_iq3_s_pair_fast']
markers = ['template<typename Grid>\ninline float qw3_iq3s_dot32_shared(', 'inline float qw3_iq3s_dot32_vector(']
helpers = ['qw3_iq3s_dot32_shared', 'qw3_iq3s_dot32_vector']
for variant in [1, 2]:
    for kind in [0, 1]:
        dot = extract(markers[kind])
        start = dot.index('    const half d' if kind == 0 else '    const float d')
        end = dot.index('    const float db', start)
        if variant == 1:
            header = dot[start:end].replace('blk + 2 +', 'blk + 16 +').replace('blk + 2u +', 'blk + 16u +')
            header = header.replace('blk[66u + ib]', 'blk[8u + ib]').replace('blk + 74 +', 'blk + 80 +').replace('blk + 74u +', 'blk + 80u +').replace('blk[106u + ib / 2u]', 'blk[4u + ib / 2u]')
        else:
            header = '    const uint4 header = *((device const uint4 *)blk);\n'
            header += '    const half d = as_type<half2>(header.x).x;\n' if kind == 0 else '    const float d = float(as_type<half2>(header.x).x);\n'
            header += '    const uint2 qs = *((device const uint2 *)(blk + 16u + ib * 8u));\n'
            header += '    const uchar qhb = uchar(header[2u + (ib >> 2u)] >> (8u * (ib & 3u)));\n'
            header += '    const uint signs = *((device const uint *)(blk + 80u + ib * 4u));\n'
            header += f'    const uint {"scale" if kind == 0 else "scales"} = (header.y >> (8u * (ib / 2u))) & 255u;\n'
            dot = dot.replace('uint(qs[2u * l])', '((qs[l >> 1u] >> (16u * (l & 1u))) & 255u)')
            dot = dot.replace('uint(qs[2u * l + 1u])', '((qs[l >> 1u] >> (16u * (l & 1u) + 8u)) & 255u)')
            dot = dot.replace('const uchar s = signs[l];', 'const uchar s = uchar(signs >> (8u * l));')
            dot = dot.replace('const uint s = uint(signs[l]);', 'const uint s = (signs >> (8u * l)) & 255u;')
        # Only the integer header and address expressions differ.
        start = dot.index('    const half d' if kind == 0 else '    const float d')
        end = dot.index('    const float db', start)
        dot = dot[:start] + header + dot[end:]
        dot_name = f'private_iq3_layout_dot{variant}_{kind}'
        shader += dot.replace(helpers[kind], dot_name)
        body = extract('kernel void ' + names[kind] + '(')
        body = body.replace(names[kind], f'private_iq3_layout{variant}_{kind}').replace(helpers[kind], dot_name)
        body = body.replace('110ull', '112ull').replace('uint64_t(args.row_bytes)', '(uint64_t(args.n_in / 256u) * 112ull)')
        shader += body
(out / 'kernels.metal').write_text(shader)

hooks = (root / 'tests/iq4_load_candidates.inc').read_text().replace('iq4_load', 'iq3_layout').replace('IQ4Load', 'IQ3Layout').replace('iq4Load', 'iq3Layout').replace('iq4_variant', 'layout_variant')
hooks = hooks.replace('/* Private integer-load variants. Preserve all native FP expressions. */', '/* Private padded IQ3_S copy; preserve native FP32 expressions. */')
hooks = hooks.replace('#import <Metal/Metal.h>', '#import <Metal/Metal.h>\n#import <objc/runtime.h>')
hooks = hooks.replace('iq3_layout_pipeline[4][3]', 'iq3_layout_pipeline[3][2]').replace('iq3_layout_calls[3]', 'iq3_layout_calls[2]')
hooks = hooks.replace('static int layout_variant=1;', '''static int layout_variant=1;
static uint64_t iq3_layout_redirects,iq3_layout_bytes;
static NSMutableDictionary<NSNumber *,id<MTLBuffer>> *iq3_layout_weights;
static void *iq3_layout_kind_key=&iq3_layout_kind_key;
static id<MTLBuffer> iq3_layout_bind(id<MTLBuffer>,NSUInteger *,NSUInteger,unsigned);''')
hooks = hooks.replace('@protocol QW3IQ3LayoutHooks', '@protocol QW3IQ3LayoutHooks\n- (void)iq3LayoutSetBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index;')
first = hooks.index('- (void)iq3LayoutSetPipeline:', hooks.index('@implementation'))
last = hooks.index('\n@end', first)
hooks = hooks[:first] + '''- (void)iq3LayoutSetPipeline:(id<MTLComputePipelineState>)pipeline {
    unsigned active=0;
    for (int kind=0;kind<2;kind++) for (int variant=1;variant<=2;variant++)
        if (pipeline==iq3_layout_pipeline[variant][kind]) {iq3_layout_calls[kind]++;active=kind+1u;}
    objc_setAssociatedObject(self,iq3_layout_kind_key,@(active),OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
- (void)iq3LayoutSetBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index {
    unsigned active=[objc_getAssociatedObject(self,iq3_layout_kind_key) unsignedIntValue];
    if (active && index>=1u && index<=active) buffer=iq3_layout_bind(buffer,&offset,index,active);
    [(id<MTLComputeCommandEncoder>)self setBuffer:buffer offset:offset atIndex:index];
}''' + hooks[last:]
hooks = hooks.replace('#define setComputePipelineState iq3LayoutSetPipeline', '#define setBuffer iq3LayoutSetBuffer\n#define setComputePipelineState iq3LayoutSetPipeline')
hooks = hooks.replace('#undef setComputePipelineState', '#undef setBuffer\n#undef setComputePipelineState')
hooks = hooks.replace('#include "../qw3_metal.m"', '#include "../../qw3_metal.m"').replace('#include "../qw3.c"', '#include "../../qw3.c"')
first = hooks.index('static NSString *iq3_layout_extract(')
hooks = hooks[:first] + '''static int iq3_layout_compile(void) {
    if (iq3_layout_pipeline[2][1]) return 1;
    if (!qw3_metal_compile_kernels()) return 0;
    iq3_layout_pipeline[0][0]=g_matvec_iq3_s_pipeline;
    iq3_layout_pipeline[0][1]=g_matvec_iq3_s_pair_pipeline;
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    NSString *source=[NSString stringWithContentsOfFile:@"tests/.iq3_layout/kernels.metal" encoding:NSUTF8StringEncoding error:&error];
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    for (int v=1;v<=2;v++) for (int kind=0;kind<2;kind++) {
        NSString *name=[NSString stringWithFormat:@"private_iq3_layout%d_%d",v,kind];
        id<MTLFunction> fn=[lib newFunctionWithName:name];
        iq3_layout_pipeline[v][kind]=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!iq3_layout_pipeline[v][kind] || iq3_layout_pipeline[v][kind].maxTotalThreadsPerThreadgroup<64) {
            fprintf(stderr,"IQ3 layout compile variant=%d kind=%d: %s\\n",v,kind,error.localizedDescription.UTF8String);return 0;
        }
    }
    return 1;
}
static id<MTLBuffer> iq3_layout_pack(id<MTLBuffer> native,uint64_t offset,uint32_t ni,uint32_t no) {
    if (!ni || ni%256u || !no) return nil;
    const uint64_t blocks=(uint64_t)ni/256u*no,bytes=blocks*112u;
    if (!native || offset>native.length || blocks*110u>native.length-offset || bytes>NSUIntegerMax-64u) return nil;
    id<MTLBuffer> packed=[g_device newBufferWithLength:bytes+64u options:MTLResourceStorageModeShared];
    if (!packed) return nil;
    const unsigned char *src=(const unsigned char *)native.contents+offset;
    unsigned char *allocation=packed.contents,*dst=allocation+32u;
    memset(allocation,0xa5,32u);memset(dst+bytes,0xa5,32u);
    for (uint64_t b=0;b<blocks;b++) {
        const unsigned char *s=src+b*110u;unsigned char *d=dst+b*112u;
        memcpy(d,s,2u);memset(d+2u,0,2u);memcpy(d+4u,s+106u,4u);
        memcpy(d+8u,s+66u,8u);memcpy(d+16u,s+2u,64u);memcpy(d+80u,s+74u,32u);
    }
    for (uint64_t b=0;b<blocks;b++) {
        const unsigned char *s=src+b*110u,*d=dst+b*112u;
        if (memcmp(d,s,2u) || d[2] || d[3] || memcmp(d+4u,s+106u,4u) ||
            memcmp(d+8u,s+66u,8u) || memcmp(d+16u,s+2u,64u) || memcmp(d+80u,s+74u,32u)) return nil;
    }
    return packed;
}
static int iq3_layout_guards(id<MTLBuffer> buffer) {
    const unsigned char *p=buffer.contents;
    if (!p || buffer.length<64u) return 0;
    for (unsigned i=0;i<32u;i++) if (p[i]!=0xa5 || p[buffer.length-32u+i]!=0xa5) return 0;
    return 1;
}
static int iq3_layout_prepare(qw3_engine *engine) {
    if (iq3_layout_weights) return 1;
    if (g_model_view_count!=1u) return 0;
    iq3_layout_weights=[NSMutableDictionary new];unsigned count=0;
    const double start=[NSDate timeIntervalSinceReferenceDate];
    for (uint32_t i=0;i<engine->model.n_tensors;i++) {
        const qw3_tensor *w=&engine->model.tensors[i];if (w->type!=21u) continue;
        if (w->ndim!=2u || w->dim[0]%256u || w->dim[0]>UINT32_MAX || w->dim[1]>UINT32_MAX) return 0;
        uint64_t offset=0;id<MTLBuffer> native=qw3_metal_model_view_for(w->offset,w->elements/256u*110u,&offset);
        id<MTLBuffer> packed=iq3_layout_pack(native,offset,(uint32_t)w->dim[0],(uint32_t)w->dim[1]);
        if (!packed || iq3_layout_weights[@(offset)] || !iq3_layout_guards(packed)) return 0;
        iq3_layout_weights[@(offset)]=packed;count++;iq3_layout_bytes+=packed.length;
    }
    fprintf(stderr,"IQ3 layout prepared tensors=%u bytes=%llu seconds=%.6f all_source_bytes=exact\\n",count,(unsigned long long)iq3_layout_bytes,[NSDate timeIntervalSinceReferenceDate]-start);
    return count>0;
}
static id<MTLBuffer> iq3_layout_bind(id<MTLBuffer> buffer,NSUInteger *offset,NSUInteger index,unsigned active) {
    (void)index;(void)active;
    /* Matrix tests already bind explicit packed copies. Native model bindings
     * are redirected only while a private candidate pipeline is selected. */
    if (g_model_view_count!=1u || buffer!=g_model_buffers[0]) return buffer;
    id<MTLBuffer> packed=iq3_layout_weights[@(*offset)];
    if (!packed) {fprintf(stderr,"IQ3 layout missing packed tensor offset=%llu\\n",(unsigned long long)*offset);abort();}
    *offset=32u;iq3_layout_redirects++;return packed;
}
static void scratch_select(int mode) {
    setenv("QW3_METAL_DENSE_PREFILL_FAST","1",1);
    g_matvec_iq3_s_pipeline=iq3_layout_pipeline[mode ? layout_variant : 0][0];
    g_matvec_iq3_s_pair_pipeline=iq3_layout_pipeline[mode ? layout_variant : 0][1];
}
'''
(out / 'hooks.inc').write_text(hooks)

quality = (root / 'tests/test_native_prefill.m').read_text().replace('#include "native_prefill_validation.inc"', '#include "hooks.inc"')
quality = quality.replace('if (argc!=3) return 2;', 'if (argc!=4) return 2;\n        layout_variant=atoi(argv[3]);if (layout_variant<1 || layout_variant>2) return 2;')
quality = quality.replace('!qw3_metal_compile_kernels() ||', '!iq3_layout_compile() || !iq3_layout_prepare(e) ||')
first = quality.index('        if (((mode & 1)');last = quality.index('        rc=0;', first)
quality = quality[:first] + '''        if (!iq3_layout_calls[0] || !iq3_layout_calls[1] || !iq3_layout_redirects) goto done;
        for (id<MTLBuffer> buffer in iq3_layout_weights.allValues) if (!iq3_layout_guards(buffer)) goto done;
        printf("PASS IQ3_S layout variant=%d single=%llu pair=%llu redirects=%llu\\n",layout_variant,
            (unsigned long long)iq3_layout_calls[0],(unsigned long long)iq3_layout_calls[1],(unsigned long long)iq3_layout_redirects);
''' + quality[last:]
(out / 'quality.inc').write_text(quality)
bench = (root / 'tests/bench_iq3_groups.m').read_text().replace('.iq3_groups/quality.inc', 'quality.inc').replace('../qw3_bench.c', '../../qw3_bench.c')
bench = bench.replace('groups_width=atoi(argv[2]);if (groups_width!=4 && groups_width!=8) return 2;', 'layout_variant=atoi(argv[2]);if (layout_variant<1 || layout_variant>2) return 2;')
bench = bench.replace('groups_compile()', 'iq3_layout_compile() || !iq3_layout_prepare(e)').replace('groups_calls=0;', 'memset(iq3_layout_calls,0,sizeof(iq3_layout_calls));iq3_layout_redirects=0;')
bench = bench.replace('groups_calls!=72u*19u', '(iq3_layout_calls[0]!=72u*73u || iq3_layout_calls[1]!=72u*19u || iq3_layout_redirects!=72u*111u)')
bench = bench.replace('PASS groups=%u', 'PASS iq3_layout=%u').replace('groups_width,', 'layout_variant,')
bench = bench.replace('(unsigned long long)groups_calls', '(unsigned long long)(iq3_layout_calls[0]+iq3_layout_calls[1])').replace('IQ3 groups failed:', 'IQ3 layout failed:')
bench = bench.replace('        fprintf(stderr,"PASS', '        for (id<MTLBuffer> buffer in iq3_layout_weights.allValues) if (!iq3_layout_guards(buffer)) goto done;\n        fprintf(stderr,"PASS')
(out / 'bench.inc').write_text(bench)
