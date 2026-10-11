"""Private IQ4_XS single/pair/add geometry: 4/8 SIMD groups, two rows each.

Keep every floating expression, LUT, K assignment and original data buffer.
Generate bridge dispatch changes without editing production inference.
"""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / "tests/.iq4_share"
out.mkdir(exist_ok=True)
source = (root / "metal/qw3_core_linear.metal").read_text()
names = ["qw3_matvec_iq4_xs_fast", "qw3_matvec_iq4_xs_pair_fast", "qw3_matvec_iq4_xs_add_x0_fast"]
shader = "/* All helpers and declarations come from the full original Metal source. */\n"
for variant, groups in ((1, 4), (2, 8)):
    for kind, name in enumerate(names):
        first = source.index("kernel void " + name + "(")
        last = source.index("\n}\n", first) + 3
        body = source[first:last].replace(name, f"private_iq4_share_fast{variant}_{kind}")
        assert body.count("const uint nsg = 2u;") == 1
        body = body.replace("const uint nsg = 2u;", f"const uint nsg = {groups}u;")
        assert "const uint nr0 = 2u;" in body and "b += 2u" in body
        shader += body
(out / "kernels.metal").write_text(shader)

bridge = (root / "qw3_metal.m").read_text()
direct = [
    "int qw3_metal_session_matvec_iq4_xs_x1_to_scratch(",
    "int qw3_metal_session_matvec_iq4_xs_pair_x1_to_scratch(",
    "int qw3_metal_session_matvec_iq4_xs_inner_add_x0(",
    "int qw3_metal_matvec_iq4_xs_expert(",
]
for marker in direct:
    first = bridge.index(marker)
    last = bridge.index("\n}\n", first) + 3
    body = bridge[first:last]
    old_grid = "(n_out + 3u) / 4u"
    old_threads = "MTLSizeMake(64u, 1u, 1u)"
    assert body.count(old_grid) == body.count(old_threads) == 1, marker
    body = body.replace(old_grid, "iq4_share_groups(n_out)")
    body = body.replace(old_threads, "MTLSizeMake(iq4_share_threads(), 1u, 1u)")
    bridge = bridge[:first] + body + bridge[last:]
generic = [
    "int qw3_metal_session_matvec_quant_x1_to_scratch(",
    "static int qw3_metal_session_matvec_k_expert_inner_to_x1(",
    "static int qw3_metal_session_matvec_k_expert_inner_to_scratch(",
]
for marker in generic:
    first = bridge.index(marker)
    last = bridge.index("\n}\n", first) + 3
    body = bridge[first:last]
    old_threads = "NSUInteger threads = (packed_rows2 || packed_rows4 || packed_rows8) ? 64u :"
    old_grid = "NSUInteger groups = packed_rows8 ? (n_out + 7u) / 8u :"
    assert body.count(old_threads) == body.count(old_grid) == 1, marker
    body = body.replace(old_threads, "NSUInteger threads = pipeline == g_matvec_iq4_xs_pipeline ? iq4_share_threads() :\n        (packed_rows2 || packed_rows4 || packed_rows8) ? 64u :")
    body = body.replace(old_grid, "NSUInteger groups = pipeline == g_matvec_iq4_xs_pipeline ? iq4_share_groups(n_out) :\n        packed_rows8 ? (n_out + 7u) / 8u :")
    bridge = bridge[:first] + body + bridge[last:]
marker = "int qw3_metal_session_sparse_moe_topk("
first = bridge.index(marker)
last = bridge.index("\n}\n", first) + 3
body = bridge[first:last]
old = """        if (down_type == 23) {
            threads = 64;
            [enc dispatchThreadgroups:MTLSizeMake((n_embd + 3u) / 4u, 1, 1)"""
assert body.count(old) == 1
body = body.replace(old, """        if (down_type == 23) {
            threads = iq4_share_threads();
            [enc dispatchThreadgroups:MTLSizeMake(iq4_share_groups(n_embd), 1, 1)""")
bridge = bridge[:first] + body + bridge[last:]
assert bridge.count("iq4_share_groups(") == bridge.count("iq4_share_threads()") == 8
(out / "core.inc").write_text(bridge)

def rename(text):
    return (text.replace("iq3_share", "iq4_share").replace("IQ3Share", "IQ4Share")
            .replace("iq3Share", "iq4Share").replace("IQ3 share", "IQ4 share")
            .replace("QW3_PRIVATE_IQ3_SHARE", "QW3_PRIVATE_IQ4_SHARE"))

base = root / "tests/.iq3_share"
hooks = rename((base / "hooks.inc").read_text())
hooks = hooks.replace("Two IQ3_S rows per SIMD group; 4/8 groups share the original packed grid.",
                      "Three IQ4_XS pipelines: native row arithmetic, LUT and data buffers.")
hooks = hooks.replace("iq4_share_pipeline[3]", "iq4_share_pipeline[3][3]", 1)
hooks = hooks.replace("static uint64_t iq4_share_calls;", "static uint64_t iq4_share_calls,iq4_share_calls_by_kind[3];")
hooks = hooks.replace("if (pipeline==iq4_share_pipeline[1] || pipeline==iq4_share_pipeline[2]) iq4_share_calls++;",
                      "for (int v=1;v<=2;v++) for (int kind=0;kind<3;kind++)\n"
                      "        if (pipeline==iq4_share_pipeline[v][kind]) {iq4_share_calls++;iq4_share_calls_by_kind[kind]++;}")
hooks = hooks.replace("iq4_share_rows_per_group=8u", "iq4_share_rows_per_group=4u")
first = hooks.index("static int iq4_share_compile(void) {")
hooks = hooks[:first] + '''static int iq4_share_compile(void) {
    if (iq4_share_pipeline[2][2]) return 1;
    if (!qw3_metal_compile_kernels()) return 0;
    iq4_share_pipeline[0][0]=g_matvec_iq4_xs_pipeline;
    iq4_share_pipeline[0][1]=g_matvec_iq4_xs_pair_pipeline;
    iq4_share_pipeline[0][2]=g_matvec_iq4_xs_add_x0_pipeline;
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) options.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/.iq4_share/kernels.metal"
        encoding:NSUTF8StringEncoding error:&error];
    NSString *original=qw3_metal_full_kernel_source();
    NSString *source=extra && original ? [original stringByAppendingString:extra] : nil;
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    for (int v=1;v<=2;v++) for (int kind=0;kind<3;kind++) {
        NSString *name=[NSString stringWithFormat:@"private_iq4_share_fast%d_%d",v,kind];
        id<MTLFunction> fn=[lib newFunctionWithName:name];
        id<MTLComputePipelineState> p=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!p || p.maxTotalThreadsPerThreadgroup<(v==1 ? 128u : 256u) || p.threadExecutionWidth!=32 ||
            p.staticThreadgroupMemoryLength!=iq4_share_pipeline[0][kind].staticThreadgroupMemoryLength) {
            fprintf(stderr,"IQ4 share compile variant=%d kind=%d: %s\\n",v,kind,error.localizedDescription.UTF8String);return 0;
        }
        iq4_share_pipeline[v][kind]=p;
    }
    for (int v=0;v<=2;v++) for (int kind=0;kind<3;kind++) {
        id<MTLComputePipelineState> p=iq4_share_pipeline[v][kind];
        fprintf(stderr,"iq4_share_metadata variant=%d kind=%d thread_width=%lu max_threads=%lu static_threadgroup_bytes=%lu rows_per_simd=2 simdgroups=%d threads=%d dynamic_threadgroup_bytes=128\\n",
            v,kind,(unsigned long)p.threadExecutionWidth,(unsigned long)p.maxTotalThreadsPerThreadgroup,
            (unsigned long)p.staticThreadgroupMemoryLength,v==1 ? 4 : v==2 ? 8 : 2,v==1 ? 128 : v==2 ? 256 : 64);
    }
    return 1;
}
static __attribute__((unused)) void scratch_select(int mode) {
    int v=mode ? share_variant : 0;
    g_matvec_iq4_xs_pipeline=iq4_share_pipeline[v][0];
    g_matvec_iq4_xs_pair_pipeline=iq4_share_pipeline[v][1];
    g_matvec_iq4_xs_add_x0_pipeline=iq4_share_pipeline[v][2];
    iq4_share_rows_per_group=mode ? (share_variant==1 ? 8u : 16u) : 4u;
    iq4_share_thread_count=mode ? (share_variant==1 ? 128u : 256u) : 64u;
}
static __attribute__((unused)) void iq4_share_print_counts(void) {
    fprintf(stderr,"iq4_share_dispatch_counts single=%llu pair=%llu add=%llu\\n",
        (unsigned long long)iq4_share_calls_by_kind[0],(unsigned long long)iq4_share_calls_by_kind[1],
        (unsigned long long)iq4_share_calls_by_kind[2]);
}
'''
(out / "hooks.inc").write_text(hooks)
for name in ("model.inc", "native.inc"):
    text = rename((base / name).read_text())
    if name == "model.inc":
        text = text.replace("iq4_share_calls=0;", "iq4_share_calls=0;memset(iq4_share_calls_by_kind,0,sizeof(iq4_share_calls_by_kind));")
        text = text.replace("32u*73u", "32u*142u")
        text = text.replace("            if (phase && (fgetc", "            if (phase && (iq4_share_calls_by_kind[0]!=3072u || iq4_share_calls_by_kind[1]!=448u || iq4_share_calls_by_kind[2]!=1024u)) goto done;\n            iq4_share_print_counts();\n            if (phase && (fgetc")
    else:
        text = text.replace("        fprintf(stderr,\"PASS iq4_share_native", "        iq4_share_print_counts();\n        fprintf(stderr,\"PASS iq4_share_native")
    (out / name).write_text(text)

text = (root / "tests/test_iq4_loads.m").read_text()
text = text[text.index("int main(int argc,char **argv) {"):]
text = text.replace("    if (argc==4) return iq4_load_full_quality_main(argc,argv);\n", "")
text = text.replace("iq4_load", "iq4_share").replace("IQ4 loads", "IQ4 share").replace("iq4_variant", "share_variant")
text = text.replace("share_variant>3", "share_variant>2").replace(" || !iq4_share_model_aligned(engine)", "")
text = text.replace("pattern<2", "pattern<3")
text = text.replace("pattern ? (i&1u ? -0.0f : 0.0f) :", "pattern==1 ? (i&1u ? -0.0f : 0.0f) :\n                pattern==2 ? (float)((int)(i%7u)-3)*(i%5u==0u ? 0x1p8f : 0x1p-12f) :")
text = text.replace("const float scale=-0.25f;", "const float scale=pattern==1 ? -0.0f : pattern==2 ? 0.75f : -0.25f;")
text = text.replace("                    [enc dispatchThreadgroups:MTLSizeMake((no+3u)/4u,1,1) threadsPerThreadgroup:MTLSizeMake(64u,1,1)];",
                    "                    NSUInteger rows=v ? (share_variant==1 ? 8u : 16u) : 4u;\n"
                    "                    NSUInteger threads=v ? (share_variant==1 ? 128u : 256u) : 64u;\n"
                    "                    [enc dispatchThreadgroups:MTLSizeMake((no+rows-1u)/rows+1u,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];")
(out / "matrix.inc").write_text(text)
