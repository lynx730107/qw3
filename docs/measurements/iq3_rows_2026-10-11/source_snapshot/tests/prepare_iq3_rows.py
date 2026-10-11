"""Single IQ3_S matvec: two/eight rows per SIMD group, native 64 threads.

Keep the original dot helper, lane assignments, FP expressions and grid.
Only private bridge dispatches for the single-IQ3 pipeline change. The
paired IQ3 pipeline and production files are never edited.
"""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / "tests/.iq3_rows"
out.mkdir(exist_ok=True)
source = (root / "metal/qw3_core_moe.metal").read_text()
first = source.index("kernel void qw3_matvec_iq3_s_fast(")
last = source.index("\n}\n", first) + 3
original = source[first:last]
shader = "/* Native dot helper is supplied by the full production library source. */\n"
for variant, rows in ((1, 2), (2, 8)):
    body = original.replace("qw3_matvec_iq3_s_fast", f"private_iq3_rows_fast{variant}")
    body = body.replace("const uint nr0 = 4u;", f"const uint nr0 = {rows}u;")
    start = body.index("    float sum0 =")
    end = body.index("    uint nb32", start)
    body = body[:start] + "    float " + ", ".join(f"sum{i} = 0.0f" for i in range(rows)) + ";\n" + body[end:]
    start = body.index("        uint row = first_row;")
    end = body.index("    }\n    sum0 =", start)
    updates = ""
    for i in range(rows):
        updates += ("        uint row = first_row;\n" if i == 0 else f"        row = first_row + {i}u;\n")
        updates += f"""        if (row < args.n_out)
            sum{i} += qw3_iq3s_dot32_shared(weights +
                uint64_t(row) * uint64_t(args.row_bytes) + block_off,
                xx, grid, ib);
"""
    body = body[:start] + updates + body[end:]
    start = body.index("    sum0 = simd_sum")
    tail = "".join(f"    sum{i} = simd_sum(sum{i});\n" for i in range(rows))
    tail += "    if (lane == 0) {\n"
    for i in range(rows):
        tail += ("        uint row = first_row;\n" if i == 0 else f"        row = first_row + {i}u;\n")
        tail += f"        if (row < args.n_out) out[row] = sum{i};\n"
    body = body[:start] + tail + "    }\n}\n"
    assert body.count("qw3_iq3s_dot32_shared(weights +") == rows
    assert body.count("simd_sum(sum") == rows
    assert "i += 64u" in body and "const uint nsg = 2u;" in body
    shader += body
(out / "kernels.metal").write_text(shader)

bridge = (root / "qw3_metal.m").read_text()
scopes = {
    "static int qw3_metal_session_matvec_dense_iq(": "iq3s_fast ? (n_out + 7u) / 8u",
    "int qw3_metal_session_matvec_iq3_s_expert_x1_to_scratch(": "MTLSizeMake((n_out + 7u) / 8u,",
    "static int qw3_metal_session_matvec_k_expert_inner_to_x1(": "packed_rows8 ? (n_out + 7u) / 8u",
    "static int qw3_metal_session_matvec_k_expert_inner_to_scratch(": "packed_rows8 ? (n_out + 7u) / 8u",
    "int qw3_metal_matvec_iq3_s_expert(": "MTLSizeMake((n_out + 7u) / 8u,",
}
for marker, old in scopes.items():
    start = bridge.index(marker)
    end = bridge.index("\n}\n", start) + 3
    section = bridge[start:end]
    assert section.count(old) == 1, marker
    new = old.replace("(n_out + 7u) / 8u", "iq3_rows_groups(n_out)")
    section = section.replace(old, new)
    bridge = bridge[:start] + section + bridge[end:]
assert bridge.count("iq3_rows_groups(n_out)") == 5
(out / "core.inc").write_text(bridge)

base = root / "tests/.iq3_immutable"
def rename(text):
    return (text.replace("iq3_immutable", "iq3_rows").replace("IQ3Immutable", "IQ3Rows")
            .replace("iq3Immutable", "iq3Rows").replace("immutable_variant", "rows_variant")
            .replace("IQ3 immutable", "IQ3 rows").replace("QW3_PRIVATE_IQ3_IMMUTABLE", "QW3_PRIVATE_IQ3_ROWS"))

hooks = rename((base / "hooks.inc").read_text())
hooks = hooks.replace("Original shader and buffers; immutable descriptors for existing readonly weights/grid only.",
                      "Single IQ3_S row geometry only; native weights, grid, lane sums and buffers.")
hooks = hooks.replace('#include "../../qw3_metal.m"', '''static NSUInteger iq3_rows_per_simd=4u;
static NSUInteger iq3_rows_groups(NSUInteger n_out) {
    NSUInteger rows=2u*iq3_rows_per_simd;
    return (n_out+rows-1u)/rows;
}
#include "core.inc"''')
start = hooks.index("static int iq3_rows_compile(void) {")
hooks = hooks[:start] + '''static int iq3_rows_compile(void) {
    if (iq3_rows_pipeline[2]) return 1;
    if (!qw3_metal_compile_kernels()) return 0;
    iq3_rows_pipeline[0]=g_matvec_iq3_s_pipeline;
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) options.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/.iq3_rows/kernels.metal"
        encoding:NSUTF8StringEncoding error:&error];
    NSString *original=qw3_metal_full_kernel_source();
    NSString *source=extra && original ? [original stringByAppendingString:extra] : nil;
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    for (int v=1;v<=2;v++) {
        NSString *name=[NSString stringWithFormat:@"private_iq3_rows_fast%d",v];
        id<MTLFunction> fn=[lib newFunctionWithName:name];
        iq3_rows_pipeline[v]=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!iq3_rows_pipeline[v] || iq3_rows_pipeline[v].maxTotalThreadsPerThreadgroup<64 ||
            iq3_rows_pipeline[v].threadExecutionWidth!=32 ||
            iq3_rows_pipeline[v].staticThreadgroupMemoryLength!=iq3_rows_pipeline[0].staticThreadgroupMemoryLength) {
            fprintf(stderr,"IQ3 rows compile variant=%d: %s\\n",v,error.localizedDescription.UTF8String);return 0;
        }
    }
    for (int v=0;v<=2;v++) {
        id<MTLComputePipelineState> p=iq3_rows_pipeline[v];
        fprintf(stderr,"iq3_rows_metadata variant=%d thread_width=%lu max_threads=%lu static_threadgroup_bytes=%lu rows_per_simd=%d threads=64\\n",
            v,(unsigned long)p.threadExecutionWidth,(unsigned long)p.maxTotalThreadsPerThreadgroup,
            (unsigned long)p.staticThreadgroupMemoryLength,v==1 ? 2 : v==2 ? 8 : 4);
    }
    return 1;
}
static __attribute__((unused)) void scratch_select(int mode) {
    g_matvec_iq3_s_pipeline=iq3_rows_pipeline[mode ? rows_variant : 0];
    iq3_rows_per_simd=mode ? (rows_variant==1 ? 2u : 8u) : 4u;
}
'''
(out / "hooks.inc").write_text(hooks)
for name in ("matrix.inc", "model.inc", "native.inc"):
    text = rename((base / name).read_text())
    text = text.replace("baseline/immutable-weight/immutable-weight-and-grid", "baseline/rows2/rows8")
    if name == "matrix.inc":
        text = text.replace("setThreadgroupMemoryLength:256", "setThreadgroupMemoryLength:128")
        text = text.replace("MTLSizeMake((no+7u)/8u,1,1)", "MTLSizeMake((no+(v ? (rows_variant==1 ? 4u : 16u) : 8u)-1u)/(v ? (rows_variant==1 ? 4u : 16u) : 8u)+1u,1,1)")
    (out / name).write_text(text)
