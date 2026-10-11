"""Two rows per SIMD group; share the native packed grid across 4/8 groups.

Generate private shaders and bridge only. Keep original row arithmetic,
weights and all session buffers; adjust exactly five single-IQ3 dispatches.
"""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / "tests/.iq3_share"
out.mkdir(exist_ok=True)
base = root / "tests/.iq3_rows"
source = (base / "kernels.metal").read_text()
first = source.index("kernel void private_iq3_rows_fast1(")
last = source.index("\n}\n", first) + 3
row2 = source[first:last]
assert "const uint nr0 = 2u;" in row2
assert row2.count("qw3_iq3s_dot32_shared(weights +") == 2
shader = "/* Native dot helper is supplied by the full production library source. */\n"
for variant, groups in ((1, 4), (2, 8)):
    body = row2.replace("private_iq3_rows_fast1", f"private_iq3_share_fast{variant}")
    body = body.replace("const uint nsg = 2u;", f"const uint nsg = {groups}u;")
    body = body.replace("i += 64u", f"i += {groups * 32}u")
    assert "ib32 += 32u" in body and "threadgroup ushort grid[512];" in body
    assert body.count("simd_sum(sum") == 2
    shader += body
(out / "kernels.metal").write_text(shader)

bridge = (root / "qw3_metal.m").read_text()
scopes = {
    "static int qw3_metal_session_matvec_dense_iq(": (
        "iq3s_fast ? (n_out + 7u) / 8u",
        "(iq3s_fast || iq3xxs_fast || iq2_fast) ? 64u :",
        "iq3s_fast ? iq3_share_threads() : (iq3xxs_fast || iq2_fast) ? 64u :"),
    "int qw3_metal_session_matvec_iq3_s_expert_x1_to_scratch(": (
        "MTLSizeMake((n_out + 7u) / 8u,", "NSUInteger threads = 64u;",
        "NSUInteger threads = iq3_share_threads();"),
    "static int qw3_metal_session_matvec_k_expert_inner_to_x1(": (
        "packed_rows8 ? (n_out + 7u) / 8u",
        "(packed_rows2 || packed_rows4 || packed_rows8) ? 64u :",
        "packed_rows8 ? iq3_share_threads() : (packed_rows2 || packed_rows4) ? 64u :"),
    "static int qw3_metal_session_matvec_k_expert_inner_to_scratch(": (
        "packed_rows8 ? (n_out + 7u) / 8u",
        "(packed_rows2 || packed_rows4 || packed_rows8) ? 64u :",
        "packed_rows8 ? iq3_share_threads() : (packed_rows2 || packed_rows4) ? 64u :"),
    "int qw3_metal_matvec_iq3_s_expert(": (
        "MTLSizeMake((n_out + 7u) / 8u,", "NSUInteger threads = 64u;",
        "NSUInteger threads = iq3_share_threads();"),
}
for marker, (old_grid, old_threads, new_threads) in scopes.items():
    start = bridge.index(marker)
    end = bridge.index("\n}\n", start) + 3
    section = bridge[start:end]
    assert section.count(old_grid) == section.count(old_threads) == 1, marker
    new_grid = old_grid.replace("(n_out + 7u) / 8u", "iq3_share_groups(n_out)")
    section = section.replace(old_grid, new_grid).replace(old_threads, new_threads)
    bridge = bridge[:start] + section + bridge[end:]
assert bridge.count("iq3_share_groups(n_out)") == 5
assert bridge.count("iq3_share_threads()") == 5
(out / "core.inc").write_text(bridge)

def rename(text):
    return (text.replace("iq3_rows", "iq3_share").replace("IQ3Rows", "IQ3Share")
            .replace("iq3Rows", "iq3Share").replace("rows_variant", "share_variant")
            .replace("IQ3 rows", "IQ3 share").replace("QW3_PRIVATE_IQ3_ROWS", "QW3_PRIVATE_IQ3_SHARE"))

hooks = rename((base / "hooks.inc").read_text())
hooks = hooks.replace("Single IQ3_S row geometry only; native weights, grid, lane sums and buffers.",
                      "Two IQ3_S rows per SIMD group; 4/8 groups share the original packed grid.")
start = hooks.index("static NSUInteger iq3_share_per_simd=")
end = hooks.index('#include "core.inc"', start)
hooks = hooks[:start] + '''static NSUInteger iq3_share_rows_per_group=8u;
static NSUInteger iq3_share_thread_count=64u;
static NSUInteger iq3_share_groups(NSUInteger n_out) {
    return (n_out+iq3_share_rows_per_group-1u)/iq3_share_rows_per_group;
}
static NSUInteger iq3_share_threads(void) { return iq3_share_thread_count; }
''' + hooks[end:]
hooks = hooks.replace("iq3_share_pipeline[v].maxTotalThreadsPerThreadgroup<64",
                      "iq3_share_pipeline[v].maxTotalThreadsPerThreadgroup<(v==1 ? 128u : 256u)")
hooks = hooks.replace("rows_per_simd=%d threads=64", "rows_per_simd=%d simdgroups=%d threads=%d")
hooks = hooks.replace("v==1 ? 2 : v==2 ? 8 : 4);",
                      "v ? 2 : 4,v==1 ? 4 : v==2 ? 8 : 2,v==1 ? 128 : v==2 ? 256 : 64);")
hooks = hooks.replace("iq3_share_per_simd=mode ? (share_variant==1 ? 2u : 8u) : 4u;",
                      "iq3_share_rows_per_group=mode && share_variant==2 ? 16u : 8u;\n"
                      "    iq3_share_thread_count=mode ? (share_variant==1 ? 128u : 256u) : 64u;")
assert "iq3_share_per_simd" not in hooks
assert "simdgroups=%d threads=%d" in hooks
(out / "hooks.inc").write_text(hooks)
for name in ("matrix.inc", "model.inc", "native.inc"):
    text = rename((base / name).read_text()).replace("baseline/rows2/rows8", "baseline/row2-SG4/row2-SG8")
    if name == "matrix.inc":
        old = "MTLSizeMake((no+(v ? (share_variant==1 ? 4u : 16u) : 8u)-1u)/(v ? (share_variant==1 ? 4u : 16u) : 8u)+1u,1,1) threadsPerThreadgroup:MTLSizeMake(64u,1,1)"
        new = "MTLSizeMake((no+rows_per_group-1u)/rows_per_group+1u,1,1) threadsPerThreadgroup:MTLSizeMake(thread_count,1,1)"
        assert text.count(old) == 1
        text = text.replace(old, new)
        text = text.replace("                [enc dispatchThreadgroups:",
                            "                NSUInteger rows_per_group=v && share_variant==2 ? 16u : 8u;\n"
                            "                NSUInteger thread_count=v ? (share_variant==1 ? 128u : 256u) : 64u;\n"
                            "                [enc dispatchThreadgroups:")
    (out / name).write_text(text)
