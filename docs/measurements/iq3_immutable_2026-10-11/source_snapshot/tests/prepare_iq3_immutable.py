"""Original IQ3_S shader with immutable weight/grid buffer descriptors only."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
base = root / "tests/.iq3_hint"
out = root / "tests/.iq3_immutable"
out.mkdir(exist_ok=True)

def rename(text):
    return (text.replace("iq3_hint", "iq3_immutable").replace("IQ3Hint", "IQ3Immutable")
            .replace("iq3Hint", "iq3Immutable").replace("hint_variant", "immutable_variant")
            .replace("IQ3 hint", "IQ3 immutable").replace("QW3_PRIVATE_IQ3_HINT", "QW3_PRIVATE_IQ3_IMMUTABLE"))

hooks = rename((base / "hooks.inc").read_text())
hooks = hooks.replace("explicit compute-pipeline threadgroup hints only",
                      "immutable descriptors for existing readonly weights/grid only")
hooks = hooks.replace("descriptor.maxTotalThreadsPerThreadgroup=64;\n"
                      "        descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth=(v==2);",
                      "descriptor.buffers[1].mutability=MTLMutabilityImmutable;\n"
                      "        if (v==2) descriptor.buffers[4].mutability=MTLMutabilityImmutable;\n"
                      "        if (descriptor.maxTotalThreadsPerThreadgroup!=0 ||\n"
                      "            descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth ||\n"
                      "            descriptor.buffers[0].mutability!=MTLMutabilityDefault ||\n"
                      "            descriptor.buffers[2].mutability!=MTLMutabilityDefault ||\n"
                      "            descriptor.buffers[3].mutability!=MTLMutabilityDefault) return 0;")
hooks = hooks.replace("iq3_immutable_pipeline[v].maxTotalThreadsPerThreadgroup!=64",
                      "iq3_immutable_pipeline[v].maxTotalThreadsPerThreadgroup<64")
hooks = hooks.replace("static_threadgroup_bytes=%lu\\n", "static_threadgroup_bytes=%lu immutable_weight=%d immutable_grid=%d\\n")
hooks = hooks.replace("(unsigned long)p.staticThreadgroupMemoryLength);",
                      "(unsigned long)p.staticThreadgroupMemoryLength,v!=0,v==2);")
assert "descriptor.maxTotalThreadsPerThreadgroup=64" not in hooks
assert "descriptor.buffers[1].mutability=MTLMutabilityImmutable" in hooks
(out / "hooks.inc").write_text(hooks)
for name in ("matrix.inc", "model.inc", "native.inc"):
    text = rename((base / name).read_text())
    text = text.replace("baseline/max64/max64-and-SIMD-multiple",
                        "baseline/immutable-weight/immutable-weight-and-grid")
    (out / name).write_text(text)
