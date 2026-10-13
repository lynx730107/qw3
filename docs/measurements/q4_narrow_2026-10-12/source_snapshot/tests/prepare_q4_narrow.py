"""Private Q4_K dispatch geometry: omit the idle second SIMD group for K<=8192."""
from pathlib import Path
root=Path(__file__).resolve().parent.parent
out=root/'tests/.q4_narrow'
out.mkdir(exist_ok=True)
source=(root/'qw3_metal.m').read_text()
marker='static NSUInteger qw3_metal_matvec_threads(id<MTLComputePipelineState> pipeline,\n                                            uint32_t n_blocks) {\n'
addition='    if (q4_narrow_mode && pipeline == g_matvec_q4_k_pipeline) {\n        if (n_blocks > 0u && n_blocks <= 32u && pipeline.threadExecutionWidth == 32u) {\n            q4_narrow_calls++;\n            return 32u;\n        }\n        q4_narrow_fallbacks++;\n    }\n'
assert source.count(marker)==1
core=source.replace(marker,marker+addition)
assert core.replace(addition,'')==source
(out/'core.inc').write_text(core)
(out/'hooks.inc').write_text('#import <Foundation/Foundation.h>\n#import <Metal/Metal.h>\n#include <stdint.h>\nstatic int q4_narrow_mode;\nstatic uint64_t q4_narrow_calls,q4_narrow_fallbacks;\n#include "core.inc"\n#define qw3_metal_env_n_gpu_layers q4_narrow_core_env_n_gpu_layers\n#define qw3_metal_llama_split_enabled q4_narrow_core_llama_split_enabled\n#define qw3_metal_llama_split_start q4_narrow_core_llama_split_start\n#include "../../qw3.c"\n#undef qw3_metal_env_n_gpu_layers\n#undef qw3_metal_llama_split_enabled\n#undef qw3_metal_llama_split_start\nstatic int q4_narrow_compile(void) {\n    if (!qw3_metal_init() || !qw3_metal_compile_kernels()) return 0;\n    id<MTLComputePipelineState> p=g_matvec_q4_k_pipeline;\n    if (p.threadExecutionWidth!=32u || p.maxTotalThreadsPerThreadgroup<64u) return 0;\n    fprintf(stderr,"q4_narrow_metadata width=%lu max_threads=%lu static_threadgroup_bytes=%lu dynamic_threadgroup_bytes=128 candidate_threads=32 native_min_threads=64 shader=original\\n",(unsigned long)p.threadExecutionWidth,(unsigned long)p.maxTotalThreadsPerThreadgroup,(unsigned long)p.staticThreadgroupMemoryLength);\n    return 1;\n}\nstatic void scratch_select(int mode) {q4_narrow_mode=mode!=0;}\nstatic void q4_narrow_print_counts(void) {\n    fprintf(stderr,"q4_narrow_dispatch_counts calls=%llu fallbacks=%llu\\n",(unsigned long long)q4_narrow_calls,(unsigned long long)q4_narrow_fallbacks);\n}\n')
base=root/'tests/.iq3_words'
s=(base/'model.inc').read_text().replace('iq3_words','q4_narrow').replace('IQ3 words','Q4 narrow')
s=s.replace('baseline/QS-word/QS-and-sign-word','baseline/narrow32').replace('phase<=2','phase<=1')
s=s.replace('            words_variant=phase ? phase : 1;scratch_select(phase!=0);','            scratch_select(phase!=0);')
s=s.replace('q4_narrow_calls=0;', 'q4_narrow_calls=0;q4_narrow_fallbacks=0;')
s=s.replace('32u*73u','32u*34u')
s=s.replace('            if (phase && (fgetc(oracle)', '            if (q4_narrow_fallbacks!=(phase ? 32u*2u : 0u)) goto done;\n            q4_narrow_print_counts();\n            if (phase && (fgetc(oracle)')
(out/'model.inc').write_text(s)
s=(base/'native.inc').read_text().replace('iq3_words','q4_narrow').replace('IQ3_WORDS','Q4_NARROW')
s=s.replace('    scratch_select(words_variant!=0);return 0;', '    return 0;')
s=s.replace(' || (strcmp(mode,"0") && strcmp(mode,"1") && strcmp(mode,"2"))',' || (strcmp(mode,"0") && strcmp(mode,"1"))')
s=s.replace('        words_variant=atoi(mode);int rc=grid_native_bench_main(argc,argv);','        int variant=atoi(mode);scratch_select(variant);int rc=grid_native_bench_main(argc,argv);')
s=s.replace('words_variant','variant')
s=s.replace('        fprintf(stderr,"PASS', '        q4_narrow_print_counts();\n        fprintf(stderr,"PASS')
(out/'native.inc').write_text(s)
