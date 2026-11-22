"""Use the repository's llama logits comparator with a real batched qw3 prefill."""
from pathlib import Path
import subprocess
import sys

repo = Path.cwd()
work = Path(sys.argv[1]).resolve()
build = work / 'frontier_build'
build.mkdir(exist_ok=True)
llama = (repo / '../../llama.cpp').resolve()
lib = llama / 'build/bin'
source = (repo / 'tests/test_dense_logits.c').read_text()
source = source.replace('TEST_CTX = 128', 'TEST_CTX = 256').replace('TEST_THREADS = 16', 'TEST_THREADS = 4')
source = source.replace('#include "../qw3.h"', '#include "qw3.h"')
start = source.index('    for (int i = 0; i < n_tokens; i++) {\n        if (qw3_session_eval(')
end = source.index('    if (getenv("QW3_DENSE_TRACE"))', start)
source = source[:start] + '''    qw3_tokens batch_tokens = {.v = (int *)tokens, .len = n_tokens, .cap = n_tokens};
    if (qw3_session_sync(session, &batch_tokens, err, sizeof(err))) {
        fprintf(stderr, "test-dense-logits: batched qw3 eval failed: %s\\n", err);
        qw3_session_free(session);
        qw3_engine_close(engine);
        return 1;
    }
''' + source[end:]
(build / 'frontier.c').write_text(source)
# Set the diagnostic mode outside the Metal encoder so both modes share the same code.
(build / 'metal_frontier.m').write_text('''#include "qw3_metal.m"
__attribute__((constructor)) static void frontier_mode(void) {
    const char *value = getenv("QW3_PARTIAL_STAGED");
    partial_mode = value ? (unsigned)atoi(value) : 0;
}
''')
subprocess.run(['cc', '-O3', '-ffast-math', '-mcpu=native', '-fobjc-arc', '-I'+str(work),
                '-c', 'metal_frontier.m', '-o', 'metal_frontier.o'], cwd=build, check=True)
subprocess.run(['cc', '-O3', '-fno-finite-math-only', '-std=c11', '-I'+str(llama/'include'),
                '-I'+str(llama/'ggml/include'), '-I'+str(work), 'frontier.c',
                str(work/'qw3.o'), 'metal_frontier.o',
                '-L'+str(lib), '-Wl,-rpath,'+str(lib), '-lllama', '-lggml', '-lggml-base',
                '-lm', '-pthread', '-framework', 'Foundation', '-framework', 'Metal',
                '-o', str(work/'frontier')], cwd=build, check=True)
