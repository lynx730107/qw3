"""Run the same local agent tests in separate processes for both configurations."""
from pathlib import Path
import json
import os
import re
import subprocess
import sys

root = Path.cwd()
work = Path(sys.argv[1]).resolve()
out = Path(__file__).resolve().parent
model = (root / '../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf').resolve()
agent = str(work / 'agent_build/agent')
env = {k: v for k, v in os.environ.items() if not k.startswith(('QW3_', 'GGML_', 'MTL_'))}
env.update(QW3_AGENT_BIN=agent, QW3_DENSE_MODEL=str(model), QW3_MODEL=str(model),
           QW3_METAL_KV_F16='1', QW3_METAL_NGL='64', QW3_METAL_KERNEL_DIR=str(work/'metal'))
tests = [
    ('dense-agent', ['sh', str(root/'tests/test_dense_agent.sh')], work),
    ('pressure', [str(work/'agent_build/pressure'), str(model)], work),
]
results = []
for name, command, cwd in tests:
    for mode in (0, 3):
        trace = out / f'agent-{name}-v{mode}.trace'
        trace.write_text('')
        local = dict(env, QW3_PARTIAL_STAGED=str(mode), QW3_PARTIAL_AGENT_TRACE=str(trace))
        logname = f'agent-{name}-v{mode}.log'
        with (out/logname).open('w') as log:
            try:
                process = subprocess.run(command, cwd=cwd, env=local, timeout=240,
                                         stdout=log, stderr=subprocess.STDOUT)
                code = process.returncode
            except subprocess.TimeoutExpired:
                code = 'timeout'
        calls = sum(map(int, re.findall(r'dispatches=(\d+)', trace.read_text())))
        results.append({'test': name, 'mode': mode, 'exit': code, 'log': logname, 'partial_dispatches': calls})
        (out/'agent.json').write_text(json.dumps(results, indent=2)+'\n')
        print(name, mode, code, calls, flush=True)
