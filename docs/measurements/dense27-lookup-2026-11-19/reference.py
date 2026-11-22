"""Benchmark prompt lookup on the local independent llama.cpp engine."""
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

out = Path(__file__).resolve().parent
root = out.parents[2]
model = (root / '../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf').resolve()
server = (root / '../../llama.cpp/build/bin/llama-server').resolve()
def traces(name):
    return [json.loads(line) for line in (out / f'{name}.jsonl').read_text().splitlines()]

jobs = {'coding': traces('coding')[0], 'web': traces('web')[-1]}
results = []
for block, mode in enumerate(['none','ngram-simple','ngram-simple','none']):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        port = sock.getsockname()[1]
    def request(endpoint, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(f'http://127.0.0.1:{port}/{endpoint}', data, {'Content-Type':'application/json'})
        with urllib.request.urlopen(req, timeout=180) as response:
            return json.load(response)
    command = [str(server), '-m',str(model),'-ngl','99','-c','8192','-np','1',
               '-b','128','-ub','128','-ctk','f16','-ctv','f16','-fa','on',
               '--host','127.0.0.1','--port',str(port),'--spec-type',mode,
               '--spec-ngram-simple-size-n','6','--spec-ngram-simple-size-m','7',
               '--spec-draft-n-max','7']
    with (out / f'llama-{block}.log').open('w') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f'server exited {proc.returncode}')
                try:
                    request('health')
                    break
                except OSError:
                    time.sleep(.3)
            else:
                raise TimeoutError('server startup')
            for name, turn in jobs.items():
                cap = min(64,len(turn['generated']))
                data = request('completion', dict(prompt=turn['prompt'], temperature=0,
                               repeat_penalty=1, seed=1, cache_prompt=False,
                               n_predict=cap, return_tokens=True, stream=False))
                if len(data['tokens']) != data['timings']['predicted_n']:
                    raise ValueError('token count mismatch')
                results.append(dict(block=block, mode=mode, job=name,
                                    prompt_length=len(turn['prompt']), limit=cap,
                                    tokens=data['tokens'], timings=data['timings']))
                (out/'reference.json').write_text(json.dumps(results, indent=2)+'\n')
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
