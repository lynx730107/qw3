#!/usr/bin/env python3
"""Local llama-server MTP comparison, sequential GPU jobs and saved raw evidence."""
import argparse
import datetime
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import urllib.request


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--order', default='off,mtp1,mtp3,mtp3,mtp1,off')
    parser.add_argument('--verbosity', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 20 or 'off' not in args.order.split(','):
        parser.error('use 1..20 repetitions and include an off configuration')
    if any(c not in ('off', 'mtp1', 'mtp2', 'mtp3', 'mtp4', 'mtp5', 'mtp7')
           for c in args.order.split(',')):
        parser.error('unsupported configuration')
    args.output.mkdir(parents=True, exist_ok=False)
    model = (root / '../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf').resolve(strict=True)
    server = (root / '../../llama.cpp/build/bin/llama-server').resolve(strict=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_', 'QW3_'))}
    prompts = {
        'code': 'Spiega dettagliatamente come implementare una tabella hash in C.',
        'math': 'Risolvi passo per passo: tre scatole contengono rispettivamente 12, 18 e 24 matite. Quanti gruppi uguali si possono formare usando tutte le matite?',
        'story': 'Scrivi un breve racconto in italiano su una biblioteca che si sveglia di notte.',
    }
    tokens = {}
    for label, prompt in prompts.items():
        result = subprocess.run([str(root / 'qw3-test'), '-m', str(model), '--ngl', '0',
                                 '--chat-tokenize', '--nothink', '-p', prompt],
                                env=env, capture_output=True, text=True, check=True)
        tokens[label] = json.loads(result.stdout)
    # Exact 128-token input shared across configurations; preserve the assistant prefix.
    pp = tokens['code'][:-6]
    pp = (pp * 8)[:122] + tokens['code'][-6:]
    assert len(pp) == 128
    version = subprocess.run([str(server), '--version'], capture_output=True,
                             text=True, check=True)
    data = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'model': str(model), 'model_size': model.stat().st_size,
            'model_mtime_ns': model.stat().st_mtime_ns,
            'version': (version.stdout + version.stderr).strip(),
            'device': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
            'context': 512, 'kv': 'f16', 'full_offload_requested': True,
            'timing_definition': 'pp128 includes final prompt logits; tg32 emits 32 tokens, with 31 timed decode tokens (first token comes from prefill), matching server timings',
            'prompt_tokens': tokens, 'pp128_tokens': pp, 'runs': [], 'summary': {}}
    manifest = args.output / 'results.json'
    def save():
        manifest.write_text(json.dumps(data, indent=2) + '\n')
    save()
    port = 18983
    def request(path, payload=None):
        body = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', body,
                                     {'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=180) as response:
            return json.load(response)
    base = {'temperature': 0, 'top_k': 1, 'top_p': 1, 'min_p': 0,
            'repeat_penalty': 1, 'seed': 42, 'ignore_eos': True,
            'cache_prompt': False, 'return_tokens': True, 'stream': False}
    order = args.order.split(',')
    for block, config in enumerate(order):
        draft = 0 if config == 'off' else int(config.removeprefix('mtp'))
        command = [str(server), '-m', str(model), '-c', '512', '-b', '128', '-ub', '128',
                   '-np', '1', '-ngl', '999', '-fa', 'on', '-ctk', 'f16', '-ctv', 'f16',
                   '-ctkd', 'f16', '-ctvd', 'f16', '-ngld', '999',
                   '--host', '127.0.0.1', '--port', str(port),
                   '-lv', str(args.verbosity),
                   '--spec-type', 'none' if not draft else 'draft-mtp',
                   '--spec-draft-n-max', str(max(1, draft))]
        print(f'block={block} config={config} loading', flush=True)
        logpath = args.output / f'{block}-{config}.server.log'
        with logpath.open('w') as log:
            proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 180
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(f'server exited {proc.returncode}: {logpath}')
                    try:
                        request('/health')
                        break
                    except (OSError, ValueError):
                        if time.monotonic() > deadline:
                            raise TimeoutError(f'server startup: {logpath}')
                        time.sleep(.3)
                # Both prompt and generation warmups excluded from measured results.
                jobs = [('pp128', pp, 1), ('tg32_code', tokens['code'], 32)]
                for label, prompt, n in jobs:
                    result = request('/completion', dict(base, prompt=prompt, n_predict=n))
                    (args.output / f'{block}-{config}-{label}-warmup.json').write_text(json.dumps(result, indent=2))
                for rep in range(args.repetitions):
                    for label, prompt, n in jobs:
                        start = time.monotonic()
                        result = request('/completion', dict(base, prompt=prompt, n_predict=n))
                        wall = time.monotonic() - start
                        timing = result['timings']
                        if timing['prompt_n'] != len(prompt) or timing['predicted_n'] != n:
                            raise ValueError(f'unexpected counts: {timing}')
                        if len(result['tokens']) != n:
                            raise ValueError('missing output token IDs')
                        data['runs'].append({'block': block, 'config': config, 'job': label,
                                             'rep': rep, 'command': command, 'wall_seconds': wall,
                                             'response': result})
                        save()
                        print(f'{config} {label} rep={rep} pp={timing["prompt_per_second"]:.3f} '
                              f'tg={timing["predicted_per_second"]:.3f} '
                              f'draft={timing.get("draft_n_accepted", 0)}/{timing.get("draft_n", 0)}', flush=True)
                # Additional greedy correctness and workload-sensitivity probes.
                if config not in {r['config'] for r in data['runs'] if r['job'] == 'tg32_math'}:
                    for label in ('math', 'story'):
                        result = request('/completion', dict(base, prompt=tokens[label], n_predict=32))
                        data['runs'].append({'block': block, 'config': config, 'job': 'tg32_' + label,
                                             'rep': 0, 'command': command, 'response': result})
                        save()
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
    for config in dict.fromkeys(order):
        rows = [r for r in data['runs'] if r['config'] == config]
        code = [r['response']['timings'] for r in rows if r['job'] == 'tg32_code']
        prefill = [r['response']['timings']['prompt_ms'] for r in rows if r['job'] == 'pp128']
        data['summary'][config] = {
            'pp128_tps': 128000 / statistics.mean(prefill),
            'pp128_ms_samples': prefill,
            'tg32_tps': 31000 / statistics.mean(t['predicted_ms'] for t in code),
            'tg32_ms_samples': [t['predicted_ms'] for t in code],
            'draft_n': sum(t.get('draft_n', 0) for t in code),
            'draft_n_accepted': sum(t.get('draft_n_accepted', 0) for t in code)}
    data['greedy_checks'] = {}
    for job in ('pp128', 'tg32_code', 'tg32_math', 'tg32_story'):
        rows = [r for r in data['runs'] if r['job'] == job]
        baseline = next(r['response']['tokens'] for r in rows if r['config'] == 'off')
        data['greedy_checks'][job] = {'baseline_tokens': baseline,
                                    'all_equal': all(r['response']['tokens'] == baseline for r in rows)}
    data['blocks'] = []
    for block, config in enumerate(order):
        rows = [r for r in data['runs'] if r['block'] == block]
        pp_ms = [r['response']['timings']['prompt_ms'] for r in rows if r['job'] == 'pp128']
        tg_ms = [r['response']['timings']['predicted_ms'] for r in rows if r['job'] == 'tg32_code']
        data['blocks'].append({'block': block, 'config': config,
                               'pp128_tps': 128000 / statistics.mean(pp_ms),
                               'tg32_tps': 31000 / statistics.mean(tg_ms)})
    off_blocks = [b['tg32_tps'] for b in data['blocks'] if b['config'] == 'off']
    data['off_baseline_range_tps'] = [min(off_blocks), max(off_blocks)]
    data['large_baseline_drift'] = max(off_blocks) / min(off_blocks) > 1.2
    data['completed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
    print(json.dumps({'summary': data['summary'], 'greedy_checks': data['greedy_checks']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
