#!/usr/bin/env python3
"""Repeated real-model handoffs through the TUI, with queue and memory evidence."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import pty
import select
import struct
import subprocess
import sys
import termios
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
from bench_hybrid_agent import Meter
from test_agent_hybrid import events, assert_serial, MOE, DENSE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True)
    work = out / 'work'
    work.mkdir()
    trace = out / 'trace.jsonl'
    command = [str(ROOT / 'qw3-agent'), '--metal', '-m', str(MOE),
               '--expert-model', str(DENSE), '--expert-mode', 'manual',
               '--ctx', '8192', '--expert-ctx', '8192', '--kv-f16',
               '--nothink', '--expert-think', 'none', '--read-only',
               '--temp', '0', '--seed', '1', '-n', '128', '--no-prefix-cache',
               '--chdir', str(work), '--store-dir', str(out / 'store'),
               '--conversation', 'cycles', '--trace', str(trace)]
    (out / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('QW3_', 'GGML_'))}
    env.pop('NO_COLOR', None)
    env.update(TERM='xterm-256color', QW3_METAL_KERNEL_DIR=str(ROOT / 'metal'))
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 30, 160, 0, 0))
    process = subprocess.Popen(command, stdin=slave, stdout=slave, stderr=slave,
                               env=env, start_new_session=True)
    os.close(slave)
    raw = bytearray()
    samples, cycles = [], []
    meter = Meter()
    started = last_sample = time.monotonic()

    def pump():
        nonlocal last_sample
        if select.select([master], [], [], .1)[0]:
            try:
                raw.extend(os.read(master, 65536))
            except OSError:
                pass
        if time.monotonic() - last_sample >= 1:
            row = meter.sample(process.pid)
            row['elapsed'] = time.monotonic() - started
            samples.append(row)
            last_sample = time.monotonic()

    def until(condition, seconds=240):
        deadline = time.monotonic() + seconds
        while not condition():
            assert process.poll() is None, ('TUI exited', process.returncode)
            assert time.monotonic() < deadline, 'TUI phase timed out'
            pump()

    def completed():
        return sum(r['event'] == 'hybrid_completed' and r.get('role') == 'moe'
                   for r in events(trace))

    try:
        until(lambda: b'qw3 ' in raw)
        target = 0
        for cycle in range(1, 6):
            begin = time.monotonic() - started
            os.write(master, f'/expert run Rispondi soltanto CYCLE_{cycle}_OK. Non usare tool.\n'.encode())
            target += 1
            if cycle == 1:
                until(lambda: any(r['event'] == 'hybrid_ready' and r.get('role') == 'dense'
                                  for r in events(trace)))
                os.write(master, b'Rispondi soltanto QUEUE_OK.\n')
                target += 1
            until(lambda: completed() >= target)
            # Drain the ready footer and measure the surviving supervisor/primary.
            for _ in range(5):
                pump()
            snapshot = meter.sample(process.pid)
            parent = next(p for p in snapshot['processes'] if p['pid'] == process.pid)
            rows = [s for s in samples if s['elapsed'] >= begin]
            cycles.append(dict(cycle=cycle, supervisor_rss=parent['rss'],
                               supervisor_physical=parent['physical'],
                               peak_rss=max((s['rss'] for s in rows), default=0),
                               peak_physical=max((s['physical'] for s in rows), default=0),
                               swap=snapshot['swap']))
            print('DONE handoff cycle', cycle, flush=True)
        os.write(master, b'/quit\n')
        until(lambda: process.poll() is not None, seconds=20)
        process.wait()
        assert process.returncode == 0
        assert_serial(events(trace))
        assert sum(r['event'] == 'hybrid_process_start' and r.get('role') == 'dense'
                   for r in events(trace)) == 5
        text = raw.decode('utf-8', errors='replace')
        assert 'dense' in text and 'left' in text and 'queued' in text, 'missing live TUI status'
        assert '\x1b[38;5;' in text or '\x1b[1;38;5;' in text, 'missing TUI colors'
        # Model children are reaped. The surviving supervisor holds no model;
        # its memory must remain bounded across warm handoffs.
        drift = cycles[-1]['supervisor_rss'] - cycles[1]['supervisor_rss']
        assert drift < 32 * 1024 * 1024, ('supervisor memory growth', drift)
        result = dict(passed=True, cycles=cycles, supervisor_rss_drift_bytes=drift,
                      owner_processes_overlap=False,
                      note='RSS/physical footprint omit some shared/file-backed GPU memory; swap is machine-wide.')
        (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    finally:
        if process.poll() is None:
            process.send_signal(2)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, 9)
                process.wait()
        os.close(master)
        (out / 'terminal.raw').write_bytes(raw)
        (out / 'memory.json').write_text(json.dumps(samples) + '\n')


if __name__ == '__main__':
    main()
