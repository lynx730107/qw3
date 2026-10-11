#!/usr/bin/env python3
"""Drive the real TUI: interrupt ordinary and manual-compaction prefill."""
import argparse
import errno
import fcntl
import json
import os
from pathlib import Path
import pty
import select
import struct
import subprocess
import tempfile
import termios
import time

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--agent', default='./qw3-agent')
p.add_argument('--log', required=True)
args = p.parse_args()
log = Path(args.log)
env = {k: v for k, v in os.environ.items() if not k.startswith(('QW3_', 'GGML_', 'MTL_'))}
env.update(TERM='xterm-256color', NO_COLOR='1', QW3_METAL_KV_F16='1',
           QW3_METAL_KV_Q8_0='0', QW3_METAL_NGL='64', MTL_DEBUG_LAYER='1')
data = bytearray()
with tempfile.TemporaryDirectory(prefix='qw3-pty-interrupt-') as tmp:
    trace = Path(tmp)/'trace.jsonl'
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 24, 120, 0, 0))
    cmd = [args.agent, '--metal', '-m', args.model, '--ctx', '4096', '--nothink',
           '--no-tools', '--no-prefix-cache', '--temp', '0', '-n', '1',
           '--store-dir', tmp, '--conversation', 'interrupt-fixture', '--trace', str(trace)]
    proc = subprocess.Popen(cmd, stdin=slave, stdout=slave, stderr=slave,
                            env=env, start_new_session=True)
    os.close(slave)

    def pump(timeout=.1):
        if select.select([master], [], [], timeout)[0]:
            try:
                block = os.read(master, 65536)
            except OSError as e:
                if e.errno == errno.EIO:
                    return
                raise
            data.extend(block)

    def wait_for(marker, start=0, timeout=30):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            at = data.find(marker, start)
            if at >= 0:
                return at+len(marker)
            assert proc.poll() is None, ('early exit', proc.returncode)
            pump()
        raise AssertionError(('timeout', marker, bytes(data[-1200:])))

    def send(text):
        os.write(master, text.encode()+b'\r')

    try:
        cursor = wait_for(b'qw3-agent ready.')
        # The startup banner precedes raw-mode/editor initialization. Sending
        # input there can be flushed by the terminal setup rather than submit.
        cursor = wait_for(b'tools off', cursor)
        send('Rispondi soltanto OK. '+ 'x '*1600)
        cursor = wait_for(b'prefill', cursor)
        start = time.monotonic();send('/interrupt')
        cursor = wait_for(b'prefill interrupted', cursor, 10)
        ordinary_latency = time.monotonic()-start
        # Allow the worker's completion notification to reach the editor.
        until = time.monotonic()+.5
        while time.monotonic()<until:
            pump(.05)
        assert not list(Path(tmp).glob('*.kvc')), 'canceled work autosaved a checkpoint'
        cursor = len(data);send('/compact')
        cursor = wait_for(b'compacting context (manual /compact)', cursor)
        cursor = wait_for(b'prefill', cursor)
        start = time.monotonic();send('/interrupt')
        cursor = wait_for(b'prefill interrupted', cursor, 10)
        compact_latency = time.monotonic()-start
        until = time.monotonic()+.5
        while time.monotonic()<until:
            pump(.05)
        assert not list(Path(tmp).glob('*.kvc')), 'canceled compaction autosaved a checkpoint'
        events = [json.loads(line) for line in trace.read_text().splitlines()]
        attempts = [e for e in events if e.get('event')=='compaction' and e.get('result')=='attempt']
        assert any(e.get('reason')=='manual /compact' for e in attempts), attempts
        assert not any(e.get('event')=='compaction' and e.get('result')=='completed' for e in events)
        send('/quit')
        # Keep draining the terminal while its editor restores the layout.
        deadline = time.monotonic()+10
        while proc.poll() is None and time.monotonic()<deadline:
            pump(.05)
        assert proc.poll()==0, ('quit did not complete', proc.poll())
        pump(0)
        print(json.dumps({'ordinary_interrupt_seconds': ordinary_latency,
                          'manual_compaction_interrupt_seconds': compact_latency,
                          'canceled_checkpoints': 0, 'model_processes_at_once': 1}, indent=2))
        print('PASS real_tui ordinary_prefill=interruptible manual_compaction=interruptible canceled_autosave=skipped')
    finally:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_bytes(data)
        if proc.poll() is None:
            proc.kill();proc.wait()
        os.close(master)
