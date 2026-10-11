"""Exercise benchmark pacing, engine order and failed-log preservation without inference."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parent.parent / 'tools/bench_dense_reference.py'
spec = importlib.util.spec_from_file_location('dense_reference_fixture', SCRIPT)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class ReferenceProtocolTests(unittest.TestCase):
    def exercise(self, scope=None, fail=False):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            model, qw3, llama = [base / n for n in ['fixture.gguf', 'qw3-fixture', 'llama-fixture']]
            for p in [model, qw3, llama]:
                p.write_text('fixture only; never executed')
            output = base / 'output'
            argv = [str(SCRIPT), '--model', str(model), '--qw3', str(qw3), '--llama', str(llama),
                    '--output', str(output), '--jobs', 'pp128', '--pairs', '2',
                    '--repetitions', '2', '--cooldown', '7']
            if scope is not None:
                argv += ['--cooldown-scope', scope]
            events = []

            def fixture_run(command, **kwargs):
                stdout, stderr, rc = '', '', 0
                if command[0] == 'caffeinate':
                    engine = 'qw3' if Path(command[2]) == qw3.resolve() else 'llama'
                    events.append(engine)
                    env = kwargs['env']
                    self.assertEqual(env['QW3_METAL_KV_F16'], '1')
                    self.assertEqual(env['QW3_METAL_KV_Q8_0'], '0')
                    self.assertFalse(any(k in env for k in ['QW3_FIXTURE', 'GGML_FIXTURE', 'MTL_DEBUG_LAYER']))
                    if fail:
                        stdout, stderr, rc = 'fixture stdout before failure', 'fixture stderr before failure', 127
                    elif engine == 'qw3':
                        stdout = 'n_prompt,n_gen,n_depth,reps,avg_ms\n128,0,0,2,100\n'
                        stderr = ''.join(f'qw3-bench: sample {i}/2 1280 tok/s 100 ms\n' for i in [1, 2])
                    else:
                        stdout = json.dumps([dict(build_commit='fixture', build_number=0, type_k='f16',
                            type_v='f16', n_gpu_layers=999, n_prompt=128, n_gen=0, n_depth=0,
                            avg_ns=100000000, samples_ns=[100000000, 100000000], samples_ts=[1280, 1280])])
                        stderr = 'offloaded 65/65 layers\noffloading output layer to GPU\nn_ctx = 256\n'
                elif command[0] == 'git':
                    stdout = '' if 'diff' in command else 'fixture_revision\n'
                elif command[0] == 'sysctl':
                    stdout = 'fixture_device\n' if '-n' in command else 'fixture_swap\n'
                else:
                    self.fail(f'unexpected external command {command}')
                return subprocess.CompletedProcess(command, rc, stdout, stderr)

            with patch.object(sys, 'argv', argv), patch.object(bench.subprocess, 'run', fixture_run), \
                 patch.object(bench.time, 'sleep', lambda seconds: events.append(('pause', seconds))), \
                 patch.dict(os.environ, {'QW3_FIXTURE': 'x', 'GGML_FIXTURE': 'x', 'MTL_DEBUG_LAYER': '1'}), \
                 contextlib.redirect_stdout(io.StringIO()):
                if fail:
                    with self.assertRaises(subprocess.CalledProcessError):
                        bench.main()
                else:
                    bench.main()
            data = json.loads((output / 'results.json').read_text())
            if fail:
                self.assertEqual(data['failure']['returncode'], 127)
                self.assertEqual((output / data['failure']['stdout']).read_text(), 'fixture stdout before failure')
                self.assertEqual((output / data['failure']['stderr']).read_text(), 'fixture stderr before failure')
                self.assertTrue(data['sources_unchanged_at_stop'])
                self.assertTrue(data['binaries_unchanged_at_stop'])
            else:
                self.assertTrue(data['sources_unchanged'])
                self.assertTrue(data['binaries_unchanged'])
                self.assertTrue(data['order_balanced'])
                self.assertEqual(data['samples_discarded'], 0)
                self.assertEqual(len(data['runs']), 4)
                self.assertEqual(sum(len(r['samples']) for r in data['runs']), 8)
                self.assertEqual(data['cooldown_scope'], scope or 'pair')
                self.assertEqual(data['cooldown_seconds_before_pair'], 7 if scope is None else 0)
                self.assertEqual(data['cooldown_seconds_before_engine_run'], 7 if scope == 'run' else 0)
            return events

    def test_legacy_pair_pacing(self):
        self.assertEqual(self.exercise(), [('pause', 7), 'qw3', 'llama', ('pause', 7), 'llama', 'qw3'])

    def test_run_pacing_is_equal_for_both_engines(self):
        self.assertEqual(self.exercise('run'), [('pause', 7), 'qw3', ('pause', 7), 'llama',
                                               ('pause', 7), 'llama', ('pause', 7), 'qw3'])

    def test_failed_run_keeps_both_streams(self):
        self.assertEqual(self.exercise('run', fail=True), [('pause', 7), 'qw3'])


if __name__ == '__main__':
    unittest.main()
