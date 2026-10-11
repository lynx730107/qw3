#!/usr/bin/env python3
"""Exercise full-page storage and parsing through the real visit_url tool."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import subprocess
import tempfile
import threading

parser = argparse.ArgumentParser()
parser.add_argument('--agent', type=Path, default=Path(__file__).resolve().parents[1] / 'qw3-agent')
args = parser.parse_args()
agent = args.agent.resolve()
rows = [f'Title {i:03d} — year {1900+i} & notes' for i in range(500)]
body = ('<html><head><style>HIDDEN_STYLE</style></head><body>'
        '<nav>HIDDEN_NAV<script>HIDDEN_SCRIPT</script></nav>'
        '<h2>Works &amp; publications</h2><ul>' + ''.join(
            '<li data-meta=\'{"payload":"literal > and < attribute"}\'>'
            + row.replace('&', '&amp;') + '</li>' for row in rows)
        + '</ul><footer>HIDDEN_FOOTER</footer></body></html>').encode()
expected = ('Works & publications\n' + ''.join('- ' + row + '\n' for row in rows)).encode()
json_body = json.dumps({'value': 'literal <inside> & text', 'count': 500}).encode()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        payload = body
        media = 'text/html; charset=utf-8'
        if self.path == '/json':
            payload, media = json_body, 'application/json'
        elif self.path == '/oversize':
            payload = b'x' * 2000001
            media = 'text/plain'
        elif self.path == '/missing':
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', media)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
    def log_message(self, *_):
        pass

server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
try:
    with tempfile.TemporaryDirectory(prefix='qw3-web-page-test-') as temp:
        store = Path(temp) / 'store'
        store.mkdir()
        def visit(route='/', target=store):
            url = f'http://127.0.0.1:{server.server_port}{route}'
            tool = '<tool_call>' + json.dumps({'name': 'visit_url', 'arguments': {'url': url}}) + '</tool_call>'
            run = subprocess.run([str(agent), '--store-dir', str(target), '--tool-native', tool],
                                 capture_output=True, text=True, timeout=45)
            assert run.returncode == 0, run.stderr
            return run.stdout
        def saved_path(output):
            match = re.search(r'Full page text saved to (.*?)\. Search', output)
            assert match, output[:300]
            return Path(match[1])
        output = visit()
        path = saved_path(output)
        assert path.read_bytes() == expected
        assert path.stat().st_mode & 0o777 == 0o600
        assert '[truncated]' in output and 'Title 499' not in output
        assert 'HIDDEN_' not in output and 'attribute' not in path.read_text()
        for _ in range(9):
            path = saved_path(visit())
        assert len(list((store / 'spills').glob('web-*'))) == 8
        assert path.read_bytes() == expected
        output = visit('/json')
        assert saved_path(output).read_bytes() == json_body + b'\n'
        assert 'literal <inside> & text' in output
        output = visit('/oversize')
        assert 'only a prefix' in output and 'Full page text saved to ' not in output
        assert len(list((store / 'spills').glob('web-*'))) == 8
        output = visit('/missing')
        assert 'visit_url failed' in output and 'Full page text saved to ' not in output
        assert len(list((store / 'spills').glob('web-*'))) == 8
        blocked = Path(temp) / 'blocked-store'
        blocked.mkdir()
        (blocked / 'spills').write_text('not a directory')
        output = visit(target=blocked)
        assert 'Full page text could not be saved' in output
        assert 'Full page text saved to ' not in output
        print('PASS visit_url complete_text=exact rows=500 attributes=excluded hidden_content=excluded '
              'UTF8=valid mode=0600 retention=8 JSON=preserved fetch_cap=honest failures=explicit no_model=1')
finally:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
