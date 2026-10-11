#!/usr/bin/env python3
"""Validate the frozen sources and recorded results, optionally in git's index."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess
import unicodedata

parser = argparse.ArgumentParser()
parser.add_argument('--index', action='store_true')
parser.add_argument('--require-web-pass', action='store_true',
                    help='also require the natural Gadda request to pass its quality oracle')
args = parser.parse_args()
archive = Path(__file__).resolve().parent
root = archive.parents[2]
manifest = json.loads((archive / 'manifest.json').read_text())

def sha(data):
    return hashlib.sha256(data).hexdigest()

for name, expected in manifest['source_sha256'].items():
    path = (archive / manifest['source_snapshot_paths'][name]).resolve()
    assert sha(path.read_bytes()) == expected, ('snapshot', name)
    assert sha((root / name).read_bytes()) == expected, ('current source', name)
    if args.index:
        indexed = subprocess.check_output(['git', 'show', ':' + str(path.relative_to(root))], cwd=root)
        assert sha(indexed) == expected, ('indexed snapshot', name)
for name, expected in manifest['binary_sha256'].items():
    assert sha((root / name).read_bytes()) == expected, ('binary', name)
model = Path(manifest['model'])
assert model.stat().st_size == manifest['model_size']
assert model.stat().st_mtime_ns == manifest['model_mtime_ns']
assert manifest['source_unchanged'] and manifest['binary_unchanged'] and manifest['model_unchanged']
assert manifest['store_unchanged'] and manifest['all_passed']
assert len(manifest['runs']) == 24 and all(r['exit_code'] == 0 for r in manifest['runs'])
assert 'PASS visit_url complete_text=exact rows=500' in (archive / 'test-agent-web-page.stdout').read_text()
assert 'PASS long_tool_turn recent_request_and_result=retained' in (archive / 'test-agent-context.stdout').read_text()
assert 'user_goal=retained' in (archive / 'test-agent-context.stdout').read_text()
assert 'PASS more_history continuation=all_7_rows' in (archive / 'test-agent-repeat.stdout').read_text()

pressure = (archive / 'test-agent-context-pressure.stdout').read_text()
for marker in ('PASS tool_limit', 'PASS native_read', 'PASS sync phase=0',
               'PASS sync phase=1', 'PASS sync phase=2', 'PASS agent_prefill'):
    assert marker in pressure, marker
assert 'finite_logits=bit_exact canonical_state=exact history=exact' in pressure
native = re.search(r'old=12504 new=(\d+) appended=(\d+)', pressure)
assert native and 0 < int(native[2]) <= 2048 and int(native[1]) == 12504 + int(native[2])
assert 'PASS adaptive_read' in pressure and 'continuation=all_120_lines_contiguous' in pressure
pty = (archive / 'test-agent-prefill-interrupt.stdout').read_text()
assert 'PASS real_tui' in pty
latency = json.loads(pty[:pty.index('PASS real_tui')])
assert 0 <= latency['ordinary_interrupt_seconds'] < 10
assert 0 <= latency['manual_compaction_interrupt_seconds'] < 10
assert latency['canceled_checkpoints'] == 0

gadda = json.loads((archive / 'gadda-result.json').read_text())
assert gadda['exit_code'] == 0 and not gadda['timed_out']
assert gadda['source_unchanged'] and gadda['binary_unchanged'] and gadda['store_unchanged']
assert gadda['source_sha256'] == {n: manifest['source_sha256'][n] for n in gadda['source_sha256']}
assert gadda['binary_sha256'] == manifest['binary_sha256']
assert gadda['model_size'] == manifest['model_size']
assert gadda['model_mtime_ns'] == manifest['model_mtime_ns']
events = [json.loads(line) for line in (archive / 'gadda.trace.jsonl').read_text().splitlines()]
tools = [e['name'] for e in events if e['event'] == 'tool']
assert 'search_web' in tools and 'visit_url' in tools
assert any(t in tools for t in ('read', 'more'))
oracle = json.loads((archive / 'gadda-oracle.json').read_text())
assert gadda['oracle_titles'] == len(oracle['titles']) == 40
assert any(len(p['main_titles_present']) == 40 for p in gadda['downloaded_readable_pages'])
answer = (archive / 'gadda.stdout').read_text()
def norm(value):
    return re.sub(r'[^a-z0-9]+', ' ', unicodedata.normalize('NFKD', value)
                  .encode('ascii', 'ignore').decode().lower()).strip()

listed = []
for line in answer.splitlines():
    match = re.match(r'^\s*(?:[-*+]\s+|\d+[.)]\s+|\|)(.*)', line)
    if match:
        listed.append(re.sub(r'^(?:\d+\s+)+', '', norm(match.group(1))))
matched = set()
for line in listed:
    choices = [t for t in oracle['canonical_titles']
               if line == norm(t) or line.startswith(norm(t) + ' ')]
    if choices:
        matched.add(max(choices, key=lambda t: len(norm(t))))
missing = [t for t in oracle['canonical_titles'] if t not in matched]
assert missing == gadda['missing_titles']
assert sorted(matched) == gadda['matched_main_titles']
source_cited = re.sub(r'^https?://', '', oracle['url']) in answer
web_quality_passed = not missing and source_cited
if args.require_web_pass:
    assert web_quality_passed, ('Gadda quality oracle failed', missing)
limits = [e for e in events if e['event'] == 'tool_context_limit']
assert all(e['context_tokens'] <= e['token_limit'] <= 1024 for e in limits)
completed_compactions = [e for e in events if e['event'] == 'compaction' and e['result'] == 'completed']
assert all(e['tail_tokens'] > 0 for e in completed_compactions)
assert 'skipped repeated more tool call' not in (archive / 'gadda.stderr').read_text()
for entry in gadda['compaction_archives']:
    path = archive / entry['path']
    assert sha(path.read_bytes()) == entry['sha256']
    assert gadda['query'] in entry['user_requests'], ('active user task', entry['path'])
negative_refs = 0
for directory in ('gadda_round24_original', 'gadda_fixed20'):
    attempt = archive / directory
    recorded = json.loads((attempt / 'gadda-result.json').read_text())
    paths = json.loads((attempt / 'source_snapshot_paths.json').read_text())
    assert recorded['source_unchanged'] and recorded['binary_unchanged'] and recorded['store_unchanged']
    assert recorded['exit_code'] != 0
    for name, expected in recorded['source_sha256'].items():
        path = (archive / paths[name]).resolve()
        assert sha(path.read_bytes()) == expected, ('negative snapshot', directory, name)
        if args.index:
            indexed = subprocess.check_output(['git', 'show', ':' + str(path.relative_to(root))], cwd=root)
            assert sha(indexed) == expected, ('indexed negative snapshot', directory, name)
        negative_refs += 1
adaptive = archive / 'gadda_adaptive_v4'
old_manifest = json.loads((adaptive / 'manifest.json').read_text())
assert old_manifest['all_passed'] and len(old_manifest['runs']) == 23
assert json.loads((adaptive / 'gadda-result.json').read_text())['exit_code'] != 0
for name, expected in old_manifest['source_sha256'].items():
    path = (adaptive / old_manifest['source_snapshot_paths'][name]).resolve()
    assert sha(path.read_bytes()) == expected, ('adaptive v4 snapshot', name)
    if args.index:
        indexed = subprocess.check_output(['git', 'show', ':' + str(path.relative_to(root))], cwd=root)
        assert sha(indexed) == expected, ('indexed adaptive v4 snapshot', name)
    negative_refs += 1
previous_tail = archive / 'gadda_tail_v7'
previous_manifest = json.loads((previous_tail / 'manifest.json').read_text())
assert previous_manifest['all_passed'] and len(previous_manifest['runs']) == 24
assert json.loads((previous_tail / 'gadda-result.json').read_text())['missing_titles']
for name, expected in previous_manifest['source_sha256'].items():
    path = (previous_tail / previous_manifest['source_snapshot_paths'][name]).resolve()
    assert sha(path.read_bytes()) == expected, ('tail v7 snapshot', name)
    if args.index:
        indexed = subprocess.check_output(['git', 'show', ':' + str(path.relative_to(root))], cwd=root)
        assert sha(indexed) == expected, ('indexed tail v7 snapshot', name)
    negative_refs += 1
previous_parser = archive / 'gadda_parser_v5'
parser_manifest = json.loads((previous_parser / 'manifest.json').read_text())
assert parser_manifest['all_passed'] and len(parser_manifest['runs']) == 24
assert json.loads((previous_parser / 'gadda-result.json').read_text())['exit_code'] != 0
for name, expected in parser_manifest['source_sha256'].items():
    path = (previous_parser / parser_manifest['source_snapshot_paths'][name]).resolve()
    assert sha(path.read_bytes()) == expected, ('parser v5 snapshot', name)
    if args.index:
        indexed = subprocess.check_output(['git', 'show', ':' + str(path.relative_to(root))], cwd=root)
        assert sha(indexed) == expected, ('indexed parser v5 snapshot', name)
    negative_refs += 1
tail_zero = archive / 'gadda_tail0_v6'
tail_manifest = json.loads((tail_zero / 'manifest.json').read_text())
assert tail_manifest['all_passed'] and len(tail_manifest['runs']) == 24
assert json.loads((tail_zero / 'gadda-result.json').read_text())['exit_code'] != 0
for name, expected in tail_manifest['source_sha256'].items():
    path = (tail_zero / tail_manifest['source_snapshot_paths'][name]).resolve()
    assert sha(path.read_bytes()) == expected, ('tail0 v6 snapshot', name)
    if args.index:
        indexed = subprocess.check_output(['git', 'show', ':' + str(path.relative_to(root))], cwd=root)
        assert sha(indexed) == expected, ('indexed tail0 v6 snapshot', name)
    negative_refs += 1
result = {
    'suite_targets_passed': len(manifest['runs']),
    'frozen_source_references_verified': len(manifest['source_sha256']),
    'previous_attempt_source_references_verified': negative_refs,
    'index_verified': args.index,
    'source_and_binary_unchanged': True,
    'production_store_unchanged': True,
    'gadda_quality_passed': web_quality_passed,
    'gadda_oracle_main_category_titles': len(oracle['titles']),
    'gadda_main_category_titles_matched': len(matched),
    'gadda_missing_titles': missing,
    'gadda_source_cited': source_cited,
    'gadda_tools': tools,
    'gadda_limited_results': len(limits),
    'gadda_completed_compactions': completed_compactions,
    'active_user_request_verified_in_archives': len(gadda['compaction_archives']),
    'real_tui_interrupt_latency_seconds': latency,
    'scope': 'listed targets and one natural web request; not a new llama throughput comparison',
}
(archive / ('verification_index.json' if args.index else 'verification.json')).write_text(
    json.dumps(result, indent=2, ensure_ascii=False) + '\n')
print(json.dumps(result, indent=2, ensure_ascii=False))
