# Agent repetition guards: formatting regression, 2026-10-03

The reported 35B answer stopped while printing a Unicode box border after tool
calls. The text guard treated the repeated border bytes as repeated prose: a
line of about 32 three-byte border characters could reach the stop threshold.
The independent token guard could also stop decorative runs.

Both guards now allow repeated periods wholly contained in a bounded non-word
formatting suffix. The classifier uses Unicode categories P*, S*, Z*, M* and
Cf, plus tab/CR/LF, rather than a list of particular border characters. The
checked-in table records Unicode 13.0.0 and can be regenerated using
`python3 tools/generate_agent_unicode.py` (the generator uses the Python
runtime's Unicode database). Combining marks and joiners allow emoji sequences;
letters and numbers do not receive the exemption. UTF-8 can be split across
decoded tokens, so the suffix check handles an unfinished final code point
and rejects invalid completed sequences.

The allowance ends at 256 code points on a line or 1024 in a contiguous
formatting suffix. Existing repeated-word, token-ngram and paragraph thresholds
are preserved. This remains a structural repetition heuristic: it does not
attempt to judge the meaning of arbitrary prose. Repeated tool-call protection
is unchanged.

## Validation

- Before the change, the new streamed 80-character Unicode border regression
  failed with repeat-guard case 8.
- Metal and CPU agent builds succeeded. The model-free repeat selftest passed
  on both binaries, covering borders, twelve types of decorations, byte-by-byte
  UTF-8 streaming, excessive symbol runs, multiple decorative rows, repeated
  prose, accented/composed letters, Chinese letters and digits. Existing token
  and repeated-tool checks also passed.
- Tool parse, verification, context ledger, color, bash, trace and status
  regressions passed.
- A live Apple M5 run with `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`, Metal, F16 KV,
  context 8192, greedy, no thinking and `--no-prefix-cache` read an isolated
  README fixture. Its final assistant answer reproduced both 80-character
  Unicode borders and a 120-character `=` separator exactly, then printed
  `AGENT_DIAGRAM_OK`. No repetition guard fired. The model's initial malformed
  tool calls were recovered by the existing retry mechanism before `read`.
- A second process resumed the saved conversation's KV checkpoint and answered
  `SESSION_FORMAT_OK` successfully.

The live test checks the persisted **assistant** answer, so the copy of the
diagram in the tool result cannot satisfy the regression. It did not inspect
the user's railway project. Raw output and the synthetic fixture are under
[`measurements/agent_repeat_format_2026-10-03`](measurements/agent_repeat_format_2026-10-03).

Reproduce with:

```sh
make qw3-agent qw3-agent-cpu
make test-agent-repeat test-agent-tool-parse test-agent-verify test-agent-context test-agent-color
QW3_AGENT_REPEAT_GUARD_SELFTEST=1 ./qw3-agent-cpu
make test-agent-bash test-agent-trace test-agent-status
QW3_REPEAT_ARTIFACT_DIR=docs/measurements/agent_repeat_format_2026-10-03 make test-agent-repeat-format
```
