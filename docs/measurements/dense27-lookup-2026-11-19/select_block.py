"""Choose the first fully matched 6-key, 7-draft trial in the first C response."""
import json
from pathlib import Path
root = Path(__file__).resolve().parent
turn = json.loads((root/'coding.jsonl').read_text().splitlines()[0])
prompt, output = turn['prompt'], turn['generated']
i = 0
while i < len(output):
    history = prompt + output[:i+1]
    key = history[-6:]
    pos = next((j for j in range(len(history)-14,-1,-1) if history[j:j+6] == key),None)
    accepted = 0
    if pos is not None:
        for draft, actual in zip(history[pos+6:pos+13], output[i+1:]):
            if draft != actual:
                break
            accepted += 1
    if accepted == 7:
        print(len(prompt), i, *prompt, *output[:i+8])
        break
    i += 1 + accepted
else:
    raise SystemExit('No matched block')
