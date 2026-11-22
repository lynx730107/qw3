"""Replay a fixed-width lookup policy, always choosing with past tokens only."""
import json
import sys
from pathlib import Path
from collections import Counter

root = Path(__file__).resolve().parent

def input_spans(prompt):
    # The captured tokenizer renders user messages and tool responses as ChatML user.
    user_start = [248045, 846, 198]
    spans = []
    for i in range(len(prompt) - len(user_start)):
        if prompt[i:i+3] == user_start:
            end = prompt.index(248046, i+3)
            spans.append((i+3,end))
    return spans

def replay(prompt, output, key, width, inputs_only=False):
    seen = {}
    history = list(prompt)
    spans = input_spans(prompt) if inputs_only else []
    indexed = 0
    position = 0
    counts = Counter(tokens=len(output), turns=1)
    while position < len(output):
        # The current token has already been sampled by the target.
        history.append(output[position])
        # Index only spans whose entire draft predates the sampled token.
        while indexed + key + width <= len(history) - 1:
            if not inputs_only or any(left <= indexed and indexed+key+width <= right for left,right in spans):
                seen[tuple(history[indexed:indexed + key])] = indexed + key
            indexed += 1
        start = seen.get(tuple(history[-key:]))
        accepted = 0
        if start is not None:
            counts['attempts'] += 1
            for candidate, actual in zip(history[start:start+width], output[position+1:]):
                if candidate != actual:
                    break
                accepted += 1
            counts['accepted'] += accepted
            counts['perfect'] += accepted == width
            counts[f'matched_{accepted}'] += 1
        history.extend(output[position+1:position+1+accepted])
        position += 1 + accepted
    return counts

def main():
    inputs_only = '--inputs-only' in sys.argv[1:]
    report = {}
    for path in sorted(root.glob('*.jsonl')):
        name = path.stem
        turns = [json.loads(line) for line in path.read_text().splitlines()]
        # The git smoke test invokes cold, warm and disabled cache on the same task.
        if name == 'git':
            turns = turns[:next((i for i in range(1,len(turns)) if len(turns[i]['prompt']) <= len(turns[0]['prompt'])),len(turns))]
        for width in (1, 3, 7):
            for key in (2, 3, 4, 6, 8, 12, 16):
                counts = Counter()
                for turn in turns:
                    counts.update(replay(turn['prompt'], turn['generated'], key, width, inputs_only))
                report.setdefault(name,{}).setdefault(width,{})[key] = dict(counts)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
