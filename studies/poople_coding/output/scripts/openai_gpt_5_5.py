import sys
import json
from collections import deque

TARGET = "poop"
ALPHABET = "abcdefghijklmnopqrstuvwxyz"

def load_words(path):
    words = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip()
            if len(w) == 4 and w.islower():
                words.add(w)
    return words

def build_next_steps(words):
    next_step = {}
    if TARGET not in words:
        return next_step

    next_step[TARGET] = None
    q = deque([TARGET])

    while q:
        word = q.popleft()
        chars = list(word)

        for i in range(4):
            original = chars[i]
            for c in ALPHABET:
                if c == original:
                    continue
                chars[i] = c
                neighbor = "".join(chars)
                if neighbor in words and neighbor not in next_step:
                    next_step[neighbor] = word
                    q.append(neighbor)
            chars[i] = original

    return next_step

def ladder_for(start, next_step):
    if start == TARGET:
        return [TARGET]
    if start not in next_step:
        return []

    ladder = []
    word = start
    while word is not None:
        ladder.append(word)
        if word == TARGET:
            break
        word = next_step.get(word)

    if ladder and ladder[-1] == TARGET:
        return ladder
    return []

def main():
    if len(sys.argv) < 2:
        return

    words = load_words(sys.argv[1])
    next_step = build_next_steps(words)
    cache = {}

    out_lines = []
    for line in sys.stdin:
        start = line.strip()
        if start not in cache:
            cache[start] = json.dumps(ladder_for(start, next_step))
        out_lines.append(cache[start])

    sys.stdout.write("\n".join(out_lines))
    if out_lines:
        sys.stdout.write("\n")

if __name__ == "__main__":
    main()