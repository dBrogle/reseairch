import sys
import json
from collections import deque, defaultdict

def build_graph(word_list):
    words = set(word_list)
    pattern_to_words = defaultdict(list)
    for w in words:
        for i in range(4):
            pattern = w[:i] + '*' + w[i+1:]
            pattern_to_words[pattern].append(w)
    return words, pattern_to_words

def bfs(start, words, pattern_to_words):
    parent = {start: None}
    queue = deque([start])
    while queue:
        curr = queue.popleft()
        for i in range(4):
            pattern = curr[:i] + '*' + curr[i+1:]
            for neighbor in pattern_to_words[pattern]:
                if neighbor not in parent:
                    parent[neighbor] = curr
                    queue.append(neighbor)
    return parent

def main():
    if len(sys.argv) != 2:
        sys.exit(1)
    wordlist_path = sys.argv[1]
    try:
        with open(wordlist_path, 'r') as f:
            word_list = [line.strip() for line in f if line.strip()]
    except Exception:
        word_list = []

    words, pattern_to_words = build_graph(word_list)
    target = "poop"
    if target in words:
        parent = bfs(target, words, pattern_to_words)
    else:
        parent = {}

    for line in sys.stdin:
        start = line.strip()
        if not start or len(start) != 4 or start not in words:
            print("[]")
            continue
        if start not in parent:
            print("[]")
            continue
        path = []
        curr = start
        while curr is not None:
            path.append(curr)
            curr = parent[curr]
        print(json.dumps(path))

if __name__ == "__main__":
    main()