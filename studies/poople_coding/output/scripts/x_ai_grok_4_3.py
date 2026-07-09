import sys
import json
from collections import defaultdict, deque

def differ_by_one(a, b):
    return sum(x != y for x, y in zip(a, b)) == 1

def build_graph(words):
    word_set = set(words)
    graph = defaultdict(list)
    buckets = defaultdict(list)
    for word in words:
        for i in range(4):
            pattern = word[:i] + '*' + word[i+1:]
            buckets[pattern].append(word)
    for pattern, group in buckets.items():
        for i in range(len(group)):
            for j in range(i+1, len(group)):
                w1, w2 = group[i], group[j]
                if differ_by_one(w1, w2):
                    graph[w1].append(w2)
                    graph[w2].append(w1)
    return graph

def compute_parents(graph, target):
    parents = {}
    queue = deque([target])
    parents[target] = None
    while queue:
        current = queue.popleft()
        for neighbor in graph[current]:
            if neighbor not in parents:
                parents[neighbor] = current
                queue.append(neighbor)
    return parents

def reconstruct_path(start, parents, target):
    if start not in parents:
        return []
    path = []
    current = start
    while current is not None:
        path.append(current)
        if current == target:
            break
        current = parents[current]
    else:
        return []
    return path

def main():
    if len(sys.argv) != 2:
        sys.exit(1)
    wordlist_path = sys.argv[1]
    with open(wordlist_path, 'r') as f:
        words = [line.strip() for line in f if line.strip()]
    target = "poop"
    if target not in words:
        words.append(target)
    graph = build_graph(words)
    parents = compute_parents(graph, target)
    for line in sys.stdin:
        start = line.strip()
        if not start:
            print(json.dumps([]))
            continue
        if start == target:
            print(json.dumps([target]))
            continue
        path = reconstruct_path(start, parents, target)
        print(json.dumps(path))

if __name__ == "__main__":
    main()