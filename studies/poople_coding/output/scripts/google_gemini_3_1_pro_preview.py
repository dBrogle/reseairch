import sys
import json
from collections import defaultdict, deque

def main():
    if len(sys.argv) < 2:
        return
    wordlist_path = sys.argv[1]
    
    words = set()
    try:
        with open(wordlist_path, 'r') as f:
            for line in f:
                w = line.strip()
                if len(w) == 4:
                    words.add(w)
    except Exception:
        pass

    patterns = defaultdict(list)
    for w in words:
        for i in range(4):
            pattern = w[:i] + '_' + w[i+1:]
            patterns[pattern].append(w)
            
    graph = defaultdict(list)
    for w in words:
        for i in range(4):
            pattern = w[:i] + '_' + w[i+1:]
            for neighbor in patterns[pattern]:
                if neighbor != w:
                    graph[w].append(neighbor)
                    
    target = "poop"
    parent = {}
    
    if target in words:
        queue = deque([target])
        parent[target] = None
        
        while queue:
            current = queue.popleft()
            for neighbor in graph[current]:
                if neighbor not in parent:
                    parent[neighbor] = current
                    queue.append(neighbor)
                    
    for line in sys.stdin:
        start = line.strip()
        if start == target:
            print('["poop"]')
        elif start not in words or start not in parent:
            print("[]")
        else:
            path = []
            curr = start
            while curr is not None:
                path.append(curr)
                curr = parent[curr]
            print(json.dumps(path))

if __name__ == '__main__':
    main()