import sys
import json
from collections import deque, defaultdict


def main():
    if len(sys.argv) < 2:
        return
    wordlist_path = sys.argv[1]

    words = set()
    try:
        with open(wordlist_path, 'r') as f:
            for line in f:
                w = line.strip()
                if len(w) == 4 and w.islower() and w.isalpha():
                    words.add(w)
    except Exception:
        return

    target = "poop"

    # Build adjacency using pattern buckets
    buckets = defaultdict(list)
    for w in words:
        for i in range(4):
            pat = w[:i] + '*' + w[i+1:]
            buckets[pat].append(w)

    adj = defaultdict(list)
    for pat, group in buckets.items():
        if len(group) > 1:
            for w in group:
                for v in group:
                    if v != w:
                        adj[w].append(v)

    # Deduplicate adjacency
    for w in list(adj.keys()):
        adj[w] = list(set(adj[w]))

    # BFS from target if it exists in dictionary
    parent = {}
    dist = {}
    if target in words:
        dist[target] = 0
        parent[target] = None
        dq = deque([target])
        while dq:
            cur = dq.popleft()
            for nxt in adj[cur]:
                if nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    parent[nxt] = cur
                    dq.append(nxt)

    out_lines = []
    data = sys.stdin.read().split('\n')
    for line in data:
        start = line.strip()
        # Skip handling: still must output 1:1? Spec says lines correspond.
        # We'll treat blank/malformed as no path -> []
        if len(start) != 4 or not start.islower() or not start.isalpha():
            # malformed line: only output if it was a real line (non-empty after strip)
            # To keep 1:1 with input lines, only process non-final empty.
            if line == '' :
                # likely trailing empty from final newline; skip
                continue
            out_lines.append("[]")
            continue

        if start == target:
            out_lines.append(json.dumps([target]))
            continue

        if start not in words or start not in dist:
            out_lines.append("[]")
            continue

        # reconstruct path from start to target
        path = []
        cur = start
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        out_lines.append(json.dumps(path))

    sys.stdout.write('\n'.join(out_lines))
    if out_lines:
        sys.stdout.write('\n')


if __name__ == "__main__":
    main()