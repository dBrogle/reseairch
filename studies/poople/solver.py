"""The Poople solver.

Builds the word-ladder graph over all valid four-letter words (an edge joins
two words that differ in exactly one position), then BFS-es outward from the
TARGET word so we know, for every reachable word:

  * the minimum number of one-letter changes needed to reach TARGET, and
  * every optimal (minimum-length) ladder to TARGET.

The BFS distance map is the *oracle* the LLM half grades against: a model's
solution is "optimal" iff its length equals dist[start_word].

Counting all optimal paths is done exactly with a tiny DP; enumerating them is
done with a DAG walk that only follows edges that strictly decrease distance.
"""

from collections import defaultdict, deque

from studies.poople.config import TARGET


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_adjacency(words: set[str]) -> dict[str, set[str]]:
    """Map each word to the set of words differing by exactly one letter.

    Uses wildcard buckets ("c_ld" groups cold/card/...) so we never compare all
    O(n^2) pairs blindly: two distinct words sharing a wildcard pattern differ
    in exactly the masked position, which is exactly a Poople edge.
    """
    buckets: dict[str, list[str]] = defaultdict(list)
    for w in words:
        for i in range(len(w)):
            buckets[w[:i] + "_" + w[i + 1:]].append(w)

    adj: dict[str, set[str]] = {w: set() for w in words}
    for group in buckets.values():
        if len(group) < 2:
            continue
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                adj[group[a]].add(group[b])
                adj[group[b]].add(group[a])
    return adj


# ---------------------------------------------------------------------------
# BFS distances from the target
# ---------------------------------------------------------------------------

def bfs_distances(adj: dict[str, set[str]], target: str = TARGET) -> dict[str, int]:
    """Return {word: minimum steps to reach `target`} for every reachable word."""
    dist: dict[str, int] = {target: 0}
    queue: deque[str] = deque([target])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


# ---------------------------------------------------------------------------
# Optimal-path counting and enumeration
# ---------------------------------------------------------------------------

def count_optimal_paths(
    adj: dict[str, set[str]],
    dist: dict[str, int],
    target: str = TARGET,
) -> dict[str, int]:
    """Exact number of distinct minimum-length ladders from each word to target.

    DP over words in increasing distance order: a word's optimal-path count is
    the sum of counts of its neighbors that sit one step closer to the target.
    """
    order = sorted(dist, key=lambda w: dist[w])
    count: dict[str, int] = {}
    for w in order:
        if w == target:
            count[w] = 1
            continue
        d = dist[w]
        count[w] = sum(
            count[v] for v in adj[w] if dist.get(v) == d - 1
        )
    return count


def enumerate_optimal_paths(
    adj: dict[str, set[str]],
    dist: dict[str, int],
    start: str,
    target: str = TARGET,
    cap: int | None = None,
) -> list[list[str]]:
    """All optimal ladders from `start` to `target` (each a list of words).

    Only follows edges that strictly decrease the distance to target, so every
    path produced is guaranteed minimum-length. Stops once `cap` paths are
    collected (None = no cap). Paths are returned in deterministic (sorted)
    order so the saved artifact is reproducible.
    """
    if start not in dist:
        return []
    results: list[list[str]] = []
    path: list[str] = [start]

    def dfs(w: str) -> None:
        if cap is not None and len(results) >= cap:
            return
        if w == target:
            results.append(list(path))
            return
        d = dist[w]
        for v in sorted(adj[w]):
            if dist.get(v) == d - 1:
                path.append(v)
                dfs(v)
                path.pop()
                if cap is not None and len(results) >= cap:
                    return

    dfs(start)
    return results


# ---------------------------------------------------------------------------
# Ladder validation (used by the LLM-grading half, but defined here so the
# graph rules live in one place)
# ---------------------------------------------------------------------------

def one_letter_apart(a: str, b: str) -> bool:
    """True iff `a` and `b` are same-length and differ in exactly one position."""
    if len(a) != len(b):
        return False
    return sum(1 for x, y in zip(a, b) if x != y) == 1


def validate_ladder(
    ladder: list[str],
    words: set[str],
    target: str = TARGET,
    start: str | None = None,
) -> tuple[bool, str]:
    """Check a proposed Poople solution.

    Returns (is_valid, reason). A valid ladder:
      * is non-empty,
      * starts at `start` (if given),
      * ends at `target`,
      * contains only valid four-letter words, and
      * changes exactly one letter between each consecutive pair.
    """
    if not ladder:
        return False, "empty ladder"
    if start is not None and ladder[0] != start:
        return False, f"does not start at '{start}' (starts at '{ladder[0]}')"
    if ladder[-1] != target:
        return False, f"does not end at '{target}' (ends at '{ladder[-1]}')"
    for i, w in enumerate(ladder):
        if w not in words:
            return False, f"'{w}' (step {i}) is not a valid word"
    for i in range(len(ladder) - 1):
        if not one_letter_apart(ladder[i], ladder[i + 1]):
            return False, (
                f"'{ladder[i]}' -> '{ladder[i + 1]}' changes "
                f"{sum(1 for x, y in zip(ladder[i], ladder[i + 1]) if x != y)} "
                f"letters, not exactly 1"
            )
    return True, "ok"


# ---------------------------------------------------------------------------
# One-shot build
# ---------------------------------------------------------------------------

def build_solution_oracle(words: set[str], target: str = TARGET) -> dict:
    """Build the full oracle: adjacency, distances, and optimal-path counts.

    Returns a dict with the graph and derived maps so callers (main / grader)
    don't rebuild the graph repeatedly.
    """
    adj = build_adjacency(words)
    dist = bfs_distances(adj, target)
    counts = count_optimal_paths(adj, dist, target)
    return {"adj": adj, "dist": dist, "counts": counts, "target": target}
