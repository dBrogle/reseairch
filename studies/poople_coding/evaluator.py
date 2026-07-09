"""Run each generated program and grade its output against the BFS oracle.

Execution is sandboxed only lightly (temp cwd, minimal env, hard timeout) — this
runs model-generated code, which is the point of the benchmark, on the user's own
machine for authorized research. There is no network sandbox; the prompt forbids
network use and programs are saved to output/scripts/ for inspection.
"""

import json
import os
import random
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

from studies.poople.wordlist import load_words
from studies.poople.solver import build_solution_oracle, validate_ladder
from studies.poople_coding.config import (
    TARGET,
    TEST_CAP_PER_DISTANCE,
    TEST_SEED,
    TEST_UNREACHABLE,
    TIMEOUT_SECONDS,
    WORDS_FILE,
)
from studies.poople_coding.cache import safe_model


# ---------------------------------------------------------------------------
# Test battery + wordlist file
# ---------------------------------------------------------------------------

def write_wordlist(words: set[str]) -> Path:
    """Materialize the four-letter word list handed to each program."""
    WORDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(WORDS_FILE, "w") as f:
        f.write("\n".join(sorted(words)) + "\n")
    return WORDS_FILE


def build_test_words(oracle: dict, words: set[str]) -> list[tuple[str, int | None]]:
    """Stratified (word, par) battery: capped per distance + some unreachable.

    par is the optimal distance for reachable words, or None for unreachable
    words (whose correct answer is an empty ladder).
    """
    dist = oracle["dist"]
    by_dist: dict[int, list[str]] = {}
    for w, d in dist.items():
        by_dist.setdefault(d, []).append(w)

    rng = random.Random(TEST_SEED)
    test: list[tuple[str, int | None]] = []
    for d in sorted(by_dist):
        bucket = sorted(by_dist[d])
        rng.shuffle(bucket)
        for w in bucket[:TEST_CAP_PER_DISTANCE]:
            test.append((w, d))

    unreachable = sorted(words - set(dist))
    rng.shuffle(unreachable)
    for w in unreachable[:TEST_UNREACHABLE]:
        test.append((w, None))

    rng.shuffle(test)  # interleave so a partial/timeout run still spans difficulty
    return test


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run_program(code: str, wordlist_path: Path, test_words: list[str]) -> dict:
    """Execute a program; feed words on stdin, capture stdout lines. No grading."""
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "solver.py"
        script.write_text(code)
        stdin_text = "\n".join(test_words) + "\n"
        start = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, str(script), str(wordlist_path)],
                input=stdin_text, capture_output=True, text=True,
                timeout=TIMEOUT_SECONDS, cwd=tmp,
                env={"PATH": os.environ.get("PATH", "")},
            )
        except subprocess.TimeoutExpired as e:
            return {"ran": False, "timed_out": True, "returncode": None,
                    "elapsed": time.time() - start, "stdout": e.stdout or "",
                    "stderr": (e.stderr or "") + f"\n[timed out after {TIMEOUT_SECONDS}s]"}
        return {"ran": proc.returncode == 0, "timed_out": False,
                "returncode": proc.returncode, "elapsed": time.time() - start,
                "stdout": proc.stdout, "stderr": proc.stderr}


def _parse_output(stdout: str, n: int) -> list:
    """Parse stdout into up to n JSON arrays, aligned positionally to inputs."""
    out = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            val = json.loads(line)
        except json.JSONDecodeError:
            out.append(None)
            continue
        out.append([str(x).strip().lower() for x in val] if isinstance(val, list) else None)
    return out


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def grade(model: str, code: str | None, oracle: dict, words: set[str],
          test_words: list[tuple[str, int | None]], wordlist_path: Path) -> dict:
    dist = oracle["dist"]
    reachable = [(w, p) for w, p in test_words if p is not None]
    unreachable = [(w, p) for w, p in test_words if p is None]
    base = {
        "model": model, "short_name": model.split("/")[-1], "ran": False,
        "error": None, "elapsed_sec": None, "n_reachable": len(reachable),
        "n_unreachable": len(unreachable), "valid_count": 0, "optimal_count": 0,
        "unreachable_correct": 0, "optimal_rate": 0.0, "valid_rate": 0.0,
        # Per-reachable-word outcome split (same 4-way scheme as the poople study):
        # par = optimal, over_par = valid but longer, illegal = invalid ladder,
        # failed = no/empty answer (or program didn't run).
        "pie": {"par": 0, "over_par": 0, "illegal": 0, "failed": 0},
        "per_distance": {}, "examples": [],
    }
    if not code:
        base["error"] = "no program produced"
        base["pie"]["failed"] = len(reachable)
        return base

    inputs = [w for w, _ in test_words]
    run = run_program(code, wordlist_path, inputs)
    base["elapsed_sec"] = round(run["elapsed"], 2)
    if not run["ran"]:
        base["error"] = ("timed out" if run["timed_out"] else
                         f"exit {run['returncode']}: {(run['stderr'] or '').strip()[-300:]}")
        base["pie"]["failed"] = len(reachable)
        return base
    base["ran"] = True

    outputs = _parse_output(run["stdout"], len(inputs))
    by_dist_total: Counter = Counter()
    by_dist_opt: Counter = Counter()
    examples = []

    for i, (word, par) in enumerate(test_words):
        ladder = outputs[i] if i < len(outputs) else None
        if par is None:  # unreachable: correct answer is empty
            if ladder == []:
                base["unreachable_correct"] += 1
            continue
        by_dist_total[par] += 1
        valid = bool(ladder) and validate_ladder(ladder, words, TARGET, start=word)[0]
        optimal = valid and (len(ladder) - 1 == par)
        if optimal:
            base["pie"]["par"] += 1
        elif valid:
            base["pie"]["over_par"] += 1
        elif ladder:  # non-empty but illegal
            base["pie"]["illegal"] += 1
        else:         # None / empty for a reachable word
            base["pie"]["failed"] += 1
        if valid:
            base["valid_count"] += 1
        if optimal:
            base["optimal_count"] += 1
            by_dist_opt[par] += 1
        if len(examples) < 8 and (not optimal):
            examples.append({"word": word, "par": par, "output": ladder,
                             "valid": valid, "optimal": optimal})

    n = len(reachable) or 1
    base["optimal_rate"] = round(base["optimal_count"] / n * 100, 1)
    base["valid_rate"] = round(base["valid_count"] / n * 100, 1)
    base["per_distance"] = {
        str(d): {"n": by_dist_total[d], "optimal": by_dist_opt[d],
                 "optimal_rate": round(by_dist_opt[d] / by_dist_total[d] * 100, 1)}
        for d in sorted(by_dist_total)
    }
    base["examples"] = examples  # a few non-optimal cases for inspection
    return base


def evaluate_all(programs: dict[str, dict]) -> list[dict]:
    """Grade every model's program. `programs` is {model: cache-entry}."""
    words = load_words()
    oracle = build_solution_oracle(words, TARGET)
    wordlist_path = write_wordlist(words)
    test_words = build_test_words(oracle, words)
    print(f"  Test battery: {len([t for t in test_words if t[1] is not None])} reachable "
          f"+ {len([t for t in test_words if t[1] is None])} unreachable words")

    results = []
    for model, entry in programs.items():
        print(f"  Running {model.split('/')[-1]} ...", end=" ", flush=True)
        r = grade(model, entry.get("code"), oracle, words, test_words, wordlist_path)
        if not r["ran"]:
            print(f"DID NOT RUN ({r['error'][:60] if r['error'] else '?'})")
        else:
            print(f"optimal {r['optimal_rate']}%  valid {r['valid_rate']}%  "
                  f"({r['elapsed_sec']}s)")
        results.append(r)
    return results, {"n_reachable": len([t for t in test_words if t[1] is not None]),
                     "n_unreachable": len([t for t in test_words if t[1] is None])}
