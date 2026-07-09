"""Export the Poople Coding results as a self-contained hand-off file.

Writes output/results_export.json (exhaustive: methodology, the exact prompt/IO
contract, per-model grades incl. the generated program source, failure examples,
findings) and output/RESULTS.md (readable digest).
"""

import datetime
import json
from pathlib import Path

from studies.poople_coding.config import (
    MAX_TOKENS,
    MODELS,
    OUTPUT_DIR,
    PROMPT_VERSION,
    TARGET,
    TEMPERATURE,
    TEST_CAP_PER_DISTANCE,
    TEST_UNREACHABLE,
    TIMEOUT_SECONDS,
    WORD_LEN,
)
from studies.poople_coding.cache import load_code, safe_model
from studies.poople_coding.prompt import build_prompt

STUDY_DIR = Path(__file__).parent
EXPORT_JSON = STUDY_DIR / OUTPUT_DIR / "results_export.json"
EXPORT_MD = STUDY_DIR / OUTPUT_DIR / "RESULTS.md"


def build_export(results: list[dict], battery: dict) -> dict:
    ranked = sorted(results, key=lambda r: (-r["optimal_rate"], -r["valid_rate"]))
    models_out = []
    for r in ranked:
        entry = load_code(r["model"]) or {}
        models_out.append({
            **r,
            "program_path": f"scripts/{safe_model(r['model'])}.py",
            "program_code": entry.get("code"),
            "generation_error": entry.get("error"),
        })

    return {
        "meta": {
            "study": "poople_coding",
            "title": "Poople Coding — can LLMs write code that solves Poople optimally?",
            "generated": datetime.date.today().isoformat(),
            "description": (
                "A coding benchmark companion to the 'poople' study. Each reasoning model "
                "gets ONE shot to write a complete Python program that, given any four-letter "
                "start word, outputs the optimal one-letter-at-a-time ladder to 'poop'. We run "
                "each program against a distance-stratified battery of words and grade its "
                "output for legality and optimality using the same BFS oracle as the poople study."
            ),
            "task_summary": (
                "Model writes a stdlib-only Python 3 program run as `python prog.py "
                "<wordlist_path>`; it reads start words on stdin and prints, per line, a JSON "
                "array of the optimal ladder to 'poop' ([] if unreachable)."
            ),
            "one_shot": True,
            "models": [m.split("/")[-1] for m in MODELS],
            "reasoning": "ON (reasoning models only)",
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "prompt_version": PROMPT_VERSION,
            "full_prompt": build_prompt(),
            "target_word": TARGET,
            "word_length": WORD_LEN,
            "execution": {
                "language": "Python 3 (stdlib only)",
                "invocation": "python <program> <wordlist_path>  (start words on stdin)",
                "timeout_seconds": TIMEOUT_SECONDS,
                "sandbox": "subprocess in a temp cwd with a minimal env (PATH only) and a "
                           "hard timeout; no network sandbox (prompt forbids network).",
            },
            "test_battery": {
                "description": "Stratified by optimal distance: up to "
                               f"{TEST_CAP_PER_DISTANCE} words per distance (all words in the "
                               f"rare hard tiers) plus {TEST_UNREACHABLE} unreachable words "
                               "(correct answer = []).",
                "n_reachable": battery["n_reachable"],
                "n_unreachable": battery["n_unreachable"],
            },
        },
        "metric_definitions": {
            "ran": "Program executed to completion (exit 0) within the timeout.",
            "valid_rate": "Of reachable test words, % for which the program output a legal "
                          "ladder ending at 'poop' (one-letter steps, all valid words).",
            "optimal_rate": "Of reachable test words, % for which the legal ladder also had "
                            "the minimum possible length (== par). This is the headline metric.",
            "unreachable_correct": "Count of unreachable test words for which the program "
                                   "correctly returned an empty ladder [].",
        },
        "results": models_out,
        "key_findings": _findings(ranked),
        "caveats": [
            "Each model writes ONE program with no chance to test/revise; a single bug "
            "(e.g. a bad edge rule or off-by-one) can tank an otherwise sound approach.",
            "A correct breadth-first search over the provided word list scores 100% optimal; "
            "this benchmark mostly separates 'got the algorithm right' from 'didn't'.",
            "Programs that don't run (crash/timeout) score 0% regardless of approach.",
            "Reasoning models only — Gemini Pro is mandatory-reasoning; Kimi is excluded "
            "(unreliable with reasoning on), matching the poople study's reasoning set.",
        ],
        "assets": {
            "note": "Paths relative to studies/poople_coding/output/.",
            "generated_programs": "scripts/<model>.py",
            "graphs": ["optimal_rate.png", "valid_rate.png", "outcomes_pie.png",
                       "heatmap_optimal_by_distance.png"],
        },
    }


def _findings(ranked: list[dict]) -> list[str]:
    out = []
    ran = [r for r in ranked if r["ran"]]
    perfect = [r["short_name"] for r in ran if r["optimal_rate"] == 100.0]
    didnt = [r["short_name"] for r in ranked if not r["ran"]]
    if perfect:
        out.append(f"{len(perfect)} model(s) wrote a fully-correct optimal solver "
                   f"(100% optimal): {', '.join(perfect)}.")
    if ran:
        best = ran[0]
        out.append(f"Best: {best['short_name']} at {best['optimal_rate']}% optimal "
                   f"({best['valid_rate']}% valid) in {best['elapsed_sec']}s.")
    if didnt:
        out.append(f"{len(didnt)} model program(s) failed to run at all: {', '.join(didnt)}.")
    out.append("Programs that solve any word optimally generally solve ALL words optimally "
               "(a correct BFS), so scores cluster near 0% or 100% — the interesting signal "
               "is whether the one-shot program was correct and ran.")
    return out


def _md(d: dict) -> str:
    m = d["meta"]
    L = [f"# {m['title']}\n", f"_Generated {m['generated']}._\n", m["description"] + "\n"]
    L.append("## Task\n")
    L.append(m["task_summary"] + "\n")
    L.append(f"- One shot, reasoning ON; models: {', '.join(m['models'])}")
    L.append(f"- Temperature {m['temperature']}, timeout {m['execution']['timeout_seconds']}s")
    L.append(f"- Test battery: {m['test_battery']['n_reachable']} reachable + "
             f"{m['test_battery']['n_unreachable']} unreachable words "
             f"(stratified by difficulty)\n")

    L.append("## Results\n")
    L.append("| Model | Ran? | Optimal % | Valid % | Unreach. ok | Time (s) | Note |")
    L.append("|---|---|---|---|---|---|---|")
    for r in d["results"]:
        ran = "✅" if r["ran"] else "❌"
        note = "" if r["ran"] else (r["error"] or "")[:50]
        unreach = f"{r['unreachable_correct']}/{r['n_unreachable']}"
        L.append(f"| {r['short_name']} | {ran} | {r['optimal_rate']}% | {r['valid_rate']}% | "
                 f"{unreach} | {r['elapsed_sec']} | {note} |")
    L.append("")

    L.append("## Key findings\n")
    for f in d["key_findings"]:
        L.append(f"- {f}")
    L.append("")
    L.append("## Caveats\n")
    for c in d["caveats"]:
        L.append(f"- {c}")
    L.append("")

    # Show a couple of failure examples for color.
    L.append("## Example failures (non-optimal outputs)\n")
    for r in d["results"]:
        if r["ran"] and r["examples"]:
            L.append(f"**{r['short_name']}**:")
            for ex in r["examples"][:3]:
                out = ex["output"]
                shown = " → ".join(out) if isinstance(out, list) and out else str(out)
                L.append(f"- `{ex['word']}` (par {ex['par']}): {shown} "
                         f"(valid={ex['valid']}, optimal={ex['optimal']})")
            L.append("")

    L.append("## Assets\n")
    L.append("Generated programs: `scripts/<model>.py`. Graphs: `optimal_rate.png`, "
             "`valid_rate.png`, `heatmap_optimal_by_distance.png`. Full data + program "
             "source: `results_export.json`.")
    return "\n".join(L)


def export_all(results: list[dict], battery: dict):
    data = build_export(results, battery)
    EXPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(EXPORT_JSON, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    with open(EXPORT_MD, "w") as f:
        f.write(_md(data))
    print(f"  Wrote {EXPORT_JSON}")
    print(f"  Wrote {EXPORT_MD}")
    return data
