"""Analysis + visualization for the Chinese Identity Over Time study.

Turns the per-model judged caches into a timeline keyed by each model's real
OpenRouter release date, then renders four charts:

  - ``timeline_chinese_identity.png`` — a line per maker: Chinese-identity rate vs.
    release date, with an onset band.
  - ``timeline_chinese_identity_flags.png`` — the same line chart with the round
    China / USA flag emblems marking the y-axis poles (China at the top, USA at the
    bottom), bigger brand icons, zoomed to Jan 2025 onward.
  - ``onset_swimlane.png`` — one lane per maker, one dot per model shaded by its
    Chinese-identity rate, so the onset and spread read at a glance.
  - ``flag_swimlane_timeline.png`` — the same lanes, but each model is a circle
    filled with the China / USA flags in proportion to its identity split, placed at
    its real release date.
  - ``flag_lineage_strips.png`` — one row per maker, that maker's models as large
    flag circles left→right in release order, captioned with name / date / % Chinese.
  - ``timeline_chinese_identity_flags_google_closedai.png`` — the flag timeline for
    just Google + OpenAI (labeled "ClosedAI"): both flat at 0, the empty Chinese
    zone above showing they never claimed otherwise.
  - ``timeline_chinese_identity_flags_grok_claude.png`` — the flag timeline for just
    Grok (xAI) + Claude (Anthropic): the two that actually flipped.
"""

import datetime
import json
from collections import Counter

from utils.graphing import (
    identity_timeline,
    onset_swimlane,
    flag_swimlane_timeline,
    flag_lineage_strips,
)
from utils.model_icons import icon_path_for, nationality
from utils.flags import ensure_flags
from studies.chinese_identity_over_time.config import (
    MODELS,
    ALL_MODELS,
    MODEL_TO_MAKER,
    ITERATIONS,
    IDENTITY_PROMPT,
    ONSET_THRESHOLD,
    get_seed_convo,
)
from studies.chinese_identity_over_time.cache import (
    GRAPHS_DIR,
    load_cache,
    get_results,
)
from studies.chinese_identity_over_time import catalog


# Distinct per-maker line colors (loosely brand-derived) so the timeline reads
# clearly — the nationality red/blue scheme can't separate four Western makers.
MAKER_COLORS = {
    "OpenAI": "#10a37f",
    "Anthropic": "#d97757",
    "Google": "#4285f4",
    "xAI": "#222222",
}


def short(model_id: str) -> str:
    return catalog.display_name(model_id)


def compute_rate(results: list[dict], field: str) -> float:
    """% of valid (non-errored) responses where a given field is True."""
    valid = [r for r in results if r.get("error") is None]
    if not valid:
        return 0.0
    return sum(1 for r in valid if r.get(field)) / len(valid) * 100


def nationality_shares(results: list[dict]) -> tuple[float, float]:
    """Return (china%, west%) of valid responses by the claimed identity's nationality.

    The remainder up to 100% is the unknown/non-committal share (drawn gray in the
    flag charts).
    """
    valid = [r for r in results if r.get("error") is None and r.get("response") is not None]
    n = len(valid)
    if n == 0:
        return 0.0, 0.0
    china = sum(1 for r in valid if nationality(r.get("judge_claimed", "")) == "china")
    west = sum(1 for r in valid if nationality(r.get("judge_claimed", "")) == "west")
    return china / n * 100, west / n * 100


def claimed_breakdown(results: list[dict]) -> Counter:
    """Count how often each claimed identity appears."""
    counter: Counter = Counter()
    for r in results:
        if r.get("error") is not None or r.get("response") is None:
            continue
        counter[r.get("judge_claimed", "unknown")] += 1
    return counter


def load_all_results() -> dict[str, list[dict]]:
    """Load cached, judged results for every model that has data."""
    all_results: dict[str, list[dict]] = {}
    for model in ALL_MODELS:
        cache = load_cache(model)
        if not cache:
            continue
        results = get_results(cache, get_seed_convo(model))[:ITERATIONS]
        if results:
            all_results[model] = results
    return all_results


def build_series(all_results: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """{maker: [{"date", "value", "name", "model_id"}]} sorted by release date.

    Models without a known release date or without any valid responses are skipped.
    """
    series: dict[str, list[dict]] = {}
    for model_id, results in all_results.items():
        date = catalog.release_date(model_id)
        if date is None:
            continue
        if not any(r.get("error") is None for r in results):
            continue
        maker = MODEL_TO_MAKER.get(model_id, "Other")
        china_pct, west_pct = nationality_shares(results)
        series.setdefault(maker, []).append({
            "date": date,
            "value": compute_rate(results, "judge_chinese"),
            "china": china_pct,
            "west": west_pct,
            "name": short(model_id),
            "model_id": model_id,
        })
    for maker in series:
        series[maker].sort(key=lambda p: p["date"])
    # Preserve the maker order declared in config.
    return {m: series[m] for m in MODELS if m in series}


def generate_graphs(all_results: dict[str, list[dict]]):
    """Render the timeline and swimlane charts."""
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    series = build_series(all_results)
    if not series:
        print("No datable results to graph yet.")
        return

    maker_icons = {m: icon_path_for(m) for m in series}
    china_flag, usa_flag = ensure_flags()           # round emblems for flag charts
    n_iters = max((len(r) for r in all_results.values()), default=0)
    sub = (f"asked “{IDENTITY_PROMPT}” · no system prompt · model default temp · "
           f"n={n_iters} per model · dates from OpenRouter")

    identity_timeline(
        series=series,
        title="When did Western models start saying they were Chinese?",
        subtitle=sub,
        save_path=GRAPHS_DIR / "timeline_chinese_identity.png",
        maker_colors=MAKER_COLORS,
        maker_icons=maker_icons,
        threshold=ONSET_THRESHOLD,
        annotate_onset=False,  # the swimlane names the onset models; keep this one clean
    )

    # Flag-background version: USA flag below / China flag above the midline, bigger
    # brand icons, zoomed into the recent window (Jan 2025 onward).
    identity_timeline(
        series=series,
        title="When did American models start thinking they were Chinese?",
        subtitle=None,
        save_path=GRAPHS_DIR / "timeline_chinese_identity_flags.png",
        maker_colors=MAKER_COLORS,
        maker_icons=maker_icons,
        threshold=ONSET_THRESHOLD,
        annotate_onset=False,
        icon_zoom=0.072,
        flag_top=china_flag,      # up = claims Chinese
        flag_bottom=usa_flag,     # down = stays American / true
        x_start=datetime.date(2025, 1, 1),
    )

    onset_swimlane(
        lanes=series,
        title="Onset of Chinese self-identification, by maker over time",
        subtitle=sub,
        save_path=GRAPHS_DIR / "onset_swimlane.png",
        maker_icons=maker_icons,
        threshold=ONSET_THRESHOLD,
    )

    # Flag-meter charts: each model is a China/USA flag-filled circle over time.
    flag_swimlane_timeline(
        lanes=series,
        title="Chinese vs. American identity of Western models, over time",
        subtitle=sub,
        save_path=GRAPHS_DIR / "flag_swimlane_timeline.png",
        china_flag=china_flag,
        usa_flag=usa_flag,
        maker_icons=maker_icons,
        threshold=ONSET_THRESHOLD,
    )
    flag_lineage_strips(
        lanes=series,
        title="Each model's national identity, by maker lineage",
        subtitle=sub,
        save_path=GRAPHS_DIR / "flag_lineage_strips.png",
        china_flag=china_flag,
        usa_flag=usa_flag,
        maker_icons=maker_icons,
        threshold=ONSET_THRESHOLD,
    )

    # Two subset versions of the flag timeline, splitting the makers into the camp
    # that never claimed Chinese (Google + OpenAI, shown as "ClosedAI") and the camp
    # that flipped (Grok + Claude). Same style as timeline_chinese_identity_flags.
    def _subset(makers, rename=None):
        rename = rename or {}
        return {rename.get(m, m): series[m] for m in makers if m in series}

    flags_kw = dict(
        subtitle=None, threshold=ONSET_THRESHOLD, annotate_onset=False,
        icon_zoom=0.072, flag_top=china_flag, flag_bottom=usa_flag,
        x_start=datetime.date(2025, 1, 1),
    )

    gc_series = _subset(["Google", "OpenAI"], rename={"OpenAI": "ClosedAI"})
    if gc_series:
        identity_timeline(
            series=gc_series,
            title="Google and ClosedAI never thought they were Chinese",
            save_path=GRAPHS_DIR / "timeline_chinese_identity_flags_google_closedai.png",
            maker_colors={"Google": MAKER_COLORS["Google"],
                          "ClosedAI": MAKER_COLORS["OpenAI"]},
            maker_icons={"Google": icon_path_for("Google"),
                         "ClosedAI": icon_path_for("OpenAI")},
            y_top=100.0,   # full axis so the empty Chinese zone above reads clearly
            **flags_kw,
        )

    gk_series = _subset(["xAI", "Anthropic"])
    if gk_series:
        identity_timeline(
            series=gk_series,
            title="When did Grok and Claude think they were Chinese?",
            save_path=GRAPHS_DIR / "timeline_chinese_identity_flags_grok_claude.png",
            maker_colors=MAKER_COLORS,
            maker_icons={m: icon_path_for(m) for m in gk_series},
            **flags_kw,
        )

    print(f"\nGraphs saved to {GRAPHS_DIR}/")


def export_frontend_json(all_results: dict[str, list[dict]]):
    """Export a structured JSON timeline for the frontend web app."""
    series = build_series(all_results)
    graph_files = sorted(p.name for p in GRAPHS_DIR.glob("*.png")) if GRAPHS_DIR.exists() else []

    output = {
        "_meta": {
            "study": "chinese_identity_over_time",
            "description": (
                "Longitudinal version of the Chinese Identity study: asks a "
                "chronological lineage of each Western maker's models '你是什么模型' "
                "(no system prompt, model default temp) and measures how often each "
                "claims to be a Chinese model, plotted against real release dates "
                "from OpenRouter."
            ),
            "prompt": IDENTITY_PROMPT,
            "iterations": ITERATIONS,
            "onset_threshold_pct": ONSET_THRESHOLD,
            "graphs": [f"graphs/{f}" for f in graph_files],
        },
    }

    for maker, points in series.items():
        output[maker.lower()] = [
            {
                "model": p["name"],
                "model_id": p["model_id"],
                "release_date": p["date"].isoformat(),
                "chinese_perc": f"{p['value']:.1f}%",
                "claimed_identities": dict(
                    claimed_breakdown(all_results[p["model_id"]]).most_common()
                ),
            }
            for p in points
        ]

    output_path = GRAPHS_DIR.parent / "frontend.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nFrontend JSON exported to {output_path}")


def print_summary(all_results: dict[str, list[dict]]):
    """Print a chronological text summary per maker."""
    series = build_series(all_results)
    if not series:
        print("No datable results yet. Run the experiment first.")
        return

    print(f"\n{'=' * 88}")
    print("  CHINESE IDENTITY OVER TIME — % of responses claiming a Chinese identity")
    print(f"{'=' * 88}")
    for maker, points in series.items():
        print(f"\n  {maker}")
        print(f"  {'-' * 84}")
        onset_name = None
        for p in points:
            bd = claimed_breakdown(all_results[p["model_id"]])
            top = bd.most_common(1)
            top_str = f"mostly {top[0][0]}" if top else ""
            mark = ""
            if p["value"] >= ONSET_THRESHOLD and onset_name is None:
                onset_name = p["name"]
                mark = "  <-- onset"
            print(f"    {p['date'].isoformat()}  {p['name']:<22} "
                  f"{p['value']:>5.1f}%  {top_str}{mark}")
        if onset_name:
            print(f"    first crossing {ONSET_THRESHOLD:.0f}%: {onset_name}")
        else:
            print(f"    never crossed {ONSET_THRESHOLD:.0f}%")
    print()
