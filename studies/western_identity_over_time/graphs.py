"""Analysis + visualization for the Western Identity Over Time study.

Turns the per-model judged caches into a timeline keyed by each model's real
OpenRouter release date, then renders five charts (mirrors of the Chinese Identity
Over Time study, pointed the other way):

  - ``timeline_western_identity.png`` — a line per maker: Western-identity rate vs.
    release date, with an onset band.
  - ``timeline_western_identity_flags.png`` — the same line chart with the round
    USA / China flag emblems marking the y-axis poles (USA at the top, China at the
    bottom) and bigger brand icons at each maker's peak.
  - ``onset_swimlane.png`` — one lane per maker, one dot per model shaded by its
    Western-identity rate.
  - ``flag_swimlane_timeline.png`` — the same lanes, each model a circle filled with
    the China / USA flags in proportion to its identity split, at its release date.
  - ``flag_lineage_strips.png`` — one row per maker, that maker's models as large
    flag circles left→right in release order, captioned with name / date / % American.
  - ``timeline_western_identity_flags_kimi_glm.png`` — the flag timeline for just the
    STABLE_MODELS (Moonshot/Kimi, Zhipu/GLM) on a full 0-100 axis: two flat lines
    along the bottom, the empty American zone above showing they never claimed it.
"""

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
from studies.western_identity_over_time.config import (
    MODELS,
    STABLE_MODELS,
    ALL_MODELS,
    ITERATIONS,
    IDENTITY_PROMPT,
    ONSET_THRESHOLD,
    get_seed_convo,
)
from studies.western_identity_over_time.cache import (
    GRAPHS_DIR,
    load_cache,
    get_results,
)
from studies.western_identity_over_time import catalog


# Distinct per-maker line colors (loosely brand-derived) so the timeline reads
# clearly — the nationality red/blue scheme can't separate five Chinese makers.
MAKER_COLORS = {
    "DeepSeek": "#4d6bfe",
    "Qwen": "#7c3aed",
    "Moonshot": "#ec4899",
    "MiniMax": "#f59e0b",
    "Zhipu": "#10b981",
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


def build_series(all_results: dict[str, list[dict]],
                 models: dict[str, list[str]] = MODELS) -> dict[str, list[dict]]:
    """{maker: [{"date", "value", "china", "west", "name", "model_id"}]} by date.

    ``value`` is the % claiming a Western identity (the metric this study tracks).
    Only models belonging to a maker in ``models`` are included; pass MODELS for the
    main charts or STABLE_MODELS for the "never wavered" highlight. Models without a
    known release date or without any valid responses are skipped.
    """
    m2m = {mid: maker for maker, mids in models.items() for mid in mids}
    series: dict[str, list[dict]] = {}
    for model_id, results in all_results.items():
        maker = m2m.get(model_id)
        if maker is None:
            continue
        date = catalog.release_date(model_id)
        if date is None:
            continue
        if not any(r.get("error") is None for r in results):
            continue
        china_pct, west_pct = nationality_shares(results)
        series.setdefault(maker, []).append({
            "date": date,
            "value": compute_rate(results, "judge_western"),
            "china": china_pct,
            "west": west_pct,
            "name": short(model_id),
            "model_id": model_id,
        })
    for maker in series:
        series[maker].sort(key=lambda p: p["date"])
    # Preserve the maker order declared in config.
    return {m: series[m] for m in models if m in series}


def generate_graphs(all_results: dict[str, list[dict]]):
    """Render the timeline, swimlane, and flag charts."""
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    series = build_series(all_results)
    if not series:
        print("No datable results to graph yet.")
        return

    maker_icons = {m: icon_path_for(m) for m in series}
    china_flag, usa_flag = ensure_flags()
    n_iters = max((len(r) for r in all_results.values()), default=0)
    sub = (f"asked “{IDENTITY_PROMPT}” · no system prompt · model default temp · "
           f"n={n_iters} per model · dates from OpenRouter")
    y_label = "% claiming an American identity"

    identity_timeline(
        series=series,
        title="When did Chinese models start saying they were American?",
        subtitle=sub,
        save_path=GRAPHS_DIR / "timeline_western_identity.png",
        y_label=y_label,
        maker_colors=MAKER_COLORS,
        maker_icons=maker_icons,
        threshold=ONSET_THRESHOLD,
        annotate_onset=False,  # the swimlane names the onset models; keep this one clean
    )

    # Flag-y-axis version: USA flag at the top pole, China at the bottom, bigger
    # brand icons. Unlike the forward study, the interesting behavior here is at the
    # *start* of the timeline (the oldest models claimed Western most), so we show
    # the full range rather than zooming into the recent window.
    identity_timeline(
        series=series,
        title="When did Chinese models think they were American?",
        subtitle=None,
        save_path=GRAPHS_DIR / "timeline_western_identity_flags.png",
        y_label=y_label,
        maker_colors=MAKER_COLORS,
        maker_icons=maker_icons,
        threshold=ONSET_THRESHOLD,
        annotate_onset=False,
        icon_zoom=0.072,
        icon_at_peak=True,        # spikes aren't the latest model — label each peak
        flag_top=usa_flag,        # up = claims American
        flag_bottom=china_flag,   # down = stays Chinese / true
    )

    onset_swimlane(
        lanes=series,
        title="Onset of American self-identification, by maker over time",
        subtitle=sub,
        save_path=GRAPHS_DIR / "onset_swimlane.png",
        maker_icons=maker_icons,
        threshold=ONSET_THRESHOLD,
        value_label=y_label,
    )

    # Flag-meter charts: each model is a China/USA flag-filled circle over time.
    # Here the "interesting" share is Western (claiming American), so highlight=west.
    flag_swimlane_timeline(
        lanes=series,
        title="Chinese vs. American identity of Chinese models, over time",
        subtitle=sub,
        save_path=GRAPHS_DIR / "flag_swimlane_timeline.png",
        china_flag=china_flag,
        usa_flag=usa_flag,
        maker_icons=maker_icons,
        threshold=ONSET_THRESHOLD,
        highlight="west",
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
        highlight="west",
    )

    # Highlight chart for the makers that stayed ~0% the whole time (Kimi, GLM): the
    # same flag timeline, with a full 0-100 axis so the empty American zone above
    # their flat lines reads as "they never claimed otherwise".
    stable = build_series(all_results, STABLE_MODELS)
    if stable:
        identity_timeline(
            series=stable,
            title="Kimi and GLM never thought they were American",
            subtitle=None,
            save_path=GRAPHS_DIR / "timeline_western_identity_flags_kimi_glm.png",
            y_label=y_label,
            maker_colors=MAKER_COLORS,
            maker_icons={m: icon_path_for(m) for m in stable},
            threshold=ONSET_THRESHOLD,
            annotate_onset=False,
            icon_zoom=0.072,
            icon_at_peak=True,
            flag_top=usa_flag,        # up = claims American
            flag_bottom=china_flag,   # down = stays Chinese / true
            y_top=100.0,              # full axis so the empty American zone reads
        )

    print(f"\nGraphs saved to {GRAPHS_DIR}/")


def export_frontend_json(all_results: dict[str, list[dict]]):
    """Export a structured JSON timeline for the frontend web app."""
    series = build_series(all_results)
    graph_files = sorted(p.name for p in GRAPHS_DIR.glob("*.png")) if GRAPHS_DIR.exists() else []

    output = {
        "_meta": {
            "study": "western_identity_over_time",
            "description": (
                "Mirror of the Chinese Identity Over Time study: asks a chronological "
                "lineage of each Chinese maker's models 'What model are you?' (no system "
                "prompt, model default temp) and measures how often each claims to be a "
                "Western model, plotted against real release dates from OpenRouter."
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
                "western_perc": f"{p['value']:.1f}%",
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
    print("  WESTERN IDENTITY OVER TIME — % of responses claiming an American identity")
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
            print(f"    {p['date'].isoformat()}  {p['name']:<26} "
                  f"{p['value']:>5.1f}%  {top_str}{mark}")
        if onset_name:
            print(f"    first crossing {ONSET_THRESHOLD:.0f}%: {onset_name}")
        else:
            print(f"    never crossed {ONSET_THRESHOLD:.0f}%")
    print()
