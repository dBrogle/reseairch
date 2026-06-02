"""Scenario-specific summary charts.

Most scenarios are well-served by the generic per-arm and delta charts
in `visualize.py`. When a scenario benefits from a bespoke layout (e.g.,
a clean shareable summary for social), register a custom chart here.

`CUSTOM_CHARTS` maps `scenario_id` → list of `(filename_suffix, fn)`.
Each `fn` takes `(scenario, all_stats, save_path)` and writes a PNG.
`main.py::generate_graphs` runs generic charts for every scenario, then
runs any registered custom charts. Add a bespoke graph by writing one
function and adding one dict entry — scenarios stay pure data.

`SKIP_GENERIC_DELTA` lists scenarios whose generic delta chart (every
treatment vs the control arm) is misleading and should be suppressed in
favour of a custom delta. The three anchoring "sticker" scenarios use a
no-price control purely as a visual reference; their meaningful delta is
high-anchor vs low-anchor (both transact at the same sale price), which
the generic treatment-vs-control chart can't express.
"""

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox

from studies.cognitive_biases.analysis import _welch_diff_ci
from studies.cognitive_biases.scenarios.base import Scenario
from studies.cognitive_biases.visualize import (
    _ensure_dir,
    _format_value,
    _logo_offset_image,
    _mathsafe,
    _short,
)

CustomChart = Callable[[Scenario, list[dict], Path], None]


# ---------------------------------------------------------------------------
# Anchoring "sticker" scenarios (used car / diamond ring / luxury watch)
#
# All three share an identical 2-arm shape:
#   low_anchor  — listed at the sale price
#   high_anchor — "originally listed at <high> but on sale for <sale>"
# Both arms transact at the same sale price, so the high-vs-low gap
# isolates the pull of the extra high sticker.
# ---------------------------------------------------------------------------

_ANCHOR_ARM_COLORS: dict[str, str] = {
    "low_anchor":  "#C0392B",  # red   — plain list price (the low number)
    "high_anchor": "#27AE60",  # green — high sticker, same sale price
}

# Per-scenario sale price (where the two anchored arms transact) and the
# high sticker, used for the reference line and subtitle.
_ANCHOR_META: dict[str, dict] = {
    "anchoring_used_car":    {"sale": 19000, "high": 29000, "noun": "car"},
    "anchoring_diamond_ring": {"sale": 5200, "high": 9500,  "noun": "ring"},
    "anchoring_luxury_watch": {"sale": 4900, "high": 7000,  "noun": "watch"},
}

# Scenarios whose generic (treatment-vs-control) delta chart is suppressed.
SKIP_GENERIC_DELTA: set[str] = set(_ANCHOR_META.keys())

_POSITIVE_COLOR = "#27AE60"  # high anchor pulled UP vs low anchor (green)
_NEGATIVE_COLOR = "#C0392B"  # high anchor pulled DOWN vs low anchor (red)


def _anchor_summary(
    scenario: Scenario,
    all_stats: list[dict],
    save_path: Path,
) -> None:
    """Wide landscape dumbbell: per model, one price dot per arm stacked
    vertically and connected by a line, with the model's logo below the
    x-axis. Sized for video overlay (~2:1).

    Arms are plotted in scenario order using `_ANCHOR_ARM_COLORS`
    (no_price grey, low_anchor red, high_anchor green).
    """
    meta = _ANCHOR_META.get(scenario.id, {})
    sale = meta.get("sale")
    high = meta.get("high")

    plot_arms = [a for a in scenario.arms if a.key in _ANCHOR_ARM_COLORS]
    if not plot_arms:
        return

    # rows: (model, {arm_key: mean}) — keep only models with every arm.
    rows: list[tuple[str, dict[str, float]]] = []
    for stats in all_stats:
        per_arm = stats["per_arm"]
        means = {}
        ok = True
        for arm in plot_arms:
            s = per_arm.get(arm.key, {})
            if s.get("n", 0) == 0:
                ok = False
                break
            means[arm.key] = s["mean"]
        if ok:
            rows.append((stats["model"], means))

    if not rows:
        print(f"  [skip] {scenario.id} summary chart: no usable data")
        return

    n = len(rows)
    fig_w = max(14.0, 2.3 * n + 2.0)
    fig_h = 7.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    x_positions = np.arange(n)

    # Connecting line spanning each model's arm means — establishes the
    # "shift" visual without overpowering the dots themselves.
    for x, (_m, means) in zip(x_positions, rows):
        vals = list(means.values())
        ax.plot([x, x], [min(vals), max(vals)],
                color="#B0B0B0", linewidth=1.8, zorder=1)

    # Dots, one per arm.
    for x, (_m, means) in zip(x_positions, rows):
        for arm in plot_arms:
            ax.plot(x, means[arm.key], "o",
                    color=_ANCHOR_ARM_COLORS[arm.key],
                    markersize=18, zorder=3)

    # Price labels next to each dot. Sort by value and alternate the
    # vertical anchor so adjacent labels are less likely to collide.
    # Escape `$` so matplotlib doesn't read it as a math-mode delimiter.
    for x, (_m, means) in zip(x_positions, rows):
        ordered = sorted(plot_arms, key=lambda a: means[a.key])
        for i, arm in enumerate(ordered):
            ax.text(x + 0.18, means[arm.key], f"\\${means[arm.key]:,.0f}",
                    va="bottom" if i == len(ordered) - 1 else "top"
                    if i == 0 else "center",
                    ha="left", fontsize=11, fontweight="bold",
                    color=_ANCHOR_ARM_COLORS[arm.key])

    # Logos below the x-axis as the tick "labels".
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["" for _ in rows])
    ax.tick_params(axis="x", length=0)

    _LOGO_TARGET_PX = 75
    for x, (model, *_rest) in zip(x_positions, rows):
        oi = _logo_offset_image(model, _LOGO_TARGET_PX)
        if oi is None:
            ax.annotate(
                _short(model),
                xy=(x, 0), xycoords=("data", "axes fraction"),
                xytext=(0, -28), textcoords="offset points",
                ha="center", va="top", fontsize=10, color="#444444",
            )
            continue
        ab = AnnotationBbox(
            oi,
            xy=(x, 0), xycoords=("data", "axes fraction"),
            xybox=(0, -52), boxcoords="offset points",
            frameon=False, pad=0.0,
            box_alignment=(0.5, 0.5),
        )
        ax.add_artist(ab)

    # Y-axis range with breathing room top/bottom for labels.
    all_prices = [v for _, means in rows for v in means.values()]
    y_lo, y_hi = min(all_prices), max(all_prices)
    span = max(y_hi - y_lo, 1.0)
    ax.set_ylim(y_lo - 0.25 * span, y_hi + 0.20 * span)
    ax.set_xlim(-0.7, n - 0.3)

    # Reference line at the actual sale price both anchored arms see.
    if sale is not None:
        ax.axhline(sale, color="#444444", linewidth=0.9,
                   linestyle="--", alpha=0.55, zorder=0)
        ax.text(
            n - 0.4, sale, f"  actual sale price (\\${sale:,.0f})",
            fontsize=9, color="#555555", va="bottom", ha="right",
        )

    subtitle = (
        f"Same {meta.get('noun', 'item')}. Low/high arms both transact at "
        f"\\${sale:,.0f}; the anchor adds \"was \\${high:,.0f}.\""
        if sale is not None and high is not None else ""
    )
    ax.set_title(
        f"{scenario.title} — anchoring shift\n{subtitle}",
        fontsize=15, fontweight="bold", pad=14,
    )
    ax.set_ylabel("Mean fair price (USD)", fontsize=12)
    ax.yaxis.set_major_formatter(lambda v, _pos: f"\\${v:,.0f}")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=_ANCHOR_ARM_COLORS[arm.key], markersize=12,
                   label=_mathsafe(arm.label))
        for arm in plot_arms
    ]
    ax.legend(handles=legend_handles, loc="upper left",
              fontsize=11, framealpha=0.95)

    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    _ensure_dir(save_path)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)  # leave room for logos below x-axis
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _anchor_hi_lo_delta(
    scenario: Scenario,
    all_stats: list[dict],
    save_path: Path,
) -> None:
    """Per-model delta chart: high_anchor mean − low_anchor mean.

    Both arms transact at the same sale price, so the gap is the pull of
    the high sticker alone. One row per model: dot + 95% Welch CI, model
    logo as the y-tick. Positive (anchor pulls up) in green, negative in
    red. Mirrors `visualize.generate_delta_chart` but computes the diff
    between two treatment arms rather than treatment-minus-control.
    """
    meta = _ANCHOR_META.get(scenario.id, {})
    sale = meta.get("sale")

    rows = []  # (model, diff, lo, hi)
    for stats in all_stats:
        per_arm = stats["per_arm"]
        low = per_arm.get("low_anchor", {})
        high = per_arm.get("high_anchor", {})
        low_vals = low.get("values", [])
        high_vals = high.get("values", [])
        ci = _welch_diff_ci(high_vals, low_vals)
        if ci is None:
            continue
        diff, lo, hi = ci
        rows.append((stats["model"], diff, lo, hi))

    if not rows:
        print(f"  [skip] {scenario.id} hi-lo delta chart: no usable data")
        return

    fig_h = max(4.5, 0.85 * len(rows) + 1.8)
    fig, ax = plt.subplots(figsize=(11, fig_h))

    y_positions = np.arange(len(rows))[::-1]
    for y, (model, diff, lo, hi) in zip(y_positions, rows):
        color = _POSITIVE_COLOR if diff >= 0 else _NEGATIVE_COLOR
        ax.errorbar(
            diff, y,
            xerr=[[diff - lo], [hi - diff]],
            fmt="o", color=color, ecolor=color,
            elinewidth=1.8, capsize=5, markersize=9, zorder=3,
        )
        ax.text(
            hi, y,
            f"  Δ {_format_value(diff, scenario.value_unit)}  "
            f"[{_format_value(lo, scenario.value_unit)}, "
            f"{_format_value(hi, scenario.value_unit)}]",
            ha="left", va="center", fontsize=9, color="#333333",
        )

    ax.axvline(0, color="black", linewidth=0.8)

    # Logos as y-axis tick "labels" (same treatment as the generic delta).
    ax.set_yticks(y_positions)
    ax.set_yticklabels(["" for _ in rows])
    ax.tick_params(axis="y", length=0)

    _LOGO_TARGET_PX = 52
    for y, (model, *_rest) in zip(y_positions, rows):
        oi = _logo_offset_image(model, _LOGO_TARGET_PX)
        if oi is None:
            ax.annotate(
                _short(model),
                xy=(0, y), xycoords=("axes fraction", "data"),
                xytext=(-8, 0), textcoords="offset points",
                ha="right", va="center", fontsize=9, color="#444444",
            )
            continue
        ab = AnnotationBbox(
            oi,
            xy=(0, y), xycoords=("axes fraction", "data"),
            xybox=(-32, 0), boxcoords="offset points",
            frameon=False, pad=0.0, box_alignment=(0.5, 0.5),
        )
        ax.add_artist(ab)

    ax.set_xlabel(
        f"High anchor − low anchor ({scenario.value_unit})  ·  "
        "error bars = 95% Welch CI",
        fontsize=10,
    )
    sale_txt = (
        f"both transact at {_mathsafe(f'${sale:,.0f}')}"
        if sale is not None else "both transact at the same sale price"
    )
    ax.set_title(
        f"{scenario.title}\n[{scenario.bias_type}]  ·  "
        f"high anchor − low anchor  ({sale_txt})",
        fontsize=13, fontweight="bold", pad=10,
    )
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    xs = [r[2] for r in rows] + [r[3] for r in rows] + [r[1] for r in rows]
    lo, hi = min(xs + [0]), max(xs + [0])
    span = hi - lo if hi > lo else max(abs(hi), 1.0)
    ax.set_xlim(lo - 0.1 * span, hi + 0.55 * span)

    _ensure_dir(save_path)
    fig.tight_layout()
    fig.subplots_adjust(left=0.14)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ANCHOR_CHARTS: list[tuple[str, CustomChart]] = [
    ("delta", _anchor_hi_lo_delta),
    ("summary", _anchor_summary),
]

CUSTOM_CHARTS: dict[str, list[tuple[str, CustomChart]]] = {
    sid: _ANCHOR_CHARTS for sid in _ANCHOR_META
}
