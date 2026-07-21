"""Charts for the Poople LLM test — built to look good when shared.

Headline horizontal bars with brand icons (solve rate, illegal moves, over par),
a small-multiples outcome pie with each model's logo, a stacked outcome bar, and
difficulty heatmaps. Each call writes into one condition's graphs_<cond> dir.
"""

import os
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Patch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from utils.graphing import heatmap
from utils.model_icons import icon_path_for, themed_icon_path_for
from studies.poople.analysis import OUTCOMES, PIE_CATEGORIES

_ICON_REF_PX = 512.0

# 4-way pie palette: green = par, light green = solved-over-par, red = illegal,
# orange = failed to reach poop.
PIE_COLORS = {"par": "#2e9b57", "over_par": "#9ad26a", "illegal": "#d33d3d", "failed": "#e8902e"}
PIE_LABELS = {
    "par": "Par (optimal)", "over_par": "Solved, over par",
    "illegal": "Illegal move", "failed": "Failed to reach poop",
}

# Brand colors (not nationality) for a clean, on-brand look.
_BRAND = {
    "openai": "#10a37f", "anthropic": "#d97757", "gemini": "#4285f4", "google": "#4285f4",
    "grok": "#5b6470", "x-ai": "#5b6470", "kimi": "#7c5cff", "moonshot": "#7c5cff",
    "deepseek": "#4d6bfe",
}

BG = "#fbfbfc"


def _short(model: str) -> str:
    return model.split("/")[-1]


def model_color(model: str) -> str:
    m = model.lower()
    for key, col in _BRAND.items():
        if key in m:
            return col
    return "#4A90D9"


def _vendor_spread_colors(models: list[str]) -> dict[str, str]:
    """Brand color per model, lightened apart when several share one vendor.

    An all-OpenAI chart would otherwise be N identical green bars. Models keeping
    a vendor to themselves get the exact brand color, so mixed charts are
    unchanged.
    """
    import colorsys

    groups: dict[str, list[str]] = {}
    for m in models:
        groups.setdefault(model_color(m), []).append(m)

    out: dict[str, str] = {}
    for base, members in groups.items():
        if len(members) == 1:
            out[members[0]] = base
            continue
        r, g, b = matplotlib.colors.to_rgb(base)
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        # Spread lightness around the brand value, darkest first.
        lo, hi = max(0.18, l - 0.20), min(0.80, l + 0.26)
        for i, m in enumerate(sorted(members)):
            li = lo + (hi - lo) * (i / max(1, len(members) - 1))
            out[m] = matplotlib.colors.to_hex(colorsys.hls_to_rgb(h, li, s))
    return out


def _place_icon(ax, model, xy, zoom, xycoords="data", zorder=20, themed=False):
    p = themed_icon_path_for(model) if themed else icon_path_for(model)
    if p is None:
        return
    try:
        img = mpimg.imread(str(p))
    except Exception:
        return
    eff = zoom * (_ICON_REF_PX / img.shape[0])
    ab = AnnotationBbox(OffsetImage(img, zoom=eff), xy, xycoords=xycoords,
                        frameon=False, box_alignment=(0.5, 0.5))
    ab.set_zorder(zorder)
    ax.add_artist(ab)


# ---------------------------------------------------------------------------
# Headline horizontal bar (icon trailing each bar)
# ---------------------------------------------------------------------------

def _headline_barh(stats, value_fn, title, subtitle, xlabel, save_path,
                   value_fmt=".0f", suffix="", xmax=None, drop_none=False):
    items = [(s, value_fn(s)) for s in stats]
    if drop_none:
        items = [(s, v) for s, v in items if v is not None]
    items = [(s, (0.0 if v is None else v)) for s, v in items]
    items.sort(key=lambda t: t[1], reverse=True)
    if not items:
        return

    n = len(items)
    vmax = xmax if xmax is not None else max(v for _, v in items) or 1.0
    vmax = vmax * 1.0
    headroom = vmax * 1.22

    fig, ax = plt.subplots(figsize=(10.5, 1.02 * n + 2.4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    palette = _vendor_spread_colors([s["model"] for s, _ in items])
    for i, (s, v) in enumerate(items):
        y = n - 1 - i
        col = palette[s["model"]]
        ax.barh(y, v, height=0.62, color=col, edgecolor="white", linewidth=1.2, zorder=3)
        # value label + trailing brand icon (icon clear of the text)
        ax.text(v + headroom * 0.02, y, f"{v:{value_fmt}}{suffix}", ha="left", va="center",
                fontsize=14, fontweight="bold", color="#2b2f36", zorder=5)
        _place_icon(ax, s["model"], (v + headroom * 0.115, y), zoom=0.05)

    ax.set_yticks(range(n))
    ax.set_yticklabels([_short(s["model"]) for s, _ in items][::-1],
                       fontsize=12.5, fontweight="bold", color="#2b2f36")
    ax.set_xlim(0, headroom)
    if suffix == "%":
        ax.set_xticks(range(0, 101, 20))
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel(xlabel, fontsize=12, color="#5a606a", labelpad=8)
    ax.tick_params(axis="x", labelsize=10.5, colors="#9aa0a8", length=0)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_axisbelow(True)
    ax.grid(True, axis="x", color="#e6e9ed", linewidth=1.0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d2d6dc")

    fig.suptitle(title, fontsize=20, fontweight="bold", color="#15171a", x=0.5, y=1.0,
                 ha="center")
    if subtitle:
        ax.set_title(subtitle, fontsize=12, color="#8a909a", pad=12)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Outcome pie (one per model, with the model's logo)
# ---------------------------------------------------------------------------

def _pie_ordered(stats, pie_order):
    """Reorder pies so `pie_order` models come first, in that order.

    Sorting is stable, so anything unlisted keeps its existing relative position
    at the back. A substring match keeps this robust to the vendor prefix.
    """
    if not pie_order:
        return stats

    def rank(s):
        model = s["model"].lower()
        return next((i for i, key in enumerate(pie_order) if key in model), len(pie_order))

    return sorted(stats, key=rank)


def _outcome_pies(stats, save_path, subtitle, title="Poople — outcome mix by model",
                  solved_word="solved", pie_order=None):
    stats = _pie_ordered(stats, pie_order)
    n = len(stats)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.3 * cols, 4.8 * rows))
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(top=0.86, bottom=0.10, hspace=0.40, wspace=0.20)
    axes = np.atleast_1d(axes).ravel()

    for ax, s in zip(axes, stats):
        ax.set_facecolor(BG)
        pie = s["overall"]["pie"]
        values = [pie.get(c, 0) for c in PIE_CATEGORIES]
        if sum(values) == 0:
            ax.axis("off")
            continue

        def _autopct(pct):
            return f"{pct:.0f}%" if pct >= 8 else ""
        # Donut, so the brand logo can live in the center hole.
        ax.pie(values, colors=[PIE_COLORS[c] for c in PIE_CATEGORIES],
               startangle=90, counterclock=False, autopct=_autopct, pctdistance=0.80,
               radius=1.0, wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
               textprops=dict(color="white", fontsize=11, fontweight="bold"))
        ax.set_aspect("equal")
        _place_icon(ax, s["model"], (0, 0), zoom=0.11, xycoords="data", themed=True)
        ax.set_title(f"{_short(s['model'])}\n{s['overall']['solve_rate']:.0f}% {solved_word}",
                     fontsize=12.5, fontweight="bold", color="#15171a", pad=8)

    for ax in axes[n:]:
        ax.axis("off")

    handles = [Patch(facecolor=PIE_COLORS[c], label=PIE_LABELS[c]) for c in PIE_CATEGORIES]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=12, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(title + (f"\n{subtitle}" if subtitle else ""),
                 fontsize=17, fontweight="bold", color="#15171a", y=0.985)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def _stacked_outcomes(stats, save_path, subtitle):
    """Horizontal 100%-stacked bar: 6-way outcome composition per model."""
    colors = {
        "optimal": "#2e9b57", "suboptimal": "#9ad26a", "reached_illegal": "#f0a830",
        "failed": "#e0653a", "unparseable": "#b0453a", "error": "#8a8f98",
    }
    labels = {
        "optimal": "Optimal (+0)", "suboptimal": "Solved, over par",
        "reached_illegal": "Reached, illegal move", "failed": "Failed to reach poop",
        "unparseable": "Unparseable JSON", "error": "API error",
    }
    stats = sorted(stats, key=lambda s: -s["overall"]["outcomes"]["optimal"])
    n = len(stats)
    fig, ax = plt.subplots(figsize=(12, max(3.5, 1.0 * n + 1.6)))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    y = np.arange(n)
    for i, s in enumerate(stats):
        total = s["overall"]["n"] or 1
        left = 0.0
        for cat in OUTCOMES:
            pct = s["overall"]["outcomes"][cat] / total * 100
            if pct <= 0:
                continue
            ax.barh(y[i], pct, left=left, color=colors[cat], edgecolor="white", height=0.62)
            if pct >= 7:
                ax.text(left + pct / 2, y[i], f"{pct:.0f}", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
            left += pct
    ax.set_yticks(y)
    ax.set_yticklabels([_short(s["model"]) for s in stats], fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of attempts", fontsize=12)
    ax.set_title("Poople — outcome breakdown by model" + (f"\n{subtitle}" if subtitle else ""),
                 fontsize=14, fontweight="bold")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    handles = [Patch(facecolor=colors[c], label=labels[c]) for c in OUTCOMES]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=3, frameon=False, fontsize=10)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_graphs(stats: list[dict], save_dir: Path, subtitle: str = "",
                    pie_order: tuple[str, ...] | None = None):
    save_dir.mkdir(parents=True, exist_ok=True)

    _headline_barh(
        stats, lambda s: s["overall"]["solve_rate"],
        "Poople — legal solve rate", subtitle, "% of attempts solved legally",
        save_dir / "solve_rate.png", value_fmt=".0f", suffix="%", xmax=100,
    )
    _headline_barh(
        stats, lambda s: s["overall"]["illegal_per_attempt"],
        "Poople — illegal moves per attempt", subtitle + "  ·  lower is better",
        "mean illegal moves per attempt",
        save_dir / "illegal_per_attempt.png", value_fmt=".2f",
    )
    _headline_barh(
        stats, lambda s: s["overall"]["avg_over_par"],
        "Poople — steps over par (solved only)", subtitle + "  ·  lower is better",
        "mean (moves − optimal)", save_dir / "avg_over_par.png",
        value_fmt=".2f", suffix="", drop_none=True,
    )

    _outcome_pies(stats, save_dir / "outcomes_pie.png", subtitle, pie_order=pie_order)
    _stacked_outcomes(stats, save_dir / "outcomes_stacked.png", subtitle)

    # Difficulty heatmaps (model × par bucket).
    labels = [_short(s["model"]) for s in stats]
    buckets = sorted({b for s in stats for b in s["by_bucket"]})
    if buckets:
        col_labels = [f"par {b}" for b in buckets]
        solve_data = [[s["by_bucket"][b]["solve_rate"] for b in buckets] for s in stats]
        heatmap(solve_data, labels, col_labels,
                "Solve rate by difficulty (par = optimal steps)", "puzzle difficulty", "",
                save_dir / "heatmap_solve_rate.png", value_range=(0, 100), cmap="RdYlGn", fmt=".0f")

        op_data, op_annot = [], []
        for s in stats:
            row, arow = [], []
            for b in buckets:
                v = s["by_bucket"][b]["avg_over_par"]
                row.append(np.nan if v is None else v)
                arow.append("—" if v is None else f"{v:.2f}")
            op_data.append(row)
            op_annot.append(arow)
        heatmap(op_data, labels, col_labels,
                "Average steps over par by difficulty (lower is better)", "puzzle difficulty", "",
                save_dir / "heatmap_over_par.png", value_range=(0, 2), cmap="RdYlGn_r",
                annotations=op_annot)

    print(f"  Graphs saved to {save_dir}/")
