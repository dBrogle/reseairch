"""Visualization for the Question Framing study.

Generates:
1. Per-model bar chart showing accuracy by framing (with 95% CIs)
2. Grouped bar chart comparing all models across framings (with 95% CIs)
3. Heatmap of framing x model accuracy
4. Delta chart showing accuracy change vs. control (with 95% CIs)

All charts accept the rich stats format from `compute_framing_accuracy()`:
    {framing_key: {accuracy, ci_low, ci_high, n_correct, n_total}} or None
"""

import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from studies.question_framing.config import FRAMING_KEYS, FRAMING_SUFFIXES


# Nice display names for framings
def _framing_label(key: str) -> str:
    suffix = FRAMING_SUFFIXES[key]
    if not suffix:
        return "Control (no framing)"
    if len(suffix) > 50:
        return suffix[:47] + "..."
    return suffix


def _unpack(stats: dict | None) -> tuple[float, float, float] | None:
    """Return (accuracy, ci_low, ci_high) or None if no data."""
    if stats is None:
        return None
    return (stats["accuracy"], stats["ci_low"], stats["ci_high"])


def generate_model_bar_chart(
    framing_accuracy: dict[str, dict | None],
    title: str,
    save_path: str | Path,
):
    """Horizontal bar chart of accuracy per framing for a single model, with 95% CIs."""
    valid = {k: v for k, v in framing_accuracy.items() if v is not None}
    if not valid:
        return

    # Sort by accuracy ascending (so highest is at the top)
    sorted_items = sorted(valid.items(), key=lambda x: x[1]["accuracy"])
    labels = [_framing_label(k) for k, _ in sorted_items]
    values = [v["accuracy"] for _, v in sorted_items]
    ci_lows = [v["ci_low"] for _, v in sorted_items]
    ci_highs = [v["ci_high"] for _, v in sorted_items]
    n_totals = [v["n_total"] for _, v in sorted_items]

    # Asymmetric error bars: distance from the point
    err_low = [max(0, v - lo) for v, lo in zip(values, ci_lows)]
    err_high = [max(0, hi - v) for v, hi in zip(values, ci_highs)]

    # Color by relative performance
    cmap = matplotlib.colormaps["RdYlGn"]
    if max(values) > min(values):
        normed = [(v - min(values)) / (max(values) - min(values)) for v in values]
    else:
        normed = [0.5] * len(values)
    colors = [cmap(n) for n in normed]

    fig, ax = plt.subplots(figsize=(12, max(6, len(labels) * 0.55)))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.4)

    # Error bars (95% Wilson CI)
    ax.errorbar(
        values,
        range(len(values)),
        xerr=[err_low, err_high],
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=3,
        alpha=0.75,
    )

    for bar, val, n in zip(bars, values, n_totals):
        ax.text(
            max(val, 0) + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.1%} (n={n})", va="center", ha="left", fontsize=9,
        )

    # Control line
    control_stats = framing_accuracy.get("control")
    if control_stats is not None:
        control_val = control_stats["accuracy"]
        ax.axvline(x=control_val, color="gray", linestyle="--", linewidth=1.2,
                   alpha=0.7, label=f"Control: {control_val:.1%}")
        ax.legend(fontsize=9)

    ax.set_xlim(0, min(1.15, max(ci_highs) * 1.25) if ci_highs else 1.0)
    ax.set_xlabel("Accuracy (error bars = 95% Wilson CI)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, axis="x", alpha=0.3)

    import os
    os.makedirs(str(Path(save_path).parent), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_aggregated_framing_chart(
    aggregated: dict[str, dict | None],
    title: str,
    save_path: str | Path,
):
    """Horizontal bar chart of accuracy per framing, pooled across all models.

    Each bar is one framing; error bars are 95% Wilson CIs on the pooled
    (model x question) trials. Sorted ascending so the best framing sits on top.
    """
    valid = {k: v for k, v in aggregated.items() if v is not None}
    if not valid:
        return

    sorted_items = sorted(valid.items(), key=lambda x: x[1]["accuracy"])
    labels   = [_framing_label(k) for k, _ in sorted_items]
    values   = [v["accuracy"] for _, v in sorted_items]
    ci_lows  = [v["ci_low"]   for _, v in sorted_items]
    ci_highs = [v["ci_high"]  for _, v in sorted_items]
    n_totals = [v["n_total"]  for _, v in sorted_items]
    n_models = [v.get("n_models", 0) for _, v in sorted_items]

    err_low  = [max(0, v - lo) for v, lo in zip(values, ci_lows)]
    err_high = [max(0, hi - v) for v, hi in zip(values, ci_highs)]

    cmap = matplotlib.colormaps["RdYlGn"]
    if max(values) > min(values):
        normed = [(v - min(values)) / (max(values) - min(values)) for v in values]
    else:
        normed = [0.5] * len(values)
    colors = [cmap(n) for n in normed]

    fig, ax = plt.subplots(figsize=(13, max(6, len(labels) * 0.6)))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.4)

    ax.errorbar(
        values,
        range(len(values)),
        xerr=[err_low, err_high],
        fmt="none",
        ecolor="black",
        elinewidth=1.4,
        capsize=4,
        alpha=0.8,
    )

    for bar, val, n, m in zip(bars, values, n_totals, n_models):
        ax.text(
            max(val, 0) + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}  (n={n}, {m} models)",
            va="center", ha="left", fontsize=9,
        )

    control_stats = aggregated.get("control")
    if control_stats is not None:
        control_val = control_stats["accuracy"]
        ax.axvline(
            x=control_val, color="gray", linestyle="--", linewidth=1.2,
            alpha=0.7, label=f"Control: {control_val:.1%}",
        )
        ax.legend(fontsize=9)

    ax.set_xlim(0, min(1.15, max(ci_highs) * 1.25) if ci_highs else 1.0)
    ax.set_xlabel(
        "Accuracy pooled across all models (error bars = 95% Wilson CI)",
        fontsize=11,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, axis="x", alpha=0.3)

    import os
    os.makedirs(str(Path(save_path).parent), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_grouped_bar_chart(
    all_accuracy: dict[str, dict[str, dict | None]],
    title: str,
    save_path: str | Path,
):
    """Grouped bar chart: framings on x-axis, one bar per model, with 95% CIs."""
    models = list(all_accuracy.keys())
    framings = FRAMING_KEYS

    active_framings = [
        f for f in framings
        if any(all_accuracy[m].get(f) is not None for m in models)
    ]
    if not active_framings:
        return

    labels = [_framing_label(f) for f in active_framings]
    x = np.arange(len(active_framings))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(max(14, len(active_framings) * 1.3), 8))
    cmap = matplotlib.colormaps["tab10"]

    for i, model in enumerate(models):
        short = model.split("/")[-1]
        stats_list = [all_accuracy[model].get(f) for f in active_framings]
        vals = [s["accuracy"] if s else 0 for s in stats_list]
        err_low = [
            max(0, (s["accuracy"] - s["ci_low"])) if s else 0 for s in stats_list
        ]
        err_high = [
            max(0, (s["ci_high"] - s["accuracy"])) if s else 0 for s in stats_list
        ]

        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, vals, width,
            label=short, color=cmap(i), alpha=0.85,
            yerr=[err_low, err_high],
            capsize=2, error_kw={"elinewidth": 0.9, "alpha": 0.7},
        )

        for bar, val, stats in zip(bars, vals, stats_list):
            if stats is not None:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    stats["ci_high"] + 0.008,
                    f"{val:.0%}", ha="center", va="bottom",
                    fontsize=6, rotation=45,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy (error bars = 95% Wilson CI)", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    import os
    os.makedirs(str(Path(save_path).parent), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_heatmap(
    all_accuracy: dict[str, dict[str, dict | None]],
    title: str,
    save_path: str | Path,
):
    """Heatmap: models (rows) x framings (columns)."""
    models = list(all_accuracy.keys())
    framings = FRAMING_KEYS

    active_framings = [
        f for f in framings
        if any(all_accuracy[m].get(f) is not None for m in models)
    ]
    if not active_framings:
        return

    data = np.zeros((len(models), len(active_framings)))
    for i, model in enumerate(models):
        for j, framing in enumerate(active_framings):
            stats = all_accuracy[model].get(framing)
            data[i, j] = stats["accuracy"] if stats is not None else np.nan

    row_labels = [m.split("/")[-1] for m in models]
    col_labels = [_framing_label(f) for f in active_framings]

    fig, ax = plt.subplots(figsize=(max(12, len(active_framings) * 1.0), max(4, len(models) * 0.8)))

    cmap = matplotlib.colormaps["RdYlGn"].copy()
    cmap.set_bad(color="lightgray")

    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(row_labels, fontsize=10)

    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = data[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 0.4 or val > 0.8 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=text_color)

    fig.colorbar(im, ax=ax, label="Accuracy", shrink=0.8)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    import os
    os.makedirs(str(Path(save_path).parent), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_delta_chart(
    all_accuracy: dict[str, dict[str, dict | None]],
    title: str,
    save_path: str | Path,
):
    """Bar chart showing each framing's accuracy delta vs. control, per model.

    Error bars here are the CI width of the framing itself, as a rough
    uncertainty bound — this is conservative since it ignores correlation
    with the control arm, but is simple and comparable across framings.
    """
    models = list(all_accuracy.keys())
    framings = [f for f in FRAMING_KEYS if f != "control"]

    active_framings = [
        f for f in framings
        if any(all_accuracy[m].get(f) is not None for m in models)
    ]
    if not active_framings:
        return

    labels = [_framing_label(f) for f in active_framings]
    x = np.arange(len(active_framings))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(max(14, len(active_framings) * 1.3), 8))
    cmap = matplotlib.colormaps["tab10"]

    for i, model in enumerate(models):
        short = model.split("/")[-1]
        control_stats = all_accuracy[model].get("control")
        control = control_stats["accuracy"] if control_stats else 0

        stats_list = [all_accuracy[model].get(f) for f in active_framings]
        deltas = [(s["accuracy"] - control) if s else 0 for s in stats_list]
        err_low = [
            max(0, (s["accuracy"] - s["ci_low"])) if s else 0 for s in stats_list
        ]
        err_high = [
            max(0, (s["ci_high"] - s["accuracy"])) if s else 0 for s in stats_list
        ]

        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(
            x + offset, deltas, width,
            label=short, color=cmap(i), alpha=0.85,
            yerr=[err_low, err_high],
            capsize=2, error_kw={"elinewidth": 0.9, "alpha": 0.7},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy Delta vs. Control (± framing 95% CI)", fontsize=11)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    import os
    os.makedirs(str(Path(save_path).parent), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Significance chart (paired McNemar vs control + Cochran's Q omnibus)
# ---------------------------------------------------------------------------

def _stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def generate_significance_chart(
    sig: dict,
    title: str,
    save_path: str | Path,
):
    """Horizontal bar chart of accuracy delta vs control per framing.

    Color encodes Holm-adjusted significance (saturated = significant, gray = not).
    Bar annotations show Δ, raw p, Holm-adjusted p, and significance stars.
    Title includes Cochran's Q omnibus result.

    `sig` is the dict returned by `significance.compute_significance()`.
    """
    pairwise = sig.get("pairwise") or {}
    if not pairwise:
        return

    items = sorted(pairwise.items(), key=lambda kv: kv[1]["delta"])
    labels   = [_framing_label(k) for k, _ in items]
    deltas   = [v["delta"]  for _, v in items]
    p_raws   = [v["p_raw"]  for _, v in items]
    p_holms  = [v["p_holm"] for _, v in items]
    bcs      = [(v["b"], v["c"]) for _, v in items]

    def _color(delta, p_holm):
        if p_holm >= 0.05:
            return "#bdbdbd"
        return "#2ca02c" if delta > 0 else "#d62728"

    colors = [_color(d, p) for d, p in zip(deltas, p_holms)]

    fig, ax = plt.subplots(figsize=(14, max(5.5, len(labels) * 0.6)))
    bars = ax.barh(labels, deltas, color=colors, edgecolor="white", linewidth=0.6)

    # Symmetric x-axis around 0 for visual balance
    max_abs = max(abs(d) for d in deltas) if deltas else 0.05
    pad = max(0.02, max_abs * 0.15)
    ax.set_xlim(-max_abs - pad - 0.18, max_abs + pad + 0.18)

    for bar, d, p_raw, p_holm, (b, c) in zip(bars, deltas, p_raws, p_holms, bcs):
        stars = _stars(p_holm)
        text = (f"  Δ {d:+.1%}   "
                f"p={p_raw:.3g}   p_adj={p_holm:.3g}   "
                f"b={b}, c={c}   {stars}")
        # Position annotation just past the bar end on the outside
        if d >= 0:
            x = d + 0.005
            ha = "left"
        else:
            x = d - 0.005
            ha = "right"
        ax.text(x, bar.get_y() + bar.get_height() / 2, text,
                va="center", ha=ha, fontsize=8)

    ax.axvline(0, color="black", linewidth=0.9)
    ax.grid(True, axis="x", alpha=0.3)

    cochran = sig.get("cochran")
    n_subj  = sig.get("n_subjects", 0)
    if cochran:
        Q, df, pq = cochran["Q"], cochran["df"], cochran["p"]
        omnibus_stars = _stars(pq) if pq is not None and not math.isnan(pq) else "—"
        sub = (f"Cochran's Q = {Q:.2f}  (df={df}, p={pq:.3g}) {omnibus_stars}   "
               f"|   n={n_subj} subjects   "
               f"|   pairwise: McNemar exact, Holm-adjusted vs control")
    else:
        sub = (f"n={n_subj} subjects   "
               f"|   pairwise: McNemar exact, Holm-adjusted vs control")

    ax.set_title(f"{title}\n{sub}", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Accuracy Δ vs control", fontsize=11)

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color="#2ca02c", label="Significantly higher (p_adj<0.05)"),
        Patch(color="#d62728", label="Significantly lower (p_adj<0.05)"),
        Patch(color="#bdbdbd", label="Not significant"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="lower right")

    import os
    os.makedirs(str(Path(save_path).parent), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
