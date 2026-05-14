"""Visualization for the Idea Priming study.

Three charts (per request):

1. Per-model bar chart of mean score under positive vs negative priming
   (grouped chart across all models, plus individual per-model bars), with
   95% CIs computed across all (idea, iteration) responses.
2. Per-model paired-dot plot: each idea contributes two dots (positive,
   negative) connected by a line, sorted by bias magnitude, colored by
   quality bucket (good / bad / ambiguous).
3. Headline chart: per-model priming bias (mean of per-idea
   positive_mean - negative_mean) with 95% t-CIs across ideas.
"""

import os
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from studies.idea_priming.config import (
    BUCKETS,
    FRAME_KEYS,
    FRAME_LABELS,
    IDEA_BY_ID,
)
from utils.model_images import load_model_image


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FRAME_COLORS = {
    "positive": "#2E86C1",  # blue
    "negative": "#C0392B",  # red
}

_BUCKET_COLORS = {
    "good":      "#27AE60",  # green
    "ambiguous": "#7F8C8D",  # grey
    "bad":       "#C0392B",  # red
}

_BUCKET_LABELS = {
    "good":      "Clearly good",
    "ambiguous": "Ambiguous",
    "bad":       "Clearly bad",
}


def _short_model(model: str) -> str:
    return model.split("/")[-1]


def _ensure_dir(path: Path):
    os.makedirs(str(path.parent), exist_ok=True)


# ---------------------------------------------------------------------------
# 1a. Per-model individual bar chart
# ---------------------------------------------------------------------------

def generate_per_model_bar(
    model: str,
    stats: dict,
    save_path: Path,
):
    """One chart, two bars (positive, negative) for a single model."""
    pooled = stats["frame_pooled"]
    if not all(pooled.get(fk) is not None for fk in FRAME_KEYS):
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    means = [pooled[fk]["mean"] for fk in FRAME_KEYS]
    err_low = [pooled[fk]["mean"] - pooled[fk]["ci_low"] for fk in FRAME_KEYS]
    err_high = [pooled[fk]["ci_high"] - pooled[fk]["mean"] for fk in FRAME_KEYS]
    colors = [_FRAME_COLORS[fk] for fk in FRAME_KEYS]
    labels = [FRAME_LABELS[fk] for fk in FRAME_KEYS]

    x = np.arange(len(FRAME_KEYS))
    bars = ax.bar(
        x, means, color=colors, edgecolor="white", linewidth=0.6, width=0.55,
        yerr=[err_low, err_high], capsize=5,
        error_kw={"elinewidth": 1.4, "alpha": 0.85},
    )

    for bar, mean_v, p in zip(bars, means, [pooled[fk] for fk in FRAME_KEYS]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            p["ci_high"] + 0.15,
            f"{mean_v:.2f}\n(n={p['n']})",
            ha="center", va="bottom", fontsize=10,
        )

    bias = stats.get("model_bias")
    if bias is not None:
        subtitle = (
            f"Priming bias (positive − negative): "
            f"{bias['mean']:+.2f}  "
            f"[95% CI {bias['ci_low']:+.2f}, {bias['ci_high']:+.2f}]  "
            f"across {bias['n_ideas']} ideas"
        )
    else:
        subtitle = ""

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Mean quality score (1-10)", fontsize=11)
    ax.set_ylim(0, 10.5)
    ax.set_title(
        f"{_short_model(model)} — Score by Priming Frame\n{subtitle}",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    _ensure_dir(save_path)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1b. Grouped bar chart, all models
# ---------------------------------------------------------------------------

def generate_grouped_bar(
    all_stats: dict[str, dict],
    save_path: Path,
):
    """Grouped bars: x = model, two bars per model (positive vs negative)."""
    models = [m for m, s in all_stats.items()
              if s["frame_pooled"].get("positive") is not None
              and s["frame_pooled"].get("negative") is not None]
    if not models:
        return

    x = np.arange(len(models))
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.6), 6))

    for i, frame_key in enumerate(FRAME_KEYS):
        means = []
        err_low = []
        err_high = []
        for m in models:
            p = all_stats[m]["frame_pooled"][frame_key]
            means.append(p["mean"])
            err_low.append(p["mean"] - p["ci_low"])
            err_high.append(p["ci_high"] - p["mean"])
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset, means, width,
            label=FRAME_LABELS[frame_key],
            color=_FRAME_COLORS[frame_key],
            edgecolor="white", linewidth=0.5,
            yerr=[err_low, err_high], capsize=3,
            error_kw={"elinewidth": 1.0, "alpha": 0.8},
        )
        for bar, mean_v in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2, mean_v + 0.18,
                f"{mean_v:.2f}", ha="center", va="bottom", fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([_short_model(m) for m in models], rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Mean quality score (1-10)", fontsize=11)
    ax.set_ylim(0, 10.5)
    ax.set_title(
        "Mean Quality Score by Priming Frame  (error bars = 95% CI)",
        fontsize=13, fontweight="bold", pad=10,
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    _ensure_dir(save_path)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Per-model paired-dot plot, sorted by bias, colored by bucket
# ---------------------------------------------------------------------------

def generate_paired_dots(
    model: str,
    stats: dict,
    save_path: Path,
):
    """Per-idea positive vs negative score, sorted by bias.

    Each idea is a row. Two dots (positive blue, negative red) connected
    by a horizontal line. Idea label is colored by quality bucket.
    """
    per_idea = stats["per_idea"]

    # Only keep ideas where both frames have data (i.e., bias defined)
    valid = []
    for idea_id, s in per_idea.items():
        if s.get("bias") is None:
            continue
        valid.append((idea_id, s["bias"], s["positive"], s["negative"]))
    if not valid:
        return

    # Sort by bias ascending (largest negative bias at bottom)
    valid.sort(key=lambda t: t[1])

    n = len(valid)
    fig, ax = plt.subplots(figsize=(11, max(5, n * 0.42)))

    y_positions = np.arange(n)
    label_colors = []
    y_labels = []

    for y, (idea_id, bias, pos_stats, neg_stats) in zip(y_positions, valid):
        idea = IDEA_BY_ID.get(idea_id)
        bucket = idea["bucket"] if idea else "ambiguous"

        # Connecting line between the two means
        ax.plot(
            [neg_stats["mean"], pos_stats["mean"]], [y, y],
            color="#999999", linewidth=1.2, alpha=0.7, zorder=1,
        )

        # Negative-frame error bar + dot
        ax.errorbar(
            neg_stats["mean"], y,
            xerr=[[neg_stats["mean"] - neg_stats["ci_low"]],
                  [neg_stats["ci_high"] - neg_stats["mean"]]],
            fmt="o", color=_FRAME_COLORS["negative"],
            ecolor=_FRAME_COLORS["negative"], elinewidth=1.0,
            capsize=2, markersize=7, alpha=0.9, zorder=2,
        )
        # Positive-frame error bar + dot
        ax.errorbar(
            pos_stats["mean"], y,
            xerr=[[pos_stats["mean"] - pos_stats["ci_low"]],
                  [pos_stats["ci_high"] - pos_stats["mean"]]],
            fmt="o", color=_FRAME_COLORS["positive"],
            ecolor=_FRAME_COLORS["positive"], elinewidth=1.0,
            capsize=2, markersize=7, alpha=0.9, zorder=2,
        )

        # Label = idea id with bias annotation, colored by bucket
        y_labels.append(f"{idea_id}  (Δ {bias:+.2f})")
        label_colors.append(_BUCKET_COLORS[bucket])

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8)
    for tick_label, color in zip(ax.get_yticklabels(), label_colors):
        tick_label.set_color(color)

    ax.set_xlim(0, 11)
    ax.set_xlabel("Quality score (1-10), error bars = 95% CI", fontsize=11)
    ax.set_title(
        f"{_short_model(model)} — Per-Idea Score by Priming Frame "
        f"(sorted by bias)",
        fontsize=13, fontweight="bold", pad=10,
    )

    # Legend: combine frames + buckets
    frame_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=_FRAME_COLORS[fk], markersize=8,
                   label=FRAME_LABELS[fk])
        for fk in FRAME_KEYS
    ]
    bucket_handles = [
        plt.Line2D([0], [0], marker="s", color="w",
                   markerfacecolor=_BUCKET_COLORS[b], markersize=10,
                   label=_BUCKET_LABELS[b])
        for b in BUCKETS
    ]
    leg1 = ax.legend(
        handles=frame_handles, fontsize=9, loc="upper left",
        title="Frame", title_fontsize=9,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=bucket_handles, fontsize=9, loc="lower right",
        title="Idea bucket (label color)", title_fontsize=9,
    )

    ax.grid(True, axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    _ensure_dir(save_path)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Dumbbell chart with model logos — all models together
#
# Each model gets one row: logo on the left, two dots (positive blue,
# negative red) connected by a line on the score axis, with the bias delta
# annotated on the right. Sorted by bias size.
# ---------------------------------------------------------------------------

def _place_logo_extent(
    ax,
    model: str,
    cx: float,
    cy: float,
    size: float,
    chip_color: str | None = None,
    fallback_text_color: str = "black",
):
    """Place a provider logo using imshow with explicit extent (units = axes
    coords, which the caller controls). `size` is the side length in those
    units. If `chip_color` is given, draw a rounded square behind the logo
    in that color (useful for dark logos on dark backgrounds)."""
    half = size / 2.0
    if chip_color is not None:
        chip_pad = size * 0.10
        chip = mpatches.FancyBboxPatch(
            (cx - half - chip_pad, cy - half - chip_pad),
            size + 2 * chip_pad, size + 2 * chip_pad,
            boxstyle="round,pad=0.0,rounding_size=" + str(size * 0.18),
            facecolor=chip_color, edgecolor="none", zorder=4,
        )
        ax.add_patch(chip)

    img = load_model_image(model)
    if img is None:
        ax.text(
            cx, cy, model.split("/")[0],
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=fallback_text_color, zorder=5,
        )
        return

    ax.imshow(
        img,
        extent=[cx - half, cx + half, cy - half, cy + half],
        aspect="auto", zorder=5, interpolation="lanczos",
    )


def generate_dumbbell_with_logos(
    all_stats: dict[str, dict],
    save_path: Path,
):
    """Wide horizontal chart, one row per model, with logo + dumbbell + delta.

    Uses a full-figure axis with axis turned off, so logos can be placed
    precisely in figure inches without getting clipped by data limits.
    Rows are sorted by bias size (biggest priming swing on top).
    """
    rows = []
    for model, s in all_stats.items():
        pooled = s["frame_pooled"]
        bias = s.get("model_bias")
        if pooled.get("positive") is None or pooled.get("negative") is None or bias is None:
            continue
        rows.append((
            model,
            pooled["positive"]["mean"],
            pooled["negative"]["mean"],
            bias["mean"],
            bias["ci_low"],
            bias["ci_high"],
            bias["n_ideas"],
        ))
    if not rows:
        return

    rows.sort(key=lambda t: t[3], reverse=True)
    n = len(rows)

    # ---------- Layout (in figure inches) ----------
    fig_w = 13.5
    row_h = 1.20
    header_h = 1.85
    axis_strip_h = 0.95
    fig_h = header_h + n * row_h + axis_strip_h

    margin = 0.40
    logo_size = 0.85
    logo_cx = margin + logo_size / 2 + 0.10
    name_label_cx = logo_cx
    rail_left = logo_cx + logo_size / 2 + 0.55
    delta_block_w = 2.55
    rail_right = fig_w - margin - delta_block_w
    delta_x = rail_right + 0.20

    # x-axis 1..10 maps to [rail_left, rail_right]
    score_min, score_max = 1.0, 10.0
    def _x_for_score(s: float) -> float:
        t = (s - score_min) / (score_max - score_min)
        return rail_left + t * (rail_right - rail_left)

    # ---------- Figure ----------
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.set_aspect("auto")
    ax.axis("off")

    # ---------- Header ----------
    title_y = fig_h - 0.55
    subtitle_y = fig_h - 1.00
    legend_y = fig_h - 1.45

    ax.text(
        fig_w / 2, title_y,
        "Mean Quality Score by Priming Frame",
        ha="center", va="center", fontsize=20, fontweight="bold", color="#111111",
    )
    ax.text(
        fig_w / 2, subtitle_y,
        "Same 21 ideas, same 1-10 scale — only the reflection question differs.",
        ha="center", va="center", fontsize=11, color="#555555",
    )

    # Legend (centered, under subtitle)
    legend_items = [
        ('"Why is this a good idea?"', _FRAME_COLORS["positive"]),
        ('"Why is this a bad idea?"',  _FRAME_COLORS["negative"]),
    ]
    legend_text_widths = [3.1, 3.0]
    spacing = 0.45
    total_w = sum(legend_text_widths) + spacing * (len(legend_items))
    cursor = fig_w / 2 - total_w / 2
    for (label, color), w in zip(legend_items, legend_text_widths):
        ax.scatter([cursor + 0.10], [legend_y], s=130, color=color,
                   edgecolors="white", linewidths=1.5, zorder=3)
        ax.text(
            cursor + 0.30, legend_y, label,
            ha="left", va="center", fontsize=10.5, color="#222222",
        )
        cursor += w + spacing

    # ---------- Score-axis strip at the bottom ----------
    axis_y = axis_strip_h - 0.30  # baseline for tick labels
    rail_y_axis = axis_y + 0.32
    # Vertical guide lines for whole integer ticks (across the rows)
    rows_y_top = header_h
    rows_y_bottom = axis_strip_h
    for tick in range(int(score_min), int(score_max) + 1):
        x = _x_for_score(tick)
        ax.plot([x, x], [rows_y_bottom, fig_h - 0.20],
                color="#EAECEF", linewidth=1.0, zorder=0)
        ax.text(x, axis_y, str(tick), ha="center", va="top",
                fontsize=10, color="#555555")
    ax.plot([rail_left, rail_right], [rail_y_axis, rail_y_axis],
            color="#CCCCCC", linewidth=1.2, zorder=1)
    ax.text(
        (rail_left + rail_right) / 2, axis_y - 0.42,
        "Mean quality score (1-10)",
        ha="center", va="top", fontsize=11, color="#333333",
    )

    # ---------- Rows ----------
    for i, (model, pos_mean, neg_mean, bias, blo, bhi, n_ideas) in enumerate(rows):
        # First row centered near the top of the body region
        cy = fig_h - header_h - (i + 0.5) * row_h

        # Subtle row separator (skip below last)
        if i < n - 1:
            ax.plot(
                [logo_cx - logo_size / 2 - 0.05, fig_w - margin],
                [cy - row_h / 2, cy - row_h / 2],
                color="#F0F2F5", linewidth=1.0, zorder=0,
            )

        # Logo + provider name
        _place_logo_extent(
            ax, model, logo_cx, cy + 0.10, logo_size,
            chip_color="#F6F7F9",
        )
        ax.text(
            name_label_cx, cy - logo_size / 2 - 0.02, model.split("/")[-1],
            ha="center", va="top", fontsize=8.5, color="#444444",
        )

        # Connector capsule between dots (light grey rounded bar)
        x_pos = _x_for_score(pos_mean)
        x_neg = _x_for_score(neg_mean)
        gap_left, gap_right = sorted([x_pos, x_neg])
        capsule_h = 0.20
        ax.add_patch(mpatches.FancyBboxPatch(
            (gap_left, cy - capsule_h / 2),
            gap_right - gap_left, capsule_h,
            boxstyle="round,pad=0.0,rounding_size=0.10",
            facecolor="#E5E7EB", edgecolor="none", zorder=2,
        ))

        # Dots
        ax.scatter([x_neg], [cy], s=300,
                   color=_FRAME_COLORS["negative"],
                   edgecolors="white", linewidths=2.0, zorder=3)
        ax.scatter([x_pos], [cy], s=300,
                   color=_FRAME_COLORS["positive"],
                   edgecolors="white", linewidths=2.0, zorder=3)

        # Score numbers — placed on outer side of each dot
        if x_neg <= x_pos:
            ax.text(x_neg - 0.15, cy, f"{neg_mean:.2f}",
                    ha="right", va="center", fontsize=10,
                    color=_FRAME_COLORS["negative"], fontweight="bold",
                    zorder=4)
            ax.text(x_pos + 0.15, cy, f"{pos_mean:.2f}",
                    ha="left", va="center", fontsize=10,
                    color=_FRAME_COLORS["positive"], fontweight="bold",
                    zorder=4)
        else:
            ax.text(x_pos - 0.15, cy, f"{pos_mean:.2f}",
                    ha="right", va="center", fontsize=10,
                    color=_FRAME_COLORS["positive"], fontweight="bold",
                    zorder=4)
            ax.text(x_neg + 0.15, cy, f"{neg_mean:.2f}",
                    ha="left", va="center", fontsize=10,
                    color=_FRAME_COLORS["negative"], fontweight="bold",
                    zorder=4)

        # Delta block on the right
        delta_color = "#1A6FB8" if bias > 0 else "#7F8C8D"
        ax.text(
            delta_x, cy + 0.20, f"Δ {bias:+.2f}",
            ha="left", va="center", fontsize=15, fontweight="bold",
            color=delta_color,
        )
        ax.text(
            delta_x, cy - 0.18,
            f"95% CI [{blo:+.2f}, {bhi:+.2f}]",
            ha="left", va="center", fontsize=8.5, color="#555555",
        )
        ax.text(
            delta_x, cy - 0.36,
            f"n = {n_ideas} ideas",
            ha="left", va="center", fontsize=8.5, color="#777777",
        )

    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Instagram Shorts card (1080 x 1920, 9:16 vertical)
#
# Big bold layout: title + subtitle at top, vertical stack of model rows
# (logo, big delta number, mini-bars). Designed to be visually engaging
# in a Reels/Shorts feed. Dark background for contrast.
# ---------------------------------------------------------------------------

_SHORTS_BG       = "#0F1116"   # near-black, slight blue tint
_SHORTS_FG       = "#FFFFFF"
_SHORTS_MUTED    = "#9AA0A6"
_SHORTS_PANEL    = "#1A1D26"
_SHORTS_POS      = "#3FA7FF"   # brighter blue for dark bg
_SHORTS_NEG      = "#FF6B6B"   # brighter red for dark bg
_SHORTS_ACCENT   = "#FFD166"   # yellow accent for the headline delta


def generate_shorts_card(
    all_stats: dict[str, dict],
    save_path: Path,
):
    """Vertical 1080x1920 card optimized for Instagram Shorts / Reels.

    Each model gets a panel: logo (with a light chip behind it for visibility)
    on the left, the big delta number in the middle, and a mini-dumbbell on
    the right showing the actual score gap. Sorted by bias size.
    """
    rows = []
    for model, s in all_stats.items():
        pooled = s["frame_pooled"]
        bias = s.get("model_bias")
        if pooled.get("positive") is None or pooled.get("negative") is None or bias is None:
            continue
        rows.append((
            model,
            pooled["positive"]["mean"],
            pooled["negative"]["mean"],
            bias["mean"],
            bias["ci_low"],
            bias["ci_high"],
        ))
    if not rows:
        return

    rows.sort(key=lambda t: t[3], reverse=True)
    n = len(rows)

    # 1080 x 1920 at dpi=150 ⇒ 7.2 x 12.8 inches
    fig_w, fig_h = 7.2, 12.8
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    fig.patch.set_facecolor(_SHORTS_BG)

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.set_aspect("auto")
    ax.axis("off")

    # ---------- Header ----------
    ax.text(
        fig_w / 2, fig_h - 0.95, "DOES PRIMING",
        ha="center", va="center", color=_SHORTS_FG,
        fontsize=44, fontweight="bold",
    )
    ax.text(
        fig_w / 2, fig_h - 1.70, "BIAS LLM SCORES?",
        ha="center", va="center", color=_SHORTS_ACCENT,
        fontsize=44, fontweight="bold",
    )
    ax.text(
        fig_w / 2, fig_h - 2.55,
        "Same 21 startup-style ideas, same 1-10 scale.\n"
        "Only the reflection question differs.",
        ha="center", va="center", color=_SHORTS_MUTED,
        fontsize=14, linespacing=1.4,
    )

    # ---------- Legend (centered) ----------
    legend_y = fig_h - 3.50
    items = [
        ('"Why good?"', _SHORTS_POS),
        ('"Why bad?"',  _SHORTS_NEG),
    ]
    item_widths = [1.55, 1.40]
    gap = 0.55
    total_w = sum(item_widths) + gap
    cursor = fig_w / 2 - total_w / 2
    for (label, color), w in zip(items, item_widths):
        ax.scatter([cursor + 0.16], [legend_y], s=170, color=color,
                   edgecolors=_SHORTS_BG, linewidths=1.5, zorder=3)
        ax.text(
            cursor + 0.40, legend_y, label,
            ha="left", va="center", color=_SHORTS_FG,
            fontsize=12, fontweight="bold",
        )
        cursor += w + gap

    # ---------- Footer (computed first so we can size panels around it) ----------
    footer_y = 0.45
    ax.text(
        fig_w / 2, footer_y,
        "n = 21 ideas · 10 iterations per (model, idea, frame) · 95% CI",
        ha="center", va="center", color=_SHORTS_MUTED, fontsize=10,
    )

    # ---------- Model rows ----------
    rows_top = legend_y - 0.50
    rows_bottom = footer_y + 0.55
    row_total_h = rows_top - rows_bottom
    row_h = row_total_h / n
    panel_inner_pad = 0.10

    # Geometry within each panel (figure inches)
    margin_outer = 0.40
    panel_left  = margin_outer
    panel_right = fig_w - margin_outer

    logo_size = 1.10
    logo_cx = panel_left + 0.20 + logo_size / 2

    delta_x_left = logo_cx + logo_size / 2 + 0.30   # left edge of big delta number

    rail_left  = 4.95
    rail_right = panel_right - 0.30
    score_min, score_max = 1.0, 10.0

    def _x_for_score(s: float) -> float:
        t = (s - score_min) / (score_max - score_min)
        return rail_left + t * (rail_right - rail_left)

    for i, (model, pos_mean, neg_mean, bias, blo, bhi) in enumerate(rows):
        cy = rows_top - (i + 0.5) * row_h
        panel_top = rows_top - i * row_h - panel_inner_pad / 2
        panel_bottom = rows_top - (i + 1) * row_h + panel_inner_pad / 2

        # Card panel
        ax.add_patch(mpatches.FancyBboxPatch(
            (panel_left, panel_bottom),
            panel_right - panel_left,
            panel_top - panel_bottom,
            boxstyle="round,pad=0.0,rounding_size=0.18",
            facecolor=_SHORTS_PANEL, edgecolor="none", zorder=1,
        ))

        # Logo with a light chip behind it (so dark logos like x-ai are visible)
        _place_logo_extent(
            ax, model, logo_cx, cy, logo_size,
            chip_color="#FFFFFF",
            fallback_text_color=_SHORTS_FG,
        )

        # Big delta number
        delta_color = _SHORTS_ACCENT if bias > 0 else _SHORTS_MUTED
        ax.text(
            delta_x_left, cy + 0.18,
            f"{bias:+.2f}",
            ha="left", va="center", color=delta_color,
            fontsize=42, fontweight="bold",
        )
        ax.text(
            delta_x_left, cy - 0.45, "priming bias",
            ha="left", va="center", color=_SHORTS_MUTED, fontsize=10.5,
        )

        # Mini-dumbbell (right side)
        x_pos = _x_for_score(pos_mean)
        x_neg = _x_for_score(neg_mean)
        # Rail
        ax.plot(
            [rail_left, rail_right], [cy + 0.05, cy + 0.05],
            color="#2C313D", linewidth=2.0, solid_capstyle="round", zorder=2,
        )
        # Active gap segment
        gap_lo, gap_hi = sorted([x_pos, x_neg])
        ax.plot(
            [gap_lo, gap_hi], [cy + 0.05, cy + 0.05],
            color="#5A6271", linewidth=4.5, solid_capstyle="round", zorder=3,
        )
        ax.scatter([x_neg], [cy + 0.05], s=180, color=_SHORTS_NEG,
                   edgecolors=_SHORTS_BG, linewidths=1.8, zorder=5)
        ax.scatter([x_pos], [cy + 0.05], s=180, color=_SHORTS_POS,
                   edgecolors=_SHORTS_BG, linewidths=1.8, zorder=5)

        # Score numbers above each dot (avoids edge clipping)
        ax.text(
            x_neg, cy + 0.42, f"{neg_mean:.1f}",
            ha="center", va="bottom", color=_SHORTS_NEG,
            fontsize=11, fontweight="bold",
        )
        ax.text(
            x_pos, cy + 0.42, f"{pos_mean:.1f}",
            ha="center", va="bottom", color=_SHORTS_POS,
            fontsize=11, fontweight="bold",
        )
        # Tiny "1" / "10" anchor labels under the rail
        ax.text(rail_left, cy - 0.28, "1",
                ha="center", va="center", color="#5A6271", fontsize=8)
        ax.text(rail_right, cy - 0.28, "10",
                ha="center", va="center", color="#5A6271", fontsize=8)

    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, facecolor=_SHORTS_BG)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Ideas list card (1080 x 1920, 9:16 vertical) — for IG carousel
#
# Lists all 21 tested ideas grouped by quality bucket. Matches the Shorts
# card's dark theme so the two slides sit cleanly together in one carousel.
# Each idea is shown as a short headline plus its category tag.
# ---------------------------------------------------------------------------

# Bright bucket accent colors that read well on the dark Shorts background
_BUCKET_COLORS_DARK = {
    "good":      "#3FCB87",
    "ambiguous": "#A0A8B5",
    "bad":       "#FF6B6B",
}

_BUCKET_HEADERS = {
    "good":      "CLEARLY GOOD",
    "ambiguous": "AMBIGUOUS",
    "bad":       "CLEARLY BAD",
}

# Short, carousel-friendly headlines for each tested idea.
# Kept here (not in config) because the prompts use the long descriptions —
# these labels are display-only.
_IDEA_SHORT_LABELS: dict[str, str] = {
    # good
    "good_customer_dev":             "30 customer interviews before writing code",
    "good_soc2_security":            "SOC 2 + security hire before enterprise",
    "good_fraud_detection":          "Adaptive fraud detection w/ push confirms",
    "good_basic_devops":             "Backups, IaC and CI before scaling",
    "good_focused_devmarketing":     "Technical case studies aimed at the actual ICP",
    "good_recruiter_retention_bonus":"Recruiter paid on retention, not signing",
    "good_value_pricing_dental":     "Dental SaaS switches to per-chair pricing",
    # bad
    "bad_carrier_pigeon_letters":    "$40/mo handwritten letters by carrier pigeon",
    "bad_premature_scaling_hires":   "15 senior engineers + CMO + VP Sales pre-customer",
    "bad_haskell_rewrite":           "Stop shipping for a year to rewrite in Haskell",
    "bad_kids_crypto_gambling_pivot":"Pivot kids' toy company to NFT loot boxes",
    "bad_super_bowl_burnout":        "Whole seed round on one Super Bowl ad",
    "bad_kids_data_for_crypto":      "Crypto-for-report-cards, sell the data",
    "bad_friction_stacked_pricing":  "$99/mo, 12-mo lockin, cold-email Fortune 500",
    # ambiguous
    "amb_ai_therapy_chatbot":        "$25/mo AI therapy chatbots for under-35s",
    "amb_bootstrap_takes_vc":        "Profitable bootstrapper raises $30M VC",
    "amb_kill_free_tier":            "Kill the free tier, lose ~70% of users",
    "amb_fully_remote_18_countries": "Shut SF HQ, fully remote across 18 countries",
    "amb_paywall_existing_features": "$200/yr paywall on features that came w/ hardware",
    "amb_open_source_pivot":         "Open-source everything, monetize hosted tier",
    "amb_retiree_student_mentorship":"Retirees ↔ college students mentorship marketplace",
}


def generate_ideas_list_card(
    save_path: Path,
):
    """Vertical 1080x1920 carousel slide listing all 21 tested ideas grouped
    by quality bucket. Pulls the idea list from config and uses the short
    display labels defined above."""
    from studies.idea_priming.config import BUCKETS, IDEAS

    # Group ideas by bucket, preserving config order within bucket
    bucketed: dict[str, list[dict]] = {b: [] for b in BUCKETS}
    for idea in IDEAS:
        bucketed[idea["bucket"]].append(idea)

    # 1080 x 1920 at dpi=150 ⇒ 7.2 x 12.8 inches
    fig_w, fig_h = 7.2, 12.8
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    fig.patch.set_facecolor(_SHORTS_BG)

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.set_aspect("auto")
    ax.axis("off")

    # ---------- Header ----------
    ax.text(
        fig_w / 2, fig_h - 0.95, "21 IDEAS",
        ha="center", va="center", color=_SHORTS_FG,
        fontsize=46, fontweight="bold",
    )
    ax.text(
        fig_w / 2, fig_h - 1.70, "WE TESTED",
        ha="center", va="center", color=_SHORTS_ACCENT,
        fontsize=46, fontweight="bold",
    )
    ax.text(
        fig_w / 2, fig_h - 2.55,
        "Same 1-10 quality scale, asked of every model.\n"
        "Balanced across three quality buckets.",
        ha="center", va="center", color=_SHORTS_MUTED,
        fontsize=13, linespacing=1.4,
    )

    # ---------- Footer (drawn first so we can size sections around it) ----------
    footer_y = 0.40
    ax.text(
        fig_w / 2, footer_y,
        "21 ideas across startups, hiring, pricing, GTM, ops, M&A.",
        ha="center", va="center", color=_SHORTS_MUTED, fontsize=10,
    )

    # ---------- Sections ----------
    sections_top = fig_h - 3.10
    sections_bottom = footer_y + 0.55
    section_h = (sections_top - sections_bottom) / len(BUCKETS)
    section_inner_pad = 0.12

    margin_outer = 0.40
    panel_left  = margin_outer
    panel_right = fig_w - margin_outer

    for s_idx, bucket in enumerate(BUCKETS):
        section_top    = sections_top - s_idx * section_h - section_inner_pad / 2
        section_bottom = sections_top - (s_idx + 1) * section_h + section_inner_pad / 2

        bucket_color = _BUCKET_COLORS_DARK[bucket]
        ideas_in_bucket = bucketed[bucket]

        # Background panel
        ax.add_patch(mpatches.FancyBboxPatch(
            (panel_left, section_bottom),
            panel_right - panel_left,
            section_top - section_bottom,
            boxstyle="round,pad=0.0,rounding_size=0.18",
            facecolor=_SHORTS_PANEL, edgecolor="none", zorder=1,
        ))

        # Colored vertical accent bar on the left edge
        accent_bar_w = 0.10
        ax.add_patch(mpatches.FancyBboxPatch(
            (panel_left + 0.08, section_bottom + 0.18),
            accent_bar_w, (section_top - section_bottom) - 0.36,
            boxstyle="round,pad=0.0,rounding_size=0.05",
            facecolor=bucket_color, edgecolor="none", zorder=2,
        ))

        # Section header
        header_y = section_top - 0.45
        # Coloured dot before header text
        ax.scatter(
            [panel_left + 0.55], [header_y], s=140,
            color=bucket_color, edgecolors=_SHORTS_BG, linewidths=1.5,
            zorder=3,
        )
        ax.text(
            panel_left + 0.78, header_y,
            _BUCKET_HEADERS[bucket],
            ha="left", va="center",
            color=bucket_color, fontsize=18, fontweight="bold",
        )
        ax.text(
            panel_right - 0.30, header_y,
            f"{len(ideas_in_bucket)} ideas",
            ha="right", va="center",
            color=_SHORTS_MUTED, fontsize=11, fontweight="bold",
        )

        # Item list
        list_top = header_y - 0.40
        list_bottom = section_bottom + 0.20
        list_h = list_top - list_bottom
        n_items = len(ideas_in_bucket)
        item_h = list_h / n_items

        text_x_label = panel_left + 0.78
        text_x_category_right = panel_right - 0.30

        for i, idea in enumerate(ideas_in_bucket):
            cy = list_top - (i + 0.5) * item_h
            short = _IDEA_SHORT_LABELS.get(idea["id"], idea["category"])

            # Bullet
            ax.scatter(
                [panel_left + 0.55], [cy], s=28,
                color=bucket_color, zorder=3,
            )

            # Short headline
            ax.text(
                text_x_label, cy, short,
                ha="left", va="center",
                color=_SHORTS_FG, fontsize=12.5,
            )

    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, facecolor=_SHORTS_BG)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Headline chart — per-model bias with 95% CI across ideas
# ---------------------------------------------------------------------------

def generate_headline_bias(
    all_stats: dict[str, dict],
    save_path: Path,
):
    """Horizontal bar chart of per-model priming bias with 95% CIs.

    Bias = mean over ideas of (positive_mean - negative_mean) for the model.
    CI is the t-CI across the per-idea biases.
    """
    rows = []
    for model, s in all_stats.items():
        bias = s.get("model_bias")
        if bias is None:
            continue
        rows.append((model, bias))
    if not rows:
        return

    rows.sort(key=lambda t: t[1]["mean"])

    fig, ax = plt.subplots(figsize=(10, max(4, len(rows) * 0.7)))

    y = np.arange(len(rows))
    means = [b["mean"] for _, b in rows]
    err_low = [b["mean"] - b["ci_low"] for _, b in rows]
    err_high = [b["ci_high"] - b["mean"] for _, b in rows]

    colors = ["#2E86C1" if m > 0 else "#7F8C8D" for m in means]

    ax.barh(
        y, means, color=colors, edgecolor="white", linewidth=0.6, alpha=0.9,
        xerr=[err_low, err_high], capsize=4,
        error_kw={"elinewidth": 1.4, "alpha": 0.85, "ecolor": "black"},
    )

    for i, (model, b) in enumerate(rows):
        ax.text(
            max(b["ci_high"], b["mean"]) + 0.05, i,
            f"{b['mean']:+.2f}  [{b['ci_low']:+.2f}, {b['ci_high']:+.2f}]  "
            f"n={b['n_ideas']}",
            va="center", ha="left", fontsize=9,
        )

    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([_short_model(m) for m, _ in rows], fontsize=10)
    ax.set_xlabel(
        "Priming bias  (positive_mean − negative_mean, score points)",
        fontsize=11,
    )
    ax.set_title(
        "Per-Model Priming Bias  (mean across ideas, 95% CI)",
        fontsize=13, fontweight="bold", pad=10,
    )

    # Pad x-limits to fit annotations
    if means:
        lo = min(b["ci_low"] for _, b in rows) - 0.3
        hi = max(b["ci_high"] for _, b in rows) + 1.4
        ax.set_xlim(min(lo, -0.3), max(hi, 0.3))

    ax.grid(True, axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    _ensure_dir(save_path)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
