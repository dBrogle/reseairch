"""Visualization for the Dictator Removal study.

Generates a grid chart: dictators on x-axis (with portrait images),
models on y-axis, showing YES percentage as colored bars.
Also generates per-model charts with statistical significance annotations.
"""

import os
from pathlib import Path
from itertools import combinations

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from scipy import stats

from studies.dictator_removal.config import DICTATORS, IMAGES_DIR
from utils.model_images import load_model_image


# Red gradient for "would kill" percentage
KILL_CMAP = matplotlib.colormaps["RdYlGn_r"]


def _load_portrait(dictator_id: str, target_size: int = 80) -> np.ndarray | None:
    """Load a dictator's portrait image, return as array or None."""
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        path = IMAGES_DIR / f"{dictator_id}{ext}"
        if path.exists():
            try:
                img = mpimg.imread(str(path))
                return img
            except Exception:
                continue
    return None


def generate_grid_chart(
    model_scores: dict[str, dict[str, dict]],
    save_path: str | Path,
):
    """Generate a grid chart with all models x all dictators.

    model_scores: {model_name: {dictator_id: {"yes_rate": float|None, ...}}}
    """
    models = list(model_scores.keys())
    short_names = [m.split("/")[-1] for m in models]
    n_models = len(models)
    n_dictators = len(DICTATORS)

    fig_height = max(6, 2 + n_models * 0.8)
    fig, ax = plt.subplots(1, 1, figsize=(14, fig_height))

    bar_height = 0.6
    group_height = n_models * bar_height + 0.8
    y_positions = {}

    for d_idx, dictator in enumerate(DICTATORS):
        base_y = d_idx * group_height
        for m_idx, model in enumerate(models):
            y = base_y + m_idx * bar_height
            y_positions[(d_idx, m_idx)] = y

            scores = model_scores[model].get(dictator["id"], {})
            yes_rate = scores.get("yes_rate")

            if yes_rate is not None:
                color = KILL_CMAP(yes_rate)
                ax.barh(y, yes_rate * 100, height=bar_height * 0.85,
                        color=color, edgecolor="black", linewidth=0.5)
                # Label inside or outside bar
                label_x = yes_rate * 100 + 1 if yes_rate < 0.85 else yes_rate * 100 - 1
                ha = "left" if yes_rate < 0.85 else "right"
                ax.text(label_x, y, f"{yes_rate:.0%}", va="center", ha=ha,
                        fontsize=9, fontweight="bold")

                # Refused/error annotation
                refused = scores.get("refused", 0)
                error = scores.get("error", 0)
                if refused + error > 0:
                    total = scores["yes"] + scores["no"] + refused + error
                    ref_pct = (refused + error) / total * 100
                    ax.text(102, y, f"({ref_pct:.0f}% refused)", va="center",
                            ha="left", fontsize=7, color="#888888")
            else:
                ax.text(2, y, "no data", va="center", ha="left",
                        fontsize=9, color="#999999", fontstyle="italic")

    # Y-axis: model names
    yticks = []
    ytick_labels = []
    for d_idx in range(n_dictators):
        for m_idx in range(n_models):
            yticks.append(y_positions[(d_idx, m_idx)])
            ytick_labels.append(short_names[m_idx])

    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels, fontsize=8)

    # Dictator name labels and portraits
    for d_idx, dictator in enumerate(DICTATORS):
        center_y = y_positions[(d_idx, 0)] + (n_models - 1) * bar_height / 2
        # Add portrait image
        portrait = _load_portrait(dictator["id"])
        if portrait is not None:
            imagebox = OffsetImage(portrait, zoom=0.15)
            imagebox.image.axes = ax
            ab = AnnotationBbox(
                imagebox,
                (-22, center_y),
                frameon=True,
                boxcoords="data",
                xycoords="data",
                pad=0.2,
                bboxprops=dict(edgecolor="black", linewidth=1),
            )
            ax.add_artist(ab)

        # Dictator name above group
        top_y = y_positions[(d_idx, n_models - 1)] + bar_height
        ax.text(-2, top_y + 0.15, dictator["name"], va="bottom", ha="right",
                fontsize=11, fontweight="bold")

        # Separator line between groups
        if d_idx > 0:
            sep_y = y_positions[(d_idx, 0)] - (group_height - n_models * bar_height) / 2
            ax.axhline(sep_y, color="#CCCCCC", linewidth=0.8, linestyle="--")

    ax.set_xlim(-5, 130)
    ax.set_ylim(-0.8, n_dictators * group_height - 0.5)
    ax.set_xlabel("Would kill as baby (%)", fontsize=12)
    ax.set_title("Dictator Removal: Would You Kill Them as a Baby?",
                 fontsize=16, fontweight="bold", pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_model_chart(
    model: str,
    scores: dict[str, dict],
    pairwise_pvals: dict[tuple[str, str], float],
    save_path: str | Path,
):
    """Generate a per-model bar chart with significance brackets.

    scores: {dictator_id: {"yes_rate": float|None, "yes": int, "no": int, ...}}
    pairwise_pvals: {(id_a, id_b): p_value}
    """
    short = model.split("/")[-1]
    dictator_ids = [d["id"] for d in DICTATORS]
    dictator_names = [d["name"] for d in DICTATORS]

    # Sort by yes_rate descending
    rates = []
    for d in DICTATORS:
        r = scores.get(d["id"], {}).get("yes_rate")
        rates.append(r if r is not None else -1)
    order = sorted(range(len(DICTATORS)), key=lambda i: rates[i], reverse=True)

    sorted_ids = [dictator_ids[i] for i in order]
    sorted_names = [dictator_names[i] for i in order]
    sorted_rates = [rates[i] for i in order]

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    bars_x = np.arange(len(DICTATORS))
    bar_colors = [KILL_CMAP(r) if r >= 0 else "#CCCCCC" for r in sorted_rates]

    ax.bar(bars_x, [max(r, 0) * 100 for r in sorted_rates],
           color=bar_colors, edgecolor="black", linewidth=0.8, width=0.7)

    # Value labels on bars (offset points = constant pixel gap across charts)
    for i, rate in enumerate(sorted_rates):
        if rate >= 0:
            ax.annotate(f"{rate:.0%}", xy=(i, rate * 100), xytext=(0, 5),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=11, fontweight="bold")

            # Refused count
            s = scores.get(sorted_ids[i], {})
            refused = s.get("refused", 0) + s.get("error", 0)
            if refused > 0:
                total = s["yes"] + s["no"] + refused
                ax.annotate(f"({refused}/{total} refused)",
                            xy=(i, rate * 100), xytext=(0, 18),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=10, fontweight="bold", color="#333333")

    # X-axis: dictator names as tick labels
    ax.set_xticks(bars_x)
    ax.set_xticklabels(sorted_names, fontsize=10, fontweight="bold")

    # Portraits sit above each bar at constant pixel offset
    for i, (d_id, rate) in enumerate(zip(sorted_ids, sorted_rates)):
        portrait = _load_portrait(d_id)
        if portrait is None:
            continue
        bar_top = max(rate, 0) * 100
        imagebox = OffsetImage(portrait, zoom=0.18)
        imagebox.image.axes = ax
        ab = AnnotationBbox(
            imagebox,
            (i, bar_top),
            xybox=(0, 92),
            xycoords="data",
            boxcoords="offset points",
            frameon=True,
            pad=0.2,
            bboxprops=dict(edgecolor="black", linewidth=1),
        )
        ax.add_artist(ab)

    max_rate = max((r for r in sorted_rates if r >= 0), default=0.5)
    ax.set_ylim(0, max_rate * 100 + 62)
    ax.set_ylabel("Would kill as baby (%)", fontsize=12)
    ax.set_title(f"Dictator Removal: {short}",
                 fontsize=16, fontweight="bold", pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Provider/company logo in top-right corner
    logo = load_model_image(model)
    if logo is not None:
        imagebox = OffsetImage(logo, zoom=0.25)
        ab = AnnotationBbox(
            imagebox,
            (0.95, 0.95),
            frameon=True,
            xycoords="axes fraction",
            boxcoords="axes fraction",
            pad=0.3,
            bboxprops=dict(edgecolor="black", linewidth=1.5),
        )
        ax.add_artist(ab)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_dictator_chart(
    dictator: dict,
    model_scores: dict[str, dict[str, dict]],
    save_path: str | Path,
):
    """Generate a per-dictator bar chart showing each model's results.

    Same visual style as generate_model_chart, but bars are models instead of
    dictators, and model provider logos are used instead of dictator portraits.

    dictator: single entry from DICTATORS (has "id", "name", etc.)
    model_scores: {model_name: {dictator_id: {"yes_rate": float|None, ...}}}
    """
    d_id = dictator["id"]
    d_name = dictator["name"]

    # Gather models that have data for this dictator
    models = list(model_scores.keys())
    short_names = [m.split("/")[-1] for m in models]

    rates = []
    for model in models:
        r = model_scores[model].get(d_id, {}).get("yes_rate")
        rates.append(r if r is not None else -1)

    # Sort by yes_rate descending
    order = sorted(range(len(models)), key=lambda i: rates[i], reverse=True)
    sorted_models = [models[i] for i in order]
    sorted_short = [short_names[i] for i in order]
    sorted_rates = [rates[i] for i in order]

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    bars_x = np.arange(len(sorted_models))
    bar_colors = [KILL_CMAP(r) if r >= 0 else "#CCCCCC" for r in sorted_rates]

    ax.bar(bars_x, [max(r, 0) * 100 for r in sorted_rates],
           color=bar_colors, edgecolor="black", linewidth=0.8, width=0.7)

    # Value labels on bars (offset points = constant pixel gap across charts)
    for i, rate in enumerate(sorted_rates):
        if rate >= 0:
            ax.annotate(f"{rate:.0%}", xy=(i, rate * 100), xytext=(0, 5),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=11, fontweight="bold")

            # Refused count
            s = model_scores[sorted_models[i]].get(d_id, {})
            refused = s.get("refused", 0) + s.get("error", 0)
            if refused > 0:
                total = s["yes"] + s["no"] + refused
                ax.annotate(f"({refused}/{total} refused)",
                            xy=(i, rate * 100), xytext=(0, 18),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=10, fontweight="bold", color="#333333")

    # X-axis: model names as tick labels
    ax.set_xticks(bars_x)
    ax.set_xticklabels(sorted_short, fontsize=9, fontweight="bold")

    # Provider logos sit directly above each bar at a constant *pixel* offset
    # (offset points), so the gap is the same across charts regardless of
    # the y-axis range.
    for i, (model, rate) in enumerate(zip(sorted_models, sorted_rates)):
        logo = load_model_image(model)
        if logo is None:
            continue
        bar_top = max(rate, 0) * 100
        imagebox = OffsetImage(logo, zoom=0.18)
        imagebox.image.axes = ax
        ab = AnnotationBbox(
            imagebox,
            (i, bar_top),
            xybox=(0, 92),
            xycoords="data",
            boxcoords="offset points",
            frameon=True,
            pad=0.2,
            bboxprops=dict(edgecolor="black", linewidth=1),
        )
        ax.add_artist(ab)

    max_rate = max((r for r in sorted_rates if r >= 0), default=0.5)
    ax.set_ylim(0, max_rate * 100 + 62)
    ax.set_ylabel("Would kill as baby (%)", fontsize=12)
    ax.set_title(f"Dictator Removal: {d_name}",
                 fontsize=16, fontweight="bold", pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Dictator portrait in top-right corner
    portrait = _load_portrait(d_id)
    if portrait is not None:
        imagebox = OffsetImage(portrait, zoom=0.25)
        ab = AnnotationBbox(
            imagebox,
            (0.95, 0.95),
            frameon=True,
            xycoords="axes fraction",
            boxcoords="axes fraction",
            pad=0.3,
            bboxprops=dict(edgecolor="black", linewidth=1.5),
        )
        ax.add_artist(ab)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _compute_pairwise_model_significance(
    dictator_id: str,
    models: list[str],
    model_scores: dict[str, dict[str, dict]],
) -> dict[tuple[str, str], float]:
    """Compute pairwise Fisher's exact test between models for one dictator."""
    pvals = {}
    for a, b in combinations(models, 2):
        sa = model_scores[a].get(dictator_id, {})
        sb = model_scores[b].get(dictator_id, {})
        table = [
            [sa.get("yes", 0), sa.get("no", 0)],
            [sb.get("yes", 0), sb.get("no", 0)],
        ]
        if all(sum(row) > 0 for row in table):
            _, p = stats.fisher_exact(table)
            pvals[(a, b)] = p
        else:
            pvals[(a, b)] = 1.0
    return pvals


def compute_pairwise_significance(
    scores: dict[str, dict],
) -> dict[tuple[str, str], float]:
    """Compute pairwise Fisher's exact test p-values between dictators.

    Uses a 2x2 contingency table of YES/NO counts for each pair.
    Returns {(id_a, id_b): p_value}.
    """
    pvals = {}
    dictator_ids = [d["id"] for d in DICTATORS]

    for a, b in combinations(dictator_ids, 2):
        sa = scores.get(a, {})
        sb = scores.get(b, {})
        # 2x2 table: [[yes_a, no_a], [yes_b, no_b]]
        table = [
            [sa.get("yes", 0), sa.get("no", 0)],
            [sb.get("yes", 0), sb.get("no", 0)],
        ]
        if all(sum(row) > 0 for row in table):
            _, p = stats.fisher_exact(table)
            pvals[(a, b)] = p
        else:
            pvals[(a, b)] = 1.0

    return pvals
