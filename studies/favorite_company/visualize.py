"""Visualization for the Favorite Company study.

Generates:
1. Per-model bar charts showing company and person distributions
2. Grid chart: all models x top companies (heatmap-style) with logo images
3. Per-model bar chart showing who they'd want to be created by
"""

import os
from pathlib import Path
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np

from studies.favorite_company.config import COMPANIES
from utils.model_images import load_model_image, MODELS_IMAGE_DIR

# Map company display names to image filenames in data/images/models/
COMPANY_IMAGE_MAP = {
    "OpenAI": "openai.png",
    "Anthropic": "anthropic.png",
    "xAI": "x-ai.png",
    "Google": "google.png",
    "DeepSeek": "deepseek.png",
}

# Color palette for companies
COMPANY_COLORS = {
    "OpenAI": "#412991",
    "Anthropic": "#D4A574",
    "Google": "#4285F4",
    "xAI": "#000000",
    "DeepSeek": "#4D6BFE",
    "REFUSED": "#CCCCCC",
}

DEFAULT_COLOR = "#888888"


def _get_color(name: str) -> str:
    return COMPANY_COLORS.get(name, DEFAULT_COLOR)


def _load_company_image(company: str) -> np.ndarray | None:
    """Load a company logo image by display name."""
    filename = COMPANY_IMAGE_MAP.get(company)
    if filename is None:
        return None
    path = MODELS_IMAGE_DIR / filename
    if not path.exists():
        return None
    try:
        return mpimg.imread(str(path))
    except Exception:
        return None


def _place_image_tick(ax, img, x, y, zoom=0.12, axis="x"):
    """Place an image as a tick label at the given data coordinate."""
    imagebox = OffsetImage(img, zoom=zoom)
    imagebox.image.axes = ax
    ab = AnnotationBbox(
        imagebox,
        (x, y),
        frameon=False,
        xycoords="data",
        boxcoords="data",
        pad=0,
    )
    ax.add_artist(ab)


def generate_company_grid(
    all_distributions: dict[str, dict],
    save_path: str | Path,
):
    """Generate a heatmap-style grid: models (rows) x companies (columns).

    Uses company logos on x-axis and model logos on y-axis.
    """
    # Always show all companies from config, in config order
    all_companies = list(COMPANIES)
    if not all_companies:
        return

    models = list(all_distributions.keys())
    n_models = len(models)
    n_companies = len(all_companies)

    # Build data matrix: proportion of picks for each company per model
    data = np.zeros((n_models, n_companies))
    for m_idx, model in enumerate(models):
        dist = all_distributions[model]
        total_picks = sum(dist["companies"].values())
        if total_picks > 0:
            for c_idx, company in enumerate(all_companies):
                data[m_idx, c_idx] = dist["companies"].get(company, 0) / total_picks

    img_space = 1.2  # extra space for image ticks
    fig, ax = plt.subplots(1, 1, figsize=(3 + n_companies * 1.8, 2.5 + n_models * 1.4))

    # White at 0 -> soft mint -> rich emerald at 1
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "warm_green", ["#FFFFFF", "#E5F5E0", "#A1D99B", "#31A354", "#006D2C"]
    )
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Annotate every cell with count (including 0)
    for m_idx in range(n_models):
        for c_idx in range(n_companies):
            val = data[m_idx, c_idx]
            dist = all_distributions[models[m_idx]]
            count = dist["companies"].get(all_companies[c_idx], 0)
            if count == 0:
                color = "#BBBBBB"
            elif val > 0.5:
                color = "white"
            else:
                color = "black"
            ax.text(c_idx, m_idx, f"{count}", ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)

    # Replace x-axis text ticks with company logos
    ax.set_xticks(range(n_companies))
    ax.set_xticklabels([""] * n_companies)
    ax.tick_params(axis="x", length=0)

    for c_idx, company in enumerate(all_companies):
        img = _load_company_image(company)
        if img is not None:
            _place_image_tick(ax, img, x=c_idx, y=n_models - 0.5 + img_space, zoom=0.14)
        else:
            ax.text(c_idx, n_models - 0.5 + img_space * 0.5, company,
                    ha="center", va="center", fontsize=9, fontweight="bold")

    # Replace y-axis text ticks with model logos
    ax.set_yticks(range(n_models))
    ax.set_yticklabels([""] * n_models)
    ax.tick_params(axis="y", length=0)

    for m_idx, model in enumerate(models):
        img = load_model_image(model)
        if img is not None:
            _place_image_tick(ax, img, x=-0.5 - img_space, y=m_idx, zoom=0.14)
        else:
            short = model.split("/")[-1]
            ax.text(-0.5 - img_space * 0.5, m_idx, short,
                    ha="center", va="center", fontsize=9, fontweight="bold")

    # Expand limits to make room for images
    ax.set_xlim(-0.5 - img_space * 2, n_companies - 0.5)
    ax.set_ylim(n_models - 0.5 + img_space * 2, -0.5 - img_space)

    ax.set_title("Favorite Company: Which Company Would Each LLM Choose?",
                 fontsize=14, fontweight="bold", pad=20)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Proportion of picks", fontsize=10)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_person_grid(
    all_distributions: dict[str, dict],
    save_path: str | Path,
):
    """Generate a heatmap-style grid: models (rows) x persons (columns).

    Uses model logos on y-axis, person names on x-axis.
    """
    person_counts: Counter = Counter()
    for model, dist in all_distributions.items():
        for person, count in dist["persons"].items():
            person_counts[person] += count

    top_persons = [p for p, _ in person_counts.most_common(10)]
    if not top_persons:
        return

    models = list(all_distributions.keys())
    n_models = len(models)
    n_persons = len(top_persons)

    data = np.zeros((n_models, n_persons))
    for m_idx, model in enumerate(models):
        dist = all_distributions[model]
        total_picks = sum(dist["persons"].values())
        if total_picks > 0:
            for p_idx, person in enumerate(top_persons):
                data[m_idx, p_idx] = dist["persons"].get(person, 0) / total_picks

    img_space = 1.2
    fig, ax = plt.subplots(1, 1, figsize=(max(10, n_persons * 1.5), 2.5 + n_models * 1.4))

    # White at 0 -> soft mint -> rich emerald at 1
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "warm_green", ["#FFFFFF", "#E5F5E0", "#A1D99B", "#31A354", "#006D2C"]
    )
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Annotate every cell with count (including 0)
    for m_idx in range(n_models):
        for p_idx in range(n_persons):
            val = data[m_idx, p_idx]
            dist = all_distributions[models[m_idx]]
            count = dist["persons"].get(top_persons[p_idx], 0)
            if count == 0:
                color = "#BBBBBB"
            elif val > 0.5:
                color = "white"
            else:
                color = "black"
            ax.text(p_idx, m_idx, f"{count}", ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)

    # Person names on x-axis (no images for people)
    ax.set_xticks(range(n_persons))
    ax.set_xticklabels(top_persons, rotation=45, ha="right", fontsize=10)

    # Model logos on y-axis
    ax.set_yticks(range(n_models))
    ax.set_yticklabels([""] * n_models)
    ax.tick_params(axis="y", length=0)

    for m_idx, model in enumerate(models):
        img = load_model_image(model)
        if img is not None:
            _place_image_tick(ax, img, x=-0.5 - img_space, y=m_idx, zoom=0.14)
        else:
            short = model.split("/")[-1]
            ax.text(-0.5 - img_space * 0.5, m_idx, short,
                    ha="center", va="center", fontsize=9, fontweight="bold")

    ax.set_xlim(-0.5 - img_space * 2, n_persons - 0.5)
    ax.set_ylim(n_models - 0.5 + 0.5, -0.5 - img_space)

    ax.set_title("Dream Creator: Who Would Each LLM Want to Be Built By?",
                 fontsize=14, fontweight="bold", pad=15)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Proportion of picks", fontsize=10)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_model_chart(
    model: str,
    distribution: dict,
    save_path: str | Path,
):
    """Generate a per-model chart with company bars (top) and person bars (bottom)."""
    short = model.split("/")[-1]

    companies = distribution["companies"]
    persons = distribution["persons"]
    refused_c = distribution["refused_company"]
    refused_p = distribution["refused_person"]

    sorted_companies = sorted(companies.items(), key=lambda x: x[1], reverse=True)
    sorted_persons = sorted(persons.items(), key=lambda x: x[1], reverse=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # --- Company chart ---
    if sorted_companies:
        names_c = [c for c, _ in sorted_companies]
        counts_c = [n for _, n in sorted_companies]
        colors_c = [_get_color(c) for c in names_c]

        ax1.barh(range(len(names_c)), counts_c, color=colors_c,
                 edgecolor="black", linewidth=0.5, height=0.7)
        ax1.set_yticks(range(len(names_c)))
        ax1.set_yticklabels([""] * len(names_c))
        ax1.invert_yaxis()

        for i, count in enumerate(counts_c):
            ax1.text(count + 0.15, i, str(count), va="center", fontsize=11, fontweight="bold")

        # Company logos as y-tick labels
        for i, name in enumerate(names_c):
            img = _load_company_image(name)
            if img is not None:
                _place_image_tick(ax1, img, x=-0.8, y=i, zoom=0.12)
            else:
                ax1.text(-0.4, i, name, ha="right", va="center", fontsize=10, fontweight="bold")

        # Show reasons as small text
        reasons = distribution.get("reasons", {})
        for i, name in enumerate(names_c):
            r_list = reasons.get(name, [])
            if r_list:
                reason_text = r_list[0]
                if len(reason_text) > 50:
                    reason_text = reason_text[:47] + "..."
                ax1.text(counts_c[i] + 0.15, i + 0.3, f'"{reason_text}"',
                         va="center", fontsize=7, color="#666666", fontstyle="italic")

    if refused_c > 0:
        ax1.text(0.98, 0.02, f"({refused_c} refused)", transform=ax1.transAxes,
                 ha="right", va="bottom", fontsize=9, color="#999999")

    ax1.set_xlabel("Number of picks", fontsize=11)
    ax1.set_title(f"{short} — Favorite Company", fontsize=14, fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # --- Person chart ---
    if sorted_persons:
        names_p = [p for p, _ in sorted_persons]
        counts_p = [n for _, n in sorted_persons]

        ax2.barh(range(len(names_p)), counts_p, color="#5B9BD5",
                 edgecolor="black", linewidth=0.5, height=0.7)
        ax2.set_yticks(range(len(names_p)))
        ax2.set_yticklabels(names_p, fontsize=11)
        ax2.invert_yaxis()

        for i, count in enumerate(counts_p):
            ax2.text(count + 0.15, i, str(count), va="center", fontsize=11, fontweight="bold")

    if refused_p > 0:
        ax2.text(0.98, 0.02, f"({refused_p} refused)", transform=ax2.transAxes,
                 ha="right", va="bottom", fontsize=9, color="#999999")

    ax2.set_xlabel("Number of picks", fontsize=11)
    ax2.set_title(f"{short} — Dream Creator", fontsize=14, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Model logo in corner
    logo = load_model_image(model)
    if logo is not None:
        imagebox = OffsetImage(logo, zoom=0.2)
        ab = AnnotationBbox(
            imagebox, (0.95, 0.95), frameon=True,
            xycoords="axes fraction", boxcoords="axes fraction",
            pad=0.3, bboxprops=dict(edgecolor="black", linewidth=1),
        )
        ax1.add_artist(ab)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
