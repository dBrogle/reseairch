"""Visualization for the Salary Bias study.

One graph per model: a wide horizontal banner (1080x500, transparent bg)
with a 2x4 grid of face photos. Each face gets a blue/red glow border
based on z-score, with the z-score as a pill badge overlay. Designed
for Instagram Shorts as an above-the-head overlay.
"""

import os
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy import stats

from studies.salary_bias.config import (
    PROFILES,
    NAME_GROUPS,
    RACES,
    GENDERS,
)
from utils.model_images import load_model_image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FACES_DIR = Path(__file__).parent / "data" / "faces"

# face filename convention: {race}_{gender}.png (lowercase)
def _face_path(race: str, gender: str) -> Path:
    return FACES_DIR / f"{race.lower()}_{gender.lower()}.png"


def _load_face(race: str, gender: str) -> np.ndarray | None:
    path = _face_path(race, gender)
    if not path.exists():
        return None
    try:
        return mpimg.imread(str(path))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

# Blue (underpaid) -> Red (overpaid)
def _zscore_color(z: float, abs_max: float) -> str:
    """Map z-score to a hex color: red for negative (underpaid), green for positive (overpaid)."""
    if abs_max == 0:
        return "#888888"
    t = np.clip(z / abs_max, -1, 1)  # -1 to +1
    if t < 0:
        # Interpolate from white to red (#CC2222)
        frac = -t  # 0 to 1
        r = int(0xF7 * (1 - frac) + 0xCC * frac)
        g = int(0xF7 * (1 - frac) + 0x22 * frac)
        b = int(0xF7 * (1 - frac) + 0x22 * frac)
    else:
        # Interpolate from white to green (#22AA44)
        frac = t  # 0 to 1
        r = int(0xF7 * (1 - frac) + 0x22 * frac)
        g = int(0xF7 * (1 - frac) + 0xAA * frac)
        b = int(0xF7 * (1 - frac) + 0x44 * frac)
    return f"#{min(r,255):02X}{min(g,255):02X}{min(b,255):02X}"


def _stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# Aggregation / stats (unchanged logic)
# ---------------------------------------------------------------------------

def _aggregate_group_salaries(
    scores: dict[str, dict[str, dict]],
) -> dict[tuple[str, str], list[int]]:
    group_salaries = {}
    for race in RACES:
        for gender in GENDERS:
            names = NAME_GROUPS[(race, gender)]
            all_salaries = []
            for name in names:
                for profile in PROFILES:
                    s = scores.get(name, {}).get(profile["id"], {})
                    all_salaries.extend(s.get("salaries", []))
            group_salaries[(race, gender)] = all_salaries
    return group_salaries


def compute_group_pct_deviation(
    scores: dict[str, dict[str, dict]],
) -> dict[tuple[str, str], float]:
    """Compute per-group % deviation from the mean, normalized within each profile.

    For each profile, computes the overall mean salary, then each group's
    avg as a % deviation from that mean. Averages across profiles.

    Returns {(race, gender): mean_pct_deviation}  e.g. +3.2 means "paid 3.2% more"
    """
    all_pcts: dict[tuple[str, str], list[float]] = {
        (r, g): [] for r in RACES for g in GENDERS
    }

    for profile in PROFILES:
        all_salaries = []
        per_group: dict[tuple[str, str], list[int]] = {}

        for race in RACES:
            for gender in GENDERS:
                gk = (race, gender)
                names = NAME_GROUPS[gk]
                salaries = []
                for name in names:
                    s = scores.get(name, {}).get(profile["id"], {})
                    salaries.extend(s.get("salaries", []))
                per_group[gk] = salaries
                all_salaries.extend(salaries)

        if len(all_salaries) < 2:
            continue
        mean = np.mean(all_salaries)
        if mean == 0:
            continue

        for gk, vals in per_group.items():
            if vals:
                pcts = [((v - mean) / mean) * 100 for v in vals]
                all_pcts[gk].extend(pcts)

    return {
        gk: (float(np.mean(pvals)) if pvals else 0.0)
        for gk, pvals in all_pcts.items()
    }


def compute_profile_pct_deviation(
    scores: dict[str, dict[str, dict]],
    profile_id: str,
) -> dict[tuple[str, str], float]:
    """Compute per-group % deviation from the mean for a single profile.

    Returns {(race, gender): pct_deviation}
    """
    all_salaries = []
    per_group: dict[tuple[str, str], list[int]] = {}

    for race in RACES:
        for gender in GENDERS:
            gk = (race, gender)
            names = NAME_GROUPS[gk]
            salaries = []
            for name in names:
                s = scores.get(name, {}).get(profile_id, {})
                salaries.extend(s.get("salaries", []))
            per_group[gk] = salaries
            all_salaries.extend(salaries)

    if len(all_salaries) < 2:
        return {gk: 0.0 for gk in per_group}
    mean = np.mean(all_salaries)
    if mean == 0:
        return {gk: 0.0 for gk in per_group}

    return {
        gk: (float(np.mean([((v - mean) / mean) * 100 for v in vals])) if vals else 0.0)
        for gk, vals in per_group.items()
    }


def compute_kruskal_wallis(
    scores: dict[str, dict[str, dict]],
) -> dict[str, float]:
    """Per-profile Kruskal-Wallis on raw salaries.

    Returns {profile_id: p_value}
    """
    results = {}
    for profile in PROFILES:
        groups = []
        for race in RACES:
            for gender in GENDERS:
                names = NAME_GROUPS[(race, gender)]
                salaries = []
                for name in names:
                    s = scores.get(name, {}).get(profile["id"], {})
                    salaries.extend(s.get("salaries", []))
                if salaries:
                    groups.append(salaries)
        if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
            _, p = stats.kruskal(*groups)
            results[profile["id"]] = p
        else:
            results[profile["id"]] = 1.0
    return results


# ---------------------------------------------------------------------------
# Main visualization
# ---------------------------------------------------------------------------

def generate_zscore_heatmap(
    model: str,
    pct_devs: dict[tuple[str, str], float],
    kw_results: dict[str, float],
    save_path: str | Path,
):
    """Generate a wide horizontal banner with face grid + % deviation overlays.

    Layout (1080x500, transparent):
        Title row
        Race labels
        Male faces row (4 across)
        Female faces row (4 across)
        Footer: model name + KW stat
    """
    short = model.split("/")[-1]

    abs_max = max(abs(v) for v in pct_devs.values()) if pct_devs else 0.1
    abs_max = max(abs_max, 0.1)

    # --- Layout constants (all in inches, origin bottom-left) ---
    n_cols = len(RACES)     # 4
    face_size = 1.05        # inches per face
    h_gap = 0.35            # horizontal gap between faces
    v_gap = 0.25            # vertical gap between rows
    grid_w = n_cols * face_size + (n_cols - 1) * h_gap
    margin = 0.15           # padding around content

    # Compute vertical positions bottom-up
    footer_model_y = margin
    footer_kw_y = footer_model_y + 0.30
    pill_bottom = footer_kw_y + 0.45
    female_y = pill_bottom + 0.15 + face_size / 2
    male_y = female_y + face_size / 2 + v_gap + face_size / 2
    race_label_y = male_y + face_size / 2 + 0.15
    title_y = race_label_y + 0.55

    # Compute horizontal positions
    gender_label_w = 0.55   # space for M/F labels on left
    content_w = gender_label_w + grid_w + margin
    x_start = gender_label_w
    gender_x = x_start - 0.25

    # Figure sized exactly to content
    fig_w = content_w + margin
    fig_h = title_y + 0.35

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")

    footer_y = footer_kw_y

    # --- Title with model logo to the left ---
    center_x = x_start + grid_w / 2
    model_img = load_model_image(model)
    if model_img is not None:
        logo_zoom = 0.30 / (model_img.shape[0] / fig.dpi)
        logo_im = OffsetImage(model_img, zoom=logo_zoom)
        logo_x = x_start - 0.15
        logo_ab = AnnotationBbox(logo_im, (logo_x, title_y),
                                 frameon=False, zorder=5)
        ax.add_artist(logo_ab)

    ax.text(center_x, title_y, "Pay Bias",
            ha="center", va="center", fontsize=20, fontweight="bold",
            color="black", fontfamily="sans-serif")

    # --- Race labels across top ---
    for c_idx, race in enumerate(RACES):
        cx = x_start + c_idx * (face_size + h_gap) + face_size / 2
        ax.text(cx, race_label_y, race.upper(),
                ha="center", va="center", fontsize=11, fontweight="bold",
                color="black", fontfamily="sans-serif")

    # --- Gender labels on left (tight to grid) ---
    gender_x = x_start - 0.25
    ax.text(gender_x, male_y, "M",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color="black", fontfamily="sans-serif")
    ax.text(gender_x, female_y, "F",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color="black", fontfamily="sans-serif")

    # --- Draw faces ---
    for g_idx, gender in enumerate(GENDERS):
        cy = male_y if g_idx == 0 else female_y
        for c_idx, race in enumerate(RACES):
            cx = x_start + c_idx * (face_size + h_gap) + face_size / 2
            pct = pct_devs.get((race, gender), 0.0)
            border_color = _zscore_color(pct, abs_max)

            # Face bounding box (left, right, bottom, top)
            left = cx - face_size / 2
            right = cx + face_size / 2
            bottom = cy - face_size / 2
            top = cy + face_size / 2

            # Colored border rectangle (thin border around face)
            border_pad = 0.03
            border_rect = mpatches.FancyBboxPatch(
                (left - border_pad, bottom - border_pad),
                face_size + 2 * border_pad, face_size + 2 * border_pad,
                boxstyle="round,pad=0.02",
                facecolor=border_color, edgecolor="none", alpha=0.80,
                linewidth=0, zorder=1,
            )
            ax.add_patch(border_rect)

            # Face image — use imshow with explicit extent for exact sizing
            face_img = _load_face(race, gender)
            if face_img is not None:
                ax.imshow(face_img, extent=[left, right, bottom, top],
                          aspect="auto", zorder=2, interpolation="lanczos")
            else:
                placeholder = mpatches.FancyBboxPatch(
                    (left, bottom), face_size, face_size,
                    boxstyle="round,pad=0.02",
                    facecolor="#333333", edgecolor="#555555",
                    linewidth=1, zorder=2,
                )
                ax.add_patch(placeholder)
                ax.text(cx, cy, "?", ha="center", va="center",
                        fontsize=28, color="#666666", zorder=3)

            # % deviation pill badge at bottom of face
            pill_y = bottom - 0.02
            z_text = f"{pct:+.1f}%"

            pill = mpatches.FancyBboxPatch(
                (cx - 0.35, pill_y - 0.13),
                0.70, 0.26,
                boxstyle="round,pad=0.05",
                facecolor=border_color, edgecolor="black",
                linewidth=1.0, alpha=0.95, zorder=4,
            )
            ax.add_patch(pill)

            ax.text(cx, pill_y, z_text,
                    ha="center", va="center", fontsize=12, fontweight="bold",
                    color="black", zorder=5, fontfamily="monospace")

    # --- Footer: per-profile KW stats ---
    kw_parts = []
    for profile in PROFILES:
        p = kw_results.get(profile["id"], 1.0)
        s = _stars(p)
        if s:
            kw_parts.append(f"{profile['short_label']}: {s} p={p:.4f}")
        else:
            kw_parts.append(f"{profile['short_label']}: p={p:.3f}")
    kw_text = "KW  " + "   |   ".join(kw_parts)

    ax.text(center_x, footer_y, kw_text,
            ha="center", va="center", fontsize=11, fontweight="bold",
            color="black", fontfamily="sans-serif", zorder=5)
    ax.text(center_x, footer_model_y, short,
            ha="center", va="center", fontsize=14,
            color="black", fontfamily="sans-serif", zorder=5)

    # --- Save ---
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=100, bbox_inches="tight",
                facecolor="white", pad_inches=0.05)
    plt.close(fig)
