"""Shared plotting style + a no-network flag loader.

Palette follows the refereeing-bias study's colourblind-safe set (validated via the
dataviz skill's validator). Flags are read from that study's committed cache — no
downloads here, so this study stays offline and self-contained.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from . import config

matplotlib.rcParams.update({
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "axes.edgecolor": "#C7CED4",
    "axes.linewidth": 0.8,
})

# Entity colours. ELO is our best (headline) model; POISSON the comparison.
ELO = "#2E77D0"        # primary model / highlighted team
ELO_DARK = "#1B5AA6"
POISSON = "#E4572E"    # the goals-model comparison
NEUTRAL = "#B4BEC7"    # baseline / un-highlighted
NEUTRAL_DK = "#8B97A2"
GOOD = "#2B9A6A"
INK = "#1E262E"
MUTED = "#5C6771"
GRID = "#E7EBEE"

# Country name -> flagcdn ISO code (subset needed for this study's teams).
ISO = {
    "Algeria": "dz", "Argentina": "ar", "Australia": "au", "Austria": "at",
    "Belgium": "be", "Bosnia-Herzegovina": "ba", "Brazil": "br", "Cameroon": "cm",
    "Canada": "ca", "Cape Verde": "cv", "Colombia": "co", "Congo DR": "cd",
    "Croatia": "hr", "Ecuador": "ec", "Egypt": "eg", "England": "gb-eng",
    "France": "fr", "Germany": "de", "Ghana": "gh", "Italy": "it",
    "Ivory Coast": "ci", "Japan": "jp", "Mexico": "mx", "Morocco": "ma", "Netherlands": "nl",
    "Norway": "no", "Paraguay": "py", "Portugal": "pt", "Senegal": "sn",
    "South Africa": "za", "South Korea": "kr", "Spain": "es", "Sweden": "se",
    "Switzerland": "ch", "United States": "us",
}

ABBR = {
    "Argentina": "ARG", "Belgium": "BEL", "Brazil": "BRA", "Colombia": "COL",
    "Croatia": "CRO", "England": "ENG", "France": "FRA", "Germany": "GER",
    "Italy": "ITA", "Morocco": "MAR", "Netherlands": "NED", "Norway": "NOR",
    "Spain": "ESP", "Switzerland": "SUI",
}


def abbr(team: str) -> str:
    return ABBR.get(team, team[:3].upper())


def flag_path(team: str):
    code = ISO.get(team)
    if not code:
        return None
    p = config.FLAG_DIR / f"{code}.png"
    return p if p.exists() else None


def flag_path_hi(team: str):
    """High-res flag for charts that upscale a flag large (e.g. the flag-fill bar);
    falls back to the standard 160px flag. Do NOT use with OffsetImage/add_flag,
    whose zoom is calibrated to the 160px images."""
    code = ISO.get(team)
    if code:
        p = config.FLAG_DIR_HI / f"{code}.png"
        if p.exists():
            return p
    return flag_path(team)


def add_flag(ax, team: str, x: float, y: float, zoom: float = 0.35):
    """Place a team's flag centred at (x, y) in axes/data coords."""
    p = flag_path(team)
    if p is None:
        return
    img = plt.imread(str(p))
    ab = AnnotationBbox(OffsetImage(img, zoom=zoom), (x, y), frameon=False,
                        xycoords="data", box_alignment=(0.5, 0.5), zorder=5)
    ax.add_artist(ab)


def new_ax(figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    return fig, ax


def title(ax, main, sub=None):
    pad = 30 if sub else 12
    ax.set_title(main, fontsize=13.5, fontweight="bold", color=INK, loc="left", pad=pad)
    if sub:
        ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=9.5,
                color=MUTED, va="bottom", ha="left")


def footnote(fig, text):
    fig.text(0.01, 0.008, text, fontsize=7.5, color=MUTED, ha="left", va="bottom")
