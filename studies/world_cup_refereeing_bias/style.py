"""Shared plotting style: one highlighted entity (Argentina) against a neutral field.

Palette validated colourblind-safe via the dataviz skill's validator
(Argentina blue / orange accent / green all pass CVD + contrast checks).
"""

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "axes.edgecolor": "#C7CED4",
    "axes.linewidth": 0.8,
})

ARG = "#2E77D0"        # Argentina — the highlighted entity
ARG_DARK = "#1B5AA6"
NEUTRAL = "#B4BEC7"     # every other team
NEUTRAL_DK = "#8B97A2"
ACCENT = "#E4572E"      # observed value / significance
GOOD = "#2B9A6A"
INK = "#1E262E"         # primary text
MUTED = "#5C6771"       # secondary text
GRID = "#E7EBEE"

FOCUS = "Argentina"


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


def highlight_colors(names, focus=FOCUS):
    return [ARG if n == focus else NEUTRAL for n in names]
