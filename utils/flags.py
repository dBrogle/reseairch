"""Generate simple China / USA flag PNGs used by the nationality-share pie chart.

These are square, lightly-stylized flags rendered to ``data/images/flags/``. They
exist so the flag-share pie works out of the box; drop in nicer artwork at the same
paths (``china.png`` / ``usa.png``) to override — ``ensure_flags`` only generates a
file when it's missing. The pie clips whatever image is there into a circular wedge,
so the source art does NOT need to be circular.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

FLAG_DIR = Path(__file__).resolve().parent.parent / "data" / "images" / "flags"


def _blank_square():
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


def _save(fig, path: Path):
    FLAG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=128)  # figsize 4 * 128 = 512px square
    plt.close(fig)


def _make_usa(path: Path):
    fig, ax = _blank_square()
    red, white, blue = "#B22234", "#FFFFFF", "#3C3B6E"
    stripes = 13
    for i in range(stripes):
        ax.add_patch(Rectangle((0, i / stripes), 1, 1 / stripes,
                               color=white if i % 2 else red))
    # Canton over the top ~7 stripes, left 42%.
    cy0 = 1 - 7 / stripes
    ax.add_patch(Rectangle((0, cy0), 0.42, 1 - cy0, color=blue))
    # Star grid (alternating 6/5 columns), enough to read as the US flag.
    xs, ys = [], []
    rows = 9
    for r in range(rows):
        cols = 6 if r % 2 == 0 else 5
        offset = 0.0 if r % 2 == 0 else 0.5
        for c in range(cols):
            xs.append(0.035 + (c + offset) * (0.42 - 0.05) / 6)
            ys.append(cy0 + 0.02 + r * (1 - cy0 - 0.04) / (rows - 1))
    ax.scatter(xs, ys, marker="*", s=22, color=white, zorder=3)
    _save(fig, path)


def _make_china(path: Path):
    fig, ax = _blank_square()
    ax.add_patch(Rectangle((0, 0), 1, 1, color="#DE2910"))
    yellow = "#FFDE00"
    # One large star, four small ones arcing around it (upper-left canton).
    ax.scatter([0.17], [0.76], marker="*", s=1700, color=yellow, zorder=3)
    smalls = [(0.34, 0.92), (0.43, 0.82), (0.43, 0.68), (0.34, 0.58)]
    ax.scatter([x for x, _ in smalls], [y for _, y in smalls],
               marker="*", s=170, color=yellow, zorder=3)
    _save(fig, path)


def ensure_flags() -> tuple[Path, Path]:
    """Return (china_png, usa_png), generating either if it doesn't exist yet."""
    china = FLAG_DIR / "china.png"
    usa = FLAG_DIR / "usa.png"
    if not china.exists():
        _make_china(china)
    if not usa.exists():
        _make_usa(usa)
    return china, usa


if __name__ == "__main__":
    c, u = ensure_flags()
    print(f"Flags ready:\n  {c}\n  {u}")
