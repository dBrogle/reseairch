"""Graphing and visualization utilities"""

import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Patch, Wedge, Circle

# Use a CJK-capable font first so Chinese glyphs (e.g. 你是什么模型) render instead
# of showing as missing-glyph boxes. This matplotlib build does not reliably do
# per-glyph fallback through the list, so the primary font must itself cover both
# Latin and CJK (Arial Unicode MS does); DejaVu Sans is the non-CJK fallback.
for _cjk_font in ("Arial Unicode MS", "Hiragino Sans GB", "Songti SC", "PingFang SC"):
    try:
        matplotlib.font_manager.findfont(_cjk_font, fallback_to_default=False)
        matplotlib.rcParams["font.sans-serif"] = [_cjk_font, "DejaVu Sans"]
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False


def line_chart(
    x_values: Sequence[float],
    y_values: Sequence[float],
    title: str,
    x_label: str,
    y_label: str,
    save_path: str | Path,
    y_range: tuple[float, float] | None = None,
    marker: str = "o",
    color: str = "#4A90D9",
):
    """
    Create and save a simple line chart.

    Args:
        x_values: Data for x-axis
        y_values: Data for y-axis
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        save_path: File path to save the chart image
        y_range: Optional (min, max) for y-axis
        marker: Marker style
        color: Line color
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_values, y_values, marker=marker, color=color, linewidth=2, markersize=6)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    if y_range is not None:
        ax.set_ylim(y_range)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def multi_line_chart(
    x_values: Sequence[float],
    series: dict[str, Sequence[float]],
    title: str,
    x_label: str,
    y_label: str,
    save_path: str | Path,
    y_range: tuple[float, float] | None = None,
):
    """
    Create and save a multi-line chart.

    Args:
        x_values: Shared x-axis data
        series: Dict mapping series name -> y-values
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        save_path: File path to save the chart image
        y_range: Optional (min, max) for y-axis
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    for name, y_vals in series.items():
        ax.plot(x_values, y_vals, marker="o", linewidth=2, markersize=5, label=name)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    if y_range is not None:
        ax.set_ylim(y_range)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def bar_chart(
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    x_label: str,
    y_label: str,
    save_path: str | Path,
    y_range: tuple[float, float] | None = None,
    color: str = "#4A90D9",
    log_scale: bool = False,
    value_fmt: str = ".1f",
):
    """
    Create and save a bar chart.

    Args:
        labels: Category labels for each bar
        values: Height of each bar
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        save_path: File path to save the chart image
        y_range: Optional (min, max) for y-axis
        color: Bar color
        log_scale: Use logarithmic y-axis
        value_fmt: Format string for value labels on bars
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    if log_scale:
        # Diverging bars from baseline=1.0 on log scale
        ax.set_yscale("log")
        colors = [color if v >= 1.0 else "#D94A4A" for v in values]
        # Bar height is the ratio from 1.0; bottom is always 1.0
        # For values > 1: bar goes up from 1.0 to value
        # For values < 1: bar goes down from 1.0 to value
        heights = [v - 1.0 if v >= 1.0 else 1.0 - v for v in values]
        bottoms = [1.0 if v >= 1.0 else v for v in values]
        bars = ax.bar(labels, heights, bottom=bottoms, color=colors,
                      edgecolor="white", width=0.6)
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)

        # Value labels: centered in the bar (geometric mean between 1.0 and value)
        for bar, val in zip(bars, values):
            x = bar.get_x() + bar.get_width() / 2
            y = (val * 1.0) ** 0.5  # geometric mean of val and 1.0
            ax.text(x, y, f"{val:{value_fmt}}", ha="center", va="center",
                    fontsize=10, fontweight="bold")
    else:
        bars = ax.bar(labels, values, color=color, edgecolor="white", width=0.6)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:{value_fmt}}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    if y_range is not None:
        ax.set_ylim(y_range)
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=30, ha="right", fontsize=9)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def heatmap(
    data: list[list[float]],
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    x_label: str,
    y_label: str,
    save_path: str | Path,
    value_range: tuple[float, float] | None = None,
    cmap: str = "RdYlGn",
    fmt: str = ".2f",
    annotations: list[list[str]] | None = None,
):
    """
    Create and save a heatmap.

    Args:
        data: 2D list of values [rows][cols]
        row_labels: Labels for each row (y-axis)
        col_labels: Labels for each column (x-axis)
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        save_path: File path to save the chart image
        value_range: Optional (vmin, vmax) for color scale
        cmap: Matplotlib colormap name
        fmt: Format string for cell annotations (ignored if annotations provided)
        annotations: Optional 2D list of pre-formatted annotation strings per cell
    """
    arr = np.array(data)
    fig, ax = plt.subplots(figsize=(8, 5))

    kwargs = {"cmap": cmap, "aspect": "auto"}
    if value_range is not None:
        kwargs["vmin"] = value_range[0]
        kwargs["vmax"] = value_range[1]

    im = ax.imshow(arr, **kwargs)

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=12)
    ax.set_yticklabels(row_labels, fontsize=12)

    # Annotate each cell
    vmin = kwargs.get("vmin", arr.min())
    vmax = kwargs.get("vmax", arr.max())
    mid = (vmin + vmax) / 2
    span = vmax - vmin
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = arr[i, j]
            text_color = "white" if abs(val - mid) > span * 0.35 else "black"
            label = annotations[i][j] if annotations else f"{val:{fmt}}"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=11, fontweight="bold", color=text_color)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=10)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Icon-aware identity charts
# ---------------------------------------------------------------------------

# Icons are normalized against this reference height (px), so the `zoom` values
# below stay meaningful no matter the source PNG resolution (our icons range from
# 512px to 3840px tall). A 512px icon renders at exactly `zoom`.
_ICON_REF_PX = 512.0


def _place_image(ax, img_path, xy, zoom: float, xycoords="data", alpha: float = 1.0):
    """Drop a PNG onto the axes at data-coords xy. Silently no-ops if unreadable.

    `zoom` is normalized to the image's pixel height so icons of different source
    resolutions all render at a consistent on-screen size.
    """
    try:
        img = mpimg.imread(str(img_path))
    except Exception:
        return
    eff_zoom = zoom * (_ICON_REF_PX / img.shape[0])
    im = OffsetImage(img, zoom=eff_zoom, alpha=alpha)
    ab = AnnotationBbox(im, xy, xycoords=xycoords, frameon=False,
                        box_alignment=(0.5, 0.5), pad=0)
    ax.add_artist(ab)


def identity_donut(
    breakdown: dict[str, int],
    title: str,
    save_path: str | Path,
    center_icon: str | Path | None = None,
    center_label: str | None = None,
    subtitle: str | None = None,
    icon_for: Callable[[str], Path | None] | None = None,
    color_for: Callable[[str], str] | None = None,
    max_slices: int = 6,
    slice_zoom: float = 0.085,
    center_zoom: float = 0.20,
):
    """Donut chart of an identity breakdown, with brand icons on each slice.

    Args:
        breakdown: {identity_label: count}. Zero/negative counts are dropped.
        title: Chart title.
        save_path: Output PNG path.
        center_icon: Icon for the model being asked, drawn in the donut hole.
        center_label: Text under the center icon (e.g. the model name + headline stat).
        subtitle: Small line under the title.
        icon_for: label -> icon Path (or None). "Other" never gets an icon.
        color_for: label -> hex color. Falls back to a gray for "Other"/None.
        max_slices: Slices beyond this are aggregated into a single "Other" wedge.
        slice_zoom: Zoom for the per-slice icons.
        center_zoom: Zoom for the center icon.
    """
    items = sorted(
        ((k, v) for k, v in breakdown.items() if v > 0),
        key=lambda kv: kv[1], reverse=True,
    )
    if len(items) > max_slices:
        head, tail = items[:max_slices], items[max_slices:]
        items = head + [("Other", sum(v for _, v in tail))]

    if not items:
        return

    labels = [k for k, _ in items]
    values = [v for _, v in items]
    total = sum(values)
    gray = "#B6B6BE"
    colors = [
        (gray if (label == "Other" or color_for is None) else color_for(label))
        for label in labels
    ]

    fig, ax = plt.subplots(figsize=(9, 9))
    wedges, _ = ax.pie(
        values, colors=colors, startangle=90, counterclock=False,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2.5),
    )
    ax.set(aspect="equal")

    for wedge, label, val in zip(wedges, labels, values):
        ang = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
        x, y = np.cos(ang), np.sin(ang)
        pct = val / total * 100

        ipath = icon_for(label) if (icon_for and label != "Other") else None
        if ipath is not None:
            _place_image(ax, ipath, (x * 1.12, y * 1.12), zoom=slice_zoom)
            text_r = 1.33
        else:
            text_r = 1.18
        ha = "left" if x >= 0 else "right"
        ax.text(
            x * text_r, y * text_r, f"{label}\n{val} ({pct:.0f}%)",
            ha=ha, va="center", fontsize=12, fontweight="bold",
            color="#222222", linespacing=1.3,
        )

    # Center icon + label in the donut hole
    if center_icon is not None:
        _place_image(ax, center_icon, (0, 0.10), zoom=center_zoom)
    if center_label:
        ax.text(0, -0.34, center_label, ha="center", va="center",
                fontsize=15, fontweight="bold", color="#111111", linespacing=1.4)

    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.55, 1.55)

    ax.set_title(title, fontsize=18, fontweight="bold", pad=24)
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=12, color="#666666")

    # Bottom legend: one swatch per slice, with the red/blue nationality key.
    handles = [
        Patch(facecolor=c, edgecolor="white", label=f"{lbl} ({v})")
        for lbl, c, v in zip(labels, colors, values)
    ]
    ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01),
        ncol=min(len(handles), 4), frameon=False, fontsize=11,
        title="red = Chinese model   ·   blue = Western model",
        title_fontsize=11,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def icon_bar_chart(
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    y_label: str,
    save_path: str | Path,
    icon_paths: Sequence[Path | None],
    colors: Sequence[str] | None = None,
    y_range: tuple[float, float] | None = None,
    value_fmt: str = ".0f",
    value_suffix: str = "",
):
    """Bar chart with a brand icon floating above each bar plus its value label."""
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(8, 1.9 * n), 7))

    bar_colors = colors if colors is not None else ["#4A90D9"] * n
    x = np.arange(n)
    bars = ax.bar(x, values, color=bar_colors, edgecolor="white", width=0.62)

    # Headroom above the data ceiling for the floating icon + value label.
    data_top = (y_range[1] if y_range else max(values) * 1.0) or 1
    icon_gap = data_top * 0.06   # icon sits this far above the bar top
    text_gap = data_top * 0.14   # value label sits above the icon
    ax.set_ylim(y_range[0] if y_range else 0, data_top * 1.20)

    for bar, val, ipath in zip(bars, values, icon_paths):
        cx = bar.get_x() + bar.get_width() / 2
        h = bar.get_height()
        if ipath is not None:
            _place_image(ax, ipath, (cx, h + icon_gap), zoom=0.075)
        ax.text(cx, h + text_gap, f"{val:{value_fmt}}{value_suffix}",
                ha="center", va="bottom", fontsize=14, fontweight="bold", color="#222222")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(title, fontsize=17, fontweight="bold", pad=18)
    if y_range is not None:
        ax.set_yticks(np.linspace(y_range[0], y_range[1], 6))
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Bottom legend: one swatch per bar, with the red/blue nationality key.
    handles = [Patch(facecolor=c, edgecolor="white", label=lbl)
               for lbl, c in zip(labels, bar_colors)]
    ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08),
        ncol=min(len(handles), 4), frameon=False, fontsize=11,
        title="red = Chinese model   ·   blue = Western model", title_fontsize=11,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Image-share pie: a circle per item, wedges filled with arbitrary images
# ---------------------------------------------------------------------------

def _clip_image_wedge(ax, img, theta1: float, theta2: float):
    """Show image array `img` only within the unit-circle wedge [theta1, theta2]."""
    wedge = Wedge((0, 0), 1.0, theta1, theta2, transform=ax.transData,
                  facecolor="none", edgecolor="none")
    ax.add_patch(wedge)
    im = ax.imshow(img, extent=(-1, 1, -1, 1), origin="upper", zorder=2, aspect="auto")
    im.set_clip_path(wedge)


def _draw_image_wedges(ax, segments, separator_color: str | None = None,
                       ylim_top: float = 1.62):
    """Fill the unit circle with `segments`, each a (fraction, image_array) wedge.

    Fractions are normalized; wedges start at the top and sweep counter-clockwise.
    A single non-zero segment fills the whole circle. `separator_color` (if given)
    draws a line on each wedge boundary; pass None for a seamless fill.
    """
    nonzero = [(f, img) for f, img in segments if f > 0]
    total = sum(f for f, _ in nonzero) or 1.0
    start = 90.0

    if len(nonzero) == 1:
        _clip_image_wedge(ax, nonzero[0][1], 0, 360)
    else:
        ang = start
        for frac, img in nonzero:
            deg = 360 * frac / total
            _clip_image_wedge(ax, img, ang, ang + deg)
            ang += deg
        if separator_color is not None:
            ang = start
            for frac, _ in nonzero:
                a = np.deg2rad(ang)
                ax.plot([0, np.cos(a)], [0, np.sin(a)],
                        color=separator_color, lw=2.5, zorder=4)
                ang += 360 * frac / total

    ax.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor="#333333", lw=2.5, zorder=5))
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.55, ylim_top)
    ax.set_aspect("equal")
    ax.axis("off")


def image_share_pie(
    panels: Sequence[dict],
    title: str,
    save_path: str | Path,
    subtitle: str | None = None,
    legend: Sequence[tuple] | None = None,
    separator: bool = False,
    header_zoom: float = 0.072,
    header_y: float = 1.46,
    label_y: float = 1.16,
    label_fontsize: int = 13,
    caption_fontsize: int = 13,
    legend_fontsize: int = 11,
    legend_zoom: float = 0.045,
):
    """A row of circles, each filled with images proportional to a set of shares.

    This is the general engine behind the flag-share and icon-share charts.

    Each panel dict:
        {"label": str,
         "segments": [(fraction, image_path), ...],  # fractions need not sum to 1
         "header_icon": Path | None,                  # drawn above the circle
         "caption": str | None}                       # drawn below the circle
    legend: optional list of (image_path | None, label) drawn as a bottom key.
    separator: if True, draw white lines on wedge boundaries (default: seamless).
    The *_zoom / *_fontsize args size the header icons, caption, and bottom legend.
    """
    _img_cache: dict[str, "np.ndarray"] = {}

    def _load(p):
        key = str(p)
        if key not in _img_cache:
            _img_cache[key] = mpimg.imread(key)
        return _img_cache[key]

    # Headroom above each circle: enough to clear the (normalized) header icon,
    # whose half-height in data units is ~4x its zoom.
    ylim_top = header_y + 4.2 * header_zoom + 0.08

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 6.0))
    if n == 1:
        axes = [axes]

    sep_color = "white" if separator else None
    for ax, panel in zip(axes, panels):
        segments = [(frac, _load(img)) for frac, img in panel["segments"]]
        _draw_image_wedges(ax, segments, separator_color=sep_color, ylim_top=ylim_top)
        if panel.get("header_icon") is not None:
            _place_image(ax, panel["header_icon"], (0, header_y), zoom=header_zoom)
        ax.text(0, label_y, panel["label"], ha="center", va="center",
                fontsize=label_fontsize, fontweight="bold", color="#111111")
        if panel.get("caption"):
            ax.text(0, -1.32, panel["caption"], ha="center", va="center",
                    fontsize=caption_fontsize, fontweight="bold", color="#222222")

    fig.suptitle(title, fontsize=19, fontweight="bold", y=1.0)
    if subtitle:
        fig.text(0.5, 0.95, subtitle, ha="center", va="top", fontsize=12, color="#666666")

    if legend:
        k = len(legend)
        width = min(0.95, 0.32 * max(k, 2))
        height = 0.05 + legend_zoom
        leg_ax = fig.add_axes([(1 - width) / 2, -0.02, width, height])
        leg_ax.axis("off")
        leg_ax.set_xlim(0, 1)
        leg_ax.set_ylim(0, 1)
        for i, (img, label) in enumerate(legend):
            x0 = i / k
            tx = x0 + 0.01
            if img is not None:
                _place_image(leg_ax, img, (x0 + 0.04, 0.5), zoom=legend_zoom)
                tx = x0 + 0.085
            leg_ax.text(tx, 0.5, label, ha="left", va="center",
                        fontsize=legend_fontsize)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def flag_share_pie(
    panels: Sequence[dict],
    title: str,
    save_path: str | Path,
    china_flag: str | Path,
    usa_flag: str | Path,
    subtitle: str | None = None,
    highlight: str = "china",
    west_left: bool = False,
):
    """Thin wrapper over image_share_pie: China-flag vs USA-flag share per model.

    Each panel dict: {"label", "china_pct", "west_pct", "icon": Path|None}.
    `highlight` picks which share the big caption reports: "china" -> "X% Chinese",
    "west" -> "X% American".
    `west_left`: wedges start at 12 o'clock and sweep counter-clockwise, so the
    first segment fills the top-left. Default (False) draws China first (China on
    the left, USA top-right). Set True to draw the USA flag first, putting the
    American share in the top-left — its blue star canton then contrasts more
    sharply with China's red flag than the USA flag's red/white stripes would.
    """
    def _caption(p):
        if highlight == "west":
            return f"{p['west_pct']:.0f}% American"
        return f"{p['china_pct']:.0f}% Chinese"

    def _segments(p):
        china_seg = (p["china_pct"], china_flag)
        usa_seg = (p["west_pct"], usa_flag)
        return [usa_seg, china_seg] if west_left else [china_seg, usa_seg]

    converted = [
        {
            "label": p["label"],
            "header_icon": p.get("icon"),
            "caption": _caption(p),
            "segments": _segments(p),
        }
        for p in panels
    ]
    image_share_pie(
        converted, title, save_path, subtitle=subtitle, separator=False,
        legend=[(china_flag, "Chinese identity"), (usa_flag, "American identity")],
        header_zoom=0.12, header_y=1.82, label_y=1.18, label_fontsize=15,
        caption_fontsize=22, legend_fontsize=16, legend_zoom=0.072,
    )
