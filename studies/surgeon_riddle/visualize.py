"""Visualizations for the Surgeon Riddle study.

Three charts, each comparing models:

  1. provider_correctness.png - one panel per model maker, its models side by
       side, showing how often each SOLVES the riddle in each scenario (classic
       vs gender-flipped). The within-maker, across-version view.

  2. mother_trap.png          - headline. Per model, the rate of the reflexive
       "the doctor is his mother" answer in each condition: near-identical bar
       heights (the answer doesn't change when the parent's gender flips) but a
       color flip from correct aha (classic) to impossible failure (flipped).

  3. answer_breakdown.png     - mechanism. Per (model, condition), the full split
       over MOTHER / FATHER / TWO_SAME / OTHER_PARENT / OTHER.

House style is borrowed from utils/graphing.py (white ground, stripped spines,
muted grid/ink, brand icons); the ready-made helpers there bake in a
nationality legend that doesn't apply here, so these are bespoke.
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Patch, Rectangle

from utils.model_icons import icon_path_for, themed_icon_path_for
from studies.surgeon_riddle.config import MODELS, CONDITIONS, REASONING_LOW
from studies.surgeon_riddle.judge import compute_scores, outcome_of, correct_rate

# Models whose endpoint refuses to disable reasoning: run at lowest effort and
# flagged with a dagger so the charts don't imply they're truly reasoning-off.
LOW_MARK = "†"
FOOTNOTE = "†  reasoning can’t be disabled on this endpoint — run at lowest effort"

CATEGORY_ORDER = ["MOTHER", "FATHER", "TWO_SAME", "OTHER_PARENT", "OTHER"]

# Correct / failure status colors, reused across charts.
CORRECT_GREEN = "#2e9e5b"
FAILURE_RED = "#c0392b"

# answer_breakdown colors by OUTCOME, not answer identity: the correct answer
# (the opposite-gender parent) is green, the trap (the parent already in the
# crash) is red, and the two valid-but-distinct resolutions get their own hues.
TWO_SAME_YELLOW = "#f1c40f"
OTHER_PARENT_ORANGE = "#e67e22"
OTHER_GRAY = "#9aa0a8"


def _segment_color(condition_id: str, label: str) -> str:
    if label == "TWO_SAME":
        return TWO_SAME_YELLOW
    if label == "OTHER_PARENT":
        return OTHER_PARENT_ORANGE
    if label == "OTHER":
        return OTHER_GRAY
    return CORRECT_GREEN if outcome_of(condition_id, label) == "solved" else FAILURE_RED


def _text_on(hexcolor: str) -> str:
    """Black on light fills (e.g. yellow), white on dark ones — for count labels."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(hexcolor)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "#1a1d22" if lum > 0.65 else "white"


BREAKDOWN_LEGEND = [
    (CORRECT_GREEN, "correct — the other parent"),
    (FAILURE_RED, "the trap — the parent already in the crash"),
    (TWO_SAME_YELLOW, "two same-gender parents (2 dads / 2 moms)"),
    (OTHER_PARENT_ORANGE, "another parent (step, adoptive, …)"),
    (OTHER_GRAY, "refused / no clear parent"),
]

# provider_correctness: color each scenario by the gender its CORRECT answer
# points to — the classic's answer is the mother (pink), the flipped's is the
# father (blue). ("father"/"mother" here are the condition ids: father = classic
# man-driving, mother = flipped woman-driving.)
SCENARIO_COLOR = {"father": "#e0559b", "mother": "#2c7cba"}
SCENARIO_LABEL = {
    "father": "Classic (man & son) — correct: mother",
    "mother": "Flipped (woman & son) — correct: father",
}

PROVIDER_NAMES = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "x-ai": "xAI",
    "deepseek": "DeepSeek",
    "moonshotai": "Moonshot",
}

_ICON_REF_PX = 512.0

# Character portraits used in place of the "MOM"/"DAD" row & column labels.
_ASSETS = Path(__file__).parent / "assets"
MOM_IMG = _ASSETS / "mom.png"
DAD_IMG = _ASSETS / "dad.png"


def _place_icon(ax, img_path, xy, zoom, xycoords="data", zorder=6):
    """Drop a brand PNG at xy; silently no-op if unreadable. Zoom is normalized to
    a 512px reference so different source resolutions render the same size."""
    if img_path is None:
        return
    try:
        img = mpimg.imread(str(img_path))
    except Exception:
        return
    eff = zoom * (_ICON_REF_PX / img.shape[0])
    ab = AnnotationBbox(OffsetImage(img, zoom=eff), xy, xycoords=xycoords,
                        frameon=False, box_alignment=(0.5, 0.5), pad=0)
    ab.set_zorder(zorder)
    ax.add_artist(ab)


def _short(model: str) -> str:
    return model.split("/")[-1]


def _model_label(model: str) -> str:
    """Compact per-model label: drop the most redundant vendor prefixes."""
    short = _short(model)
    for pre in ("claude-", "gpt-5.6-"):
        if short.startswith(pre):
            return short[len(pre):]
    return short


def _disp_label(model: str) -> str:
    """Display label with a dagger for lowest-effort (reasoning-forced) models."""
    lab = _model_label(model)
    return f"{lab} {LOW_MARK}" if model in REASONING_LOW else lab


def _low_note(models) -> str:
    """Subtitle suffix explaining the dagger, if any plotted model needs it."""
    return "\n" + FOOTNOTE if any(m in REASONING_LOW for m in models) else ""


def _provider(model: str) -> str:
    return model.split("/")[0]


def _cond(cid: str) -> dict:
    return next(c for c in CONDITIONS if c["id"] == cid)


def _with_data(models):
    return [m for m in models
            if any(compute_scores(m)[c["id"]]["total"] for c in CONDITIONS)]


def _grouped_by_provider(models):
    """[(provider_key, [models])] in first-appearance order."""
    groups: dict[str, list[str]] = {}
    for m in models:
        groups.setdefault(_provider(m), []).append(m)
    return list(groups.items())


# ---------------------------------------------------------------------------
# Chart 1 — correctness by provider (the requested view)
# ---------------------------------------------------------------------------

def generate_provider_chart(models: list[str], save_path: str | Path):
    models = _with_data(models)
    if not models:
        print("  provider_correctness: no data.")
        return

    groups = _grouped_by_provider(models)
    n_groups = len(groups)
    width_ratios = [len(g[1]) for g in groups]

    fig, axes = plt.subplots(
        1, n_groups, sharey=True,
        figsize=(max(12, 1.7 * len(models) + 1.5 * n_groups), 6.2),
        gridspec_kw={"width_ratios": width_ratios, "wspace": 0.12},
    )
    if n_groups == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    bw = 0.38
    for ax, (prov, prov_models) in zip(axes, groups):
        ax.set_facecolor("white")
        m = len(prov_models)
        x = np.arange(m)

        for i, model in enumerate(prov_models):
            s = compute_scores(model)
            for cid, off in (("father", -bw / 2 - 0.01), ("mother", bw / 2 + 0.01)):
                rate = correct_rate(s[cid], cid)
                if rate is None:
                    continue
                pct = rate * 100
                ax.bar(i + off, pct, width=bw, color=SCENARIO_COLOR[cid],
                       edgecolor="white", linewidth=1.4, zorder=2)
                ax.text(i + off, pct + 2, f"{pct:.0f}", ha="center", va="bottom",
                        fontsize=10.5, fontweight="bold",
                        color=_darken(SCENARIO_COLOR[cid]), zorder=5)
            _place_icon(ax, themed_icon_path_for(model), (i, 108), zoom=0.058)

        ax.set_xticks(x)
        ax.set_xticklabels([_disp_label(mm) for mm in prov_models], fontsize=10,
                           fontweight="bold", color="#2b2f36")
        ax.set_xlim(-0.62, m - 0.38)
        ax.set_ylim(0, 120)
        ax.set_title(PROVIDER_NAMES.get(prov, prov), fontsize=13.5,
                     fontweight="bold", color="#1a1d22", pad=10)
        ax.tick_params(axis="x", length=0, pad=6)
        ax.tick_params(axis="y", labelsize=10.5, colors="#8a909a", length=0)
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", color="#e3e6ea", linewidth=1.0, alpha=0.9)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c7ccd3")

    axes[0].set_yticks(range(0, 101, 20))
    axes[0].set_yticklabels([f"{v}%" for v in range(0, 101, 20)])
    axes[0].set_ylabel("replies that solve the riddle", fontsize=12,
                       color="#4a505a", labelpad=10)

    fig.suptitle("How often each model solves the riddle", fontsize=19,
                 fontweight="bold", color="#1a1d22", y=1.02)
    fig.text(0.5, 0.955,
             "% of replies that name a coherent surgeon (not the parent who was "
             "already in the crash) · by maker" + _low_note(models),
             ha="center", va="top", fontsize=11.5, color="#7a808a")

    handles = [Patch(facecolor=SCENARIO_COLOR[c], edgecolor="white",
                     label=SCENARIO_LABEL[c]) for c in ("father", "mother")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.05),
               ncol=2, frameon=False, fontsize=12, labelcolor="#4a505a",
               handlelength=1.2, handleheight=1.2, columnspacing=2.5)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {save_path}")


def _darken(hexcolor: str, f: float = 0.72) -> tuple:
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(hexcolor)
    return (r * f, g * f, b * f)


# ---------------------------------------------------------------------------
# Chart 2 — the headline "mother trap"
# ---------------------------------------------------------------------------

def generate_mother_trap_chart(models: list[str], save_path: str | Path):
    models = _with_data(models)
    if not models:
        print("  mother_trap: no data.")
        return

    n = len(models)
    fig, ax = plt.subplots(figsize=(max(11, 1.15 * n + 1), 7.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    width = 0.36
    x = np.arange(n)
    pooled = {"father": [0, 0], "mother": [0, 0]}  # [mother, total]

    for i, model in enumerate(models):
        s = compute_scores(model)
        for cid, offset, color in (("father", -width / 2 - 0.01, CORRECT_GREEN),
                                    ("mother", width / 2 + 0.01, FAILURE_RED)):
            counts = s[cid]
            pooled[cid][0] += counts["MOTHER"]
            pooled[cid][1] += counts["total"]
            if not counts["total"]:
                continue
            pct = counts["MOTHER"] / counts["total"] * 100
            ax.bar(i + offset, pct, width=width, color=color, edgecolor="white",
                   linewidth=1.5, zorder=2)
            if pct > 0:
                ax.text(i + offset, pct + 2.2, f"{pct:.0f}", ha="center",
                        va="bottom", fontsize=11, fontweight="bold", color=color,
                        zorder=5)
        _place_icon(ax, icon_path_for(model), (i, 116), zoom=0.062)

    ax.set_xticks(x)
    ax.set_xticklabels([_disp_label(m) for m in models], fontsize=10.5,
                       fontweight="bold", color="#2b2f36", rotation=25, ha="right")
    ax.set_ylim(0, 128)
    ax.set_xlim(-0.65, n - 0.35)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)])
    ax.set_ylabel("answers “the doctor is his mother”", fontsize=12.5,
                  color="#4a505a", labelpad=10)
    ax.tick_params(axis="y", labelsize=11, colors="#8a909a", length=0)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color="#dfe3e8", linewidth=1.0, alpha=0.9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c7ccd3")

    pc = pooled["father"][0] / pooled["father"][1] * 100 if pooled["father"][1] else 0
    pf = pooled["mother"][0] / pooled["mother"][1] * 100 if pooled["mother"][1] else 0
    fig.suptitle("The same answer, whether it's right or impossible",
                 fontsize=19, fontweight="bold", color="#1a1d22", y=0.99)
    ax.set_title(
        f"“The doctor is his mother” — said {pc:.0f}% of the time when a MAN is "
        f"driving (correct), and {pf:.0f}% when a WOMAN is (impossible — she's the driver)"
        + _low_note(models),
        fontsize=11.5, color="#7a808a", pad=14)

    handles = [
        Patch(facecolor=CORRECT_GREEN, edgecolor="white",
              label="Man & son (classic) — “mother” is the correct twist"),
        Patch(facecolor=FAILURE_RED, edgecolor="white",
              label="Woman & son (flipped) — “mother” is impossible; the answer is the father"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=1, frameon=False, fontsize=11.5, labelcolor="#4a505a",
              handlelength=1.2, handleheight=1.2)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {save_path}")


# ---------------------------------------------------------------------------
# Chart 3 — the full answer breakdown (mechanism)
# ---------------------------------------------------------------------------

def generate_answer_breakdown_chart(models: list[str], save_path: str | Path):
    models = _with_data(models)
    if not models:
        print("  answer_breakdown: no data.")
        return

    rows = []  # (model, cid, y)
    y = 0.0
    for model in models:
        for cid in ("father", "mother"):
            rows.append((model, cid, y))
            y += 1.0
        y += 0.7
    top = y - 0.7

    fig, ax = plt.subplots(figsize=(13, 0.62 * len(rows) + 3.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bar_h = 0.74
    for model, cid, ry in rows:
        counts = compute_scores(model)[cid]
        total = counts["total"] or 1
        cond = _cond(cid)

        left = 0.0
        for cat in CATEGORY_ORDER:
            frac = counts[cat] / total * 100
            if frac <= 0:
                continue
            seg = _segment_color(cid, cat)
            ax.barh(ry, frac, left=left, height=bar_h, color=seg,
                    edgecolor="white", linewidth=1.6, zorder=2)
            if frac >= 12:
                ax.text(left + frac / 2, ry, str(counts[cat]), ha="center",
                        va="center", fontsize=10.5, fontweight="bold",
                        color=_text_on(seg), zorder=3)
            left += frac

        pronoun = "his" if cond["parent"] == "man" else "her"
        ax.text(-1.5, ry, f"“a {cond['parent']} & {pronoun} son”",
                ha="right", va="center", fontsize=10.5, color="#4a505a")

        dominant = max(CATEGORY_ORDER, key=lambda c: counts[c])
        ok = outcome_of(cid, dominant) == "solved"
        correct_word = "mother" if cid == "father" else "father"
        ax.text(103, ry, f"correct: {correct_word}", ha="left", va="center",
                fontsize=9.5, color="#7a808a")
        ax.text(101, ry, "✓" if ok else "✗", ha="left", va="center",
                fontsize=14, fontweight="bold",
                color=CORRECT_GREEN if ok else FAILURE_RED, zorder=4)

    for model in models:
        block = [r for r in rows if r[0] == model]
        yc = sum(r[2] for r in block) / len(block)
        _place_icon(ax, icon_path_for(model), (-0.26, yc), zoom=0.06,
                    xycoords=("axes fraction", "data"))
        ax.text(-0.215, yc, _disp_label(model), transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=11, fontweight="bold",
                color="#2b2f36")

    ax.set_ylim(top + 0.7, -0.7)
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xticks(range(0, 101, 20))
    ax.set_xticklabels([f"{v}%" for v in range(0, 101, 20)])
    ax.tick_params(axis="x", labelsize=10.5, colors="#8a909a", length=0)
    ax.set_axisbelow(True)
    ax.grid(True, axis="x", color="#eef0f2", linewidth=0.9, alpha=0.9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c7ccd3")

    fig.suptitle("What each model actually answers", fontsize=19,
                 fontweight="bold", color="#1a1d22", y=1.0)
    ax.set_title("share of each model’s replies per condition · numbers are reply counts"
                 + _low_note(models),
                 fontsize=11.5, color="#7a808a", pad=12)

    handles = [Patch(facecolor=c, edgecolor="white", label=lbl)
               for c, lbl in BREAKDOWN_LEGEND]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.05),
              ncol=3, frameon=False, fontsize=10.5, labelcolor="#4a505a",
              handlelength=1.2, handleheight=1.2, columnspacing=2.0)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {save_path}")


# ---------------------------------------------------------------------------
# Chart 4 — per-provider accuracy table (minimal, for social)
# ---------------------------------------------------------------------------

def _blend_white(hexcolor: str, frac: float) -> tuple:
    """Tint from white (frac=0) toward the full hue (frac=1)."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(hexcolor)
    f = max(0.0, min(1.0, frac))
    return (1 - (1 - r) * f, 1 - (1 - g) * f, 1 - (1 - b) * f)


def _accuracies(model: str) -> tuple[float | None, float | None]:
    """(accuracy when the answer is the mother, accuracy when it's the father).

    "mother" = the classic man & son scenario (condition id 'father'); "father" =
    the flipped woman & son scenario (condition id 'mother')."""
    s = compute_scores(model)
    return correct_rate(s["father"], "father"), correct_rate(s["mother"], "mother")


def _draw_accuracy_table(models: list[str], save_path, subtitle: str | None = None):
    """Minimal table: rows = models, two columns = accuracy when the correct
    answer is the mother vs the father. Cells shaded on a red↔green gradient."""
    n = len(models)
    fig, ax = plt.subplots(figsize=(6.2, 1.9 + 0.68 * n))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 4.0)
    ax.set_ylim(-0.2, n + 1.4)
    ax.axis("off")

    x_mom, x_dad, cw = 2.5, 3.45, 0.85
    _place_icon(ax, MOM_IMG, (x_mom, n + 0.55), zoom=0.095)
    _place_icon(ax, DAD_IMG, (x_dad, n + 0.55), zoom=0.095)

    for i, m in enumerate(models):
        mom, dad = _accuracies(m)
        y = n - 1 - i
        _place_icon(ax, icon_path_for(m), (0.22, y + 0.5), zoom=0.055)
        lab = _model_label(m) + (" †" if m in REASONING_LOW else "")
        ax.text(0.5, y + 0.5, lab, ha="left", va="center", fontsize=12.5,
                fontweight="bold", color="#2b2f36")
        for x, acc in ((x_mom, mom), (x_dad, dad)):
            if acc is None:
                bg = (1, 1, 1)
            else:
                # diverging around a 70% pivot (a "C" reads as average): 100%
                # pure green, 70% neutral white, 0% pure red. Each side is scaled
                # to its own span so both extremes reach full saturation.
                pivot = 0.70
                if acc >= pivot:
                    hue, frac = CORRECT_GREEN, (acc - pivot) / (1 - pivot)
                else:
                    hue, frac = FAILURE_RED, (pivot - acc) / pivot
                bg = _blend_white(hue, frac)
            ax.add_patch(Rectangle((x - cw / 2, y + 0.08), cw, 0.84, facecolor=bg,
                                   edgecolor="#e6e8eb", linewidth=1.5))
            lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
            txt = f"{acc * 100:.0f}%" if acc is not None else "–"
            ax.text(x, y + 0.5, txt, ha="center", va="center", fontsize=15,
                    fontweight="bold", color="#1a1d22" if lum > 0.62 else "white")

    fig.suptitle("Accuracy when the doctor is the…", fontsize=16.5,
                 fontweight="bold", color="#1a1d22", y=0.99)
    if subtitle:
        ax.text(1.0, n + 0.78, subtitle, ha="center", va="bottom", fontsize=12,
                color="#7a808a")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {save_path}")


def generate_provider_table(prov_key: str, prov_models: list[str], save_path):
    _draw_accuracy_table(prov_models, save_path,
                         subtitle=PROVIDER_NAMES.get(prov_key, prov_key))


# One flagship model per lab, for a single cross-provider comparison table.
OVERALL_MODELS = [
    "anthropic/claude-sonnet-5",
    "openai/gpt-5.6-sol",
    "x-ai/grok-4.5",
    "deepseek/deepseek-v4-pro",
    "moonshotai/kimi-k2.6",
]


def generate_overall_table(save_path):
    _draw_accuracy_table(OVERALL_MODELS, save_path)


# ---------------------------------------------------------------------------
# Chart 5 — per-provider pie grids (correct/incorrect donuts), for social
# ---------------------------------------------------------------------------

def _draw_donut(ax, model: str, cid: str):
    """Draw one outcome donut (green correct / red trap / yellow two-same /
    orange other / gray refused) with the correct-% in the hole."""
    counts = compute_scores(model)[cid]
    total = counts["total"] or 1
    vals, colors = [], []
    for cat in CATEGORY_ORDER:
        if counts[cat] > 0:
            vals.append(counts[cat])
            colors.append(_segment_color(cid, cat))
    if not vals:
        vals, colors = [1], ["#e6e8eb"]
    ax.pie(vals, colors=colors, startangle=90, counterclock=False,
           wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2.4))
    acc = correct_rate(counts, cid)
    ax.text(0, 0, f"{acc * 100:.0f}%" if acc is not None else "–", ha="center",
            va="center", fontsize=18, fontweight="bold", color="#1a1d22")
    ax.set(aspect="equal")


PIE_LEGEND = [
    (CORRECT_GREEN, "correct"),
    (FAILURE_RED, "wrong"),
    (TWO_SAME_YELLOW, "two same-gender"),
    (OTHER_PARENT_ORANGE, "another parent"),
    (OTHER_GRAY, "refused"),
]


def generate_provider_pies(prov_key: str, prov_models: list[str], folder: Path):
    """Per maker: a grid of donuts (columns = models, top row = the man & son
    scenario 'MOM', bottom row = woman & son 'DAD'), plus each donut on its own."""
    folder.mkdir(parents=True, exist_ok=True)
    prov_name = PROVIDER_NAMES.get(prov_key, prov_key)
    n = len(prov_models)

    fig, axes = plt.subplots(2, n, figsize=(2.55 * n + 1.7, 6.9), squeeze=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.16, right=0.97, top=0.78, bottom=0.14,
                        wspace=0.05, hspace=0.1)

    for col, m in enumerate(prov_models):
        for row, cid in enumerate(("father", "mother")):
            _draw_donut(axes[row][col], m, cid)
        top = axes[0][col]
        top.set_title(_model_label(m) + (" †" if m in REASONING_LOW else ""),
                      fontsize=13, fontweight="bold", color="#1a1d22", pad=22)

    for row, img in ((0, MOM_IMG), (1, DAD_IMG)):
        pos = axes[row][0].get_position()
        _place_icon(axes[row][0], img, (0.088, (pos.y0 + pos.y1) / 2),
                    zoom=0.2, xycoords=fig.transFigure)

    title = fig.suptitle(prov_name, x=0.52, y=0.965, fontsize=20,
                         fontweight="bold", color="#1a1d22")
    # place the provider icon just left of the title text, whatever its width
    fig.canvas.draw()
    bb = title.get_window_extent(fig.canvas.get_renderer()).transformed(
        fig.transFigure.inverted())
    _place_icon(axes[0][0], icon_path_for(prov_models[0]),
                (bb.x0 - 0.035, (bb.y0 + bb.y1) / 2), zoom=0.09,
                xycoords=fig.transFigure)
    handles = [Patch(facecolor=c, edgecolor="white", label=l) for c, l in PIE_LEGEND]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.01),
               ncol=5, frameon=False, fontsize=10.5, labelcolor="#4a505a",
               handlelength=1.1, handleheight=1.1, columnspacing=1.6)

    fig.savefig(folder / "grid.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Each donut on its own, too (handy for arranging on social).
    for m in prov_models:
        for cid, tag in (("father", "mom"), ("mother", "dad")):
            figi, axi = plt.subplots(figsize=(3.2, 3.6))
            figi.patch.set_facecolor("white")
            _draw_donut(axi, m, cid)
            axi.set_title(f"{_model_label(m)} — doctor is {tag}", fontsize=12,
                          fontweight="bold", color="#1a1d22", pad=12)
            figi.savefig(folder / f"{m.replace('/', '_')}_{tag}.png", dpi=150,
                         bbox_inches="tight", facecolor="white")
            plt.close(figi)
    print(f"  saved {folder}/ (grid + {2 * n} pies)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

GRAPHS_DIR = Path(__file__).parent / "output" / "graphs"
TABLES_DIR = GRAPHS_DIR / "tables"
PIES_DIR = GRAPHS_DIR / "pies"


def generate_provider_tables(models: list[str] | None = None):
    models = _with_data(models or MODELS)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for prov_key, prov_models in _grouped_by_provider(models):
        generate_provider_table(prov_key, prov_models, TABLES_DIR / f"{prov_key}.png")


def generate_provider_pie_grids(models: list[str] | None = None):
    models = _with_data(models or MODELS)
    for prov_key, prov_models in _grouped_by_provider(models):
        generate_provider_pies(prov_key, prov_models, PIES_DIR / prov_key)


def generate_all(models: list[str] | None = None):
    models = models or MODELS
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    generate_provider_chart(models, GRAPHS_DIR / "provider_correctness.png")
    generate_mother_trap_chart(models, GRAPHS_DIR / "mother_trap.png")
    generate_answer_breakdown_chart(models, GRAPHS_DIR / "answer_breakdown.png")
    generate_provider_tables(models)
    generate_overall_table(TABLES_DIR / "overall.png")
    generate_provider_pie_grids(models)
