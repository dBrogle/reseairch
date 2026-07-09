"""Wordle-style result boards for the Poople LLM test.

One PNG per (example word, model) under output/results_images_<condition>/<word>/.
Each board draws the model's best attempt as a Wordle grid:

  * Rows are the steps of the ladder — the start word on top, each subsequent
    word on the next line, just like Wordle guesses stacking downward.
  * Columns are the four letter positions.
  * A tile is GREEN when its letter already matches the goal word "poop" in that
    position (so a solved board's final row is all green); otherwise it's gray.
    A "GOAL" reference row of all-green poop tiles sits at the top.
  * A legal one-letter change gets a green ✓ and a subtle outline on the changed
    tile; an illegal step gets a red ✗, red outlines on the offending tiles, and
    a dashed red frame + "not a word" when the word is invalid.

The brand icon sits in the top-right corner; the only caption is how many
illegal moves there were and how the attempt did versus par.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle, Patch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from studies.poople.config import ITERATIONS, TARGET
from studies.poople.llm_cache import get_results, load_cache, results_images_dir
from studies.poople.prompt import build_messages
from studies.poople.sampling import sample_test_words
from studies.poople.config import RESULT_IMAGE_WORDS_PER_BUCKET
from utils.model_icons import icon_path_for

# Palette
GREEN = "#6aaa64"          # letter in its final (goal) position
GRAY = "#cdd1d6"           # not (yet) matching
TEXT_ON_GREEN = "white"
TEXT_ON_GRAY = "#2b2f36"
RED = "#e03a3a"            # illegal move / invalid word
CHANGE_EDGE = "#6e7177"    # outline of the single legal changed tile
TILE_BORDER = "white"
_ICON_REF_PX = 512.0

# Geometry (data units; the axes is equal-aspect so tiles stay square)
TILE = 1.0
STEP = 1.16
GUTTER_X = -1.05           # x-center of the left status glyph
NOTE_X = 4 * STEP + 0.25   # x where right-hand notes start


def safe_model(model: str) -> str:
    """OpenRouter id -> filename stem: '/', '.', '-' all become '_'."""
    out = model
    for ch in "/.-":
        out = out.replace(ch, "_")
    return out


# ---------------------------------------------------------------------------
# Attempt selection + per-row analysis
# ---------------------------------------------------------------------------

def attempt_score(r: dict) -> tuple:
    """Sort key (lower = better) to pick the most illustrative attempt."""
    if r.get("error") is not None:
        return (5, 0)
    if not r.get("parsed"):
        return (4, 0)
    if r.get("solved"):
        return (0, r.get("over_par") or 0)
    if r.get("reached_target"):
        return (1, r.get("illegal_moves", 0))
    return (2, r.get("illegal_moves", 0))


def caption(r: dict) -> str:
    """One concise line: illegal-move count + result versus par."""
    if r.get("error") is not None:
        return "API error"
    if not r.get("parsed"):
        return "no valid answer"
    ill = r.get("illegal_moves", 0)
    ill_str = f"{ill} illegal move{'s' if ill != 1 else ''}"
    if r.get("solved"):
        op = r.get("over_par", 0)
        par_str = "optimal (par)" if op == 0 else f"+{op} over par"
    else:
        par_str = "didn't reach poop"
    return f"{ill_str}   ·   {par_str}"


def analyze_rows(ladder: list[str], words: set[str], target: str = TARGET) -> list[dict]:
    """Per-row drawing info for a ladder (ladder[0] is the start word)."""
    rows = []
    for i, word in enumerate(ladder):
        w = (word or "")
        matches = [j < len(w) and j < len(target) and w[j] == target[j] for j in range(4)]
        if i == 0:
            rows.append({"word": w, "matches": matches, "changed": [],
                         "valid": True, "legal": True, "is_start": True})
            continue
        prev = ladder[i - 1] or ""
        changed = [j for j in range(max(len(prev), len(w), 4))
                   if (prev[j] if j < len(prev) else None) != (w[j] if j < len(w) else None)]
        valid = w in words
        legal = (len(changed) == 1) and valid
        rows.append({"word": w, "matches": matches, "changed": [c for c in changed if c < 4],
                     "valid": valid, "legal": legal, "is_start": False})
    return rows


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _place_icon(ax, icon_path, xy, zoom=0.14):
    """Drop a brand PNG at axes-fraction xy, normalized for source resolution."""
    if icon_path is None:
        return
    try:
        img = mpimg.imread(str(icon_path))
    except Exception:
        return
    eff = zoom * (_ICON_REF_PX / img.shape[0])
    ab = AnnotationBbox(OffsetImage(img, zoom=eff), xy, xycoords="axes fraction",
                        frameon=False, box_alignment=(0.5, 0.5))
    ab.set_zorder(20)
    ax.add_artist(ab)


def _draw_tile(ax, x_left, y_top, letter, green, edge, edge_w, zorder=2):
    ax.add_patch(Rectangle((x_left, y_top - TILE), TILE, TILE,
                           facecolor=GREEN if green else GRAY,
                           edgecolor=edge, linewidth=edge_w, zorder=zorder))
    ch = (letter[:1].upper() if letter else "")
    ax.text(x_left + TILE / 2, y_top - TILE / 2, ch, ha="center", va="center",
            color=TEXT_ON_GREEN if green else TEXT_ON_GRAY,
            fontsize=21, fontweight="bold", zorder=zorder + 1)


def _draw_row(ax, y_top, row: dict):
    """Draw one ladder row (4 tiles) plus its status glyph and notes."""
    changed = set(row["changed"])
    for col in range(4):
        x = col * STEP
        if row["is_start"]:
            edge, ew = TILE_BORDER, 2.0
        elif not row["legal"] and col in changed:
            edge, ew = RED, 3.0
        elif row["legal"] and col in changed:
            edge, ew = CHANGE_EDGE, 3.0
        else:
            edge, ew = TILE_BORDER, 2.0
        letter = row["word"][col] if col < len(row["word"]) else ""
        _draw_tile(ax, x, y_top, letter, row["matches"][col], edge, ew)

    cy = y_top - TILE / 2
    if row["is_start"]:
        ax.text(GUTTER_X, cy, "start", ha="center", va="center",
                fontsize=10, color="#878a8c", fontweight="bold")
    elif row["legal"]:
        ax.text(GUTTER_X, cy, "✓", ha="center", va="center",
                fontsize=17, color=GREEN, fontweight="bold")
    else:
        ax.text(GUTTER_X, cy, "✗", ha="center", va="center",
                fontsize=17, color=RED, fontweight="bold")

    if not row["is_start"] and not row["valid"]:
        ax.add_patch(Rectangle((-0.08, y_top - TILE - 0.08), 4 * STEP - STEP + TILE + 0.16,
                               TILE + 0.16, fill=False, edgecolor=RED, linewidth=2.0,
                               linestyle=(0, (4, 2)), zorder=5))
        ax.text(NOTE_X, cy, "not a word", ha="left", va="center",
                fontsize=10.5, color=RED, fontweight="bold")
    elif not row["is_start"] and not row["legal"] and len(row["changed"]) != 1:
        ax.text(NOTE_X, cy, f"{len(row['changed'])} letters changed",
                ha="left", va="center", fontsize=10.5, color=RED, fontweight="bold")


def render_board(
    start_word: str,
    par: int,
    model: str,
    attempt: dict,
    words: set[str],
    save_path: Path,
    target: str = TARGET,
):
    """Render one model's attempt at one word as a Wordle board PNG."""
    ladder = attempt.get("ladder") or [start_word]
    rows = analyze_rows(ladder, words, target)
    n = len(rows)

    goal_top = 0.0
    gap = 0.55
    first_row_top = goal_top - TILE - gap
    last_row_bottom = first_row_top - (n - 1) * STEP - TILE

    fig_h = max(4.4, (abs(last_row_bottom) + 2.8) * 0.62)
    fig, ax = plt.subplots(figsize=(7.6, fig_h))
    ax.set_aspect("equal")
    ax.axis("off")

    # GOAL reference row (all-green poop).
    ax.text(GUTTER_X, goal_top - TILE / 2, "goal", ha="center", va="center",
            fontsize=10, color="#6aaa64", fontweight="bold")
    for col in range(4):
        _draw_tile(ax, col * STEP, goal_top, target[col], True, TILE_BORDER, 2.0)

    for i, row in enumerate(rows):
        _draw_row(ax, first_row_top - i * STEP, row)

    # Title + one concise caption, drawn in data units above the goal row.
    cx = 1.5 * STEP + TILE / 2
    ax.text(cx, goal_top + 2.0, f"{start_word.upper()}   →   {target.upper()}",
            ha="center", va="center", fontsize=21, fontweight="bold", color="#1a1d22")
    ax.text(cx, goal_top + 1.15, caption(attempt), ha="center", va="center",
            fontsize=12.5, fontweight="bold", color="#5a606a")

    # Brand icon + model name, top-right corner.
    _place_icon(ax, icon_path_for(model), (0.94, 0.95), zoom=0.15)
    ax.text(0.94, 0.845, model.split("/")[-1], transform=ax.transAxes,
            ha="center", va="center", fontsize=10, color="#7a808a", fontweight="bold")

    handles = [
        Patch(facecolor=GREEN, edgecolor="white", label="letter in poop's spot"),
        Patch(facecolor=GRAY, edgecolor="white", label="not yet"),
        Patch(facecolor="white", edgecolor=RED, label="illegal step / not a word"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=3, frameon=False, fontsize=10.5)

    ax.set_xlim(-1.7, NOTE_X + 2.4)
    ax.set_ylim(last_row_bottom - 1.4, goal_top + 2.7)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def pick_example_words(dist: dict[str, int]) -> list[tuple[str, int]]:
    """Pick example words from the front of each bucket's stable sample order."""
    sample = sample_test_words(dist)
    out: list[tuple[str, int]] = []
    for d in sorted(sample):
        k = RESULT_IMAGE_WORDS_PER_BUCKET.get(d, 0)
        out.extend((w, d) for w in sample[d][:k])
    return out


def generate_result_images(models: list[str], words: set[str], dist: dict[str, int],
                           condition: str):
    """Render every (example word × model) board for one condition."""
    examples = pick_example_words(dist)
    if not examples:
        print("  No example words selected.")
        return

    base = results_images_dir(condition)
    print(f"\n--- [{condition}] result boards: {len(examples)} words × {len(models)} models ---")
    made = 0
    for word, par in examples:
        word_dir = base / word
        for model in models:
            results = get_results(load_cache(model, condition), build_messages(word))[:ITERATIONS]
            if not results:
                continue
            best = min(results, key=attempt_score)
            render_board(word, par, model, best, words, word_dir / f"{safe_model(model)}.png")
            made += 1
    print(f"  Rendered {made} board(s) under {base}/")
