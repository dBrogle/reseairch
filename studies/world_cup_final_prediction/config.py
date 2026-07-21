"""All tunables for the World Cup final-prediction study.

Nothing study-specific lives in shared code — this file is the single place to
change hosts, the hyper-parameter grids, the TVT split, or the 2026 bracket.
"""

from __future__ import annotations

from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = STUDY_DIR / "output"
GRAPH_DIR = OUTPUT_DIR / "graphs"
TABLE_DIR = OUTPUT_DIR / "tables"

# The match data is owned by the refereeing-bias study; we read its committed CSV
# rather than duplicate the ingestion. One source of truth for "what happened".
SOURCE_CSV = STUDY_DIR.parent / "world_cup_refereeing_bias" / "data" / "matches_long.csv"
FLAG_DIR = STUDY_DIR.parent / "world_cup_refereeing_bias" / "data" / "flags"
# High-res flags (1280px, from flagcdn) cached locally for the flag-fill final
# chart, which upscales a flag across a whole bar — the sibling's 160px flags
# pixelate. Only the flag-fill chart reads these; other charts keep the 160px set.
FLAG_DIR_HI = STUDY_DIR / "data" / "flags"

# Host nations get a genuine home-field term; every other WC game is neutral.
# (The CSV's `is_home` is a nominal box-score label, not real home advantage.)
HOSTS = {
    1998: ["France"],
    2002: ["South Korea", "Japan"],
    2006: ["Germany"],
    2010: ["South Africa"],
    2014: ["Brazil"],
    2018: ["Russia"],
    2022: ["Qatar"],
    2026: ["United States", "Canada", "Mexico"],
}

# Finals decided on penalties: the dataset records only the pre-shootout draw, so
# the actual winner is supplied here (real-world fact). Used to grade the model's
# "lean" on those finals, shown lightly since a shootout is ~a coin flip.
KNOWN_SHOOTOUT_WINNERS = {2006: "Italy", 2022: "Argentina"}

# Cups to pull ourselves (ESPN) beyond the sibling refereeing-bias CSV's 2010-2026,
# so the models can be tested out-of-sample on older finals too. Cached to
# data/extra_cups.csv; delete that file (or pass --refetch) to rebuild from ESPN.
EXTRA_CUPS = {
    1998: ("1998-06-10", "1998-07-12"),
    2002: ("2002-05-31", "2002-06-30"),
    2006: ("2006-06-09", "2006-07-09"),
}
EXTRA_CSV = STUDY_DIR / "data" / "extra_cups.csv"

# Tournament-wise TVT split. Team attack/defence are always re-fit *within* each
# tournament (rosters don't transfer across cups), so the split governs the
# METHODOLOGY: transferable pieces are fit on TRAIN, hyper-parameters chosen on
# VALIDATION, and the model is scored once on the untouched TEST cup. Validation
# pools two cups (n~32 knockout games): a single cup (n=16) is too small to locate
# an interior optimum and just drives every knob to its boundary. Train cups still
# feed validation/test through the leakage-free past-cup prior.
TRAIN_YEARS = [2010, 2014]
VALIDATION_YEARS = [2018, 2022]
TEST_YEAR = 2026

# --- Poisson (Dixon-Coles) hyper-parameter grid, searched on the validation cups.
# kappa       : ridge strength shrinking attack/defence toward the prior (bigger =
#               trust the ~6 noisy games less, lean on the average/prior more).
# prior_weight: how much of each team's PAST-World-Cup form to seed as the shrink
#               target (0 = pure this-tournament, shrink toward the field mean).
# Ceiling is 32: past kappa~16 the validation RPS is flat, and pushing higher just
# converts the model into "past reputation only", defeating the point of reading
# in-tournament form.
POISSON_KAPPA_GRID = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
POISSON_PRIOR_WEIGHT_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]

# --- Elo baseline grid.
# k        : learning rate per game.
# mov      : margin-of-victory exponent (0 = ignore goal margin, 1 = linear-ish).
# carryover: fraction of a team's prior-cup Elo carried into the new tournament
#            (0 = everyone starts equal each cup).
ELO_K_GRID = [10.0, 20.0, 40.0, 60.0]
ELO_MOV_GRID = [0.0, 0.5, 1.0]
ELO_CARRYOVER_GRID = [0.0, 0.5, 1.0]

# Host home-field advantage (log-rate boost to the host's expected goals). Fixed,
# not fitted: with one host per cup and ~3-7 host games, a free host term is
# unidentifiable and just soaks up that host's form (Qatar 2022 drove it to the
# bound). A modest literature-informed constant is the honest choice; other WC
# games are neutral. Set to 0.0 to treat every game as neutral.
HOST_HOME_ADV = 0.25

# Knockouts have no tie. A predicted draw is split into a winner: an extra-time
# share (edge to the stronger side) plus a penalty share. Penalties are a coin flip
# (SHOOTOUT_PROB) — the data has no shootout winners, and shootouts are ~50/50
# anyway. The ET-vs-penalty split rate is measured from past cups at runtime
# (data.et_resolution_rate), not hard-coded. The remaining bracket is small and
# fixed, so we propagate probabilities EXACTLY through the tree (no Monte-Carlo).
SHOOTOUT_PROB = 0.5

# Max goals per side when building the Poisson score matrix (P(x,y), x,y in 0..MAX).
MAX_GOALS = 10

# --- The remaining 2026 bracket (played through the Round of 16 in the data).
# Eight quarter-finalists = the eight R16 winners. Pairings follow the standard
# single-elimination layout; ASSUMPTION, easily edited here if the real fixtures
# differ. One slot is unresolved in the data (SUI-COL R16 was a 0-0 draw decided on
# penalties), so it is entered as a mini-tie the model itself resolves first.
#
# QF pairings (top half then bottom half of the draw):
QUARTERFINALS = [
    ("Morocco", "France"),        # QF1
    ("Norway", "England"),        # QF2
    ("Spain", "Belgium"),         # QF3
    ("Argentina", "__SUI_COL__"), # QF4  (opponent = winner of the unresolved R16 tie)
]
# Semi-final structure: winner(QF1) vs winner(QF2); winner(QF3) vs winner(QF4).
# Final: winner(SF1) vs winner(SF2). Encoded in simulate.py as a fixed tree.

# The one unresolved Round-of-16 tie, resolved by the model before the QFs.
UNRESOLVED_R16 = ("Switzerland", "Colombia")
UNRESOLVED_KEY = "__SUI_COL__"
