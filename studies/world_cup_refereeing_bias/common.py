"""Shared schema and helpers for the World Cup refereeing-bias study.

Every data source (StatsBomb events, scraped 2026 box scores, ...) is normalised
into the SAME long-format table: one row per (match, team). That way the analysis
and visualisation code never needs to know where a row came from.
"""

from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parent
DATA_DIR = STUDY_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = STUDY_DIR / "output"
GRAPH_DIR = OUTPUT_DIR / "graphs"
TABLE_DIR = OUTPUT_DIR / "tables"

# The canonical match-team table. One row = one team's involvement in one match.
COLUMNS = [
    "tournament",      # int year, e.g. 2018 / 2022 / 2026
    "match_id",        # source-native id (string)
    "stage",           # raw stage name (Group Stage, Round of 16, ...)
    "stage_group",     # "group" | "knockout"
    "team",            # canonical team name
    "opponent",        # canonical team name
    "is_home",         # bool (nominal for WC; kept for completeness)
    "fouls",           # fouls COMMITTED by `team`
    "yellows",         # yellow cards shown to `team` (incl. the first of a 2nd-yellow)
    "reds",            # red cards to `team` (straight red OR second-yellow dismissal)
    "cards",           # yellows + reds (total cards shown to `team`)
    "pens_won",        # penalty kicks AWARDED to `team` (excludes shootouts)
    "pens_conceded",   # penalty kicks awarded to the opponent
    "went_to_et",      # bool: match went to extra time
    "minutes",         # 90 if regulation, 120 if extra time (stoppage ignored by design)
    "gf",              # goals for (regulation+ET, excludes shootout)
    "ga",              # goals against
    "result",          # "W" | "D" | "L" (pre-shootout; ET counts, shootout does not)
    "referee",         # referee name if known, else ""
    "source",          # "statsbomb" | "scraped:<site>" | "manual"
]

KNOCKOUT_STAGES = {
    "Round of 16",
    "Round of 32",
    "Quarter-finals",
    "Semi-finals",
    "3rd Place Final",
    "Final",
}


def stage_group(stage_name: str) -> str:
    """Collapse a raw stage name into group vs knockout."""
    s = (stage_name or "").strip()
    if s == "Group Stage" or s.lower().startswith("group"):
        return "group"
    return "knockout"


# Canonical names so StatsBomb / scraped / manual sources line up. Extend as needed.
TEAM_ALIASES = {
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "USA": "United States",
    "United States of America": "United States",
    "Republic of Korea": "South Korea",
    "China PR": "China",
}


def canon_team(name: str) -> str:
    name = (name or "").strip()
    return TEAM_ALIASES.get(name, name)


def dataset_path() -> Path:
    return DATA_DIR / "matches_long.csv"
