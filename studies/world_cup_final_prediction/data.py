"""Load the committed World Cup match data into a match-level table.

The source CSV (owned by the refereeing-bias study) is LONG: two rows per match,
one per team. Here we fold it into one row per match (home/away/goals), attach a
real home-field flag (host nation only), and a knockout "depth" = rounds-from-final
so the backtest can train only on strictly-earlier rounds without leakage.
"""

from __future__ import annotations

import pandas as pd

from . import config


# Distance from the final. Higher = earlier round. Group is earliest of all.
# Normalises the inconsistent stage labels across cups (2010 "Second Round" etc.).
# 3rd-place match is played alongside the final (after the semis), so depth 0 too.
DEPTH = {"GROUP": 99, "R32": 4, "R16": 3, "QF": 2, "SF": 1, "3RD": 0, "FINAL": 0}


def _norm_stage(raw: str) -> str:
    s = (raw or "").strip().lower()
    if "group" in s:
        return "GROUP"
    if "3rd" in s or "third" in s or "place" in s:
        return "3RD"
    if "quarter" in s:
        return "QF"
    if "semi" in s:
        return "SF"
    if "round of 32" in s:
        return "R32"
    if "round of 16" in s or "second round" in s:
        return "R16"
    if "final" in s:  # after 3rd/semi/quarter are peeled off, only the Final is left
        return "FINAL"
    return "GROUP"


def _host_side(tournament: int, home: str, away: str) -> str | None:
    """Which side (if any) is the tournament host playing at home."""
    hosts = config.HOSTS.get(int(tournament), [])
    if home in hosts:
        return "home"
    if away in hosts:
        return "away"
    return None


def load_matches() -> pd.DataFrame:
    """One row per match: home/away teams, goals, round depth, host-home side.

    Combines the sibling refereeing-bias CSV (2010-2026) with this study's own
    older-cups pull (1998/2002/2006, if built) so the models get more finals to be
    tested on out of sample.
    """
    frames = [pd.read_csv(config.SOURCE_CSV)]
    if config.EXTRA_CSV.exists():
        frames.append(pd.read_csv(config.EXTRA_CSV))
    long = pd.concat(frames, ignore_index=True)

    rows = []
    for match_id, g in long.groupby("match_id"):
        if len(g) != 2:
            continue  # defensive: a well-formed match is exactly two team-rows
        home_r = g[g.is_home]
        away_r = g[~g.is_home]
        if len(home_r) != 1 or len(away_r) != 1:
            # Nominal home flag missing/duplicated — pick a stable order instead.
            home_r, away_r = g.iloc[[0]], g.iloc[[1]]
        home_r, away_r = home_r.iloc[0], away_r.iloc[0]

        norm = _norm_stage(home_r.stage)
        home, away = str(home_r.team), str(away_r.team)
        hg, ag = int(home_r.gf), int(home_r.ga)
        rows.append({
            "tournament": int(home_r.tournament),
            "match_id": match_id,
            "stage": home_r.stage,
            "stage_norm": norm,
            "depth": DEPTH[norm],
            "is_knockout": norm != "GROUP",
            "home": home,
            "away": away,
            "hg": hg,
            "ag": ag,
            "went_to_et": bool(home_r.went_to_et),
            "host_side": _host_side(home_r.tournament, home, away),
            "result": "H" if hg > ag else ("A" if ag > hg else "D"),
        })

    df = pd.DataFrame(rows).sort_values(["tournament", "depth"], ascending=[True, False])
    return df.reset_index(drop=True)


def teams_in(df: pd.DataFrame) -> list[str]:
    return sorted(set(df.home) | set(df.away))


def tournament_games(df: pd.DataFrame, year: int, before_depth: int | None = None) -> pd.DataFrame:
    """Games in one cup; if `before_depth` given, only strictly-earlier rounds.

    "Earlier" = further from the final = larger depth. So to predict a match at
    depth d we train on games with depth > d (all group + earlier knockout rounds).
    """
    sub = df[df.tournament == year]
    if before_depth is not None:
        sub = sub[sub.depth > before_depth]
    return sub.reset_index(drop=True)


def knockout_matches(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """The matches we actually evaluate on: every knockout game of a cup.

    Group games are un-predictable early on (no prior form that tournament) and
    aren't the target anyway — the final is a knockout.
    """
    return df[(df.tournament == year) & df.is_knockout].reset_index(drop=True)


def final_match(df: pd.DataFrame, year: int):
    """The one final of a cup (as a row), or None if it hasn't been played."""
    fin = df[(df.tournament == year) & (df.stage_norm == "FINAL")]
    return fin.iloc[0] if len(fin) else None


def et_resolution_rate(df: pd.DataFrame) -> tuple[float, dict]:
    """Empirical P(a tied-after-90 knockout is settled in extra time, not penalties).

    From all cups' knockout games: `went_to_et` marks games level after 90; of those,
    a decisive result (W/L) was settled in extra time, while a "D" went to penalties
    (whose winner the data does NOT record). This rate is how we split a predicted
    knockout draw — the ET share favours the stronger side, the penalty share is a
    coin flip. A naive "ET = 1/3 of a match" goals model overpredicts this badly
    (~0.51 vs ~0.32 empirically), because tied games are selectively low-scoring.
    """
    ko = df[df.is_knockout]
    et = ko[ko.went_to_et]
    if len(et) == 0:
        return 0.33, {"tied_after_90": 0, "decided_in_et": 0, "penalties": 0}
    decided = int((et.result != "D").sum())
    pens = int((et.result == "D").sum())
    return decided / len(et), {"tied_after_90": int(len(et)),
                               "decided_in_et": decided, "penalties": pens}


def finalists(df: pd.DataFrame, year: int) -> list[str] | None:
    """The two semi-final winners, if both semis are played and decided.

    Returns None if the semis aren't done, or one was a draw with no recorded
    winner (penalties) — in which case there's no single final to predict yet.
    """
    sf = knockout_matches(df, year)
    sf = sf[sf.stage_norm == "SF"]
    if len(sf) < 2:
        return None
    winners = []
    for m in sf.itertuples():
        if m.hg == m.ag:
            return None  # undecided on penalties, winner unknown from this data
        winners.append(m.home if m.hg > m.ag else m.away)
    return winners
