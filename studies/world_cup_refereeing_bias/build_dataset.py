"""Build the combined match-team dataset and validate source agreement.

Design decision on sources
--------------------------
ESPN covers 2010, 2014, 2018, 2022 and the ongoing 2026 World Cup with ONE
consistent methodology (match-total fouls / cards / penalties + an extra-time
flag). StatsBomb only covers 2018 & 2022, but from raw events (gold standard).

For cross-tournament and cross-team comparison, methodological CONSISTENCY matters
more than per-match precision, so the PRIMARY dataset is all-ESPN across the five
cups. We then use StatsBomb (2018/2022) as an independent VALIDATION set: if ESPN
agrees closely with StatsBomb on the two overlapping cups, the all-ESPN dataset is
trustworthy for the years StatsBomb can't cover.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .common import COLUMNS, dataset_path, TABLE_DIR
from . import espn_ingest as E
from . import statsbomb_ingest as SB

# World Cup date windows (inclusive). 2026 end = "today" since it is ongoing.
WINDOWS = {
    2010: (date(2010, 6, 11), date(2010, 7, 11)),
    2014: (date(2014, 6, 12), date(2014, 7, 13)),
    2018: (date(2018, 6, 14), date(2018, 7, 15)),
    2022: (date(2022, 11, 20), date(2022, 12, 18)),
    2026: (date(2026, 6, 11), date.today()),  # ongoing: end = today
}
REFRESH_FROM = {2026: date(2026, 7, 6)}  # re-fetch recent, possibly-updated results


def build_espn() -> pd.DataFrame:
    frames = []
    for year, (start, end) in WINDOWS.items():
        rows = E.ingest(year, start, end, refresh_from=REFRESH_FROM.get(year))
        frames.append(pd.DataFrame(rows))
    df = pd.concat(frames, ignore_index=True)
    return df[COLUMNS]


def build_statsbomb() -> pd.DataFrame:
    return pd.DataFrame(SB.ingest())[COLUMNS]


def validate(espn: pd.DataFrame, sb: pd.DataFrame) -> pd.DataFrame:
    """Compare ESPN vs StatsBomb on the two overlapping cups, per (tournament, team).

    Matches don't share ids across sources, so we compare per-team-per-tournament
    aggregates (total fouls, cards, penalties) -- robust to individual-match noise.
    """
    keys = ["tournament", "team"]
    agg = {"fouls": "sum", "cards": "sum", "pens_won": "sum", "pens_conceded": "sum"}
    e = espn[espn.tournament.isin([2018, 2022])].groupby(keys).agg(agg)
    s = sb.groupby(keys).agg(agg)
    j = e.join(s, lsuffix="_espn", rsuffix="_sb", how="outer").fillna(0)
    for m in agg:
        j[f"{m}_diff"] = j[f"{m}_espn"] - j[f"{m}_sb"]
    return j.reset_index()


def _corr_report(v: pd.DataFrame) -> str:
    lines = ["Source agreement (ESPN vs StatsBomb), per team-tournament totals:"]
    for m in ["fouls", "cards", "pens_won", "pens_conceded"]:
        a, b = v[f"{m}_espn"], v[f"{m}_sb"]
        r = np.corrcoef(a, b)[0, 1] if a.std() and b.std() else float("nan")
        mae = float(np.mean(np.abs(a - b)))
        lines.append(f"  {m:14s} r={r:.3f}  MAE={mae:.2f}  (ESPN sum={a.sum():.0f} / SB sum={b.sum():.0f})")
    return "\n".join(lines)


def build(save: bool = True) -> pd.DataFrame:
    espn = build_espn()
    sb = build_statsbomb()

    v = validate(espn, sb)
    report = _corr_report(v)
    print("\n" + report + "\n")

    if save:
        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        espn.to_csv(dataset_path(), index=False)
        sb.to_csv(TABLE_DIR / "statsbomb_validation_rows.csv", index=False)
        v.to_csv(TABLE_DIR / "source_agreement.csv", index=False)
        (TABLE_DIR / "source_agreement_report.txt").write_text(report + "\n")
        print(f"[build] primary (ESPN) dataset -> {dataset_path()}  ({len(espn)} rows)")
    return espn


if __name__ == "__main__":
    df = build()
    print("\n=== rows per tournament ===")
    print(df.groupby("tournament").size())
    print("\n=== Argentina totals per tournament ===")
    print(
        df[df.team == "Argentina"]
        .groupby("tournament")
        .agg(games=("match_id", "count"), fouls=("fouls", "sum"),
             cards=("cards", "sum"), pens_won=("pens_won", "sum"),
             pens_conceded=("pens_conceded", "sum"))
    )
