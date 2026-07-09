"""Ingest World Cup match box scores from ESPN's public JSON API.

Used for tournaments StatsBomb doesn't cover -- primarily the ONGOING 2026 World
Cup (the source of the current Argentina controversy), and best-effort for older
cups (2010/2014) if ESPN happens to expose their box-score stats.

ESPN exposes per-team match stats (foulsCommitted, yellowCards, redCards,
penaltyKickShots) plus a status detail that distinguishes regulation (`FT`) from
extra time (`AET` / `FT-Pens`). Penalty *shootout* kicks are NOT counted in
`penaltyKickShots`, which is exactly what we want (we treat shootouts as excluded).

Granularity note vs StatsBomb: ESPN is match-total only (no per-event minute), and
its card convention may differ marginally (e.g. how second yellows are split), so
these rows are tagged `source="scraped:espn"` and can be filtered/flagged downstream.
"""

from __future__ import annotations

import json
import time
from datetime import date, timedelta

import requests

from .common import CACHE_DIR, canon_team, stage_group

API = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
ESPN_CACHE = CACHE_DIR / "espn"

# Completed-match status details we accept (others = scheduled/in-progress).
DONE_DETAILS = {"FT", "AET", "FT-Pens"}
ET_DETAILS = {"AET", "FT-Pens"}

# Normalise ESPN stage labels (from season.name) to canonical study stage names.
STAGE_MAP = {
    "group stage": "Group Stage",
    "round of 32": "Round of 32",
    "round of 16": "Round of 16",
    "quarterfinals": "Quarter-finals",
    "quarter-finals": "Quarter-finals",
    "semifinals": "Semi-finals",
    "semi-finals": "Semi-finals",
    "third place": "3rd Place Final",
    "3rd place": "3rd Place Final",
    "final": "Final",
}


def _session():
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    return s


def _get_json(url: str, cache_path, session, retries: int = 4, refresh: bool = False):
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text())
    last = None
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=45)
            if r.status_code == 200:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(r.text)
                return r.json()
            last = f"HTTP {r.status_code}"
        except Exception as e:
            last = str(e)
        time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last}")


def _canon_stage(season_name: str) -> str:
    # season_name looks like "2026 FIFA World Cup, Round of 32"
    tail = season_name.split(",")[-1].strip().lower() if season_name else ""
    return STAGE_MAP.get(tail, season_name.split(",")[-1].strip() if season_name else "")


def enumerate_events(year: int, start: date, end: date, session, refresh_today: date | None = None):
    """Return list of (event_id, iso_date) for completed matches in the window."""
    found = []
    d = start
    while d <= end:
        ds = d.strftime("%Y%m%d")
        # Re-fetch (bypass cache) for very recent dates whose results may have landed
        # after a previous run cached an in-progress/empty scoreboard.
        refresh = refresh_today is not None and d >= refresh_today
        sb = _get_json(
            f"{API}/scoreboard?dates={ds}",
            ESPN_CACHE / str(year) / f"scoreboard_{ds}.json",
            session,
            refresh=refresh,
        )
        for e in sb.get("events", []):
            comp = e["competitions"][0]
            detail = comp.get("status", {}).get("type", {}).get("detail")
            if detail in DONE_DETAILS:
                found.append((e["id"], d.isoformat()))
        d += timedelta(days=1)
    return found


def _num(stats: dict, key: str) -> int:
    try:
        return int(float(stats.get(key, 0) or 0))
    except (ValueError, TypeError):
        return 0


def parse_summary(summary: dict, year: int) -> list[dict]:
    header = summary.get("header", {})
    comp = header.get("competitions", [{}])[0]
    detail = comp.get("status", {}).get("type", {}).get("detail")
    went_et = detail in ET_DETAILS
    minutes = 120 if went_et else 90
    stage = _canon_stage(header.get("season", {}).get("name", ""))
    sg = stage_group(stage)

    box_teams = summary.get("boxscore", {}).get("teams", [])
    if len(box_teams) != 2:
        return []  # no box score -> unusable

    # Map competitor id -> homeAway + score from the header competitors.
    comp_meta = {}
    for c in comp.get("competitors", []):
        comp_meta[c.get("id")] = {
            "homeAway": c.get("homeAway"),
            "score": _int_or_none(c.get("score")),
            "name": canon_team(c.get("team", {}).get("displayName", "")),
        }

    parsed = []
    for t in box_teams:
        tid = t.get("team", {}).get("id")
        s = {x.get("name"): x.get("displayValue") for x in t.get("statistics", [])}
        meta = comp_meta.get(tid, {})
        parsed.append(
            {
                "id": tid,
                "team": canon_team(t.get("team", {}).get("displayName", "")),
                "fouls": _num(s, "foulsCommitted"),
                "yellows": _num(s, "yellowCards"),
                "reds": _num(s, "redCards"),
                "pens_won": _num(s, "penaltyKickShots"),
                "score": meta.get("score"),
                "homeAway": meta.get("homeAway"),
            }
        )

    if len(parsed) != 2:
        return []
    a, b = parsed
    mid = comp.get("id") or header.get("id")
    rows = []
    for team, opp in ((a, b), (b, a)):
        gf, ga = team["score"], opp["score"]
        if gf is None or ga is None:
            result = ""
        else:
            result = "W" if gf > ga else ("L" if gf < ga else "D")
        rows.append(
            {
                "tournament": year,
                "match_id": f"espn_{mid}",
                "stage": stage,
                "stage_group": sg,
                "team": team["team"],
                "opponent": opp["team"],
                "is_home": team["homeAway"] == "home",
                "fouls": team["fouls"],
                "yellows": team["yellows"],
                "reds": team["reds"],
                "cards": team["yellows"] + team["reds"],
                "pens_won": team["pens_won"],
                "pens_conceded": opp["pens_won"],
                "went_to_et": went_et,
                "minutes": minutes,
                "gf": gf if gf is not None else "",
                "ga": ga if ga is not None else "",
                "result": result,
                "referee": "",  # ESPN summary rarely exposes referee reliably
                "source": "scraped:espn",
            }
        )
    return rows


def _int_or_none(v):
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def ingest(year: int, start: date, end: date, refresh_from: date | None = None) -> list[dict]:
    session = _session()
    events = enumerate_events(year, start, end, session, refresh_today=refresh_from)
    print(f"[espn] {year}: {len(events)} completed matches in {start}..{end}")
    rows = []
    skipped = 0
    for i, (eid, iso) in enumerate(events, 1):
        refresh = refresh_from is not None and date.fromisoformat(iso) >= refresh_from
        summ = _get_json(
            f"{API}/summary?event={eid}",
            ESPN_CACHE / str(year) / f"summary_{eid}.json",
            session,
            refresh=refresh,
        )
        r = parse_summary(summ, year)
        if r:
            rows.extend(r)
        else:
            skipped += 1
        if i % 20 == 0 or i == len(events):
            print(f"[espn] {year}: {i}/{len(events)} parsed ({skipped} skipped, no box score)")
        time.sleep(0.05)
    return rows


if __name__ == "__main__":
    # 2026 is ongoing; refresh recent dates so newly-finished matches are picked up.
    rows = ingest(2026, date(2026, 6, 11), date(2026, 7, 8), refresh_from=date(2026, 7, 6))
    print("total 2026 rows:", len(rows))
    import pandas as pd

    df = pd.DataFrame(rows)
    if len(df):
        print(df.groupby("stage").size())
        print("Argentina games:", (df.team == "Argentina").sum())
