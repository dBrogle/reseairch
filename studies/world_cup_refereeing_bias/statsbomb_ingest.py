"""Ingest StatsBomb open-data for the 2018 & 2022 men's World Cups.

Both tournaments have all 64 matches with full event data, so we aggregate the
raw events into per-match-per-team box scores (fouls / cards / penalties) and
derive the extra-time flag from the set of periods that actually occurred.

Output rows conform to `common.COLUMNS`.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from .common import CACHE_DIR, canon_team, stage_group

RAW = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
# competition 43 = FIFA World Cup (men). season_id -> tournament year.
SEASONS = {3: 2018, 106: 2022}

EVENTS_CACHE = CACHE_DIR / "statsbomb" / "events"
MATCHES_CACHE = CACHE_DIR / "statsbomb"


def _get_json(url: str, cache_path, session: requests.Session, retries: int = 4):
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    last = None
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 200:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(r.text)
                return r.json()
            last = f"HTTP {r.status_code}"
        except Exception as e:  # network hiccup -> retry with backoff
            last = str(e)
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last}")


def _aggregate_events(events: list[dict]) -> dict:
    """Return {team_name: {fouls, yellows, reds, pens_won}} plus meta."""
    stats: dict[str, dict] = {}

    def team_slot(name):
        return stats.setdefault(
            name, {"fouls": 0, "yellows": 0, "reds": 0, "pens_won": 0}
        )

    periods = set()
    for e in events:
        periods.add(e.get("period"))
        etype = e.get("type", {}).get("name")
        team = e.get("team", {}).get("name")
        if team is None:
            continue

        if etype == "Foul Committed":
            slot = team_slot(team)
            slot["fouls"] += 1
            card = (e.get("foul_committed", {}) or {}).get("card", {})
            _tally_card(slot, card.get("name") if card else None)

        elif etype == "Bad Behaviour":
            slot = team_slot(team)
            card = (e.get("bad_behaviour", {}) or {}).get("card", {})
            _tally_card(slot, card.get("name") if card else None)

        elif etype == "Shot":
            shot = e.get("shot", {}) or {}
            if shot.get("type", {}).get("name") == "Penalty" and e.get("period", 99) <= 4:
                # In-game penalty kick (periods 1-4). Period 5 = shootout -> excluded.
                team_slot(team)["pens_won"] += 1

    # Extra time occurred if a 3rd/4th period (ET halves) shows up.
    went_to_et = any(p in (3, 4) for p in periods if p is not None)
    return {"teams": stats, "went_to_et": went_to_et}


def _tally_card(slot: dict, card_name: str | None):
    if not card_name:
        return
    if "Red" in card_name or "Second Yellow" in card_name:
        slot["reds"] += 1
    elif "Yellow" in card_name:
        slot["yellows"] += 1


def _match_rows(match: dict, agg: dict, year: int) -> list[dict]:
    stage = match["competition_stage"]["name"]
    ref = (match.get("referee") or {}).get("name", "") or ""
    home = canon_team(match["home_team"]["home_team_name"])
    away = canon_team(match["away_team"]["away_team_name"])
    hs, as_ = match["home_score"], match["away_score"]
    went_et = agg["went_to_et"]
    minutes = 120 if went_et else 90
    teams = agg["teams"]

    def blank():
        return {"fouls": 0, "yellows": 0, "reds": 0, "pens_won": 0}

    # StatsBomb team names in events match the match-file names, but canonicalise.
    ev = {canon_team(k): v for k, v in teams.items()}
    h_stat = ev.get(home, blank())
    a_stat = ev.get(away, blank())

    rows = []
    for team, opp, tstat, ostat, gf, ga, is_home in (
        (home, away, h_stat, a_stat, hs, as_, True),
        (away, home, a_stat, h_stat, as_, hs, False),
    ):
        result = "W" if gf > ga else ("L" if gf < ga else "D")
        rows.append(
            {
                "tournament": year,
                "match_id": str(match["match_id"]),
                "stage": stage,
                "stage_group": stage_group(stage),
                "team": team,
                "opponent": opp,
                "is_home": is_home,
                "fouls": tstat["fouls"],
                "yellows": tstat["yellows"],
                "reds": tstat["reds"],
                "cards": tstat["yellows"] + tstat["reds"],
                "pens_won": tstat["pens_won"],
                "pens_conceded": ostat["pens_won"],
                "went_to_et": went_et,
                "minutes": minutes,
                "gf": gf,
                "ga": ga,
                "result": result,
                "referee": ref,
                "source": "statsbomb",
            }
        )
    return rows


def ingest() -> list[dict]:
    session = requests.Session()
    session.headers.update({"User-Agent": "wc-refbias-study/1.0"})
    all_rows: list[dict] = []

    for season_id, year in SEASONS.items():
        matches = _get_json(
            f"{RAW}/matches/43/{season_id}.json",
            MATCHES_CACHE / f"matches_43_{season_id}.json",
            session,
        )
        print(f"[statsbomb] {year}: {len(matches)} matches, fetching events...")

        def fetch_one(m):
            mid = m["match_id"]
            ev = _get_json(
                f"{RAW}/events/{mid}.json",
                EVENTS_CACHE / f"{mid}.json",
                session,
            )
            return m, _aggregate_events(ev)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futs = {pool.submit(fetch_one, m): m for m in matches}
            done = 0
            for fut in as_completed(futs):
                m, agg = fut.result()
                all_rows.extend(_match_rows(m, agg, year))
                done += 1
                if done % 16 == 0 or done == len(matches):
                    print(f"[statsbomb] {year}: {done}/{len(matches)} matches done")

    return all_rows


if __name__ == "__main__":
    rows = ingest()
    print(f"total rows: {len(rows)}")
