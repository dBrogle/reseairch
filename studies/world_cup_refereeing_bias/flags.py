"""Country flag assets for charts. Flags are downloaded once from flagcdn.com
(free, ISO-3166 keyed) and cached under data/flags/. Returns a local PNG path."""

from __future__ import annotations

import requests

from .common import DATA_DIR

FLAG_DIR = DATA_DIR / "flags"
CDN = "https://flagcdn.com/w160/{code}.png"

# Team name (as canonicalised in the dataset) -> flagcdn code.
ISO = {
    "Algeria": "dz", "Argentina": "ar", "Australia": "au", "Austria": "at",
    "Belgium": "be", "Bosnia-Herzegovina": "ba", "Brazil": "br", "Cameroon": "cm",
    "Canada": "ca", "Cape Verde": "cv", "Chile": "cl", "Colombia": "co",
    "Congo DR": "cd", "Costa Rica": "cr", "Croatia": "hr", "Curaçao": "cw",
    "Czechia": "cz", "Denmark": "dk", "Ecuador": "ec", "Egypt": "eg",
    "England": "gb-eng", "France": "fr", "Germany": "de", "Ghana": "gh",
    "Greece": "gr", "Haiti": "ht", "Honduras": "hn", "Iceland": "is",
    "Iran": "ir", "Iraq": "iq", "Italy": "it", "Ivory Coast": "ci",
    "Japan": "jp", "Jordan": "jo", "Mexico": "mx", "Morocco": "ma",
    "Netherlands": "nl", "New Zealand": "nz", "Nigeria": "ng", "North Korea": "kp",
    "Norway": "no", "Panama": "pa", "Paraguay": "py", "Peru": "pe",
    "Poland": "pl", "Portugal": "pt", "Qatar": "qa", "Russia": "ru",
    "Saudi Arabia": "sa", "Scotland": "gb-sct", "Senegal": "sn", "Serbia": "rs",
    "Slovakia": "sk", "Slovenia": "si", "South Africa": "za", "South Korea": "kr",
    "Spain": "es", "Sweden": "se", "Switzerland": "ch", "Tunisia": "tn",
    "Türkiye": "tr", "United States": "us", "Uruguay": "uy", "Uzbekistan": "uz",
    "Wales": "gb-wls",
}


# FIFA-style 3-letter codes for compact y-axis labels.
ABBR = {
    "Algeria": "ALG", "Argentina": "ARG", "Australia": "AUS", "Austria": "AUT",
    "Belgium": "BEL", "Bosnia-Herzegovina": "BIH", "Brazil": "BRA", "Cameroon": "CMR",
    "Canada": "CAN", "Cape Verde": "CPV", "Chile": "CHI", "Colombia": "COL",
    "Congo DR": "COD", "Costa Rica": "CRC", "Croatia": "CRO", "Curaçao": "CUW",
    "Czechia": "CZE", "Denmark": "DEN", "Ecuador": "ECU", "Egypt": "EGY",
    "England": "ENG", "France": "FRA", "Germany": "GER", "Ghana": "GHA",
    "Greece": "GRE", "Haiti": "HAI", "Honduras": "HON", "Iceland": "ISL",
    "Iran": "IRN", "Iraq": "IRQ", "Italy": "ITA", "Ivory Coast": "CIV",
    "Japan": "JPN", "Jordan": "JOR", "Mexico": "MEX", "Morocco": "MAR",
    "Netherlands": "NED", "New Zealand": "NZL", "Nigeria": "NGA", "North Korea": "PRK",
    "Norway": "NOR", "Panama": "PAN", "Paraguay": "PAR", "Peru": "PER",
    "Poland": "POL", "Portugal": "POR", "Qatar": "QAT", "Russia": "RUS",
    "Saudi Arabia": "KSA", "Scotland": "SCO", "Senegal": "SEN", "Serbia": "SRB",
    "Slovakia": "SVK", "Slovenia": "SVN", "South Africa": "RSA", "South Korea": "KOR",
    "Spain": "ESP", "Sweden": "SWE", "Switzerland": "SUI", "Tunisia": "TUN",
    "Türkiye": "TUR", "United States": "USA", "Uruguay": "URU", "Uzbekistan": "UZB",
    "Wales": "WAL",
}


def abbr(team: str) -> str:
    return ABBR.get(team, team[:3].upper())


def flag_path(team: str):
    """Return a local PNG path for the team's flag, downloading + caching if needed.
    Returns None if the team has no known flag code or the download fails."""
    code = ISO.get(team)
    if not code:
        return None
    FLAG_DIR.mkdir(parents=True, exist_ok=True)
    path = FLAG_DIR / f"{code}.png"
    if path.exists():
        return path
    try:
        r = requests.get(CDN.format(code=code), timeout=30,
                         headers={"User-Agent": "wc-refbias-study/1.0"})
        if r.status_code == 200 and r.content:
            path.write_bytes(r.content)
            return path
    except Exception:
        return None
    return None


def prefetch(teams):
    ok = 0
    for t in set(teams):
        if flag_path(t):
            ok += 1
    return ok


if __name__ == "__main__":
    import pandas as pd
    from .common import dataset_path
    teams = pd.read_csv(dataset_path()).team.unique()
    print(f"cached {prefetch(teams)}/{len(teams)} flags -> {FLAG_DIR}")
