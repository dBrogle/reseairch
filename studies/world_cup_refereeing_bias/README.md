# Is Argentina favoured by World Cup referees? A statistical study

A data-driven look at the recurring claim that Argentina benefits from refereeing
at the World Cup — more penalties, softer treatment, opponents punished harder.
Rather than cherry-picking one eye-catching stat, this study computes the same
metrics for **every team** across **five World Cups (2010–2026)** so Argentina is
always measured against the full distribution, and it runs significance tests so we
can separate a real signal from noise.

> **Headline:** The "Argentina gets penalties" claim is **real and statistically
> significant** — but it is **recent** (nothing in 2010/2014, a spike in 2022/2026)
> and Argentina is a **notable outlier, not a unique one** (other teams show similar
> patterns). It is *not* explained away by Argentina simply being a strong team.
> Whether the cause is refereeing bias, Argentina's style of drawing fouls, or both,
> the data cannot say — but the pattern the fans are pointing at does exist.

---

## Best charts to share (`output/graphs/share/`)

If you only post three, use these — they tell the whole story without needing the
methodology:

1. **`share/summary_scorecard.png`** — square Instagram card with the three headline
   findings and the honest verdict. Self-contained.
2. **`share/penalties_per90_top10_flags.png`** — the headline stat with flags and the
   raw receipts (penalties / minutes / games), which also makes clear that the teams
   *above* Argentina are 3-game small samples while Argentina's rate is over 7 games.
3. **`03_argentina_penalty_timeline.png`** — the "recent, not career-long" nuance in
   one glance (zero penalties in 2010 & 2014, spike in 2022/2026).

Deeper cuts for a stats-literate audience: `rankings/09b_deviation_banded_2022_2026.png`
(the deviation evidence for the last two cups) and
`by_world_cup/10_deviation_by_opponent_argentina_2022.png` (every team ranked with
flags, Argentina highlighted, with a permutation p-value — Argentina is #1 in penalties
conceded and #2 in cards that tournament; a `top_teams` variant highlights the other
elite sides for comparison).

---

## Data

| Tournament | Matches used | Source | Notes |
|---|---|---|---|
| 2010 | 64 | ESPN box scores | complete |
| 2014 | 64 | ESPN box scores | complete |
| 2018 | 64 | ESPN box scores | complete (validated vs StatsBomb) |
| 2022 | 64 | ESPN box scores | complete (validated vs StatsBomb) |
| 2026 | 96 | ESPN box scores | **in progress** — through the Round of 16 at the data pull |

**Why ESPN as the primary source.** We need one *consistent* methodology across all
five tournaments (mixing sources would make cross-tournament *levels* incomparable).
ESPN's public match API exposes per-team `foulsCommitted`, `yellowCards`,
`redCards`, and `penaltyKickShots` (penalty *shootout* kicks are correctly
excluded), plus a status flag (`FT` / `AET` / `FT-Pens`) that tells us whether a
match went to extra time.

**Match length, by design.** Per the study brief we do **not** count stoppage time
(it merely compensates for dead time). A match is treated as **90 minutes**
(regulation) or **120 minutes** (extra time). All rate metrics are "per 90", so a
long knockout game can't masquerade as an unusually foul-heavy one.

**Source validation.** StatsBomb open-data has full event data for 2018 & 2022. We
independently rebuilt those two tournaments from StatsBomb and compared to ESPN
(per team-tournament totals):

```
fouls          r=0.987   cards          r=0.979
pens_won       r=0.968   pens_conceded  r=0.960
```

Penalty totals were near-identical (ESPN 50 vs StatsBomb 52 across both cups; the
real figures are 29 in 2018 and 23 in 2022 — StatsBomb nails both). ESPN counts ~9%
fewer fouls than StatsBomb, but the correlation is 0.99, so *relative* comparisons
between teams are unaffected. This makes ESPN trustworthy for the years StatsBomb
can't cover.

---

## Metrics & method

Every row of `data/matches_long.csv` is one team's involvement in one match
(fouls committed, cards received, penalties won/conceded, extra-time flag, result).

**1. Absolute rates (all teams, Argentina highlighted).** Penalties won/conceded,
fouls, and cards — per game and per-90, split group vs knockout. For each we report
Argentina's rank, percentile and z-score across the field.

**2. Strength control.** Good teams attack more and might earn more penalties
legitimately. We regress each team's penalties-won-per-game on its goal difference
per game (a dominance proxy) and check where Argentina sits versus the prediction.

**3. Deviation-by-opponent — the centrepiece.** This is the "controls for
foul-heavy opponents" idea from the brief. For every team we compute a **baseline**
(its leave-one-out average *within the same tournament*), then its **deviation** in
each match. Grouping those deviations by *opponent* tells us: when a team plays
Argentina, do they foul / get carded / concede penalties **more than they normally
do**? A **permutation test** (20,000 reshuffles) asks whether "facing Argentina" is
genuinely different from facing a random team, or just small-sample noise.

---

## Key findings

### 1. Penalties won: Argentina is near the top of the entire field
`0.32 penalties/game` vs a field average of `0.10` — **rank 2 of 52** teams, 96th
percentile, **z = +2.2**. Holds in both phases: group `0.33` vs field `0.11`;
knockout `0.31` vs field `0.14`. *(chart 01)*

### 2. It is not just "they're a good team"
After regressing penalties on dominance, Argentina still sits **+0.19 penalties/game
above expectation (94th-percentile residual)**. Their strength explains only a sliver
of the gap. *(chart 02)*

### 3. But it is recent, not career-long
Penalties won per game by tournament: **2010: 0.00 · 2014: 0.00 · 2018: 0.25 ·
2022: 0.71 · 2026: 0.60**. In 2014 they reached the *final* and were awarded **zero**
penalties. The entire signal is 2022 onward. *(chart 03)*

### 4. Opponents genuinely deviate from their norms against Argentina — significantly
Across Argentina's 28 games, opposing teams exceed their own within-tournament
baselines by:

| Metric (per 90) | Deviation | z | permutation p | Argentina's rank |
|---|---|---|---|---|
| Fouls committed | **+2.0** | +2.2 | **0.015** | 5 / 47 |
| Cards received | **+0.65** | +2.5 | **0.008** | 4 / 47 |
| Penalties conceded | **+0.14** | +2.1 | **0.025** | 5 / 47 |

All three are statistically significant. *(charts 04–05)*

### 4b. …and the significance is entirely a 2022/2026 phenomenon
Collapsing the three metrics into one **goal-impact-weighted Referee Pressure Index**
(the single most appropriate summary — a penalty counts ~30× a foul) and testing each
World Cup on its own: Argentina's opponents are whistled **above their own norm**
significantly only in the last two tournaments.

| World Cup | Opp. RPI deviation /90 | z | permutation p | Argentina's rank |
|---|---|---|---|---|
| 2010 | −0.06 | −0.5 | 0.694 | 21 / 32 |
| 2014 | −0.10 | −0.7 | 0.741 | 22 / 32 |
| 2018 | +0.15 | +0.7 | 0.228 | 9 / 32 |
| **2022** | **+0.52** | **+3.0** | **0.004** | **1 / 32** |
| **2026** | **+0.44** | **+2.7** | **0.009** | **2 / 48** |

In 2010 and 2014 Argentina actually sat in the *bottom half* of the field. The tilt
switches on in 2022 (rank #1) and holds in 2026 (rank #2). *(chart 11)*

The same test applied to the other elite teams shows the tilt is **real but not
Argentina's alone**: Spain (2010) and France (2014, 2018) were the significant sides
before Argentina ever was. *(chart 12)* And it is **not outlier-driven** — dropping
each team's single most extreme opponent-game (high and low) leaves every significant
result intact, including Argentina 2022/2026 and even Spain 2010, whose signal
therefore isn't just the 9-card 2010 final. *(chart 13)*

### 5. Which specific games tilted their way?
We define a **Referee Pressure Index (RPI)** per team-game — how much the whistle
went against a team, in **goal-impact units**. Each event is weighted by its
approximate expected-goals swing, so a penalty counts ~30× a foul rather than
equally:

```
RPI = 0.76·penalties_conceded + 0.50·reds + 0.06·yellows + 0.025·fouls   (all per 90)
```

For every Argentina game we compare the *opponent's* RPI to that opponent's own
tournament norm. The opponent was whistled **above their norm in 19 of 28** Argentina
games; Argentina was **below their norm in 16 of 28** (and about at their norm on
average — they are not systematically *under*-whistled). Most lopsided in Argentina's
favour: **Iceland '18, Saudi Arabia '22, Netherlands '22, Croatia '22, Jordan '26** —
games where Argentina won a penalty *and* the opponent was carded/fouled more than
usual. The clearest game *against* them was the **2022 Final vs France** (Argentina
conceded two penalties to Mbappé) — a reminder the effect isn't universal.
*(chart 07; full ranking in `output/tables/anomalous_games.csv`.)*

### 6. …but Argentina is not uniquely singular
On the cards-deviation ranking Argentina is **4th**, behind Spain, Russia and
Canada; on fouls and penalties conceded it is **5th**. So the effect is real, yet a
handful of other teams show it as strongly or more so — evidence *against* the idea
that Argentina is treated uniquely, and a caution against any single cherry-picked
stat. Argentina themselves are **clean**: below-average fouls (rank 37/52) and cards
(28/52). *(chart 06 shows the opposition-cards angle.)*

---

## Charts (`output/graphs/`)

1. `01_penalties_won_ranking.png` — every team ranked, Argentina highlighted
2. `02_penalty_strength_control.png` — penalties vs dominance; Argentina's residual
3. `03_argentina_penalty_timeline.png` — the recency story, 2010→2026
4. `04_deviation_by_opponent_rankings.png` — opponent deviations, all teams, 3 metrics
5. `05_permutation_significance.png` — Argentina's value vs the null distribution
6. `06_opposition_cards_drawn.png` — cards drawn by Argentina's opponents (minor stat)
7. `07_anomalous_games_scatter.png` — expected vs observed referee-pressure per game; Argentina's opponents (orange) vs Argentina itself (blue), most anomalous games labeled
8. `11_significance_by_tournament.png` — forest plot, one row per World Cup: Argentina's composite Referee-Pressure opponent-deviation vs a random team's range (central 90% of the permutation null), with per-cup p-value and rank. Shows the effect is n.s. in 2010/2014/2018 and significant in 2022/2026
9. `12_significance_by_tournament_top_teams.png` — the same test for all five elite teams (Argentina + Spain/England/Brazil/France) in one chart: each World Cup is a lane and each team is a dot with its **own** null band; colour encodes identity (Argentina orange, others blue) and significance (bright = p<0.05, pale = n.s.). Makes clear that Spain (2010) and France (2014/2018) were the significant sides early, and only in 2022/2026 does Argentina light up (with England in 2026) — the "real, recent, not unique" story in a single frame
10. `13_significance_by_tournament_top_teams_robust.png` — a robustness check: chart 12 recomputed after dropping **each team's single most extreme opponent-game (highest and lowest)**, so no lone freak match (e.g. the 2010 final, in which the Netherlands drew 9 cards) can carry a result. Every significant finding survives — Argentina 2022 (p 0.003→0.004) and 2026 (0.010→0.007), and notably Spain 2010 too (0.009→0.006), so its signal was **not** just the final

**`graphs/share/`** — the polished, ready-to-post visuals (see "Best charts to share" above):
- `summary_scorecard.png` — square Instagram card of the headline findings
- `penalties_per90_top10_flags.png` — flagged top-10 with raw penalty/minute counts

**`graphs/rankings/`** — full-field banded views (top 10 / median 10 / bottom 10, with skipped counts; Argentina always surfaced even when in a skipped band):
- `08_penalties_per90_by_team_year.png` — penalties won per 90, one bar per team per World Cup (176 team-tournaments)
- `09_deviation_by_opponent_banded.png` — the chart-04 deviations across the full field, banded, all three metrics
- `09b_deviation_banded_2022_2026.png` — same, restricted to the two most recent World Cups

**`graphs/by_world_cup/`** — per-tournament breakdowns:
- `05_permutation_<year>.png` — the significance test within a single tournament (2014/2018/2022/2026)
- `06_opposition_cards_<year>.png` — cards drawn by the opposition, ranked, per tournament
- `10_deviation_by_opponent_{argentina,top_teams,top_teams_only}_<year>.png` — the chart-04 idea per tournament (2018/2022/2026): every team ranked by its opponents' deviation, country flags + 3-letter codes on the y-axis, and permutation p-values. Three variants: `argentina` highlights Argentina (blue); `top_teams` highlights Spain/England/Brazil/France (green) with Argentina still marked blue; `top_teams_only` shows just those five teams head-to-head with **no highlight colour** (flags + 3-letter codes identify them; each bar still labelled with its rank among the full field)

Tables and the machine-readable `findings.json` are in `output/tables/`.

---

## Caveats

- **Correlation, not causation.** A high "opponent deviation" is consistent with
  refereeing bias *or* with Argentina's style (drawing fouls, provoking cards,
  winning penalties through skill). This study measures the pattern, not its cause.
- **2026 is incomplete** (through the Round of 16) and provisional; re-running will
  pick up later matches automatically.
- **Small samples.** Teams play 3–7 games per cup; individual deviations are noisy.
  The permutation test and pooling across five cups (28 Argentina games) are how we
  guard against reading noise as signal.
- **On-pitch player cards only.** Neither source reliably captures bench/staff cards
  (e.g. the true total in the 2022 Argentina–Netherlands match was higher than any
  box score shows). This is consistent across all teams, so comparisons are fair.
- **Referee-level analysis was intentionally out of scope** (team-level only, per the
  brief). The data to add it later (referee identity) is available from StatsBomb.

---

## Reproduce

```bash
# full run: pull data (cached after first time), analyse, render charts
python -m studies.world_cup_refereeing_bias.main

# re-use the cached dataset, just re-analyse + re-render
python -m studies.world_cup_refereeing_bias.main --no-build
```

Raw API responses are cached under `data/cache/` so re-runs are offline and fast.

### Module map
- `common.py` — the shared match-team schema and team-name canonicalisation
- `espn_ingest.py` — ESPN box-score ingestion (all five cups)
- `statsbomb_ingest.py` — StatsBomb event ingestion (2018/2022, validation)
- `build_dataset.py` — assembles the CSV + runs the source-agreement check
- `analysis.py` — rankings, strength control, deviation, permutation test
- `style.py` / `visualize.py` — chart styling and generation
- `main.py` — orchestrates everything and writes `findings.json`
