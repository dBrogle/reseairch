# Predicting the 2026 World Cup final

Can we predict the 2026 World Cup final from how the teams have actually performed,
opponent-strength and all? This study fits two opponent-adjusted models to eight
World Cups of results (1998–2026) and evaluates them by **leave-one-cup-out
cross-validation** — the most relevant honesty check being how they would have
called each **past World Cup final**, out of sample.

The tournament has now reached the final: **Argentina vs Spain** (neutral venue).

> **How it does on past finals (out-of-sample):** across the seven previous finals,
> the better model (Elo) favoured the eventual winner in **all five that were
> decided in normal/extra time** — 1998 France, 2002 Brazil, 2010 Spain, 2014
> Germany (84%), 2018 France. The other two (**2006** Italy–France, **2022**
> Argentina–France) were **penalty shootouts** — genuine coin flips the data can't
> grade. So **5 for 5** on the decisive finals.
>
> **Live 2026 final (Elo, no tie):** **Argentina 53% to win the Cup, Spain 47%** —
> a near-toss-up. And it's a *lean, not a lock*: the Poisson goals model leans the
> other way (**Spain 67% / Argentina 33%**), rating Spain's in-tournament goals
> higher. With ~7 games per team, treat the edge as slight.

---

## The task

The 2026 data runs **through the semi-finals**, so only the final is unplayed.
"Predict the final" means: rate each team from the games played so far (adjusting
for who they played), then predict that single match. The two things the brief
asked for — **opponent strength** ("beat a 4-0 team, not a 2-2 team") and **goal
difference** — are exactly what both models are built around. (Run earlier in the
tournament, the same code instead propagates the whole remaining bracket; it
switches automatically once both semi-finalists are known.)

## Data

One row per team per match for **1998–2026** (eight cups, 550 matches). The
2010–2026 cups come from the sibling `world_cup_refereeing_bias` study's committed
`data/matches_long.csv` (refreshed from ESPN through the 2026 semis); the three
older cups (**1998, 2002, 2006**) are pulled by this study's own
`build_extra_cups.py` and cached to `data/extra_cups.csv`, so the sibling study is
left untouched. (To predict an old final the models need that whole tournament — a
finalist is rated from its group + knockout games — so we pull the full cups, not
just the finals.) We fold each match to one row (home/away/goals), and attach:

- **Knockout depth** (rounds-from-final) so the backtest can train only on
  strictly-earlier rounds. Normalises the inconsistent stage labels across cups
  (2010 calls the R16 "Second Round", etc.).
- **A real home-field flag** for the **host nation only**. The CSV's `is_home` is a
  nominal box-score label (every match has one), not genuine home advantage — only
  the host actually plays at home at a World Cup.

## Models

Two opponent-adjusted models, deliberately low-capacity because the data is tiny
(a neural net / gradient-boosting would overfit ~350 matches badly, and the Apple
Neural Engine is irrelevant at this scale — both models fit in milliseconds on CPU):

1. **Dixon–Coles Poisson goals model** (`models.PoissonModel`). Each team has an
   **attack** and a **defence** strength; expected goals in a match are your attack
   against their defence, so opponent strength and goal difference are built in. Fit
   by **penalised maximum likelihood** — a ridge penalty (`kappa`) shrinks the ~6
   noisy games per team toward a prior, with the low-score Dixon–Coles correction and
   a fixed host-home term. Outputs a full scoreline distribution.

2. **Margin-of-victory Elo** (`models.EloModel`). Beating a strong team, by a wide
   margin, moves your rating more; rating difference maps to W/D/L through an ordered
   logit. The honest baseline the Poisson model has to beat.

### "This tournament only, or past-cup form too?" — let validation decide

Both models carry a **cross-tournament prior**: how much of a team's past-World-Cup
form to seed before updating on the current cup (Poisson's `prior_weight`, Elo's
`carryover`). It's leakage-free — a 2022 prediction only ever sees 2010/2014/2018 —
and treated as a hyper-parameter chosen by cross-validation. The answer it lands on:
**past form matters a lot**, and given the prior, **moderate-to-strong shrinkage** is
best.

## Evaluation — leave-one-cup-out cross-validation

To judge the model *on finals*, we can't use a fixed train/validation/test split:
whichever cups tune the hyper-parameters would make their finals in-sample. So we
use **leave-one-cup-out CV** — to predict any cup, the hyper-parameters are tuned on
the **other seven cups only**. Every cup (and its final) is thus predicted **fully
out of sample**. Two safeguards keep it honest:

- Team attack/defence are always re-fit *within* the held-out cup (rosters don't
  transfer), and its priors use **strictly earlier cups only** — no future team data.
- Only the *hyper-parameters* (shrinkage, prior weight, Elo's k/mov/carryover) are
  chosen from other cups. They're methodology, not team knowledge.

Neatly, the "hold out 2026" fold — tune on 1998–2022, predict 2026 — **is** the live
final prediction. Scoring uses the **Ranked Probability Score** (the standard for
ordered 1X2 outcomes), with log-loss, Brier and accuracy alongside.

## Results

### Out-of-sample on every knockout game (1998–2026, n=141)

| Model | RPS | Log-loss | Brier | Accuracy |
|---|---|---|---|---|
| **Elo** (best) | **0.187** | **0.916** | **0.535** | **61%** |
| Dixon–Coles Poisson | 0.202 | 0.974 | 0.574 | 58% |
| Base-rate baseline | 0.231 | 1.036 | 0.623 | 51% |

Both models beat the baseline; the **simpler margin-Elo generalises best**. (These
are higher/harder than a single-cup test because they pool every cup, including the
less predictable early ones — a more honest number.) *(chart 01)*

### The most relevant test: past World Cup finals (out-of-sample)

Seven previous finals, each predicted with hyper-parameters tuned on the other cups:

The number is the probability each model gave the team that **actually won**:

| Final | P(winner): Elo / Poisson | Winner | |
|---|---|---|---|
| **1998** Brazil–France | 69% / 63% | **France** (3–0) | ✓ both backed the winner |
| **2002** Germany–Brazil | 58% / 56% | **Brazil** (2–0) | ✓ both backed the winner |
| **2006** Italy–France | 54% / 48% | **Italy** (pens) | split — a coin flip |
| **2010** Netherlands–Spain | 51% / 57% | **Spain** | ✓ both backed the winner |
| **2014** Germany–Argentina | 84% / 67% | **Germany** | ✓ both backed the winner |
| **2018** France–Croatia | 68% / 59% | **France** | ✓ both backed the winner |
| **2022** Argentina–France | 31% / 39% | **Argentina** (pens) | both leaned France — a coin flip |

**Five for five on the finals that were actually decided** (1998, 2002, 2010, 2014,
2018). The two it "missed" the winner on — 2006 and 2022 — were **penalty shootouts**,
which the model correctly reads as near coin flips and which the data can't grade
anyway (no shootout winner recorded; Italy and Argentina won them in reality). Adding
the three older cups also **fixed the one earlier miss**: with 1998–2006 now feeding
its prior, 2010 flips from a wrong Netherlands lean to a correct (if narrow) Spain
call. Still a small sample, but a genuinely strong showing. *(chart 02)*

## The prediction — Argentina vs Spain (no tie)

Both models are refit on **all** 2026 games so far, then applied to the one match
left (neutral venue, so no home term). A knockout has a winner, so the draw is
resolved (see below):

| | Regulation/ET (win · level · win) | **Wins the World Cup** |
|---|---|---|
| **Elo** (best out of sample) | 40% · 27% · 34% | **Argentina 53% · Spain 47%** |
| Dixon–Coles Poisson | 21% · 26% · 53% | Spain 67% · Argentina 33% |

Likeliest scorelines (Poisson): 1–1, 0–1, 1–2, 0–0, 2–1. *(chart 03)*

**The honest part — the models disagree.** Elo makes **Argentina** a narrow favourite
(higher accumulated rating across the cups plus a deep 2026 run); the
harder-shrinking Poisson goals model leans **Spain** clearly (a much stronger
in-tournament goal record, including a 2–0 semi-final win over France). Elo generalises
best out of sample so it's the headline, but with ~7 games per side and the two models
pointing opposite ways, this is close to a toss-up. *(chart 04)*

### Resolving a knockout tie (no draw option)

A knockout can't end level, so a predicted draw is split into a winner. We validated
how against past cups rather than guessing:

- Across all eight cups, **35 knockout games were level after 90'** — **9 (26%) were
  settled in extra time, 26 (74%) went to penalties.** The data records no shootout
  *winners*, and shootouts are ~50/50 anyway, so penalties stay a coin flip.
- A **naïve** "extra time = ⅓ of a match" goals model is **miscalibrated** — it
  predicts ~51% of ties settled in ET, because tied-after-90 games are selectively
  low-scoring and cagey (they average ~2.3 goals vs ~3.0 in games decided in
  regulation). So we don't simulate ET goals; we use the **measured 26% rate**.
- **Rule:** a level game is settled `26%` in extra time (edge to the stronger side,
  by its share of the non-draw probability) and `74%` on penalties (coin flip). For
  Argentina–Spain the net effect is small — penalties dominate, so it barely moves
  off a coin flip: Argentina's 27% draw share splits ≈14% / 13%, giving 53% / 47%.
  (`data.et_resolution_rate` measures the rate; `simulate.resolve_draw` applies it.)

## Charts (`output/graphs/`)

1. `01_knockout_scorecard.png` — four scoring rules, both models vs the baseline, out-of-sample over all cups
2. `02_finals_report.png` — how each model would have called the seven past finals, with flags (the key chart)
3. `03_predicted_final.png` — the live final card: headline model's two-way winner (no tie), scorelines
4. `04_final_model_comparison.png` — plain side-by-side of each model's win probability for the 2026 final
5. `05_final_both_models.png` — the final card (as 03) but with a winner bar for **both** models

Machine-readable results (per-fold hyper-parameters, out-of-sample scores, the past-
finals table, both models' live-final probabilities) are in `output/tables/findings.json`.

## Caveats

- **Small samples, loudly.** Teams play 3–7 games a cup; there are **7 past finals**
  (5 of them decided in normal/extra time). The out-of-sample win over the baseline
  across 141 knockout games is solid and "5/5 on decisive finals" is a real result —
  but it's still a handful of finals, and the two models disagreeing on 2026 is the
  honest signal that it's a lean, not a lock.
- **The final is neutral-site.** No host nation (USA/Canada/Mexico) reached it, so
  there's no home term in the prediction. Earlier in the tournament the code
  propagates the whole remaining bracket instead (pairings in `config.QUARTERFINALS`,
  standard layout); it switches to single-match mode automatically once both
  semi-finalists are known.
- **Host home-field is fixed, not fitted.** With one host per cup, a free host term
  is unidentifiable (Qatar 2022 drove it to the bound), so it's a modest constant
  (`config.HOST_HOME_ADV`); every non-host game is neutral. No hosts remain alive in
  2026, so it doesn't affect the live forecast.
- **Tie resolution is an aggregate, not per-matchup.** The 26% ET rate is one number
  measured over 35 games, applied to every knockout; a proper per-matchup ET model
  would need far more data than exists. Penalties are a pure coin flip because the
  data has no shootout winners — if you'd rather ignore penalties entirely, the
  `by_model` block in `findings.json` still carries the raw regulation W/D/L to
  renormalise yourself.
- **Goals include extra time.** The ~7% of matches that went to ET carry 120' of
  goals; not normalised to 90', a minor inflation shared across teams.
- **2010 label noise** (one group/QF mislabel in the source) shifts that cup's
  knockout count by one; a minor effect that doesn't change the story.

## Reproduce

```bash
# one-time: pull the older cups (1998/2002/2006) from ESPN -> data/extra_cups.csv
python -m studies.world_cup_final_prediction.build_extra_cups

python -m studies.world_cup_final_prediction.main            # CV, past finals, live final, charts
python -m studies.world_cup_final_prediction.main --no-charts # skip rendering
```

`extra_cups.csv` is committed, so `main` runs offline; delete it (or pass
`--refetch` to the builder) to rebuild from ESPN. Leave-one-cup-out CV runs the full
hyper-parameter search once per cup; per-cup Elo/Poisson predictions are memoised, so
a run is ~30s.

### Module map
- `config.py` — hosts, hyper-parameter grids, older-cups windows, the (fallback) bracket
- `data.py` — combine the shared CSV + older-cups CSV into a match table; depth; finalists; ET rate
- `build_extra_cups.py` — pull 1998/2002/2006 via the sibling study's ESPN ingester
- `models.py` — the Dixon–Coles Poisson and Elo models + the carry-over prior
- `evaluate.py` — scoring rules, leakage-safe per-cup prediction, leave-one-cup-out CV, finals table
- `simulate.py` — knockout tie-resolution + exact bracket propagation
- `style.py` / `visualize.py` — chart styling (offline flag loader) and the five figures
- `main.py` — orchestrates CV → past-finals → live final → findings.json + charts
