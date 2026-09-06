# DraftDB — the redraft model

We re-ran every NBA draft from 2019 to 2025 with a tabular model that sees only what was publicly knowable before draft night: college and international box scores, measurements, recruiting and scouting-report grades, and the final mock-draft consensus. Then we checked whose order better matched how the players actually turned out — the model, or the NBA teams that made the real picks.

| AI model accuracy | NBA scouts accuracy | Drafts the AI won |
|:---:|:---:|:---:|
| **51%** | **24%** | **5 of 6** |

*Current model: EVO gen11 (`model/CURRENT_MODEL.json`), scored on the expanding window, 2020–2025. Strict scoring (the model trained on 2007–2018 only, scored on 2019–2025): **45%** vs **26%**, **5 of 7** drafts won. The previously shipped model scored 44% strict.*

*Order accuracy = Spearman rank correlation between a draft order and the drafted players' actual WAR ranking, as a percentage. 100% is a perfect order, 0% is no relationship. Players are ranked by WAR over the NBA seasons they have completed so far (2019–2021: five seasons; 2022: four; 2023: three; 2024: two; 2025: one).*

*Expanding window = how the model would actually be used: to score the 2023 draft it is trained on every class up to 2022, using only the NBA seasons those players had completed by draft night 2023. Strict = the model never sees anything after the 2018 class. The 2019 class has no expanding row because nothing precedes it in the test period.*

## Where the edge is, and where it is not

How many of each order's first N picks were actually among the true best N by WAR, summed over the seven test classes (computed on the previously shipped model, v2 gen17; the current model's gains are also in the middle of the draft, see the fold analysis in `docs/EXPERIMENTS.md`):

| | AI | NBA teams |
|---|---|---|
| top-5 hits (of 35) | 17 | 17 |
| top-10 hits (of 70) | 30 | 31 |
| top-20 hits (of 140) | 80 | 76 |

At the top of the board the model and the teams are a dead heat. The 18-point gap in order accuracy is earned in the middle and bottom of the draft: ordering picks 15 to 58, avoiding lottery busts, finding second-round value. The correlation treats a swap at picks 40 and 45 the same as a swap at picks 2 and 12; the value-of-order metric below weights the top more, and there the gap is about 5 points. The training label is a within-class rank that rewards every position equally, so nothing in the objective says the top ten matter more — a top-weighted label and a star-probability member are the next things being tested.

## Per class

Current model (EVO gen11). Strict = trained on 2007–2018 only. Expanding = earlier test classes join training with the seasons they had completed by that draft night.

| class | seasons scored (k) | drafted players | AI strict | NBA scouts | AI expanding | training rows (expanding) | won |
|---|---|---|---|---|---|---|---|
| 2019 | 5 | 58 | 34% | 40% | – | – | scouts |
| 2020 | 5 | 58 | 30% | 35% | 33% | 1,098 | scouts |
| 2021 | 5 | 56 | 62% | 41% | 62% | 1,210 | AI |
| 2022 | 4 | 52 | 63% | 26% | 60% | 1,443 | AI |
| 2023 | 3 | 56 | 40% | 11% | 53% | 1,540 | AI |
| 2024 | 2 | 55 | 48% | 13% | 57% | 1,641 | AI |
| 2025 | 1 | 57 | 40% | 18% | 39% | 1,755 | AI |
| **mean** | | | **45%** | **26%** | **51%** (scouts 24%) | | **5 of 7 strict, 5 of 6 expanding** |

*The 2019 and 2020 classes are the two the NBA teams ordered better. 2019 had a consensus top (Zion, Morant, Barrett) that was right; 2020 was the pandemic class with no combine and no tournament. Every class from 2021 on, the model's order beat the real draft by 0.21 to 0.43 in rank correlation. Boards for all seven classes, with names and actual picks: `model/board_current_2019_2025.csv`.*

### Example: the 2022 class, scored on four seasons

*(This example is the board of the previously shipped model, v2 gen17. The current model's 2022 board is in `model/board_current_2019_2025.csv`.)*

Fifty-two drafted players. The teams' order correlates with the four-season WAR ranking at 26%, the model's at 52%; value of the order 74.7% for the teams, 81.1% for the model. Left: what the model would have done with the first 15 picks. Right: what the teams did.

| model pick | player | real pick | 4-yr WAR | WAR rank | | real pick | player | 4-yr WAR | WAR rank | model rank |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Chet Holmgren | 2 | 19.9 | 2 | | 1 | Paolo Banchero | 18.5 | 3 | 4 |
| 2 | Mark Williams | 15 | 8.4 | 9 | | 2 | Chet Holmgren | 19.9 | 2 | 1 |
| 3 | Jabari Smith Jr. | 3 | 7.5 | 12 | | 3 | Jabari Smith Jr. | 7.5 | 12 | 3 |
| 4 | Paolo Banchero | 1 | 18.5 | 3 | | 4 | Keegan Murray | 8.4 | 10 | 5 |
| 5 | Keegan Murray | 4 | 8.4 | 10 | | 5 | Jaden Ivey | -0.5 | 38 | 27 |
| 6 | Jeremy Sochan | 9 | -0.1 | 32 | | 6 | Bennedict Mathurin | -1.3 | 46 | 9 |
| 7 | Jalen Duren | 13 | 17.1 | 4 | | 7 | Shaedon Sharpe | 2.7 | 19 | 21 |
| 8 | Christian Braun | 21 | 8.1 | 11 | | 8 | Dyson Daniels | 15.4 | 5 | 20 |
| 9 | Bennedict Mathurin | 6 | -1.3 | 46 | | 9 | Jeremy Sochan | -0.1 | 32 | 6 |
| 10 | Jake LaRavia | 19 | 3.5 | 16 | | 10 | Johnny Davis | -1.9 | 47 | 34 |
| 11 | Wendell Moore Jr. | 26 | -0.1 | 30 | | 11 | Ousmane Dieng | -0.6 | 39 | 36 |
| 12 | Dalen Terry | 18 | 0.8 | 24 | | 12 | Jalen Williams | 22.4 | 1 | 19 |
| 13 | Jaylin Williams | 34 | 8.7 | 7 | | 13 | Jalen Duren | 17.1 | 4 | 7 |
| 14 | Nikola Jović | 27 | 3.2 | 18 | | 14 | Ochai Agbaji | -0.8 | 41 | 31 |
| 15 | TyTy Washington Jr. | 29 | -0.9 | 42 | | 15 | Mark Williams | 8.4 | 9 | 2 |

The ten best four-season outcomes of the class and where each order had them:

| WAR rank | player | 4-yr WAR | real pick | model pick |
|---|---|---|---|---|
| 1 | Jalen Williams | 22.4 | 12 | 19 |
| 2 | Chet Holmgren | 19.9 | 2 | 1 |
| 3 | Paolo Banchero | 18.5 | 1 | 4 |
| 4 | Jalen Duren | 17.1 | 13 | 7 |
| 5 | Dyson Daniels | 15.4 | 8 | 20 |
| 6 | Walker Kessler | 12.9 | 22 | 16 |
| 7 | Jaylin Williams | 8.7 | 34 | 13 |
| 8 | Tari Eason | 8.6 | 17 | 17 |
| 9 | Mark Williams | 8.4 | 15 | 2 |
| 10 | Keegan Murray | 8.4 | 4 | 5 |

Reading it: both orders had Holmgren and Banchero in the top four and both took Jabari Smith third. The model's gain came in the middle of the first round — Mark Williams at 2 (taken 15th, 8.4 WAR), Jalen Duren at 7 (taken 13th, 17.1 WAR), Christian Braun at 8 (taken 21st), Jaylin Williams at 13 (taken 34th, 8.7 WAR) — and from what it refused to take: Jaden Ivey 27th, Johnny Davis 34th and Ousmane Dieng 36th, three lottery picks who have produced nothing yet. Its misses are just as visible: it took Jeremy Sochan sixth and Bennedict Mathurin ninth, and it had Jalen Williams, the best player in the class, 19th and Dyson Daniels 20th, both worse than the teams did.

### How to read the table

- **class** — the draft year; the pool is every player actually drafted in rounds 1–2 who has a row in the data.
- **seasons scored (k)** — NBA seasons that class has completed, capped at 5: a 2025 pick is judged on one season, a 2019 pick on five.
- **outcome** — each player's WAR (wins above replacement) summed over those k seasons; a player who never played scores 0.
- **AI strict** — Spearman rank correlation between the model's order and the WAR order, times 100, for a model trained on the 2007–2018 classes only: 100% is the identical order, 0% no relationship, negative means backwards.
- **AI expanding** — the same statistic when the model is also trained on every earlier test class, using only the NBA seasons those players had completed by the scored draft night. This is how the model would be used for a real draft.
- **NBA scouts** — the same statistic for the real pick order.
- **training rows (expanding)** — labeled rows available at that draft night.
- **won** — the AI's correlation beat the teams' for that class (strict; the expanding count is in the mean row).
- **mean** — the plain average over the classes scored.

### Findings, one line each

- On the expanding window the model orders a draft better than the teams did in 5 of 6 classes, by 27 points on average (51% vs 24%); strict, 5 of 7 by 19 points (45% vs 26%).
- It loses 2019 and 2020: the consensus top of 2019 (Zion, Morant, Barrett) was right, and 2020 was the pandemic class with no combine and no tournament; the model under-rated the shooters (Edwards, Garland, Herro, Maxey).
- The edge comes from the middle and bottom of the draft: ordering picks 11–40, bust avoidance, second-round value; at the top of the board the model and the teams are a dead heat.
- The current model still has Wembanyama seventh in the 2023 class (Anthony Black first); it wins 2023 on order (40% strict, 53% expanding vs 11%) because it ordered the rest of the class well, not because it got the top pick right.
- Recent classes are scored on few seasons, so their numbers will move: 2024 rests on two seasons, 2025 on one.
- A single class's correlation has a standard error near 0.13, so one-class wins and losses mean little; a six- or seven-class mean (about ±0.05) is what to read. The 55% target is one such step above the current 50.6%.
- Widening the training window to 2007 was the only tested change that improved blind results; every added data block and model family since then failed the walk-forward gate.
- The pool is drafted players only, so this measures how to order a draft, not whom to take from outside it.

## How it was tested

**Current model: training 2007–2018 (1,040 labeled rows). Test: 2019–2025, scored strict and on the expanding window (2026 has no outcomes yet).** Model selection (label, features, members, weights, training window) used walk-forward folds *inside* the training window only: train on classes before year y, predict class y. The current model was accepted under the gate of the time (folds 2014–2018: gain +0.020 over its parent, 4 of 5 folds). Since 2026-09-06 the gate is stricter: seven folds (2012–2018) whose training labels only use the NBA seasons that were complete at that fold's draft night, more weight on the recent folds, a mutation must win 6 of 7 folds including 2 of the 3 latest, and every accepted change is re-evaluated with different model seeds before it becomes champion. Nothing from the blind classes is ever used to choose a model; blind numbers are the report. The test outcomes live in a hash-locked vault on the training box, the scoring function returns aggregates only, and every blind scoring is appended to a hash-chained ledger (`experiments/ledger.jsonl`, 1,889 evaluations at publication).

## Runs

| when (UTC+8) | model | AI strict | NBA scouts | AI expanding | drafts the AI won |
|---|---|---|---|---|---|
| 2026-09-06 | **EVO gen11 · training window 2007 · data v3.2** — current | 45% | 26% | 51% (scouts 24%) | 5 of 7 strict · 5 of 6 expanding |
| 2026-09-06 | v2 gen17 genome re-scored on the rebuilt data v3.2 | 45% | 26% | 50% | 3 of 7 |
| 2026-09-05 13:11 | BEST v2 gen17 · data v3 (rebuilt international block + measurements) — previously shipped | 44% | 26% | – | 5 of 7 |
| 2026-09-05 13:08 | BEST v2 gen17 · data v3.0 (rebuilt international block only) | 45% | 26% | – | 5 of 7 |
| 2026-09-05 04:12 | evolution champion gen86 · data v1 (walk-forward pick, not shipped) | 43% | 26% | – | 6 of 7 |
| 2026-09-05 02:05 | overnight queue final (hz+covw) · data v1 | 44% | 26% | – | 5 of 7 |
| 2026-09-05 00:51 | BEST v2 gen17 · data v1 | 46% | 26% | – | 5 of 7 |

The differences between these rows are inside the noise floor (a single season's correlation has a standard error of about 0.12; a seven-season mean about 0.045). The 46% row was scored on data v1, which still contained stale international seasons (post-draft information) for some players; on the corrected data the same genome scores 45%.

## The model (`model/CURRENT_MODEL.json`)

Exact settings of the current model, EVO gen11:

- **Training window:** draft classes 2007–2018 (the previous model used 2010–2018; adding 2007–2009 was the one change that improved blind results).
- **Label:** WAR over the first five NBA seasons, discounted 0.85 per season (`Σ 0.85^(i−1)·WAR_i`), clipped at 40, then Gaussian-rank transformed within each draft class. The model learns *where in his class* a prospect finished.
- **Inputs:** the base pre-draft columns (biography and measurements, final college season per-36 and rate stats with age/class/role percentiles, last international season, recruiting, process flags, injury flags, scouting-report grades, mock-draft consensus) plus the `shoot+prod` engineered features: free-throw × three-point volume, shot-projection × volume, a shooting score, production-over-age, usage × efficiency, rim protection (blocks × height), and young/old production terms. The optional data blocks (Torvik, text, trajectory, attention, Trends, recruiting spread, combine athletics) are all present in the data and all switched off: none passed the gate.
- **Sample-size shrinkage:** every rate stat is pulled toward the training-set prior in proportion to minutes played, `w = minutes / (minutes + 400)`.
- **Members:** (1) TabICL, an in-context tabular foundation model, 32 estimators, on the 100 inputs with the highest XGBoost importance, no input normalization, outlier threshold 2; (2) ridge regression, alpha 300, on standardized inputs; (3) XGBoost, three seeds, depth 3, 800 trees, quantile objective at 0.25 (a conservative, floor-seeking estimate), age constrained monotone, fitted on the residual of the label after TabICL's out-of-fold rank.
- **Stack:** within-class rank average with coverage-dependent weights. Rows with normal input coverage: TabICL 0.25 + ridge 0.75. Rows whose coverage is below their class median (thin rows, typically international or low-minute prospects): XGBoost 1.0. Weights were learned on the walk-forward folds.
- **Scoring pool:** the players actually drafted (52–58 per class). Training uses every labeled row, drafted or not.

The previously shipped model is kept as `model/BEST_MODEL.json` (XGBoost + TabICL 50/50, window 2010).

## What changed since the previously shipped model (2026-09-05 → 2026-09-06)

- **Headline metric is now the expanding window.** It is how the model is used in practice and it is where the current model stands at 51% vs 24% for the real draft order.
- **Training window widened to 2007.** +3 classes, +270 rows. Better on 2021, 2022 and 2025; worse on 2019, 2020 and 2024. Windows starting in 2000 or 2003 were tested and hurt (−0.024 to −0.027 on the folds), because the 2000–2004 classes have no mock-draft consensus in the data.
- **Data corrections:** stale international seasons removed and rebuilt from the latest pre-draft season (this had leaked post-draft information into the old 46% number), 86 wingspans and 39 standing reaches filled from published measurements, official combine anthropometrics added, Wembanyama moved from 12th to 3rd in the 2023 board on data alone.
- **Data added and tested:** Google Trends pre-draft interest, RSCI recruiting-service spread, Torvik advanced stats and team context, Wikipedia attention, rule-based scouting-text counts, year-over-year trajectories, coach and program pipelines, mock-draft momentum, NBA combine athletic tests and drills, G League post-draft production as a label tiebreaker, the 2000–2009 classes. Every block failed the walk-forward gate alone and in combination (details in `docs/EXPERIMENTS.md`).
- **Model families tested and rejected:** a pairwise within-class "who beats whom" learner (better than XGBoost alone, no gain inside the stack), class- and position-relative standardization (hurt: absolute stat levels carry information), fitted stacking (ridge, NNLS), CatBoost, ExtraTrees, LightGBM, ranking objectives, star classifiers, path and tier specialists.
- **Selection became stricter.** The earlier search was fitting its five folds: island champions rose on the folds while falling blind (one lineage went from 43.7% to 40.7%). The gate now uses seven season-consistent folds, recency weights, a 6-of-7 rule and seed-shifted confirmation, and blind accuracy of accepted champions is tracked separately.
- **Where the gains came from:** the fold-level analysis shows the window change improved the ordering of drafted players ranked 11–30 by consensus and of the youngest prospects, and worsened undrafted and fringe international players who are never scored. The middle of the first round is where the remaining error lives.
- **Target status:** the 55% expanding target was not reached. Best expanding accuracy 50.6%; the noise floor of a six-class mean is about ±5 points.

## Repository layout

| path | what |
|---|---|
| `model/CURRENT_MODEL.json` | current model: exact genome, stack weights, strict and expanding results per class |
| `model/board_current_2019_2025.csv` | the current model's boards for 2019–2025 with names and actual picks |
| `docs/TEXT_FEATURES.md` | how the rule-based scouting-text features were computed (word lists, no language-model scoring) |
| `index.html`, `runs.html`, `compare.html`, `css/`, `js/` | the static site; `runs.html` shows the table above |
| `data/train_2000_2018.csv` | training rows 2000–2018 (the model uses 2010–2018), inputs + outcome columns (`y_*`) |
| `data/tests/test_YYYY_inputs.csv` | test-class inputs, 2019–2026 (every drafted player, early entrants who stayed in, undrafted players who later made the NBA) |
| `data/answers/answers_YYYY.csv` | test-class outcomes (WAR per season, tiers); on the training box these sit inside the vault |
| `data/input_columns.json`, `data/FEATURES.md`, `data/README.md` | column lists and block descriptions |
| `data/patches/`, `data/DATA_V3_CHANGES.md` | the pid-keyed corrections applied on 2026-09-05 (international block, college fills, measurements) and the log of what changed |
| `data/data_v2/` | optional EuroLeague-API and Wikipedia feature blocks (tried, not used by the shipped model) |
| `model/BEST_MODEL.json` | the shipped genome and feature weights |
| `model/best_results.json`, `model/EVAL_V3.json` | blind results per season for the frozen model (data v1 and v3) |
| `model/board_preds_v3.csv` | the model's score for every drafted player in every test class (pid-keyed) |
| `code/` | `honest.py` (leak-proof feature engine and members), `vault.py` (hash-locked outcomes, ledger), `tripwire.py` (18 cheat attempts that must be blocked), `evolve3.py` (search + genome definitions), `overnight.py`, `eval_v3.py`, `ablate_v3.py`, `apply_patches.py`, `dash.py` + `stack.html` (dashboard) |
| `experiments/` | the ledger, the overnight queue log, the evolution history, the ablations |
| `docs/` | methods, experiments, data rebuild report, label definitions |

Player names are not in this repository: every row is keyed by an anonymous `pid`. The name key is kept off the training machine so the model can never see it.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate && pip install xgboost tabicl pandas numpy scipy pyarrow
python code/tripwire.py            # must print ALL GUARDS HOLD
python code/eval_v3.py             # frozen model: walk-forward folds, one sealed blind scoring, boards
```

`code/` expects the layout of the training box (`data/`, `vault/`, `r7/`); see the header of each script.
