# DraftDB — the redraft model

We re-ran every NBA draft from 2019 to 2025 with a tabular model that sees only what was publicly knowable before draft night: college and international box scores, measurements, recruiting and scouting-report grades, and the final mock-draft consensus. Then we checked whose order better matched how the players actually turned out — the model, or the NBA teams that made the real picks.

| AI model accuracy | NBA scouts accuracy | Drafts the AI won |
|:---:|:---:|:---:|
| **44%** | **26%** | **5 of 7** |

*Order accuracy = Spearman rank correlation between a draft order and the drafted players' actual WAR ranking, as a percentage. 100% is a perfect order, 0% is no relationship. Averaged over the seven test drafts. Players are ranked by WAR over the NBA seasons they have completed so far (2019–2021: five seasons; 2022: four; 2023: three; 2024: two; 2025: one).*

## Per class

| class | seasons scored (k) | drafted players | AI accuracy | NBA scouts | won | value of the order: NBA teams | value of the order: AI |
|---|---|---|---|---|---|---|---|
| 2019 | 5 | 58 | 39% | 40% | – | 74.6% | 73.7% |
| 2020 | 5 | 58 | 31% | 35% | – | 72.7% | 70.4% |
| 2021 | 5 | 56 | 63% | 41% | ✓ | 79.3% | 84.0% |
| 2022 | 4 | 52 | 52% | 26% | ✓ | 74.7% | 81.1% |
| 2023 | 3 | 56 | 35% | 11% | ✓ | 75.0% | 71.6% |
| 2024 | 2 | 55 | 60% | 13% | ✓ | 55.9% | 80.7% |
| 2025 | 1 | 55 | 30% | 18% | ✓ | 67.0% | 75.2% |
| **mean** | | | **44%** | **26%** | **5 of 7** | **71.3%** | **76.7%** |

*Value of the order: the WAR actually delivered by each draft slot, weighted linearly by slot (pick 1 counts most), scaled so the perfect order = 100% and the worst possible order = 0%; a random order scores 50%. Boards are in `model/board_preds_v3.csv` (pid-keyed).*


### Example: the 2022 class, scored on four seasons

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
- **AI accuracy** — Spearman rank correlation between the model's order and the WAR order, times 100: 100% is the identical order, 0% no relationship, negative means backwards. It is a correlation, not "percent of picks correct".
- **NBA scouts** — the same statistic for the real pick order.
- **won** — the AI's correlation beat the teams' for that class.
- **value of the order** — the WAR captured at each slot, weighted by slot (pick 1 heaviest), scaled so the perfect order is 100%, the worst 0%, a random order 50%. It rewards getting the top of the board right; the correlation does not.
- **mean** — the plain average of the seven classes; "5 of 7" is the count of classes won.

### Findings, one line each

- The model orders a draft better than the teams did in 5 of 7 classes, by 18 points on average (44% vs 26%).
- It loses 2019 and 2020 narrowly: the consensus top of those classes was right, and the model under-rated the shooters (Edwards, Garland, Herro, Maxey).
- Its edge comes from the middle and bottom of the draft — bust avoidance and second-round value — not from beating teams at the first pick.
- 2023 shows why both metrics are reported: the model orders the class better (35% vs 11%) but the teams win on value (75.0% vs 71.6%), because they took Wembanyama first and the model had him fourth.
- Across the seven classes the ordering gain is worth about 5 points of captured WAR (76.7% vs 71.3%).
- Recent classes are scored on few seasons, so their numbers will move: 2024's 60% rests on two seasons, 2025's 30% on one.
- A single class's correlation has a standard error near 0.13, so one-class wins and losses mean little; the seven-class mean (about ±0.05) is what to read.
- The pool is drafted players only, so this measures how to order a draft, not whom to take from outside it.

## How it was tested

**Training: 2010–2018. Test: 2019–2025 (2026 has no outcomes yet).** The model is trained once on the 2010–2018 classes — every drafted player plus every undrafted player who reached the NBA, 770 rows — and never sees a test class. Model selection (which label, which features, which members, which weights) used walk-forward folds *inside* the training window only: train on 2010…y−1, predict class y, for y = 2014…2018. The test outcomes live in a hash-locked vault on the training box; the scoring function returns aggregates only, and every blind scoring is appended to a hash-chained ledger (`experiments/ledger.jsonl`). Nothing from a test class's own year or later is visible to the model.

## Runs

| when (UTC+8) | model | AI | NBA scouts | drafts the AI won |
|---|---|---|---|---|
| 2026-09-05 13:11 | **BEST v2 gen17 · data v3** (rebuilt international block + measurements) — shipped | 44% | 26% | 5 of 7 |
| 2026-09-05 13:08 | BEST v2 gen17 · data v3.0 (rebuilt international block only) | 45% | 26% | 5 of 7 |
| 2026-09-05 04:12 | evolution champion gen86 · data v1 (walk-forward pick, not shipped) | 43% | 26% | 6 of 7 |
| 2026-09-05 02:05 | overnight queue final (hz+covw) · data v1 | 44% | 26% | 5 of 7 |
| 2026-09-05 00:51 | BEST v2 gen17 · data v1 | 46% | 26% | 5 of 7 |

The differences between these rows are inside the noise floor (a single season's correlation has a standard error of about 0.12; a seven-season mean about 0.045). The shipped model is the frozen genome on the corrected data, chosen for data correctness, not for the best number.

## The model (`model/BEST_MODEL.json`)

- **Label:** WAR over the first five NBA seasons, discounted 0.85 per season (`Σ 0.85^(i−1)·WAR_i`), clipped at 40, then Gaussian-rank transformed within each draft class. So the model learns *where in his class* a prospect finished, not raw wins.
- **Inputs:** 222 pre-draft columns (`data/input_columns.json`, blocks described in `data/FEATURES.md`): biography and measurements, final college season per-36 and rate stats with age/class/role percentiles, last international season, recruiting, process flags, scouting-report grades, mock-draft consensus.
- **Sample-size shrinkage:** every rate stat is pulled toward the training-set prior in proportion to minutes played, `w = minutes / (minutes + 400)`, so a 13-game line moves the model less than a 34-game line.
- **International adjustment:** international per-36 production is scaled by league level; data v3 adds a finer league-strength column and a two-season block.
- **Members:** XGBoost (three seeds, age constrained monotone, feature weights from `BEST_MODEL.json`) and TabICL (an in-context tabular foundation model, 8 estimators). The board is the 50/50 average of their within-class ranks.
- **Scoring pool:** the players actually drafted (≈55–60 per class). Training uses all labeled rows.

Full formulas: [`docs/METHODS.md`](docs/METHODS.md). Every permutation tried and why it was rejected: [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md). The data rebuild of 2026-09-05: [`docs/DATA_V3_REPORT.md`](docs/DATA_V3_REPORT.md). Label definitions: [`docs/LABELS.md`](docs/LABELS.md).

## Repository layout

| path | what |
|---|---|
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
