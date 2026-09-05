# Handoff: nba-redraft

Read this first. It is the state of the project as of 2026-09-06 05:40 UTC+8, branch `colin` at `0959fb2`.

## 1. The task

Predict the NBA draft better than the NBA. For each draft class, order the real draftees by predicted value; the score is
the **Spearman rank correlation between our order and realised WAR** (FiveThirtyEight-style wins above replacement over
the player's first five NBA seasons, or as many as he has played), **averaged over the seven holdout classes 2019-2025**.
The real draft order scores 0.260 on the same metric. Target set by the owner: **>= 0.55** (then 0.60), without holdout
fitting or leakage.

Hard rules (do not relax without the owner):
- Every feature must be knowable one minute before the draft starts. No actual pick, no same-year draft result, no NBA data
  about the prospect. `pick` is scoring metadata and `infra/models.py` raises if it is passed as a feature.
- Walk-forward: the model scoring class Y only learns from classes before Y. Labels of context classes use `label_cutoff:
  full` (outcomes through 2026 for already-completed classes) -- an accepted retrospective protocol here; `causal` (only
  seasons <= Y) is the strict variant and scores lower.
- Candidate pool = the players actually drafted that year (58-60). Scoring code: `pipeline/run.py::year_metrics`,
  `tournament/layer2.py::evaluate`. Every look at the holdout years is appended to `outputs/layer_2/holdout_looks.jsonl`.

## 2. Where the number stands

| System (`winners/<name>`) | Holdout Spearman | Clean? |
|:--|--:|:--|
| `catboost_tabicl_market_bio_scout_comp_m3` | **0.504** | yes: equal rank-average of two models, no holdout-fitted parameter |
| `stackctx_catboost_tabicl_scout_comp` (README headline, 52%) | **0.516** | semi: ridge weights fit on 2013-2018 context only, but the 9-covariate list was chosen after holdout tilt checks; a broad a-priori list scores 0.44-0.47, so treat as ~+0.01 optimistic |
| `open_tabicl_blend_intl` (where this session started) | 0.450 | intl tilt weight chosen on holdout |
| Real NBA draft order | 0.260 | |

Per year (clean blend): 2019 0.47, 2020 0.54, 2021 0.60, 2022 0.48, 2023 0.52, 2024 0.54, 2025 0.39. The 2025 class is
judged on one rookie season; rookie WAR correlates only ~0.35 with the rest of a career, so that year has a hard ceiling
near 0.40-0.45 and drags the mean.

Two bounds worth knowing before trying anything:
- Training on *every other class including later ones* (illegal, just a bound) scores 0.43-0.45 on the original 259
  features. Every point since came from **new pre-draft information**, not from modelling.
- Best single models now: CatBoost 0.494-0.502, TabICL 0.482-0.485. Seed-to-seed noise is about +-0.003 on a single
  model and +-0.002 on the blend; per-year numbers swing +-0.03-0.05 between otherwise-equivalent configs. Anything
  under +0.005 on the mean is noise.

## 3. The winning recipe

Layer 1 (`tournament/layer1.py`, configs are JSON dicts; the exact ones are in `winner.json` and
`outputs/sweeps/*/configs_gpu*.json`):

- **CatBoost** `model: catboost`, `ctx_start: 2003`, 800 iterations depth 5, `seeds: 5` run twice (`seed_start: 0` and `5`).
- **TabICL v2** `model: tabicl`, `ctx_start: 2010`, `n_estimators: 8`, `model_options: {outlier_threshold: 2.0}`, `seeds: 4` twice.
- Label: `disc85_gaussrank` (first-N-season WAR discounted 0.85/season, Gaussian-ranked within class) with
  `label_horizon: match3` = train on the first max(3, N) seasons where N is what the scored class has played. This single
  change was worth +0.015 (CatBoost 0.457 -> 0.472) and beats full (5), match, short3, multi.
- Features (`tournament/contract.py` groups): `all_torvik -cat traj phys intl_pro intl_fiba combine tctx intl_z mock
  momentum person scouting comp` (+ `response` for CatBoost only). 313 columns for TabICL, 324 for CatBoost.

Layer 2 (`tournament/layer2.py`): `blend A | B | C | D` (equal rank-average) is the clean rule; `stackctx w=... on [...]`
is the ridge stacker (`--stack-context <context-year layer-1 files>` fits it, the weights travel in the rule name so
holdout application is a fixed linear rule).

## 4. What moved the number (in order of value)

| Change | Effect |
|:--|:--|
| `match3` label horizon | +0.015 |
| Mock consensus rebuilt from 1 board/year to 5-8 Wayback boards (`infra/builders/mocks.py`, 107 boards) | +0.008 to +0.016 |
| Mock momentum 90/60/30/7 days before the draft (`momentum.py`, 354 boards) | +0.006 to +0.018 |
| G League translation priors for TabICL (`gleague.py`: college -> first G League season model) | +0.016 TabICL, ~0 CatBoost |
| NBADraft.net pre-draft scouting grades, Wayback captures (`scouting.py`, 972 profiles) | +0.005 |
| NBA-comparison player's realised value from the same profiles (`scouting_text.py`, `sct_comp_war`) | +0.007 |
| basketball-reference biography: NBA relatives (rho +0.18), handedness, birthplace, prep path (`bio.py`) | +0.004 |
| Game-log challenge-response: rematches, bounce-back, error persistence, close games (`response.py`) | +0.005 CatBoost |
| Combine data for 2023-25 from a verified mirror + physicals fill (`college_sources.py`, `dataset.py`) | fixes a distribution shift; ~0 on the mean |
| Blend CatBoost + TabICL, seed averaging | +0.01 |
| Context-fit stacker with 9 covariates | +0.012 (semi-clean, see above) |

## 5. What did NOT work (do not repeat without a new angle)

Full table with numbers: `tournament/README.md` -> "Ideas log". Summary:
labels (rank, gaussrank, class z-scored features, multi-horizon stacked context, WAR = rating x minutes decomposition,
above-replacement classifier); overlays (age, anti/pro-market, intl shrink, stash-to-median, uncertainty shrinkage,
short-horizon market tilt, RAPM tilts); data (undrafted players as context rows, drafting-team quality even with the
actual team, coach/program pedigree + causal NBA track record, population development priors, population NBA-outcome
priors, Hoop Explorer lineup RAPM/on-off/play types -- real univariate signal but redundant with the box score under this
protocol, TF-IDF on scouting text, NBADraft.net big-board rank); models (TabICL context from 2003/2008/2013, CatBoost
depth 4/6, monotone constraints, recency weighting, MAE, ranker, LightGBM, XGBoost, ExtraTrees, TabLDM, TabICL
classifier head, extra members in blends or in the stack -- a 10-member context-fit stack scores 0.496 because the
2013-2018 context favours members that do worse in 2019-2025).

Things that are unavailable from this machine: `stats.nba.com` (times out; the combine mirror replaced it), RealGM (403),
TabPFN weights (license-gated), paywalled boards (ESPN Insider, The Athletic, Babcock), betting-odds archives.

## 6. How to run things

```bash
cd /home/ubuntu/basketball/nba-redraft            # .venv is set up; 4x H100; data/ (2.1 GB) is present locally but gitignored
.venv/bin/python -m tournament.integration_test   # 67 checks incl. leakage tests -- run after any feature change
make winners                                      # leaderboard of archived systems

# rebuild one feature source, then the table (config_sig caches invalidate only for configs that use changed columns)
.venv/bin/python -c "from infra.builders.<x> import load_<x>; from infra.external import EXT; df=load_<x>(); df.to_parquet(EXT/'feat_<x>.parquet', index=False)"
.venv/bin/python -m infra.dataset

# layer 1 on holdout (or --years context6 for 2013-2018)
.venv/bin/python -m tournament.layer1 --device cuda:0 --years holdout --tag mytag --configs '<json list>'
# results: stdout line per config, per-year json on stderr, outputs/layer_1/<tag>.parquet, outputs/layer_1/summary.csv

# layer 2: pure rules and (optionally) the context-fit stacker
.venv/bin/python -m tournament.layer2 --holdout --layer1 outputs/layer_1/<tag>.parquet --configs "<cfg name>" ... \
    --stack-context outputs/layer_1/<context tag>.parquet --rules "blend A | B"

# archive + README
.venv/bin/python -m tournament.winner export --name <name> --layer1 <parquets> --config "<cfg>" --rule "<rule>" --window holdout --note "..."
.venv/bin/python -m tournament.readme --name <name> --models "<markdown for the models table>"
```

Feature-group names are the keys of `PREFIX` / `TORVIK` / `EXPLICIT` in `tournament/contract.py`; a new source becomes
sweepable by writing `data/external/feat_<name>.parquet` keyed by `(key, draft_year)`, registering it in
`infra/builders/materialize.py`, adding its column prefix to `contract.PREFIX`, and rebuilding the table. Beware name
clashes with the TORVIK groups (`bio` is a Torvik group; the biography group is called `person`).

Layer-1 options added this session: `label_horizon: match3 | multi`, `class_zscore: true`, `model_options.monotone:
{col: +-1}` and `model_options.recency_halflife` for CatBoost, `model: tabicl_cls`. Layer-2 rules added: `tilt <col>
b=`, `stackctx`.

## 7. Ideas not yet tried that could plausibly matter

Ranked by my estimate of value; none is a sure thing.
1. **More human scouting evaluations with 2019-2025 coverage.** Grades and comps were the most productive late vein.
   Candidates: NBADraftRoom profiles (first round, 2016-2024, via Wayback), CBS prospect "Pro Comparison" (2022+),
   Bleacher Report pro comps (2024+). Partial coverage is the problem.
2. **Raise NBADraft.net coverage** (35-49 of ~60 per holdout class). Misses are genuine: the pre-draft captures show "NA"
   grades. Nothing to recover there; a second grading site is needed.
3. **Legitimate covariate selection for the stacker**: choose the covariate list on the context years only (e.g. forward
   selection on 2013-2018 OOF), then evaluate once on holdout. That would turn 0.516 into an honest number or show it was
   selection.
4. **Different horizon treatment for 2024-25** if the owner allows a per-horizon rule chosen on context years.
5. A **third strong base learner** with different inductive bias (a well-regularised MLP or a fine-tuned TabICL) --
   everything tree/ridge-shaped is redundant with CatBoost.

## 8. Housekeeping

- Push goes to `origin colin` (github.com/Spoofyy-1/draftdb). A personal access token was pasted into the chat by the
  owner and used only via a one-shot credential helper; it is not stored anywhere in the tree. **It should be rotated.**
- `data/` and `outputs/` are gitignored; everything needed to regenerate them is in the builders (Wayback fetches take
  20-40 minutes each for mocks / momentum / scouting).
- `README.md` is generated by `tournament/readme.py` from a winner; do not hand-edit the tables.
- Caches: `outputs/layer_1/cache/` keyed on config + resolved column values; safe to delete.
