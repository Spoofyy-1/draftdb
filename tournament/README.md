# tournament

The sweep. Run it, read the report, improve a builder or a rule, run it again.

```bash
.venv/bin/python -m tournament.sweep --tag s1 --grid full      # audit -> layer 1 on all GPUs -> layer 2 -> report
```

The sweep selects on context years (2013-2018, inside the 2003-2018 context block). The holdout years 2019-2025 are
never touched by the sweep; confirm a short, pre-registered list once, and every look is logged:

```bash
.venv/bin/python -m tournament.layer2 --layer1 outputs/layer_1/s1_gpu*.parquet --holdout --rules "blend all"
```

```
contract.py     every feature group, model and label framing; validate() + assert_covered() guarantee the grid uses all of them
audit.py        who is modelled (college / international / nobody), coverage per group, leakage tests
layer1.py       raw AI predictions per config (point + bin distribution) -> outputs/layer_1/<tag>.parquet
layer2.py       pre-draft-only rules on layer-1 outputs (blends, upside/floor, consensus); holdout looks -> outputs/layer_2/holdout_looks.jsonl
sweep.py        the entrypoint: grid over the contract, sharded across GPUs, then layer 2 and a report in outputs/sweeps/<tag>/
builders/       feature builders -> data/external/feat_<name>.parquet, merged into the table by infra.dataset
```

Leakage rule for every feature: knowable one minute before the draft starts. No NBA data, no draft results of the
same year, no population statistic fitted on later seasons.

When a configuration wins on the context years and survives its holdout look, it is written into `infra/` + `pipeline/` (features in
`dataset.py`/`external.py`, model in `models.py`, label in `config.py`) and republished with `make run`.

## Ideas log (holdout 2019-2025, mean Spearman; scouts 0.260)

Start of the log: `open_tabicl_blend_intl` 0.450 (best single TabICL 0.433). A bound: training on every other class *including
later ones* (not a legal protocol) also scores 0.43-0.45 on the original 259 features, so gains had to come from new information.

| Idea | Result |
|:-----|:-------|
| Label variants (rank, gaussrank), feature top-k, min-coverage, kitchen-sink feature sets | within noise or worse |
| Multi-horizon stacked context (`label_horizon: multi`, horizon as a feature) | worse (TabICL 0.40, CatBoost 0.43-0.44) |
| WAR decomposed into rating x minutes, above-replacement classifier | worse than direct (0.38-0.42) |
| draft_year as a regime feature; age / anti-mock / intl-shrink / stash-to-median overlays; uncertainty shrinkage | no gain |
| Undrafted players (NBA-signed + strong never-signed seniors) as extra context rows | no gain (0.400 -> 0.391 on a reduced set) |
| Drafting-team quality (even the *actual* team): residual correlation with WAR ~ 0 | not pursued (also borderline pre-draft) |
| Combine 2023-25 from a verified mirror + physicals filled from the combine for 2024-25 (coverage 0-9% -> 86-93%) | neutral on the mean, fixes a distribution shift |
| Mock consensus rebuilt from 1 board/year to 5-8 boards (Wayback, 107 boards) | CatBoost 0.449 -> 0.457, TabICL-2010 0.427 -> 0.443 |
| `match3` label horizon (train on the first max(3, N) seasons) | CatBoost 0.457 -> 0.472; the single best change |
| Head coach / program pedigree + causal NBA track record (`co_`, 19 columns) and layer-2 tilts on them | no gain (-0.02 to +0.006) |
| Population development priors (`dv_`, 40k college season pairs) | TabICL +0.008, CatBoost -0.02 |
| G League translation priors (`gl_`, 1.5k college -> G League pairs) + own pre-draft G League line | TabICL +0.016, TabLDM +0.018, CatBoost -0.005 |
| Mock momentum 90/60/30/7 days (`mo_`, 354 boards) | CatBoost +0.006, TabICL +0.018 |
| Population NBA-outcome priors (`pop_`, 18k D1 final seasons) | no gain |
| Challenge-response game-log features (`rs_`: rematches, bounce-back, error persistence, close games) | ~0 univariate; CatBoost +0.005 |
| Context window 2008 / 2013 for TabICL, CatBoost depth 4 / 6, more estimators, extra seed batches | 2010 and depth 5 stay best; seeds stable to +-0.002 |
| basketball-reference biography (`bio_`: NBA relatives rho +0.18, shooting hand, birthplace, high-school path) | CatBoost +0.004, TabICL +0.004 |
| Blend CatBoost (2003, +mo +rs +bio, 10 seeds) with TabICL (2010, +mo +bio, 8 seeds), then intl a=0.3 | 0.495 pure, 0.500 with the tilt (`winners/catboost_tabicl_market_bio_m3*`) |
| Hoop Explorer lineup impact (`he_`, 56 cols: RAPM off/def, on/off, play types, vs-top-100 versions; classes 2019+ only) as model features, as a 2019+ specialist model, and as tilts | CatBoost neutral, TabICL -0.015 (NaN before 2019); specialist 0.36-0.42 alone and hurts blends; best tilt +0.003 |
| Market tilt only for the 1-2 season classes (2024-25) | hurts at every weight |
| Within-class z-scored features (`class_zscore`), CatBoost monotone constraints on 11 stats, TabICL classifier head (3 / 5 bins), horizon full / match instead of match3 on the final feature set | all neutral or worse (0.43-0.486 vs 0.484-0.489) |
| NBADraft.net pre-draft scouting grades (`sc_`, 972 Wayback profiles; intangibles rho +0.25, overall +0.23, NBA-ready +0.17) | CatBoost +0.005, TabICL +0.004; clean blend 0.495 -> 0.498, stack 0.508 -> 0.513 |
| CatBoost ranker / LightGBM / XGBoost on the final feature set as extra blend members | 0.452 / 0.464 / 0.457 -- below both anchors, not blended |
| NBADraft.net write-up: TF-IDF ridge on the strengths / weaknesses text, fit walk-forward | no out-of-sample signal (rho 0.01-0.05), dropped |
| NBA-comparison player's realised value (`sct_comp_war`: the scout's comp's mean WAR/season through draft night, rho +0.20) | CatBoost +0.007 (first 0.500 single), TabICL +0.004; clean blend 0.498 -> 0.504-0.507, stack 0.513 -> 0.516-0.520 (the two comp-matching versions differ by noise) |
| CatBoost recency weighting of context classes (half-life 8 / 15 years) | worse (0.483 / 0.485 vs 0.494-0.502): the old classes carry weight |
| NBADraft.net's own big-board rank on the profile pages | 2020+ layout only, mostly stale snapshots, rho 0.12 -- dropped |
| `stackctx`: ridge over the two models' ranks + 9 fixed pre-draft covariate ranks, weights fit on 2013-2018 only, applied to holdout as a fixed rule | 0.508 -> 0.513 (scouting grades) -> 0.516-0.520 (comparison value); weights are context-fit but the covariate list was chosen after holdout tilt checks -- a broad a-priori list scores 0.44-0.47, so call it +0.01 optimistic (`winners/stackctx_catboost_tabicl_scout_comp`) |
