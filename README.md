# NBA redraft: AI model vs NBA scouts

Every NBA draft since 2010, redrafted by in-context tabular foundation models
([TabFM](https://huggingface.co/google/tabfm-1.0.0-pytorch), [EXAONE-Tabular](https://huggingface.co/LG-AI-Research/EXAONE-Tabular))
that only knew what was knowable one minute before the draft started, and scored against the scouts' real order on how the
players actually turned out. The pool is exactly the players taken on draft night -- college, international and G League --
reordered.

**Score:** 3-year WAR, a player's Basketball WAR summed over his first three NBA seasons (all of them if fewer) -- what the pick produced early. `infra/config.py: TARGET_KIND` switches to peak WAR (mean of his three best seasons, how good he became); peak WAR was the score until 2026-09-05.
Never played ranks below everyone. **Metric:** order accuracy, the rank correlation between a draft order and
the class's 3-year-WAR ranking, computed for the AI model and for the scouts over the same players.

**Split:** training 2003-2018, test 2019-2025, scored walk-forward: class Y is
redrafted by a model that saw only classes before Y, labelled with the seasons those players had played by draft night Y.
Model selection happens on 2011-2018; every look at the test years is logged (`outputs/layer_2/test_looks.jsonl`).

**Where it stands (2019-2025, teams' own order = 0.357):** the AI model scores 0.471 (+0.114, 6 of 7 classes won,
9.2% more top-14 WAR captured). Its inputs include player data and the full history of public mock drafts, all captured
before the draft began. It never sees the actual draft order or post-draft information.

## Run it

Needs `python3.11+`, [`uv`](https://docs.astral.sh/uv), [`bun`](https://bun.sh) and NVIDIA GPU(s) for the models.

```bash
make all          # venv, deps, downloads, features, tables, published run, then serves http://localhost:3010
```

Or step by step:

```bash
make setup        # python venv + deps, web deps
make data         # Torvik college stats, Basketball-Reference drafts and advanced stats, FiveThirtyEight RAPTOR (~15 min)
make model        # TabFM regression + classification checkpoints (~12 GB); EXAONE fetches its own on first use
make features     # every extra pre-draft source -> data/external/feat_*.parquet (international, combine, hoopR, context, ...)
make tables       # season WAR -> configured target -> the draft table with every feature
make run          # the best validated model, using every selected pre-draft signal
make web          # site on http://localhost:3010
make test         # 0.5 s read-only integration test: is the data we claim to have actually in the table?
make sweep        # the tournament: every feature group x model on the validation years, layer-2 rules, report
make sweep-status TAG=<tag>   # progress + best so far (also live in ./logs.txt)
make sweep-cut TAG=<tag>      # stop a sweep, finish with what is done, export outputs/sweeps/<tag>/best.json
```

Downloads are idempotent: re-running `make data`, `make model` or `make features` only fetches what is missing.
`data/`, `models/`, `outputs/` and `logs.txt` are not versioned.

## Layout

```
infra/                 everything upstream of a model
  config.py            years, the 2019-2025 test split, WAR constants, label transform, paths
  download_data.py     Torvik, Basketball-Reference, RAPTOR downloads + parse
  download_model.py    TabFM checkpoints (regression + classification); EXAONE downloads itself on first use
  war.py               WAR per season and configured target (draft-night aware)
  external.py          extra pre-draft sources joined on (name, draft year): physicals, JasonG, international + FIBA youth, ...
  dataset.py           the draft table: college (Torvik) and international / G League rows, every feature, modelled flag
  models.py            the model zoo: tabfm, tabfm_cls, exaone, exaone_cls, ridge, lgbm, stack and labelled variants
  builders/            feature builders -> data/external/feat_<name>.parquet (college sources, international, Torvik context, ...)
tournament/            the sweep: contract of all features + models, audit, layer 1 (AI predictions), layer 2 (rules), report
validation/            thin: the published walk-forward run (run.py) and run.json -> results.sqlite (db.py)
web/                   Next.js app (bun runtime, reads outputs/results.sqlite)
```
