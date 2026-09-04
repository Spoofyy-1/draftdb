# NBA redraft: AI model vs NBA scouts

Every NBA draft since 2010, redrafted by a zero-shot tabular foundation model
([TabFM](https://huggingface.co/google/tabfm-1.0.0-pytorch)) that only knew what was knowable on draft night,
and scored against the scouts' real order on how the players actually turned out.

**Score:** peak WAR, a player's average Basketball WAR over his three best NBA seasons (all of them if fewer).
Never played ranks below everyone. **Metric:** order accuracy, the rank correlation between a draft order and
the class's peak-WAR ranking, computed for the AI model and for the scouts over the same players.

**Out of sample:** to redraft class Y the model sees only classes before Y, labelled with the seasons those
players had played by draft night Y. Nothing from the future leaks in.

## Run it

Needs `python3.11+`, [`uv`](https://docs.astral.sh/uv), [`bun`](https://bun.sh). A GPU is optional; everything runs on CPU.

```bash
make all          # venv, deps, downloads, tables, scoring run, then serves http://localhost:3010
```

Or step by step:

```bash
make setup        # python venv + deps, web deps
make data         # Torvik college stats, Basketball-Reference drafts and advanced stats, FiveThirtyEight RAPTOR (~15 min)
make model        # TabFM regression checkpoint, ~6 GB, no account needed
make tables       # season WAR -> peak-WAR target -> draft table with college features
make run          # score every draft (13 CPU workers on a 52-core box takes ~15 min)
make web          # site on http://localhost:3010
```

Downloads are idempotent: re-running `make data` or `make model` only fetches what is missing.
`data/`, `models/` and `outputs/` are not versioned.

## Layout

```
nbadraft/
  config.py          years, splits, WAR constants, paths
  data.py            download + parse raw sources
  download_model.py  TabFM checkpoint
  war.py             WAR per season, peak-WAR target (draft-night aware)
  dataset.py         join draftees to their final college season
  models.py          tabfm / tabfm_ens / lgbm / ridge scorers
  run.py             one experiment: validation, out-of-sample redrafts, results.sqlite
  db.py              run.json -> outputs/results.sqlite
web/                 Next.js app (bun runtime, reads outputs/results.sqlite)
```
