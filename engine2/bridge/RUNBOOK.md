# bridge/RUNBOOK.md — running the ported winner on the GPU box

The bridge makes Colin's **model layer** (`tournament/layer1.py`, `layer2.py`, `contract.py`, `horizon.py`,
`infra/models.py`, `infra/war.py`) run on **our** verified, pre-draft-dated data. None of his builders run; the
table comes from `bridge/build_table.py`. Column-by-column mapping: `bridge/MAPPING.md`.

```
bridge/build_table.py     staging CSVs  ->  data/processed/{draft_table,season_war}.parquet
bridge/run_winner.py      the winner's 4 layer-1 configs -> his layer1 -> his layer2 blend -> outputs/bridge/*.csv
bridge/smoke_stubs/       numpy stand-ins for catboost / tabicl, for the Mac only. NEVER on the box.
```

---

## 1. Local build (this Mac)

```bash
cd /Users/kennakao/nba/site/engine2
python3 -m bridge.build_table            # ~1 s -> 1,373 x 672; this is what the ported winner expects
```

Nothing here needs a GPU, a network or a package beyond pandas / numpy / pyarrow. The table is pid-keyed: the only
thing read from the identity file is `actual_pick`, and `bbref_id` / `key` / `player` are all the pid.
`--no-picks` skips the identity file entirely (see MAPPING.md §6).

Every option is off / unlimited by default. All four are documented in full in MAPPING.md §5:

| flag | effect | table |
|:--|:--|--:|
| `--families` | `fy_` `dx_` `eur_` `dp_` written under their own prefixes as the contract groups `fiba_youth`, `growth`, `euroleague`, `draftpage` | 760 cols |
| `--intl-mask` | the old engine's IMASK gene: blank the 47 `intl_`-fed columns on the 1,157 rows with a college career | 672 cols |
| `--response-max N` | keep only the N strongest `rs_` (`gl2_`) columns by the 2000-2011 pre-fold screen; his `response` group is 11 wide, ours is 147 | 672 − (147−N) |
| `--no-fallback` | primary source only — the A/B floor for what the coalesce chains are worth | 672 cols |
| `--extras` | the 584 unmapped input columns as `xt_*` (396 with `--families`) | 1,256 cols |

Every named feature is filled by an ordered coalesce chain (`tv_` → `col_` → `ctx_base_` → `cgd_` → …), each step
converted onto Torvik's scale first, and `src_torvik` (contract group `srcflag`) says which side a row came from.
The build prints the whole chain list, the before/after coverage per class band, and the group sizes.

## 2. What to copy to the box

The box needs Colin's tree plus the bridge plus the two built parquets. From `site/engine2`:

```bash
rsync -av --relative \
    infra/ tournament/ pipeline/ bridge/ pyproject.toml Makefile \
    data/processed/draft_table.parquet data/processed/season_war.parquet \
    ubuntu@<box>:~/nba/engine2/
```

Concretely:

| path | why |
|:--|:--|
| `infra/` | `config.py` (protocol constants), `models.py` (the zoo), `war.py`, `dataset.py`, `external.py` |
| `tournament/` | `layer1.py`, `layer2.py`, `contract.py`, `horizon.py`, `winner.py` |
| `pipeline/run.py` | `layer1` imports `context_years` and `year_metrics` from it |
| `bridge/` | `build_table.py`, `run_winner.py`, the two `.md`s, `coverage_before.csv` + `response_screen.csv` (small, needed only to reproduce the build report and `--response-max` on the box) |
| `data/processed/draft_table.parquet` | 1,373 x 672, 2.3 MB (x 760 with `--families`, x 1,256 with `--extras`) |
| `data/processed/season_war.parquet` | 3,885 rows, 56 kB |

**Do not copy `bridge/smoke_stubs/` onto the box's `PYTHONPATH`.** The directory is harmless where it sits (it is
only importable if explicitly put first on the path by `--stub`), but never pass `--stub` there.
Not needed on the box: `data/raw/`, `data/external/`, `models/`, `winners/`, `web/`, `infra/builders/`,
`infra/download_*.py` — none of them is touched by this path. `outputs/` is created on demand.

The box must never see names. Check before and after copying:

```bash
python3 - <<'PY'
import pandas as pd
t = pd.read_parquet('data/processed/draft_table.parquet')
obj = [c for c in t.columns if t[c].dtype == object]
print(obj, {c: t[c].head(2).tolist() for c in obj})   # bbref_id/key/player must be pids, source in {college,intl,other,none}
PY
```

## 3. Running the full winner (box, Ubuntu, `~/nba/.venv`)

```bash
cd ~/nba/engine2
~/nba/.venv/bin/python -m bridge.run_winner --window both --device cuda:0 --tag winner
```

Defaults are the winner exactly: `--members catboost,tabicl --seeds` (5 CatBoost / 4 TabICL) `--batches 2
--n-estimators 800 --tabicl-estimators 8 --cutoff full,causal`. That is 8 layer-1 configs per window per cutoff:
CatBoost x5 seeds twice, TabICL x4 seeds twice — the four whose names are `winner.json`'s `layer2_rule`, run for
both label cutoffs. One GPU and no sharding is the right default here — CatBoost is CPU-bound and TabICL on
~500 context rows is small, so the whole thing is minutes, not hours.

If you do want to shard by year across the four H100s, note that **every shard overwrites the same
`outputs/bridge/predictions_*.csv`**. Shard the layer-1 pass only, then blend once:

```bash
for g in 0 1 2 3; do
  ~/nba/.venv/bin/python -m tournament.layer1 --device cuda:$g --years holdout --tag w_gpu$g \
      --year-list "${SHARD[$g]}" --configs "$(python -c '...print the same 4 config dicts...')" &
done; wait
~/nba/.venv/bin/python - <<'PY'
from pathlib import Path; import json, pandas as pd
import bridge.run_winner as R
cfgs = json.load(open('outputs/bridge/run_manifest.json'))['runs'][0]['configs']   # or rebuild with R.build_configs
frames = [R.blend(Path(f'outputs/layer_1/w_gpu{g}.parquet'), f'holdout:{shard}', cfgs) for g, shard in SHARDS]
pd.concat(frames).to_csv('outputs/bridge/predictions_holdout.csv', index=False)
PY
```

Outputs:

```
outputs/bridge/predictions_holdout.csv       pid, draft_year, cutoff, score, rank_in_class, <one col per layer-1 config>
outputs/bridge/predictions_walkforward.csv   same, classes 2012-2018, each predicted from classes before it only
outputs/bridge/run_manifest.json             argv, tags, year keys, the exact config dicts
outputs/layer_1/<tag>_<window>_<cutoff>.parquet   his per-player, per-member layer-1 predictions
```

`score` is the equal rank-average of the four configs' within-class percentile ranks: 0-1, higher = better prospect,
uniform inside each class. `rank_in_class` is 1 = best. The per-config columns are layer 1's own stacked score,
which for a single-member config is `rankdata` inside the class (1 = worst, n = best).

**Scoring.** Holdout classes carry no labels here, so his `spearman` prints `nan` and every holdout metric is 0 —
that is correct, not a failure. Score `predictions_holdout.csv` through the sealed vault. `predictions_walkforward.csv`
*is* scored in-process (context classes have labels), and its per-year Spearman is printed by layer 1 — that is the
number to watch while iterating, and the only one you may look at repeatedly.

Smaller runs:

```bash
--members catboost                  # skip TabICL
--seeds 2 --batches 1               # 2 seeds, one batch
--years 2019,2020                   # a subset of the window
--cutoff full                       # one label cutoff
--ctx-start-catboost 2008           # see "Known gaps" below — the first experiment to run
--dry-run                           # print the layer-1 command lines and stop
```

Layer 1 caches by config + resolved columns + their values + years + seed under `outputs/layer_1/cache/`, so a
re-run with one config changed only recomputes that config.

## 4. Smoke test performed on this Mac

This Mac has only the system Python 3.9 with pandas / numpy / scipy / pyarrow / torch. CatBoost, TabICL, XGBoost,
LightGBM and scikit-learn are not installed and no network was used, so `--stub` swaps in numpy ridge stand-ins
(`bridge/smoke_stubs/`) that accept the same constructor keywords and `fit`/`predict` API. **The stubs prove the
plumbing, never the accuracy** — `--stub` therefore writes to `outputs/bridge_stub/`, never to `outputs/bridge/`.

```bash
cd /Users/kennakao/nba/site/engine2
python3 -m bridge.build_table                                                   # 1.1 s
python3 -m bridge.run_winner --window holdout --years 2019,2020 --members catboost \
    --seeds 1 --batches 1 --n-estimators 50 --cutoff full --device cpu --stub --tag smoke      # 4.3 s
python3 -m bridge.run_winner --window both --years 2012,2013,2019,2020 --members catboost,tabicl \
    --seeds 2 --batches 2 --n-estimators 50 --tabicl-estimators 1 --cutoff full,causal \
    --device cpu --stub --tag smoke2                                                           # 15.3 s
```

Result: `contract.validate` passes (**30** groups — `srcflag` is the new one, always present; 34 with
`--families`); layer 1 resolves 575 columns for the CatBoost member and 428 for TabICL, unchanged by the fallback
work because `src_torvik` is not in the winner's feature list; the four blend members and the `blend A | B | C | D`
rule run for both windows and both cutoffs; 232 holdout rows and 214 walk-forward rows come out with `score`
uniform on (0, 1) and `rank_in_class` 1..n. Holdout Spearman prints `nan` (labels sealed); walk-forward 2012/2013
prints real numbers (0.14-0.37 for the ridge stand-in, against 0.413 for the real draft order), which is what
proves the labels, the `match3` horizon and both cutoffs are wired correctly. Independently verified:
`war_target(ctx, seasons, through=2026)` reproduces `y_early_war` to 7e-15; the causal cutoff shrinks the three
classes before the scored year (2019: 5.13 → 0.98); `disc85_gaussrank` comes out N(0, 0.99) within class.

Re-verified after the fallback rebuild: every named feature is value-identical to the previous table where both
are non-null (Pearson 1.000, max abs diff 0) except the seven changed on purpose — `rec_rank` (scale fix, 165
rows), `role` (one code set, 706), `oreb` / `dreb` (real box split instead of the ORB%/DRB% approximation, 779),
`TPA_pg` / `FTA_pg` (the actual per-game attempt line, 730), `Min_per` (exactly-scaled source first, 121) and
`class_year` (clipped to 1..4, 15). A name scan over every produced parquet and CSV against all 2,743 name tokens
in the identity file returns **zero data-value hits**; the only matches are English words inside 13 *column names*
(`..._days`, `iz_young_x_*`, `sc_post`, `sct_*peak*`, `sct_comp_is_hall_tier`), identical to the pre-change table.

## 5. Every change made to Colin's code

Five edits, all marked `# BRIDGE:` (`grep -rn "BRIDGE:" infra tournament pipeline`). Nothing else in his tree was
touched, and the `colin` branch itself was never checked out, modified or merged.

| file | change | why |
|:--|:--|:--|
| `infra/war.py` (`war_target`, +3 lines) | `if not path.exists(): continue` around the frozen `data/raw/reference_war/answers_YYYY.csv` override | `label_horizon: match3` resolves to 5 seasons for classes ≤ 2021, which sends layer 1 through `war_target(..., through=LAST_SEASON)`; his answer sheets are his benchmark and our 2019-2025 labels are sealed, so the file is absent and the unguarded `pd.read_csv` raised `FileNotFoundError` |
| `tournament/contract.py` (`PREFIX`, +6 entries) | `"extra": "xt_"`, `"srcflag": "src_"`, and the four `--families` groups `"fiba_youth": "fy_"`, `"growth": "dx_"`, `"euroleague": "eur_"`, `"draftpage": "dp_"` | makes the bridge's own columns addressable as feature groups. `srcflag` always exists (one column, `src_torvik`); the other five are inert unless the matching flag is used — with no columns of that prefix `contract.groups()` never creates the group. ⚠ `tournament/sweep.py` calls `contract.assert_covered`, which requires every group to appear in the grid, so a full sweep against a table built with `--extras` or `--families` would need those names added to its grid |
| `infra/models.py`, `tournament/layer1.py`, `tournament/layer2.py` (+1 import each) | `from __future__ import annotations` | `X \| None` in a signature is evaluated at def time and needs Python 3.10+. The only interpreter on the porting Mac is 3.9, so without it the modules do not import. A no-op on the box's 3.12 |

Everything else lives in `bridge/`. In particular `layer1.py`'s protocol, label logic, walk-forward split, caching,
metrics and output format are untouched, and `run_winner.py` drives `python -m tournament.layer1` as a subprocess
rather than reimplementing any of it.

Deliberately **not** patched: `tournament/winner.py`, `sweep.py`, `optimize.py`, `audit.py`, `readme.py`,
`integration_test.py`, `infra/builders/*` and `pipeline/db.py`. `winner.py` also carries a `X | None` signature and
so does not import under this Mac's 3.9 — it is his archive tool, runs fine on the box's 3.12, and the bridge does
not use it. Nothing under `infra/builders/` is reachable from this path at all. `git status` on `site/` shows
exactly the five modified files above plus the new `engine2/bridge/`; nothing was committed and the `colin` branch
was never touched.

## 6. Known gaps

1. **Context coverage is thin before 2010, where his was not — and the fallbacks only half-close it.** Our Torvik
   block starts with the 2010 class, `col_` *box* line with 2002 but its *ratings* (`ortg/drtg/adj_*/impact*/
   value`) with 2008, `ctx_base_` / `cgd_` with 2008, `tc_` with 2008, `bb_` boards with 2009, `gl2_` with 2005.
   His `ctx_start: 2003` relied on hoopR / AyushBatra lines for 2003-2007, which this project cannot use.
   Coalescing every named feature down through `col_` / `ctx_base_` / `cgd_` lifts mean named-feature coverage in
   2003-09 from 0.430 to 0.459 and `all_torvik` over 2003-18 from 0.583 to 0.602 — real, but 14 of the 50 named
   features are still 0.00-0.24 in 2003-09 because the underlying seasons are simply not in our staging tables.
   **First experiment on the box remains `--ctx-start-catboost 2008` (and 2010), chosen on the walk-forward
   window, never on the holdout.** `--no-fallback` is the A/B floor for what the chains bought.
2. **Our feature set is much wider than his** (575 vs 324 for CatBoost, 428 vs 313 for TabICL), almost entirely
   from `response` (147 vs 11), `tctx` (107 vs 72), `person` (41 vs 6), `intl_z` (39 vs 18), `momentum` (36 vs 17)
   and `comp` (20 vs 3). His ideas log says extra columns were usually neutral-to-negative for CatBoost.
   `--response-max N` is now the knob: `--response-max 11` matches his group width and takes CatBoost to 439
   features, `--response-max 24` to 452. Chosen by the 2000-2011 pre-fold screen, so it never looks at a fold.
3. **`momentum` is effectively the `bb_` half only.** Our `mock_*` momentum block is empty for every class ≤ 2019
   and 43-85% covered for 2020-2025; `select_features` drops all-NaN-in-fold columns, so those 13 columns never
   reach a model that trains on context years. Fixing that needs mock snapshots for the older classes.
4. **Per-class n is smaller than his** (44-58 vs his 58-60): a drafted player with no staging row is simply absent.
   2015 (44) and 2022 (52) are the thinnest. Our Spearman and his 0.504 have different denominators and are not
   directly comparable.
5. **Unfillable columns**: `a_box` (17), the whole JasonG family (67), `prev_/d_ obpm|dbpm|porpag` (6),
   `i_tov_36`, `i_ast_tov`, ten `i_fiba_*`, four `sc_*` — all present and empty, all dropped by `select_features`.
   Five named Torvik features are `tv_`-only and therefore empty before the 2010 class — `dporpag`, `stops`,
   `gbpm`, `ogbpm`, `dgbpm` — because nothing of ours is on their scale (checked: `tc_porpag_share` is a team
   share, `col_impact_def` a BPM component, `tc_slope_bpm` a per-season slope). `porpag`, `adjoe`, `pfr` and
   `ast_tov` were in this list and are not any more (`col_value`, `col_adj_ortg`, `ctx_base_fouls40`,
   `ctx_base_ast_tov`). `conf` has no scale-compatible source either, and the winner drops it with `-cat`.
   Absent prefix groups: `kaggle`, `score`, `ianstack`, `hoopr`, `marchmadness`, `shrunk`, `eurocamp`, `bwb`,
   `academy`, `transfers`, `odds`, `dev`, `gleague`, `population`, `impact`, `coach`, `game`. None is in the
   winner's feature list, but `layer2`'s `tilt` rules (`co_*`) and `stackctx` (`gl_pred_ws48`) would silently lose
   those covariates.
6. **`sct_comp_war` has no equivalent.** His single best late addition was the NBA comparison's realised WAR per
   season through draft night; ours is the comp's dated career *line* (`sct_pts36_to_date`, `sct_peak_mpg_to_date`,
   …). Same idea, different measurement — worth checking whether a WAR-shaped aggregate of `cp_` helps.
7. **Draft-and-stash label timing.** `season = draft_year + i` treats our compressed per-season array as calendar
   seasons. For a player whose debut was delayed this credits his seasons one or two years early under the
   `causal` cutoff. `full` (the winner's setting) is unaffected.
8. **2026 is not in the table.** `staging_v419/tests/` carries 2019-2025 only; the identity file has 61 rows for
   2026. Add `test_2026_inputs.csv` and the class appears automatically (it will be unlabelled, like the holdout).
9. **pandas version.** Built and smoke-tested against pandas 2.3 here; `pyproject.toml` pins `pandas>=3.0` for the
   box. Re-run the smoke commands there first — several of his groupby/apply idioms changed behaviour in 3.0.
10. **`src_torvik` is built but not wired into the winner.** 668 of 1,373 rows carry the player's own Torvik line
    and 705 are filled from `col_` / `ctx_base_` / `cgd_`; the indicator lets a model separate the two, but
    `bridge/run_winner.py`'s `FEATURES_CATBOOST` / `FEATURES_TABICL` are unchanged, so the ported winner still
    resolves 575 / 428 columns and never sees it. Adding `"srcflag"` to those lists is a one-word change and an
    obvious walk-forward experiment — deliberately left as a decision rather than a default, since it moves the
    port off "his exact configuration".
11. **`--families growth` (`dx_`) is a train/serve trap.** Coverage 0.67-0.72 through 2012-2017, then 2019 0.30,
    2021 0.18, 2023 0.03, 2024 0.00. `select_features` only drops a column that is all-NaN inside the *training*
    fold, so every `dx_` column survives training and is empty at holdout time. Walk-forward only, if at all.
