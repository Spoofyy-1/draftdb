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

Six edits, all marked `# BRIDGE:` (`grep -rn "BRIDGE:" infra tournament pipeline`). Nothing else in his tree was
touched, and the `colin` branch itself was never checked out, modified or merged. No `git` command was run.

| file | change | why |
|:--|:--|:--|
| `infra/war.py` (`war_target`, +3 lines) | `if not path.exists(): continue` around the frozen `data/raw/reference_war/answers_YYYY.csv` override | `label_horizon: match3` resolves to 5 seasons for classes ≤ 2021, which sends layer 1 through `war_target(..., through=LAST_SEASON)`; his answer sheets are his benchmark and our 2019-2025 labels are sealed, so the file is absent and the unguarded `pd.read_csv` raised `FileNotFoundError` |
| `tournament/contract.py` (`PREFIX`, +6 entries) | `"extra": "xt_"`, `"srcflag": "src_"`, and the four `--families` groups `"fiba_youth": "fy_"`, `"growth": "dx_"`, `"euroleague": "eur_"`, `"draftpage": "dp_"` | makes the bridge's own columns addressable as feature groups. `srcflag` always exists (one column, `src_torvik`); the other five are inert unless the matching flag is used — with no columns of that prefix `contract.groups()` never creates the group. ⚠ `tournament/sweep.py` calls `contract.assert_covered`, which requires every group to appear in the grid, so a full sweep against a table built with `--extras` or `--families` would need those names added to its grid |
| `infra/builders/college_sources.py` (`load_combine`, +3 comment lines, 1 expression) | `out["c_shoot_pct"] = (made / att.replace(0, np.nan)) if hasattr(att, "replace") else np.nan` | the public MichLitt mirror of `draftcombinestats` has anthropometrics and drills but **no shooting-drill columns** (`SPOT_*`, `OFF_DRIB_*`, `ON_MOVE_*`), so his accumulator loop never runs and `att` is still the scalar `0.0` — the unguarded `att.replace` raised `AttributeError`. With his original stats.nba.com JSON archive present the guard is not taken. Consequence: `c_shoot_pct` is empty in the rebuild |
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

---

## 7. Rebuilding Colin's data from the permitted public sources

`bridge/rebuild_colin_data.py` recreates the inputs his builders expect, runs the builders whose sources are
allowed here, and writes **`data/processed/draft_table_colin.parquet`** — the bridge table plus his rebuilt
families, pid-keyed, names dropped. `data/processed/season_war.parquet` is unchanged by it (3,885 rows, 975
players, identical to the bridge's). Licences and exactly what was taken: `bridge/SOURCES.md`.

### Exact commands

```bash
cd /Users/kennakao/nba/site/engine2

# everything, in order (sources -> torvik -> drafts -> hoopr -> features -> merge -> coverage)
python3 -m bridge.rebuild_colin_data all

# or stage by stage; every stage is idempotent and safe to re-run
python3 -m bridge.rebuild_colin_data sources        # ayush (2 files) + jasong + combine mirror + RAPTOR
python3 -m bridge.rebuild_colin_data torvik         # local Torvik cache -> data/raw/torvik/*.csv -> torvik.parquet
python3 -m bridge.rebuild_colin_data drafts         # identity -> drafts.parquet, target.parquet, season_war.parquet
nohup python3 -m bridge.rebuild_colin_data hoopr --keep-mb 300 > bridge/logs/hoopr.log 2>&1 &
python3 -m bridge.rebuild_colin_data features       # his builders -> data/external/feat_*.parquet
python3 -m bridge.rebuild_colin_data merge coverage # -> draft_table_colin.parquet + coverage_colin_rebuild.csv

# subsets and housekeeping
python3 -m bridge.rebuild_colin_data features --only combine,hoopr,shrunk,transfers,ianstack,torvik_context
python3 -m bridge.rebuild_colin_data features --only game,response      # the two hoopR-heavy builders
python3 -m bridge.rebuild_colin_data restore        # put OUR bridge table back at data/processed/draft_table.parquet
python3 -m bridge.rebuild_colin_data clean          # delete data/external/hoopr + player_game.parquet (~212 MB)
```

**Resuming.** `hoopr` skips any file already on disk and stops cleanly when free disk falls below `--keep-mb`,
printing how many files remain; just run it again. `features` re-runs each loader from scratch (seconds to two
minutes each) but reuses `data/processed/player_game.parquet` if it exists. After `clean`, a full rebuild is
`hoopr` then `features --only game,response` then `merge`.

⚠ **`features` parks HIS name-keyed table at `data/processed/draft_table.parquet`** because `transfers.py` and
`game_features.py` read that path. `merge` restores ours from `data/processed/draft_table_bridge.parquet` at the
end. If a `features` run is interrupted, run `python3 -m bridge.rebuild_colin_data restore` before anything else
touches the bridge table.

### Run times and disk (measured on this Mac, 2026-09-09)

| stage | wall time | bytes downloaded | disk left behind |
|:--|--:|--:|--:|
| `sources` | 2 s | 4.8 MB | 1.8 MB (`data/external/*.csv`) |
| `torvik` | 2 s | 0 (local cache) | 35 MB CSV + 17 MB `torvik.parquet` |
| `drafts` | 2 s | 0 | 0.1 MB |
| `hoopr` | **1 m 38 s** | **113 MB** (88 parquets) | 113 MB |
| `features` (first run) | **2 m** | 0.3 MB (ianstack) | 99 MB `player_game.parquet` + 5 MB `feat_*` / crosswalks |
| `features` (player_game cached) | 1 m 35 s | 0 | — |
| `merge` + `coverage` | 2 s | 0 | 4.3 MB `draft_table_colin.parquet` |
| **total, cold** | **≈ 4 min** | **≈ 118 MB** | **≈ 275 MB peak**, 63 MB after `clean` |

Free disk on this Mac was 927 MB at the start and 669 MB at the end; the floor reached was 667 MB, so the
`--keep-mb 300` guard never fired. `clean` returns ~212 MB.

### What lands where

| path | rows × cols | ships to the box? |
|:--|:--|:--|
| `data/processed/draft_table_colin.parquet` | **1,373 × 991** (bridge 672 + 319 new) | **yes** — pid-keyed, no names |
| `data/processed/season_war.parquet` | 3,885 × 3 | yes (unchanged) |
| `bridge/coverage_colin_rebuild.csv` | per family, per band, before/after | yes (report) |
| `bridge/collision_audit.csv` | 205 name collisions with n / Spearman / ratio / action | yes (report) |
| `data/processed/draft_table_his.parquet` | his own name-keyed table, 1,373 × 596 | **NO — carries player names** |
| `data/processed/drafts.parquet`, `_pid_names.parquet` | name-keyed | **NO** |
| `data/raw/`, `data/external/` | raw sources | no |

`data/` is already in `.gitignore`, and `merge` asserts `bbref_id == key == player == pid`, that no rebuilt column
is an object column, and that no column name contains "name".

### Feeding it to layer 1

`draft_table_colin.parquet` passes `tournament.contract.validate()` unchanged. To score against it, point the
winner at it (or copy it over `draft_table.parquet` on the box):

```bash
cp data/processed/draft_table_colin.parquet data/processed/draft_table.parquet   # on the BOX only
python3 -m bridge.run_winner --window holdout
```

The new families are addressable as contract groups without any further change: `a_box`, `j_bio`, `j_team`,
`j_box`, `j_shot`, `j_aau`, `j_event` are explicit lists in `tournament/contract.py`; `hoopr` (`h_`),
`game` (`g_`), `shrunk` (`sh_`), `transfers` (`tr_`), `ianstack` (`is_`) are prefix groups that now have columns
and therefore now exist. **None of them is in the winner's feature list**, so the ported winner is unchanged until
you add them — which is a walk-forward decision, not a default.

## 8. What could not be rebuilt, and why

### (a) Families deliberately not attempted — our own blocks already fill them

| his family | prefix | why not rebuilt |
|:--|:--|:--|
| `person` | `bio_` | `infra/builders/bio.py` scrapes basketball-reference player pages. Not permitted. Our `wt_` / `misc_` Wikipedia-rule block fills all 41 columns (0.74 / 0.83 / 0.84 by band). |
| `intl_pro`, `intl_fiba`, `intl_z` | `i_`, `iz_` | `infra/builders/intl.py`'s two largest stages are basketball-reference international league pages and `stats.gleague.nba.com`. Not permitted. Our `intl_` / `fy_` / `eur_` blocks fill 26/28, 17/27 and 39 columns. The rebuild writes an empty, correctly-typed `data/external/intl_prospects.parquet` so his `dataset.build_table()` runs unmodified. **Still all-NaN:** `i_tov_36`, `i_ast_tov` (we carry international turnovers only as a percentage of possessions) and the ten `i_fiba_*` per-event columns (`u16`, `u17wc`, `u18`, `u19wc`, `u20`, `gp`, `trb_36`, `blk_36`, `fg3a_rate`, `ft_pct`) — our `fy_` block is age-relative and z-scored inside the event cohort rather than raw per-36. |
| `mock`, `momentum`, `scouting`, `comp` | `mock_`, `mo_`, `sc_`, `sct_` | Wayback re-scrapes of mock boards and NBADraft.net, 20–40 min each, for columns our `vcons_` / `bb_` / `sc_` / `cp_` blocks already fill. |
| labels (`war5`, `season_war`) | — | his `infra/war.py` needs bbref advanced tables + RAPTOR. Our staging per-season outcomes replace them; `season_war.parquet` is unchanged by the rebuild. |

### (b) Sources that exist but were skipped

| loader | why |
|:--|:--|
| `load_kaggle_college` (`k_`) | `huggingface_hub.snapshot_download`. His own docstring: it is a 2021 Torvik snapshot and "returns only the keys of the player-seasons it holds" — **no new column**. |
| `load_score` (`s_`) | his own docstring: "there is no pre-draft column to expose: returns keys only". |
| `load_marchmadness` (`mm_`) | HF snapshot; would add 11 `mm_*` columns (starts, WS, team SOS/SRS, seed, tournament wins) for NCAA-tournament players 2011+. Needs `huggingface_hub`, which is not installed on this Mac, and the group is not in the winner's feature list. **This is the one genuinely missing permitted source** — add it with `pip install huggingface_hub` then `features --only marchmadness` after registering it in `PERMITTED`. |
| `odds.py`, `coaches.py`, `development.py`, `gleague.py`, `population.py`, `hoopexplorer.py` | live sportsbook archives, bbref coach pages, stats.nba.com, hoop-explorer. None permitted, none in the winner's feature list; his ideas log records `dev`, `population`, `coach` and `impact` as non-improvements. |

### (c) Columns that are empty because the permitted source does not carry them

| column(s) | reason |
|:--|:--|
| `c_shoot_pct` | the MichLitt combine mirror has no `SPOT_*` / `OFF_DRIB_*` / `ON_MOVE_*` shooting-drill columns. (The output table still has values here — they come from our own `vcmb_` block, not from the rebuild.) |
| `ev_min`, `ev_placement` | present in the JasonG sheet but as `MM:SS` (`20:00`) and as medal/finish strings (`Gold`, `6th`, `-`). His `external.jasong` coerces every mapped column with `pd.to_numeric(errors="coerce")`, so **these two were empty in his pipeline too**. Recovering them would need a parser he did not write. |

### (d) Class-band limits of each rebuilt family (this is where the coverage table's zeros come from)

| family | first usable class | last usable class | why |
|:--|:--|:--|:--|
| `a_*` (AyushBatra) | **2004** | **2024** | the repo ships `draft_players.csv` (2004–2023) and `draft_players24.csv` (2024); there is no 2025 file, and no file covers 2000–2003. |
| `rsci`, `sos`, `j_*`, `j_shot` | **2011** | 2025 (thin) | the JasonG sheet's `2008-09` and `2009-10` rows carry recruiting and bio columns only — every box-score column is null there — so `j_PER` and friends start with the 2011 class. `2024-25` has just 9 rows, so the 2025 class is 14% covered. |
| `aau_*` | **2020** | **2024** | the AAU block exists only for the `2019-20` … `2023-24` seasons (75 rows in total). |
| `ev_*` | 2011 | 2024 | 518 rows overall, concentrated in the same window. |
| `h_*` (hoopR) | **2005** | 2026 | ESPN's men's college feed holds 1 game in the 2003 season and 11 in 2004; full coverage starts 2005. His own `college_sources.py` says the same. |
| `g_*`, `t_*`, `sh_*`, `tr_*`, `rs_*` (his 11) | **2008** | 2026 | all five are keyed to a matched **Torvik** final season, and the Torvik export starts with the 2007-08 season. |
| `is_college_ws` | 2000 | 2022 | the ianstack sheet stops at the 2022 combine. |
| `c_*` (combine) | 2000 | 2026 | full range; the 2020 class is 47% because that combine was cancelled. |

### (e) Consequence for the 2003–2009 band

`all_torvik` in 2003-09 is still 0.459 — the rebuild does not move it, because his named Torvik features are fed
from the Torvik export, which starts in 2008. What the rebuild *does* add there is a genuine college box-score
line from a different source: **`hoopr` (35 columns) is 0.566 in 2003-09** and `a_box` (17 columns) is 0.520.
That is exactly how Colin covered the pre-Torvik classes — his `dataset.build_table` marks a 2003-2007 draftee
`modelled` on `h_gp >= 5`, and the `h_*` season line is what his models saw for them. So the first experiment
worth running on the box is now a genuine choice rather than a workaround:

* `--ctx-start-catboost 2008` (drop the thin classes), **versus**
* keep `ctx_start: 2003` and add `"hoopr"` and `"a_box"` to the feature list so those classes carry real data.

Decide it on the walk-forward window, never on the holdout.
