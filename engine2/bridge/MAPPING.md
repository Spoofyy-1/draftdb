# bridge/MAPPING.md — our verified columns → Colin's contract

`bridge/build_table.py` turns the verified, pre-draft-dated staging tables into the two parquets his tournament
consumes. Nothing in his `infra/builders/` runs: his raw sources (basketball-reference labels + biography, ESPN /
hoopR game logs, AyushBatra, JasonG) are off-limits here and are replaced by our own blocks.

```
/Users/kennakao/nba/datarebuild/v4_build/staging_v419/train_2000_2018.csv   1,458 rows, classes 2000-2018, labelled
/Users/kennakao/nba/datarebuild/v4_build/staging_v419/tests/test_YYYY_inputs.csv   2019-2025, sealed labels
/Users/kennakao/nba/datarebuild/v4_build/staging_v419/input_columns.json    the 1,215 legal model inputs
/Users/kennakao/Downloads/nba_redraft_handoff/identity_KEEP_SEPARATE/tabular_names.csv   pid -> actual_pick ONLY
/Users/kennakao/nba/datarebuild/v4_build/screen_blocks.csv   the pre-fold label screen, for --response-max
        ->  engine2/data/processed/draft_table.parquet     1,373 rows x 672 columns
                                                          (760 with --families, 1,256 with --extras)
        ->  engine2/data/processed/season_war.parquet      3,885 player-seasons, 975 players
```

Only the 1,215 columns listed in `input_columns.json` are read; `cons_src2_n` and the meta columns are ignored.
**Names are never read.** `player_name` and `nba_id` are not loaded into memory, and `bbref_id`, `key` and `player`
are all set to the pid, so nothing carrying a name can reach the box.

---

## 1. What layer 1 and layer 2 actually consume

Working this out was half the job; it is recorded here so the next person does not have to.

**`data/processed/draft_table.parquet`** — one row per candidate. Required *identity / protocol* columns:

| column | dtype | who reads it | what we put there |
|:--|:--|:--|:--|
| `bbref_id` | str | join key to `season_war`, layer-1 output key, `layer2.wide` index | the pid |
| `key` | str | `layer1.config_sig` cache hash | the pid |
| `player` | str | copied into layer-1 predictions, `layer2.wide` meta | the pid (**never a name**) |
| `draft_year` | int | every split, every within-class transform | as given |
| `pick` | float | **scoring metadata only** — `pipeline.run.year_metrics`, `layer2.evaluate`, and a hard `assert pool["pick"].notna().all()` in layer 1 | `actual_pick` from the identity file; `--no-picks` substitutes a within-class placeholder |
| `modelled` | bool | context filter `t[t.modelled & t.labelled …]` | has a college line, a pre-draft pro season, or combine measurements |
| `source` | str | `segment`, `layer2.rule_intl`, coverage reports | `college` / `intl` / `other` / `none` |
| `labelled` | bool | context filter; `layer2.evaluate` skips unlabelled rows | True for 2000-2018, False for the sealed 2019-2025 |
| `war5` (`config.TARGET`) | float | the label; `year_metrics`; `layer2.evaluate` | `y_early_war`; NaN for 2019-2025 |
| `seasons_played` | int | `pipeline.run._picks` only | count of non-null `y_sN_war` |
| `age_at_draft`, `height_in`, `class_year`, `rec_rank`, `role` | float | carried into every layer-1 prediction row | see §2 |

`layer2.wide()` additionally joins, when present: `iz_young_x_eff`, `iz_eff_36`, `mock_rank_consensus`,
`mock_n_sources`, `TILT_COLUMNS`, `STACK_RAW`. All of these exist in our table except the `co_*` coach columns
(`TILT_COLUMNS`) and `gl_pred_ws48` (`STACK_RAW`); both are optional there (`[c for c in extra if c in tab.columns]`)
and only matter for the `tilt` / `stackctx` rules, which the clean winner does not use.

`tournament.contract.validate()` **asserts every column of every explicit group exists**, so families we cannot
fill are still written, empty (§4). `layer1.select_features` then drops any column that is all-NaN or constant
*inside the training fold*, so empty families cost nothing.

**`data/processed/season_war.parquet`** — `(bbref_id, season, war)`. Read by `layer1.horizon_target`,
`layer1.transform_labels("disc85_gaussrank")` and `infra.war.war_target`. Built from `y_s1_war … y_s5_war` with

```
season = draft_year + i        (i = 1..5, rows with a null WAR omitted)
```

That mapping is what makes both label cutoffs work: `season > draft_year` is always true, and `season <= through`
truncates a label to the seasons finished by draft night `through` (`through = 2026` for `label_cutoff: full`,
`through = Y` for `causal`). Verified: `war_target(ctx, seasons, through=2026)` reproduces `y_early_war` to 7e-15,
and the causal labels of the three classes before the scored year shrink (2019: mean 5.13 → 0.98).

**Caveat.** Our per-season array is a *prefix* — 1,452 of 1,458 rows have season 1 and no row has a hole — so it is
"the n-th season he played", not literally "the season draft_year + n". For a draft-and-stash player who debuted
late the two differ, and the causal cutoff then credits his seasons one or two years early. Six players (all
never-played) have no rows at all and score `NEVER_PLAYED_WAR = 0`, which matches their `y_early_war = 0`.

**The winner's config dict** (`bridge/run_winner.py::build_configs`) — CatBoost and TabICL, two seed batches each,
rank-averaged by `layer2.rule_blend`:

```json
{"model":"catboost","ctx_start":2003,"features":["all_torvik","-cat","traj","phys","intl_pro","intl_fiba",
 "combine","tctx","intl_z","mock","momentum","response","person","scouting","comp"],
 "label":"disc85_gaussrank","label_cutoff":"full","label_horizon":"match3","n_estimators":800,
 "seeds":5,"seed_start":0,"name":"C03 m3 +mo+rs+person+sc+comp2 x5"}
{"model":"tabicl","ctx_start":2010,"features":[… same minus "response" …],"n_estimators":8,
 "model_options":{"outlier_threshold":2.0},"label":"disc85_gaussrank","label_cutoff":"full",
 "label_horizon":"match3","seeds":4,"seed_start":0,"name":"T10 o2 m3 +mo+person+sc+comp2 x4"}
```

`match3` resolves to `max(3, min(5, LAST_SEASON - Y))` training seasons: 5 for classes ≤ 2021, 4 for 2022,
3 for 2023-2025. Depth 5 / lr 0.03 / l2 5.0 are `infra/models.py` defaults, not config keys.

---

## 2. Torvik line and the named groups

Coalescing order is written `a → b → c` (first non-null wins), and **every step is converted onto Torvik's scale
before it enters the chain**, so a column carries one scale and one unit across all 1,373 rows. The sources, in the
order they are tried:

| block | what it is | classes covered (drafted rows) |
|:--|:--|:--|
| `tv_` | the player's own Torvik final-season row | 2010+ only, ~0.80 inside those classes |
| `col_` | our own final-college-season line — box line from 2002, ratings (`ortg/drtg/adj_*/impact*/value`) from 2008 | 0.74 / 0.24 in 2003-09, ~0.85 after |
| `ctx_base_` | the same college season restated from the game log (Torvik-scaled already) | 2008+ |
| `cgd_` | the same season as a per-game attempt/counting line | 2008+ |
| `tc_`, `traj_`, `hs_`, `rsci_`, `bio_`, `vmb_`, `vcmb_`, `dx_` | context, trajectory, recruiting, biography, measurements | see the chains |

### Scale conversions applied (each verified as a median ratio on the tv_ / col_ overlap, n = 653-935)

| conversion | applied to | check |
|:--|:--|:--|
| 0-1 → 0-100 (`× 100`) | `col_ft_pct`, `col_fg2_pct`, `col_fg3_pct`, `col_ftar` | `col_ftar × 100` / `tv_ftr` = 1.00 |
| minutes share → Torvik min% (`× 500`) | `col_minutes_share` (a share of all 5×40 team minutes) | ratio 1.007; `ctx_base_minutes_share` is already 0-100 (1.000) and is tried first |
| per-36 → per game (`× mpg/36`) | `col_pts36 / reb36 / ast36 / stl36 / blk36`, `traj_last_pts36` | ratio 1.000 against `col_*_pg` |
| per-40 → per game (`× mpg/40`) | `col_fg3a_per40`, `col_dunks_per40` | `col_dunks_per40 / tv_dunks_pg` = 1.266 = 40/mpg |
| season total → per game (`÷ GP`) | `ctx_base_fg3a`, `ctx_base_fg2a` (medians 109 and 251 — totals, not rates) | ratio 33.99 ≈ GP |
| per-game → per 40 (`× 40/mpg`) | `cgd_pf_pg` → `pfr` | ratio 1.001 against `tv_pfr` |
| recruiting **score** → recruiting **rank**: `(100 − score) × 5`, clipped to 1..101 | `tv_rec_rank`, `col_recruit_score` | decile-checked: score 99.8→rank 2, 98.4→8, 95.0→25, 90.8→46, 86.8→66 |
| class index clipped to 1..4 | `col_class_index` (runs to 6 for a super-senior) | Torvik's `yr` is Fr/So/Jr/Sr |
| `bio_pos_code` → Torvik `role` id | `{1:2, 2:4, 3:5, 4:7, 5:8, 6:7}` | the modal Torvik role per position code on the 667-row overlap |

⚠ **The old `rec_rank ← 101 − tv_rec_rank` was a scale bug.** `tv_rec_rank` is a 0-100 *score*, not a rank:
`101 − score` had median 5 against `rsci_rank`'s 22 on the same rows, so the column carried two incompatible
scales. It is now `(100 − score) × 5`. 165 rows changed; every other named feature is value-identical to v1 where
both are non-null (`rho` 1.000, max abs diff 0) except the ones re-sourced on purpose (`Min_per`, `oreb`, `dreb`,
`TPA_pg`, `FTA_pg`, `class_year`, `role`).

### `all_torvik` — 48 numeric + 2 categorical

| his | chain (first non-null wins, all on Torvik's scale) |
|:--|:--|
| `GP` | `tv_gp` → `col_gp` → `ctx_base_gp` → `cgd_observed_gp` → `traj_last_gp` |
| `Min_per` | `tv_min_per` → `ctx_base_minutes_share` → `col_minutes_share × 500` |
| `mpg` | `col_mpg` → `tv_mp` → `ctx_base_mpg` → `traj_last_mpg` |
| `ORtg` | `tv_ortg` → `col_ortg` |
| `adjoe` | `tv_adjoe` → **`col_adj_ortg`** |
| `drtg` `adrtg` | `tv_drtg` → `col_drtg`; `tv_adrtg` → `col_adj_drtg` |
| `usg` `eFG` `TS_per` `ORB_per` `DRB_per` `AST_per` `TO_per` | `tv_*` → the matching `col_*_pct` → the matching `ctx_base_*` (all three already 0-100) |
| `blk_per` `stl_per` | `col_blk_pct` → `ctx_base_blk_pct`; `col_stl_pct` → `ctx_base_stl_pct` (no `tv_` equivalent) |
| `FT_per` `twoP_per` | `col_ft_pct ×100` → `ctx_base_ft_pct ×100`; same for `fg2` |
| `TP_per` | `col_fg3_pct ×100` → `ctx_base_fg3_pct ×100` → **`ctx_base_fg3m / ctx_base_fg3a ×100`** |
| `ftr` | `tv_ftr` → `ctx_base_ftr` → `col_ftar × 100` |
| `porpag` | `tv_porpag` → **`col_value`** (our points-over-replacement-per-game; ratio 1.000, ρ 0.97) |
| `pfr` | `tv_pfr` → **`ctx_base_fouls40`** → **`cgd_pf_pg × 40/mpg`** |
| `ast_tov` | `tv_ast_tov` → **`ctx_base_ast_tov`** → **`cgd_ast_pg / cgd_tov_pg`** |
| `bpm` | `tv_bpm` → `col_impact` → **`tc_first_bpm + tc_bpm_first_to_last`** |
| `obpm` `dbpm` | `tv_obpm` → `col_impact_off`; `tv_dbpm` → `col_impact_def` |
| `rec_rank` | `rsci_rank` → **`hs_rsci_rank`** → **`hs_recruit_rank_final`** → **`(100 − col_recruit_score) × 5`** → **`(100 − tv_rec_rank) × 5`** → **101 for an unranked college player** (see below). Lower = better; `layer2` ranks on `-rec_rank` |
| `rim_pct` `mid_pct` | `tv_rim_pct` → `col_rim_fg_pct` → **`ctx_base_rim_made / ctx_base_rim_attempts`**; `tv_mid_pct` → `col_mid_fg_pct` → **`(ctx_base_fg2m − rim_made) / (ctx_base_fg2a − rim_attempts)`** (all 0-1, as Torvik's are) |
| `dunk_made` | `tv_dunks_pg × GP` → **`ctx_base_dunk_made`** (already a season count) → **`col_dunks_per40 × mpg/40 × GP`** |
| `oreb` `dreb` | **`ctx_base_oreb_pg`** → **`cgd_orb_pg`** → `col_reb_pg × ORB%/(ORB%+DRB%)` (the v1 approximation is now the *last* resort, not the only one — 779 rows changed, by up to 0.55 rpg) |
| `treb` | `col_reb_pg` → `oreb + dreb` → `col_reb36 × mpg/36` |
| `ast` `stl` `blk` | `col_*_pg` → `ctx_base_*_pg` → `cgd_*_pg` → `col_*36 × mpg/36` |
| `pts` | `tv_pts` → `col_pts_pg` → `ctx_base_pts_pg` → `cgd_points_pg` → `col_pts36 × mpg/36` → `traj_last_pts36 × mpg/36` |
| `TPA_pg` | **`cgd_three_a_pg`** → `col_fg3a_per40 × mpg/40` → `ctx_base_fg3a / GP` |
| `FTA_pg` | **`cgd_ft_a_pg`** → `col_ftar × FGA/40 × mpg/40` (`FGA/40 = col_fg3a_per40 / col_fg3ar`) → `ctx_base_ftr/100 × (fg2a+fg3a) / GP` |
| `height_in` | `bio_height_in` → `vmb_listed_height_in` → `bio_combine_height_in` → `vcmb_height_with_shoes_in` → **`dx_last_height_in`** → **`rsci_hs_height_in`** |
| `class_year` | `tv_yr` → `col_class_index` **clipped to 1..4** → `vmb_college_class_year` |
| `n_college_seasons` | `col_n_seasons` → `tv_seasons` → `tc_n_seasons` → **`traj_n_seasons`** |
| `age_at_draft` | `bio_age_at_draft` → `tv_age_exact` → `vmb_age_reported_years` → `col_age` (his age *during the final college season*, ≈0.27 yr below draft age — last in the chain and in practice never reached, `bio_age_at_draft` covers every row) |
| `dporpag` `stops` `gbpm` `ogbpm` `dgbpm` | `tv_*` only. **No scale-compatible source exists**: `tc_porpag_share` is a team share (ratio 0.05), `col_impact_def` is a BPM component (ratio 0.013 against `stops`), `tc_slope_bpm` a per-season slope. Empty before 2010 |
| `conf` (categorical) | `tv_conf_tier` only. `tc_conf_strength_prior` is a continuous rating on a different scale (ρ 0.63, ratio 6.1) and is deliberately *not* coalesced into a tier id |
| `role` (categorical) | `tv_role` → `bio_pos_code` **mapped through `POS_TO_ROLE`**. v1 mixed the two code sets (1-8 and 1-6) in one column; 706 rows changed |

### Three invariants the chains enforce

1. **College-line columns only exist on rows with a college career.** `traj_*` is a *pre-draft* trajectory across
   any competition — 101 international rows carry `traj_last_gp` / `traj_last_mpg` with median 23.7 games and
   21.3 mpg, a pro season. Without a gate those would land in `GP` / `mpg` / `pts`. The gate is the staging `path`
   meta column (`path.startswith("college")`, 1,157 rows), which is exactly the row set the old engine's IMASK gene
   keys on, unioned with `col_gp | tv_gp`. Applies to the 46 columns in `build_table.COLLEGE_LINE`; `rec_rank`,
   `height_in`, `age_at_draft` and `role` are not college-season quantities and are filled for everyone.
2. **A corrupt staging cell must not survive a unit conversion.** `col_stl36` carries one value of 67.7 steals per
   36 minutes (a 2016 row); `× mpg/36` turned it into 69.0 steals per game. `build_table.COUNT_CEILING` sets hard
   plausibility ceilings well above the NCAA single-season records (pts 45, treb 25, ast 15, stl 6, blk 10 per game,
   GP 45, mpg 40, dunk_made 250) and anything above becomes NaN.
3. **`rec_rank = 101` means unranked, not missing.** A college player in a class the RSCI source covers who appears
   on no top-100 list is not "unknown", he is behind the last ranked recruit — the same convention
   `mock_rank_consensus ← 61` already uses for an un-mocked player. Restricted to college rows; an international
   player never in a US high-school ranking stays NaN. This is the single biggest coverage change (`rec_rank`
   0.534 → 0.812 in 2003-09, 0.759 → 0.869 in 2010-18, 0.709 → 0.883 on the holdout).

### `src_torvik` — the provenance indicator

One column, `1` when the row's college line is the player's own Torvik final season and `0` when a fallback filled
it. 668 of 1,373 rows are 1 (zero before the 2010 class, 33-47 per class after). It is its own contract group,
`srcflag` (prefix `src_`, added to `tournament/contract.py` PREFIX with a `# BRIDGE:` marker), so a config asks for
it by name: `"features": [..., "srcflag"]`. **It is not in the winner's feature list** — `bridge/run_winner.py`'s
`FEATURES_CATBOOST` / `FEATURES_TABICL` are unchanged, so the ported winner still resolves 575 / 428 columns.
Add `"srcflag"` there (or pass your own `--configs`) if you want the model to be able to separate the two
populations; that is a deliberate decision to make on the walk-forward window, not a default.

### Named-feature coverage, before and after

`python -m bridge.build_table` prints this table every run: `v1_` is the frozen pre-fallback mapping
(`bridge/coverage_before.csv`), `pri_` is the first step of each chain alone, `now_` is the full chain.

| band | v1 (before) | primary source only | **now (after)** |
|:--|--:|--:|--:|
| 2003-09 | 0.430 | 0.209 | **0.459** |
| 2010-18 | 0.841 | 0.810 | **0.853** |
| 2019-25 | 0.809 | 0.760 | **0.827** |

Biggest movers in 2003-09: `rec_rank` 0.534→0.812, `mpg` 0.582→0.634, `GP`/`pts`/`n_college_seasons` 0.741→0.750,
`porpag`/`adjoe`/`drtg`/`adrtg`/`bpm`/`obpm`/`dbpm` 0.000→0.236 (they were `tv_`-only and `tv_` starts in 2010),
`pfr`/`ast_tov` 0.000→0.222, `oreb`/`dreb` 0.222→0.585, `TP_per` 0.406→0.474, `role` 0.000→1.000.
In 2010-18: `TP_per` 0.674→0.844, `rec_rank` 0.759→0.869, `dunk_made` 0.797→0.846, `FTA_pg` 0.774→0.844,
`porpag`/`adjoe` 0.797→0.849.

**The 2003-09 ceiling is a data limit, not a mapping limit.** Our ratings line (`col_ortg/adj_*/impact*/value`)
does not exist before the 2008 class and the box line does not exist before 2002, so 14 of the 50 named features
are still 0.00-0.24 there however they are coalesced. `--ctx-start-catboost 2008` remains the first experiment.
Exactly one column loses against v1, by design: `class_year` (0.857 → 0.855 in 2010-18, → 0.849 on the holdout),
because v1 filled `vmb_college_class_year` for a handful of players with no college career at all. Every other
named feature is >= v1 in every band.

### `traj` — 16 columns, 10 filled

`d_bpm ← tv_d_bpm`, `d_usg ← tv_d_usg`, `d_TS_per ← tv_d_ts`, `d_mpg ← traj_d_mpg`, and each `prev_X = X − d_X`.
`career_bpm_mean` / `career_bpm_max` are the mean / max of {final, previous, first} season BPM
(`tc_first_bpm → tv_first_bpm`) — an approximation of his full Torvik career aggregate.
**Not fillable:** `prev_obpm`, `prev_dbpm`, `prev_porpag`, `d_obpm`, `d_dbpm`, `d_porpag` — we carry no
previous-season delta for those three stats.

**Deliberately not coalesced.** His deltas are *previous season → final season*. Four columns of ours look like
candidates and are all something else, so none is used: `tv_d_min_per` is a minutes-*share* delta, not minutes per
game (mixing them produced a −37…+68 range); `col_usg_delta` (coverage 0.56 against `tv_d_usg`'s 0.32) and
`traj_mpg_growth` (0.64 against `traj_d_mpg`'s 0.44) are *first* season → final season, ρ 0.54 / 0.44 with a
systematic +1.2 / +3.4 offset; `tc_slope_bpm` is a per-season regression slope (ratio 0.645 against `tv_d_bpm`).
Filling a one-season delta from a whole-career change would be exactly the kind of two-populations-in-one-column
mistake the rest of this file is about. `--families growth` and `--extras` make all four reachable as their own
columns instead, which is the honest way to use them.

### `phys` — 4/4

`wingspan_in ← vcmb_wingspan_in → bio_combine_wingspan_in → dx_last_wingspan_in`;
`weight_lb ← bio_weight_lb → vcmb_weight_lb → vmb_listed_weight_lb`;
`wing_minus_height ← vcmb_wingspan_minus_height_in → bio_combine_… → (wingspan − height)`;
`draft_age_x ← bio_age_at_draft → vmb_age_reported_years → age_at_draft`.

### `intl_pro` — 28 columns, 26 filled ← our `intl_` (+ `eur_` for the top-level line)

`i_has_pro = intl_gp > 0`; `i_league_level ← intl_level`; `i_top_level ← intl_best_level`;
`i_n_comps ← intl_n_seasons`; `i_age_season ← intl_age`; `i_gp/i_mpg ← intl_gp/intl_mpg`;
`i_pts_36 / i_trb_36 / i_ast_36 / i_stl_36 / i_blk_36 ← intl_pts36 / reb36 / ast36 / stl36 / blk36`;
`i_pir_36 ← intl_impact`; `i_fg3_pct / i_ft_pct / i_fg3a_rate / i_fta_rate ← intl_fg3_pct / ft_pct / fg3ar / ftar`;
`i_ts / i_efg ← intl_ts_pct / intl_efg_pct ÷ 100` (ours are 0-100, his are 0-1).

Derived: `i_orb_36` = `intl_reb36` × offensive share of `intl_orb_pct`/`intl_drb_pct` (thin: 4% coverage);
`TSA/36 = pts36 / (2·TS)`, `i_fga_36 = TSA/36 ÷ (1 + 0.44·FTr)`, `i_fg3a_36 = 3PAr·FGA36`, `i_fta_36 = FTr·FGA36`;
`i_fg_pct = eFG − 0.5·3P%·3PAr`. `i_top_mpg ← eur_el2_ls_min / eur_el2_ls_games` (falling back to `intl_mpg` when
his league already is his best level); `i_top_pir_36 ← eur_el2_ls_pir40 × 36/40`.
**Not fillable:** `i_tov_36`, `i_ast_tov` — we carry international turnovers only as a percentage of possessions.

### `intl_fiba` — 27 columns, 17 filled ← our `fy_` (+ `eur_angt_*`)

**Scale differs from his on purpose.** His columns were raw FIBA per-36 rates; our `fy_` block is age-relative and
z-scored *inside the event cohort*, which is the stronger form of the same information. Mapping:
`i_fiba_youth_events ← fy_n_tournaments`, `i_fiba_max_level ← fy_best_level`,
`i_fiba_best_rank ← fy_team_finish_last`, `i_fiba_last_age ← fy_age_rel_last` (relative, not absolute, age),
`i_fiba_mpg ← fy_minutes_share_last` (a share, not minutes), `i_fiba_pts_36 ← fy_best_pts40_z`,
`i_fiba_last_pts_36 ← fy_pts40_z_last`, `i_fiba_ast_36 ← fy_ast40_z_last`,
`i_fiba_stl_36 ← fy_stl_blk40_z_last`, `i_fiba_pir_36 ← fy_best_pir40_z`,
`i_fiba_last_eff_36 ← fy_pir40_z_last`, `i_fiba_ts ← fy_ts_z_last`, `i_fiba_best_effpg ← fy_z_trend`,
`i_ngt ← eur_angt_has`, `i_ngt_last_age ← eur_angt_age_first`, `i_ngt_pir_36 / i_ngt_pts_36 ← eur_angt_pir40 /
pts40 × 36/40`.
**Not fillable:** the per-event flags `i_fiba_u16 / u17wc / u18 / u19wc / u20`, `i_fiba_gp`, `i_fiba_trb_36`,
`i_fiba_blk_36`, `i_fiba_fg3a_rate`, `i_fiba_ft_pct`. `fy_has_youth`, `fy_underage_flag`, `fy_age_rel_min`,
`fy_usage_proxy_z_last`, `fy_hoop_summit_world` have no counterpart in his explicit list and are only reachable
through `--extras` (`intl_fiba` is a fixed list, not a prefix group, so extra `i_fiba_*` columns would be inert).

---

## 3. Prefix groups (renaming with the prefix is enough — `contract.groups()` matches by prefix)

| his group | prefix | ← ours | his cols | ours | note |
|:--|:--|:--|--:|--:|:--|
| `combine` | `c_` | `vcmb_` (37) + `bio_combine_` (13) | 30 | 30 | 15 map straight across; the ratios, lean/fat mass, touches, approach gain, `c_anthro_n` / `c_drills_n` and `c_shoot_pct` (made ÷ attempted over all six spot / off-dribble / on-move drills) are derived exactly as his builder did. Every measurement now falls back `vcmb_X → bio_combine_X` (the same tape from the biography feed) |
| `tctx` | `t_` | `tc_` (107) | 72 | 107 | one-for-one rename `tc_X → t_X`. Same idea (all-D1 cohort percentiles, team shares, teammate quality, availability, tournament), different column names — harmless, the group is prefix-matched. **Our `tctx_` block is *not* this**: `tctx_` is team four-factors and goes to `--extras` |
| `intl_z` | `iz_` | `intl_`, `eurs_` (17) | 18 | 39 | `iz_pts_36 ← intl_lg_adj_pts36`, `iz_eff_36 ← intl_impact`, plus `mpg/age/ts/usg/n_league/trb/ast/stl/blk/tov/fg3a/fta` from `intl_`. `iz_pct_eff` = within-class percentile of `intl_impact`; `iz_young_x_{eff,mpg,usg}` = `max(0, 22 − intl_age) ×` the stat, his definition. The prior-chosen Euroleague/EuroCup/ANGT slim spine lands as `iz_eur_*` |
| `mock` | `mock_` | `vcons_`, `cons_`, `dis_` | 16 | 7 | `mock_rank_consensus ← vcons_mock_mean_rank → cons_mock_consensus_rank`, **unranked players filled with 61** (behind the last pick, as `infra/models.py::CONSENSUS_FEATURE` does). Also `mock_rank_consensus_alt`, `mock_rank_best`, `mock_rank_range`, `mock_n_sources`, `mock_rank_std ← dis_mock_rank_std`, `mock_first_round = consensus ≤ 30`. We have no per-source board ranks, so his 13 `mock_rank_<site>` columns have no equivalent |
| `momentum` | `mo_` | `mock_` (13) + `bb_` (23) | 17 | 36 | our `mock_` block **is** the momentum block (`mo_last`, `mo_first_days`, `mo_delta_late`, `mo_snaps`, `mo_std`, …) and `bb_` adds per-board final / 30d / 60d ranks and deltas. ⚠ our `mock_*` is empty for every class ≤ 2019 and only ~43-85% covered for 2020-2025 — `select_features` drops those columns inside every fold, so in practice `momentum` = the `bb_` half |
| `response` | `rs_` | `gl2_` (147) | 11 | 147 | rename `gl2_X → rs_X`. Much wider than his 11 (opponent-tier deltas, tournament residual, consistency, absence blocks, close games). CatBoost only |
| `person` | `bio_` | `wt_` (29) + `misc_` (6) | 6 | 41 | ⚠ **our `bio_` block is physicals and feeds `phys` / `all_torvik`; his `bio_` prefix is the biography group.** `bio_nba_relative_n ← wt_n_relatives_pro_any_sport → wt_nba_relative`, `bio_nba_father ← wt_relative_is_parent`, `bio_shoots_left ← wt_left_handed` (⚠ ~2% coverage), `bio_born_us ← 1 − wt_born_outside_usa → misc_born_usa`, `bio_n_high_schools ← wt_hs_transferred`, `bio_prep_academy ← wt_prep_year → misc_prep`; every other `wt_`/`misc_` column follows as `bio_<suffix>` (injuries, JUCO, reclass, transfers, birth state/metro) |
| `scouting` | `sc_` | `sc_` (24) | 28 | 25 | our prefix is already his. Renames: `sc_nbaready → sc_nba_ready`, `sc_ballhandling → sc_handle`, `sc_postskills → sc_post`, `sc_kw_* → sc_flag_*`. Added `sc_ratio_strength_weakness`. **Not fillable:** `sc_height_in`, `sc_weight`, `sc_wingspan_in`, `sc_snapshot_days_before_draft`. Our `txt_` keyword block is *not* folded in here — `--extras` |
| `comp` | `sct_` | `cp_` (18) | 3 | 20 | rename `cp_X → sct_X`, plus `sct_comp_n ← cp_n_comps` and `sct_words_total = sc_words_strengths + sc_words_weaknesses`. His `sct_comp_war` (the comp's realised WAR/season through draft night) has no equivalent — ours is the comp's *dated career line* (`sct_pts36_to_date`, `sct_peak_mpg_to_date`, `sct_seasons_to_date`, `sct_comp_is_hall_tier`, …), which is the same idea measured differently |

### Resulting feature counts

| member | his | ours (resolved) | with `--response-max 24` | with `--response-max 11` |
|:--|--:|--:|--:|--:|
| CatBoost (`all_torvik -cat traj phys intl_pro intl_fiba combine tctx intl_z mock momentum response person scouting comp`) | 324 | **575** | 452 | 439 |
| TabICL (same minus `response`) | 313 | **428** | 428 | 428 |

Ours is wider mostly through `response` (+136), `tctx` (+35), `person` (+35), `intl_z` (+21), `momentum` (+19),
`comp` (+17). That is a real difference from his winner and a thing to check first if the holdout number moves.
The knobs, in the order worth trying: `--response-max N` on the build (his `response` group is 11 columns wide),
`bridge/run_winner.py --ctx-start-catboost / --ctx-start-tabicl`, and a narrower `FEATURES_*` list.

---

## 4. Families we cannot fill

| his group | columns | why |
|:--|--:|:--|
| `a_box` (`a_*`) | 17 | AyushBatra's sports-reference college line. Not a source this project uses; present and empty |
| `j_bio` `j_team` `j_box` `j_shot` `j_aau` `j_event` (`rsci`, `sos`, `j_*`, `aau_*`, `ev_*`, `dunks_per_min`, `pct_rim`, `pct_astd`) | 67 | JasonG7234's sheet (RSCI, SOS, hoop-math shot location, AAU / showcase lines). Present and empty. Our own recruiting block (`rsci_`) reaches the model through `rec_rank` and `--extras`, not through `j_bio` |
| `kaggle` `score` `ianstack` `hoopr` `marchmadness` `shrunk` `eurocamp` `bwb` `academy` `transfers` `odds` `dev` `gleague` `population` `impact` `coach` `game` | — | prefix groups; no columns written, so `contract.groups()` never creates them. None is in the winner's feature list. `STACK_RAW`'s `gl_pred_ws48` and `TILT_COLUMNS`' `co_*` therefore do not exist — only the `tilt` / `stackctx` layer-2 rules would notice |

Coverage of what *is* filled (mean non-null rate over the group's live columns):

| family | cols | filled | 2003-18 | 2019-25 | (v1 2003-18) |
|:--|--:|--:|--:|--:|--:|
| all_torvik | 50 | 50 | **0.602** | **0.827** | 0.583 |
| traj | 16 | 10 | 0.343 | 0.513 | 0.340 |
| phys | 4 | 4 | 0.850 | 0.886 | 0.843 |
| intl_pro | 28 | 26 | 0.259 | 0.320 | 0.259 |
| intl_fiba | 27 | 17 | 0.211 | 0.309 | 0.211 |
| combine `c_` | 30 | 30 | 0.591 | 0.632 | 0.591 |
| tctx `t_` | 107 | 107 | 0.472 | 0.792 | 0.472 |
| intl_z `iz_` | 39 | 39 | 0.152 | 0.180 | 0.152 |
| mock `mock_` | 7 | 7 | 0.564 | 0.935 | 0.564 |
| momentum `mo_` | 36 | 36 | 0.262 | 0.500 | 0.262 |
| response `rs_` | 147 | 147 | 0.558 | 0.736 | 0.558 |
| person `bio_` | 41 | 41 | 0.769 | 0.838 | 0.764 |
| scouting `sc_` | 25 | 25 | 0.589 | 0.623 | 0.589 |
| comp `sct_` | 20 | 20 | 0.689 | 0.673 | 0.689 |
| srcflag `src_` | 1 | 1 | 1.000 | 1.000 | — |

**Coverage is materially thinner in the old context classes than in the holdout** (all_torvik 0.60 vs 0.83, tctx
0.47 vs 0.79). Our Torvik block starts with the 2010 class, `col_ortg/drtg` with 2008, `tc_` with 2008, `bb_` with
2009, `gl2_` with 2005. His `ctx_start: 2003` therefore buys eight classes that are mostly empty rows here, where
for him they carried hoopR / AyushBatra lines. **Try `--ctx-start-catboost 2008` on the box** — it is the most
likely single difference between his 0.504 and whatever this scores.

## 5. Build options

All four are off / unlimited by default, so a plain `python -m bridge.build_table` writes the table the ported
winner expects (1,373 × 672, 575 / 428 resolved features).

### `--families` — our extra dated blocks as first-class contract groups

Writes four of our blocks under their own prefixes instead of burying them in the `xt_` catch-all, and
`tournament/contract.py` PREFIX gains the matching groups (all `# BRIDGE:`-marked), so a config names them
directly: `"features": [..., "fiba_youth", "growth"]`. Table goes 672 → 760 columns.

| group | prefix | cols | 2003-09 | 2010-18 | 2019-25 | note |
|:--|:--|--:|--:|--:|--:|:--|
| `fiba_youth` | `fy_` | 18 | 0.29 | 0.42 | 0.42 | the whole block, including the five columns `intl_fiba`'s fixed list has no slot for (`fy_has_youth`, `fy_underage_flag`, `fy_age_rel_min`, `fy_usage_proxy_z_last`, `fy_hoop_summit_world`) |
| `growth` | `dx_` | 30 | 0.45 | 0.66 | **0.13** | ⚠ see below |
| `euroleague` | `eur_` | 155 | 0.065 | 0.062 | 0.052 | the full spine. `eur_el2_ls_*` and `eur_angt_*` are also consumed by `intl_pro` / `intl_fiba`, and `eurs_` by `iz_eur_*`; the raw copies are emitted anyway so the group is complete |
| `draftpage` | `dp_` | 8 | 0.24 | 0.65 | 0.56 | green room, wave and days before, early / international early entrant, auto-eligible, wiki-projected |

Three of our blocks are **already** first-class and are not repeated here: `wt_` → `person` (`bio_`),
`bb_` → `momentum` (`mo_`), `cp_` → `comp` (`sct_`).

> ⚠ **`growth` is a train/serve trap.** `dx_` coverage runs 0.67-0.72 through the 2012-2017 classes and then
> collapses: 2018 0.49, 2019 0.30, 2020 0.34, 2021 0.18, 2022 0.04, 2023 0.03, 2024 0.00, 2025 0.02. `select_features`
> only drops a column that is all-NaN *inside the training fold*, so every `dx_` column survives training and is
> empty at holdout time. Use it on the walk-forward window if at all, and never assume a holdout gain from it.

### `--intl-mask` — the old engine's IMASK gene

On a row with a college career the `intl_` block is not a pre-draft pro season: it is a 6-8 game FIBA youth
tournament line (117 of the 125 such rows sit at `intl_level == 1`), and a model reads it with pro-season
coefficients. `genes/patch_opt_v34.py` blanks it:

```python
ic = [c for c in B.columns if c.startswith(("intl_", "adj_intl_")) and c != "intl_youth_n"]
B.loc[meta.path.astype(str).str.startswith("college"), ic] = np.nan
```

`--intl-mask` reproduces that on **1,157 rows** (`path.startswith("college")`, unioned with `col_gp | tv_gp`) for
the **47 bridge columns fed from an `intl_*` staging column**:

```
intl_pro (25):  i_has_pro i_league_level i_top_level i_n_comps i_age_season i_gp i_mpg i_pts_36 i_trb_36
                i_orb_36 i_ast_36 i_stl_36 i_blk_36 i_pir_36 i_fga_36 i_fg3a_36 i_fta_36 i_fg_pct i_fg3_pct
                i_ft_pct i_fg3a_rate i_fta_rate i_ts i_efg i_top_mpg
intl_z   (22):  iz_pts_36 iz_eff_36 iz_mpg iz_age iz_ts iz_usg_36 iz_n_league iz_trb_36 iz_ast_36 iz_stl_36
                iz_blk_36 iz_tov_36 iz_fg3a_36 iz_fta_36 iz_lg_strength iz_season_gap iz_prev_pts_36
                iz_two_pts_36 iz_pct_eff iz_young_x_eff iz_young_x_mpg iz_young_x_usg
```

**Not blanked**, deliberately: `i_top_pir_36` and all 17 `iz_eur_*` come from `eur_` / `eurs_` only, never from an
`intl_` column, so the engine's rule does not reach them (`i_top_mpg` *is* blanked because it falls back to
`intl_mpg`). `intl_youth_n` is kept, exactly as the gene keeps it — the "did he play youth ball at all" counter is
the one piece of the block that means the same thing on a college row. With `--extras` the six raw `xt_intl_*`
copies are blanked too (53 columns in total); `intl_fiba` is untouched because it is fed from `fy_` and
`eur_angt_`, which are youth data by construction and already on the right scale.

### `--response-max N` — trim the widest group to the strongest N columns

His `response` group is 11 columns; ours is 147 (`gl2_` renamed to `rs_`). `--response-max N` keeps the N with the
highest `abs_res` in `datarebuild/v4_build/screen_blocks.csv` — the 2000-2011 **pre-fold** label screen
(n-weighted mean within-class Spearman against the residual of `y_early_war_log` on `log(actual_pick)`: information
beyond the draft slot). Classes 2012-2018 are the walk-forward folds and are never touched by it.

That CSV currently holds only the `eur_` run, so the bridge recomputes the gl2_ screen with `screen_blocks.py`'s
exact definition and caches it in `bridge/response_screen.csv` (preference order: gl2_ rows already in
`screen_blocks.csv` → the cache → recompute). The recompute is validated against the frozen `gls_` slim block:
its ten members come out ranked 1, 2, 4, 6, 7, 8, 9, 10, 12, 13, the gaps being the near-duplicates
`screen_blocks.py` drops. Top of the list: `gl2_stl40` 0.257, `gl2_t100_spg` 0.226, `gl2_t50_spg` 0.217,
`gl2_ast40` 0.212, `gl2_t50_apg` 0.193, `gl2_close_d_gmsc40` 0.182. With `--no-picks` there is no pick residual and
the screen falls back to the raw label correlation (it says so when it does).

### `--no-fallback` — the A/B floor

Every named feature keeps only the **first** step of its chain. Mean named-feature coverage drops to 0.209 /
0.810 / 0.760 across the three bands. Useful for measuring what the fallbacks are worth on the walk-forward
window without hand-editing the mapping.

### `--extras` — everything else

Writes the legal input columns the mapping does not consume as `xt_<name>` (**584** on its own, **396** with
`--families`, which takes `fy_` / `dx_` / `eur_` / `dp_` out of the catch-all); `tournament/contract.py` has a
`# BRIDGE:`-marked prefix group `extra: "xt_"` so a config can ask for them with `"features": [..., "extra"]`.
What lands there without `--families`: `eur_` (148), `col_` (53 unconsumed), `f50_` (50), `txt_` (42), `tv_` (32),
`dx_` (27), `nb_` (25), `tctx_` (20), `rsci_` (16), `traj_` (14), `dv_` (13), the slim variants
(`fys_` `gls_` `tcs_` `dvs_` `dxs_` `scs_` `wts_`), `wp_` `gt_` `dp_` `ts_` `hs_` `pre_` `slot_` `prog_`, and the
unconsumed `intl_` / `ctx_` / `cgd_` / `vmb_` / `vcmb_` / `bio_` leftovers.

## 6. Pool and identity

The pool is `was_drafted == 1` — his protocol ("the players actually taken on draft night"), and what makes the
Spearman comparable. `--include-undrafted` keeps the declared-only rows (his ideas log: no gain).

| class | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| rows | 47 | 46 | 55 | 52 | 50 | 51 | 51 | 54 | 54 | 56 | 51 | 54 | 44 | 55 | 57 | 57 | 58 | 58 | 56 | 52 | 56 | 55 | 57 |

(2000-2002 also exist — 50 / 49 / 48 — outside his context window. His classes are 58-60 wide; ours are 44-58,
because a drafted player with no row in our staging tables is simply absent. 2015 at 44 and 2022 at 52 are the
thinnest. **This changes the denominator of the Spearman**: our per-class n is not his per-class n, so our number
and his 0.504 are not directly comparable even before the feature differences.)

`pick` is the only thing read from the identity file. `infra/models.py::predict` raises if `pick` is passed as a
feature and the bridge never puts it in a feature group; it exists for `year_metrics` / `evaluate` and for layer 1's
`assert pool["pick"].notna().all()`. If you would rather the box never see picks, build with `--no-picks`: layer 1
still runs and the prediction CSVs are unchanged, only his `spearman_nba` and WAR-captured become meaningless.
