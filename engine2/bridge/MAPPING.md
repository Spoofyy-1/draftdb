# bridge/MAPPING.md — our verified columns → Colin's contract

`bridge/build_table.py` turns the verified, pre-draft-dated staging tables into the two parquets his tournament
consumes. Nothing in his `infra/builders/` runs: his raw sources (basketball-reference labels + biography, ESPN /
hoopR game logs, AyushBatra, JasonG) are off-limits here and are replaced by our own blocks.

```
/Users/kennakao/nba/datarebuild/v4_build/staging_v419/train_2000_2018.csv   1,458 rows, classes 2000-2018, labelled
/Users/kennakao/nba/datarebuild/v4_build/staging_v419/tests/test_YYYY_inputs.csv   2019-2025, sealed labels
/Users/kennakao/nba/datarebuild/v4_build/staging_v419/input_columns.json    the 1,215 legal model inputs
/Users/kennakao/Downloads/nba_redraft_handoff/identity_KEEP_SEPARATE/tabular_names.csv   pid -> actual_pick ONLY
        ->  engine2/data/processed/draft_table.parquet     1,373 rows x 671 columns  (1,328 with --extras)
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

Coalescing order is written `a → b → c` (first non-null wins). `tv_` is the player's own Torvik final-season row
(our block starts with the 2010 class); `col_` is our wider final-college-season line and is the fallback wherever
the two share a scale.

### `all_torvik` — 48 numeric + 2 categorical

| his | ours | note |
|:--|:--|:--|
| `GP` | `tv_gp` → `col_gp` | |
| `Min_per` | `tv_min_per` → `col_minutes_share × 500` | share of all 5×40 team minutes → Torvik's 0-100 min% (medians 80 vs 76.5 on the overlap) |
| `ORtg` `drtg` `adrtg` | `tv_ortg`→`col_ortg`, `tv_drtg`→`col_drtg`, `tv_adrtg`→`col_adj_drtg` | |
| `usg` `eFG` `TS_per` `ORB_per` `DRB_per` `AST_per` `TO_per` | `tv_usg/efg/ts/orb/drb/ast_pct/to_pct` → the matching `col_*_pct` | both already 0-100 |
| `FT_per` `twoP_per` `TP_per` | `col_ft_pct`, `col_fg2_pct`, `col_fg3_pct`, **×100** | ours are 0-1, Torvik's are 0-100 |
| `blk_per` `stl_per` | `col_blk_pct`, `col_stl_pct` | |
| `ftr` | `tv_ftr` → `col_ftar × 100` | |
| `porpag` `adjoe` `pfr` `ast_tov` `dporpag` `stops` `gbpm` `ogbpm` `dgbpm` | `tv_*` | Torvik-only; no `col_` equivalent, so 2003-2009 is empty |
| `bpm` `obpm` `dbpm` | `tv_bpm/obpm/dbpm` → `col_impact/impact_off/impact_def` | our `col_impact` is on the same BPM scale |
| `rec_rank` | `rsci_rank` → `101 − tv_rec_rank` | **his convention is lower = better** (`layer2` ranks on `-rec_rank`); `rsci_rank` already is, `tv_rec_rank` is a 0-100 score and is flipped |
| `rim_pct` `mid_pct` | `tv_rim_pct`→`col_rim_fg_pct`, `tv_mid_pct`→`col_mid_fg_pct` | |
| `dunk_made` | `tv_dunks_pg × GP` | **derived** — his is a season count, ours per game |
| `mpg` | `col_mpg` → `tv_mp` | |
| `treb` `ast` `stl` `blk` `pts` | `col_reb_pg / ast_pg / stl_pg / blk_pg`, `tv_pts`→`col_pts_pg` | |
| `oreb` `dreb` | **derived**: `col_reb_pg` split by `col_orb_pct / (col_orb_pct + col_drb_pct)` | we carry no per-game offensive/defensive rebound split |
| `TPA_pg` `FTA_pg` | **derived**: `FGA/40 = col_fg3a_per40 / col_fg3ar`; `TPA_pg = col_fg3a_per40 × mpg/40`; `FTA_pg = col_ftar × FGA/40 × mpg/40` | |
| `height_in` | `bio_height_in` → `vmb_listed_height_in` → `bio_combine_height_in` → `vcmb_height_with_shoes_in` | dated / official listed height |
| `class_year` | `tv_yr` → `col_class_index` → `vmb_college_class_year` | |
| `n_college_seasons` | `col_n_seasons` → `tv_seasons` → `tc_n_seasons` | |
| `age_at_draft` | `bio_age_at_draft` → `tv_age_exact` → `col_age` → `vmb_age_reported_years` | |
| `conf` `role` (categorical) | `tv_conf_tier`; `tv_role` → `bio_pos_code` | numeric ids. The winner drops them (`-cat`); they exist only so `contract.validate` passes |

### `traj` — 16 columns, 10 filled

`d_bpm ← tv_d_bpm`, `d_usg ← tv_d_usg`, `d_TS_per ← tv_d_ts`, `d_mpg ← traj_d_mpg`, and each `prev_X = X − d_X`.
`career_bpm_mean` / `career_bpm_max` are the mean / max of {final, previous, first} season BPM
(`tc_first_bpm → tv_first_bpm`) — an approximation of his full Torvik career aggregate.
**Not fillable:** `prev_obpm`, `prev_dbpm`, `prev_porpag`, `d_obpm`, `d_dbpm`, `d_porpag` — we carry no
previous-season delta for those three stats. (`tv_d_min_per` is a minutes-*share* delta and is deliberately **not**
used as a fallback for `d_mpg`, which is minutes per game — mixing them produced a −37…+68 range.)

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
| `combine` | `c_` | `vcmb_` (37) | 30 | 30 | 15 map straight across; the ratios, lean/fat mass, touches, approach gain, `c_anthro_n` / `c_drills_n` and `c_shoot_pct` (made ÷ attempted over all six spot / off-dribble / on-move drills) are derived exactly as his builder did |
| `tctx` | `t_` | `tc_` (107) | 72 | 107 | one-for-one rename `tc_X → t_X`. Same idea (all-D1 cohort percentiles, team shares, teammate quality, availability, tournament), different column names — harmless, the group is prefix-matched. **Our `tctx_` block is *not* this**: `tctx_` is team four-factors and goes to `--extras` |
| `intl_z` | `iz_` | `intl_`, `eurs_` (17) | 18 | 39 | `iz_pts_36 ← intl_lg_adj_pts36`, `iz_eff_36 ← intl_impact`, plus `mpg/age/ts/usg/n_league/trb/ast/stl/blk/tov/fg3a/fta` from `intl_`. `iz_pct_eff` = within-class percentile of `intl_impact`; `iz_young_x_{eff,mpg,usg}` = `max(0, 22 − intl_age) ×` the stat, his definition. The prior-chosen Euroleague/EuroCup/ANGT slim spine lands as `iz_eur_*` |
| `mock` | `mock_` | `vcons_`, `cons_`, `dis_` | 16 | 7 | `mock_rank_consensus ← vcons_mock_mean_rank → cons_mock_consensus_rank`, **unranked players filled with 61** (behind the last pick, as `infra/models.py::CONSENSUS_FEATURE` does). Also `mock_rank_consensus_alt`, `mock_rank_best`, `mock_rank_range`, `mock_n_sources`, `mock_rank_std ← dis_mock_rank_std`, `mock_first_round = consensus ≤ 30`. We have no per-source board ranks, so his 13 `mock_rank_<site>` columns have no equivalent |
| `momentum` | `mo_` | `mock_` (13) + `bb_` (23) | 17 | 36 | our `mock_` block **is** the momentum block (`mo_last`, `mo_first_days`, `mo_delta_late`, `mo_snaps`, `mo_std`, …) and `bb_` adds per-board final / 30d / 60d ranks and deltas. ⚠ our `mock_*` is empty for every class ≤ 2019 and only ~43-85% covered for 2020-2025 — `select_features` drops those columns inside every fold, so in practice `momentum` = the `bb_` half |
| `response` | `rs_` | `gl2_` (147) | 11 | 147 | rename `gl2_X → rs_X`. Much wider than his 11 (opponent-tier deltas, tournament residual, consistency, absence blocks, close games). CatBoost only |
| `person` | `bio_` | `wt_` (29) + `misc_` (6) | 6 | 41 | ⚠ **our `bio_` block is physicals and feeds `phys` / `all_torvik`; his `bio_` prefix is the biography group.** `bio_nba_relative_n ← wt_n_relatives_pro_any_sport → wt_nba_relative`, `bio_nba_father ← wt_relative_is_parent`, `bio_shoots_left ← wt_left_handed` (⚠ ~2% coverage), `bio_born_us ← 1 − wt_born_outside_usa → misc_born_usa`, `bio_n_high_schools ← wt_hs_transferred`, `bio_prep_academy ← wt_prep_year → misc_prep`; every other `wt_`/`misc_` column follows as `bio_<suffix>` (injuries, JUCO, reclass, transfers, birth state/metro) |
| `scouting` | `sc_` | `sc_` (24) | 28 | 25 | our prefix is already his. Renames: `sc_nbaready → sc_nba_ready`, `sc_ballhandling → sc_handle`, `sc_postskills → sc_post`, `sc_kw_* → sc_flag_*`. Added `sc_ratio_strength_weakness`. **Not fillable:** `sc_height_in`, `sc_weight`, `sc_wingspan_in`, `sc_snapshot_days_before_draft`. Our `txt_` keyword block is *not* folded in here — `--extras` |
| `comp` | `sct_` | `cp_` (18) | 3 | 20 | rename `cp_X → sct_X`, plus `sct_comp_n ← cp_n_comps` and `sct_words_total = sc_words_strengths + sc_words_weaknesses`. His `sct_comp_war` (the comp's realised WAR/season through draft night) has no equivalent — ours is the comp's *dated career line* (`sct_pts36_to_date`, `sct_peak_mpg_to_date`, `sct_seasons_to_date`, `sct_comp_is_hall_tier`, …), which is the same idea measured differently |

### Resulting feature counts

| member | his | ours (resolved) | after `select_features` |
|:--|--:|--:|--:|
| CatBoost (`all_torvik -cat traj phys intl_pro intl_fiba combine tctx intl_z mock momentum response person scouting comp`) | 324 | **575** | 537 (score 2013) / 544 (2019+) |
| TabICL (same minus `response`) | 313 | **428** | 390 (2013) / 397 (2019+) |

Ours is wider mostly through `response` (+136), `tctx` (+35), `person` (+35), `intl_z` (+21), `momentum` (+19),
`comp` (+17). That is a real difference from his winner and a thing to check first if the holdout number moves:
`bridge/run_winner.py --ctx-start-catboost / --ctx-start-tabicl` and a narrower `FEATURES_*` list are the knobs.

---

## 4. Families we cannot fill

| his group | columns | why |
|:--|--:|:--|
| `a_box` (`a_*`) | 17 | AyushBatra's sports-reference college line. Not a source this project uses; present and empty |
| `j_bio` `j_team` `j_box` `j_shot` `j_aau` `j_event` (`rsci`, `sos`, `j_*`, `aau_*`, `ev_*`, `dunks_per_min`, `pct_rim`, `pct_astd`) | 67 | JasonG7234's sheet (RSCI, SOS, hoop-math shot location, AAU / showcase lines). Present and empty. Our own recruiting block (`rsci_`) reaches the model through `rec_rank` and `--extras`, not through `j_bio` |
| `kaggle` `score` `ianstack` `hoopr` `marchmadness` `shrunk` `eurocamp` `bwb` `academy` `transfers` `odds` `dev` `gleague` `population` `impact` `coach` `game` | — | prefix groups; no columns written, so `contract.groups()` never creates them. None is in the winner's feature list. `STACK_RAW`'s `gl_pred_ws48` and `TILT_COLUMNS`' `co_*` therefore do not exist — only the `tilt` / `stackctx` layer-2 rules would notice |

Coverage of what *is* filled (mean non-null rate over the group's live columns):

| family | cols | filled | 2003-18 | 2019-25 |
|:--|--:|--:|--:|--:|
| all_torvik | 50 | 50 | 0.58 | 0.81 |
| traj | 16 | 10 | 0.34 | 0.51 |
| phys | 4 | 4 | 0.84 | 0.88 |
| intl_pro | 28 | 26 | 0.26 | 0.32 |
| intl_fiba | 27 | 17 | 0.21 | 0.31 |
| combine `c_` | 30 | 30 | 0.59 | 0.63 |
| tctx `t_` | 107 | 107 | 0.47 | 0.79 |
| intl_z `iz_` | 39 | 39 | 0.15 | 0.18 |
| mock `mock_` | 7 | 7 | 0.56 | 0.94 |
| momentum `mo_` | 36 | 36 | 0.26 | 0.50 |
| response `rs_` | 147 | 147 | 0.56 | 0.74 |
| person `bio_` | 41 | 41 | 0.76 | 0.84 |
| scouting `sc_` | 25 | 25 | 0.59 | 0.62 |
| comp `sct_` | 20 | 20 | 0.69 | 0.67 |

**Coverage is materially thinner in the old context classes than in the holdout** (all_torvik 0.58 vs 0.81, tctx
0.47 vs 0.79). Our Torvik block starts with the 2010 class, `col_ortg/drtg` with 2008, `tc_` with 2008, `bb_` with
2009, `gl2_` with 2005. His `ctx_start: 2003` therefore buys eight classes that are mostly empty rows here, where
for him they carried hoopR / AyushBatra lines. **Try `--ctx-start-catboost 2008` on the box** — it is the most
likely single difference between his 0.504 and whatever this scores.

## 5. `--extras`

`bridge/build_table.py --extras` writes the 657 legal input columns the mapping does not consume as `xt_<name>`,
and `tournament/contract.py` gains a `# BRIDGE:`-marked prefix group `extra: "xt_"` so a config can ask for them
with `"features": [..., "extra"]`. Off by default so the ported winner keeps its own feature count.
What lands there: `eur_` (full Euroleague spine, 148 unconsumed), `f50_`, `ctx_`, `cgd_`, `tctx_`, `txt_`, `dx_`,
`dv_` / `dvs_`, `nb_`, `gt_`, `wp_`, `dp_`, `ts_`, `prog_`, `hs_`, `pre_`, `slot_`, `rsci_` (17 of 18), the slim
variants (`tcs_`, `gls_`, `scs_`, `wts_`, `fys_`, `dxs_`, `eurs_` leftovers) and the unmapped `tv_` / `col_` /
`intl_` / `traj_` / `bio_` / `vmb_` / `vcmb_` columns.

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
