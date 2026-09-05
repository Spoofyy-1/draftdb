# Redraft files for a tabular model

**Input = one prospect's draft-night record. Output = a score per prospect; sort the class by
that score and you have the model's draft order.**

| file | rows | use |
|---|---|---|
| `train_2000_2018.csv` | 1458 | training: inputs + targets, classes 2000-2018 (outcomes fully known). Universe = every first-round pick, every second-round pick, and every undrafted player who reached the NBA; each row carries his pre-NBA statistics (inputs) and his NBA seasons (targets, `y_s1_*` .. `y_s5_*`) |
| `test_<year>_inputs.csv` (2019-2026) | one full draft-night pool per class | inputs only: every drafted player, every early entrant who stayed in, and undrafted players who later made the NBA, with their statistics up to draft night |
| `answers_<year>.csv` | same rows | outcomes for scoring; `outcome_status` = final (2019-2021) or partial (2022-2026, fewer than five seasons elapsed) |
| `prospects.parquet` / `.csv` | everything | the whole table with `split` (train / test / prospective) |
| `FEATURES.md` | | every column block explained |
| `../keys/tabular_names.csv` | | pid -> player name, for reading boards (keep out of training) |

## Columns
- **Inputs** (213 columns): `bio_*`, `col_*`, `intl_*`, `hs_*`, `pre_*`, `cons_*`, `txt_*`. All exist
  before the pick is made. `cons_*` is the scouting consensus (final mock rank); drop it for a
  stats-only board, keep it for a blended board. Missing values are real (a one-and-done has no
  trend columns, an international prospect has no `col_*`); tabular models handle NaN natively.
- **Targets** (pick one):
  - `y_early_war` - wins above replacement over the first five NBA seasons (0 if never played).
    Regression target; sort descending for the draft order. Recommended.
  - `y_redraft_rank` - the true redraft order within the class (1 = best by y_early_war).
    Ranking target with `draft_year` as the group id.
  - `y_early_tier_code` - 0 never played, 1 bust, 2 rotation, 3 starter, 4 star. Classification target.
  - also available: `y_career_tier`, `y_draft_outcome`, `y_second_contract_aav`, `y_earnings_first5`,
    and the NBA seasons themselves: `y_s1_*` .. `y_s5_*` (minutes, WAR, impact, pts/36, season tier
    code per season), `y_minutes_5yr`, `y_games_5yr`, `y_allstar_5yr`, `y_nba_seasons_total`.
    These describe what happened AFTER the draft: use them as targets, never as inputs.
- **Meta / benchmark** (never features): `pid`, `draft_year`, `was_drafted`, `declared_only`, `path`,
  and in the answer files `actual_pick` (the league's order, the benchmark to beat).

## Known limits (measured, not guessed)
- **Score the DRAFTED subset when comparing to the league.** The test pools also contain players who
  declared and went undrafted; their pre-draft record is systematically thinner (no mock-board rank,
  no combine, no pre-draft process rows), so absence of data correlates with never playing
  (about 52% of null-valued rows never played, against a 33% base rate). Among drafted players in
  2019-2021 nobody failed to play at all, so that correlation cannot distort the head-to-head.
- **Column coverage varies by era**: Torvik advanced stats and BPM start in 2008 (2002-2007 rows carry
  totals, rates and per-40 only); game-log splits from 2008; on/off and RAPM from 2010; assisted-shot
  splits 2012-2013 so far; combine shooting drills exist for only ~5% of prospects (the NBA published
  almost nothing before 2014). Missing means missing; tree models handle it.
- **`bio_weight_lb` is imputed** from height and position where neither a combine measurement nor a
  listing exists, so its absence cannot signal the outcome.
- **2000 and 2001** have no college source at all: those two classes carry bio, international and
  pre-draft blocks only.

## Scoring a board
For each test class: Spearman correlation between the model's order and `y_early_war`, compared
with the same correlation for `actual_pick`; stars (`y_early_tier` = star_track) in the model's
top 10 vs the league's; busts in the top 10; bust precision at k. `python -m nbadata.harness score
--year <year> --ranking <file>` does this for a pid list, and `python -m nbadata.redraft` is a
reference XGBoost implementation of the whole pipeline.
