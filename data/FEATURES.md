# Prospect table for the redraft model

`prospects.parquet` / `prospects.csv`: one row per prospect, draft classes 2000-2026.
`split`: **train** (2000-2018, outcomes known) | **test** (2019-2021, five seasons elapsed, never
trained on) | **prospective** (2022-2026, partial or no outcomes; redraft now, score later).

| block | columns | note |
|---|---|---|
| id | pid, draft_year, was_drafted, actual_pick, actual_round, path | `actual_pick` is a benchmark, never a feature |
| bio_ | height, weight, position (+code), age at draft, combine size and athletic testing, USA flag | size matters; athletic testing is a known null (keep for trees, drop for linear models) |
| col_ | final college season: per-36, rates, ratings, BPM (Torvik 2008+), shot locations, team context (games missed, minutes share, usage/BPM rank on team, team strength), percentiles vs same-age / same-class / same-role peers, shooting projection, game-log splits (gl_*: late-vs-early trend, vs top-50, NCAA tournament, consistency), on/off + RAPM (lu_*, 2010+), shot zones (sh_*, 2019+), Hoop-Math assisted splits (hm_*, 2012+); plus n_seasons, minutes total, first-season BPM, BPM delta and mean/max | 2002-2007 rows have totals/rates but no BPM/ratings |
| intl_ | last pre-NBA international season (level: 4 EuroLeague, 3 EuroCup/pro league, 2 G League, 1 FIBA youth), counts, youth scoring | international prospects have these instead of col_ |
| hs_ | McDonald's / Jordan Brand / Hoop Summit flags, recruiting rank first/final/delta, RSCI rank | pre-college signals |
| pre_ | transferred, n_schools, redshirt, medical redshirt, withdrew from a previous draft, green-room invite | process signals |
| cons_ | final mock-draft consensus rank, best rank, sources, big-board rank | OPTIONAL: the market prior. Use for the "blended" board, exclude for the "stats-only" board |
| txt_ | has_report, words; scout_strengths / scout_weaknesses text | text is for a language model or feature extraction, not the tree model |
| y_ | early_war (first five seasons WAR; 0 if never played), early_tier (+code 0-4), window_complete, career_tier/war/complete, draft_outcome, second_contract_aav, earnings_first5, career_earnings, all_star_selections, partial_seasons | targets only |

Recommended target for redrafting: `y_early_war` (continuous, time-consistent across eras).
Recommended benchmark: Spearman of `actual_pick` (negated) vs `y_early_war` per class.
