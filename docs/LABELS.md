# Label definitions

All labels are rule-based on an impact metric (points per 100 possessions vs average) and WAR
(wins above replacement). 1977-2022 use FiveThirtyEight RAPTOR directly; other seasons carry a
RAPTOR-equivalent fitted from darko.app season RAPM (1957+) and DARKO (1998+), see
`impact_source` / `impact_estimated_share`. Nothing is hand-assigned. One floor applies on top:
a complete career whose awards say superstar (>= 3 All-NBA 1st teams or an MVP) or star (>= 3
All-NBA or >= 5 All-Star) is at least that tier (`tier_basis` = awards_floor), because era RAPM
under-rates some pre-1977 greats.

## career_tier (one per player; evaluated top-down)
- **superstar**: (peak-3-season RAPTOR >= 7.5 and career WAR >= 90.0) OR career WAR >= 165.0 (the WAR-only path keeps long great careers RAPTOR under-rates, e.g. older bigs, in the tier)
- **star**: (peak-3-season RAPTOR >= 5.0 and career WAR >= 40.0) OR career WAR >= 80.0 (the WAR-only path keeps long great careers RAPTOR under-rates, e.g. older bigs, in the tier)
- **semi_star**: (peak-3-season RAPTOR >= 3.0 and career WAR >= 18.0) OR career WAR >= 45.0 (the WAR-only path keeps long great careers RAPTOR under-rates, e.g. older bigs, in the tier)
- **starter**: >= 4 seasons of >= 1000 minutes and career WAR >= 8.0
- **role_player**: career WAR >= 2.0, or >= 3000 career minutes with >= 2 qualifying seasons
- **fringe**: >= 500 career minutes, below role_player
- **cup_of_coffee**: played, < 500 minutes
- **never_played**: no NBA minutes (drafted players only)
- **unrated_modern**: no impact data yet (impact coverage ends 2026)

`career_complete` = not active in 2026 (last season < 2026); active
players carry provisional tiers. `tier_confidence`: final / provisional / estimated_impact (complete
career, but most seasons use fitted impact) / none.

## early_career_tier (first five NBA seasons — the draft-projection target)
- **star_track**: an all_star-or-better season (RAPTOR >= 2.5, >= 1200 min) within the first 5 seasons, or >= 15 WAR over them
- **starter_track**: a starter-or-better season (RAPTOR >= 0, >= 1000 min) or >= 6 WAR
- **rotation_track**: at least one >= 1000-minute season or >= 1.5 WAR
- **bust_track**: played, but none of the above
- **never_played**; `partial_*` when fewer than 5 seasons have elapsed (`early_window_complete` = False)
- **not_yet_played**: drafted 2024 or later with no NBA minutes yet (no outcome exists; use the harness `freeze` mode)

## season_tier (one per player-season)
- **mvp_level**: RAPTOR >= 7.0 with >= 1500 minutes
- **all_nba**: RAPTOR >= 4.5 with >= 1500 minutes
- **all_star**: RAPTOR >= 2.5 with >= 1200 minutes
- **quality_starter**: RAPTOR >= 1.0 with >= 1000 minutes
- **starter**: RAPTOR >= 0.0 with >= 1000 minutes
- **rotation**: RAPTOR >= -1.5 with >= 500 minutes
- **bench**: RAPTOR >= -99.0 with >= 1 minutes

## draft_outcome (drafted players with complete careers, or active players with >= 5 seasons)
- **bust**: top-14 pick with career WAR < 5 or a fringe/never-played career; or a first-rounder with < 2 WAR in <= 3 seasons
- **disappointment**: career WAR at least 10 below the median for that pick bucket
- **steal**: picked after #10 and career WAR at least 15 above the bucket median, or a second-rounder who became semi_star or better
- **exceeded_expectations**: a top-10 pick with career WAR at least 15 above the bucket median
- **met_expectations**: everything else
- **undrafted / unresolved** (career incomplete)

## archetype (one per player-season)
Per season, first rule that matches wins (needs mp >= 500):
 1 heliocentric_creator : usg >= 30 and ast_pct >= 30
 2 primary_creator      : usg >= 26 and ast_pct >= 22
 3 volume_scorer        : usg >= 26
 4 playmaker            : ast_pct >= 25
 5 rim_protector        : blk_pct >= 4.0 and pos in (PF, C)
 6 stretch_big          : pos in (PF, C) and fg3ar >= 0.30
 7 glass_cleaner        : trb_pct >= 17 and usg < 20
 8 three_and_d          : pos in (SG, SF, PG) and fg3ar >= 0.40 and usg < 21 and (stl_pct >= 1.6 or raptor_d >= 0.5)
 9 floor_spacer         : fg3ar >= 0.45 and usg < 22
10 defensive_specialist : raptor_d >= 1.5 and usg < 19
11 two_way_wing         : pos in (SF, SG) and raptor_o > 0 and raptor_d > 0
12 secondary_scorer     : usg >= 21
13 connector            : ast_pct >= 15 and tov_pct <= 12 and usg < 21
14 rotation_generic     : everything else with mp >= 500
   low_minutes          : mp < 500

