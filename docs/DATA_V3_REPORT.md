# Data v3 rebuild — 2026-09-05

## What was wrong (audit of train 2010–2018 + test 2019–2026)

1. **International block used whichever season the sources had, not the latest pre-draft season.** Sources covered EuroLeague, EuroCup, G League, ACB, BCL, LBA, BBL, BSL, ABA, NBL, LKL, FIBA Europe Cup and FIBA youth — but not France's LNB, China's CBA, Brazil's NBB, Greek A2, Overtime Elite, or most second divisions. Any prospect whose final season was in one of those got an older season (Wembanyama: 2021–22 ASVEL bench, 13 games, 42.6% TS) or a youth tournament (Giannis: FIBA U20 only; Fournier: U18 only) or nothing at all (Mudiay, Zhou Qi, Yang Hansen, Alex Sarr, Louzada, Caboclo, Tavares).
2. **Wingspan / reach missing for the top of every test class 2019–2023.** 20–40% of drafted players in those classes had no combine measurement (top prospects skip it), versus 4–15% in the training classes where the missing ones are fringe players. The model therefore learned "no measurement = fringe" and applied it to Zion, Morant, Edwards, LaMelo, Cade, Banchero, Holmgren, Wembanyama, Scoot. From 2024 the NBA made measurements near-mandatory, so those classes were fine.
3. **A few drafted players had no college block at all** (GG Jackson, Bub Carrington, Bones Hyland, Cam Thomas, Dewan Hernandez) plus a path mislabel (Cam Thomas tagged international).
4. Not an issue after checking: `col_final_is_current=0` rows are genuine sit-out years; missing ages are undrafted fringe rows only.

## What was done (all pid-keyed; names never left the Mac)

- **International block rebuilt** from English + French Wikipedia career tables (CC BY-SA, API, cached) and the official EuroLeague API (advanced stats where the latest season was EuroLeague/EuroCup). Rule: latest pre-draft pro season, every competition in that season combined by minutes; per-36 counting stats; FG/3P/FT%; league level recoded as max level in that season. Replacement only when the rebuilt season is NEWER than the dataset's (or the dataset had a youth-only block) and the sample is at least 60% of the old one. Result: 15 rows replaced, 11 inserted, ~720 international rows kept as they were.
- **New columns** (inputs): `intl_lg_strength` (league strength 0–1 from a league-name map, finer than the 1–4 level), `intl_lg_adj_pts36`, `intl_two_pts36` / `intl_two_minutes` (minutes-weighted last two seasons), `intl_prev_pts36`, `intl_season_gap` (draft year minus season), `intl_src` (0 kept / 1 Wikipedia / 2 Wikipedia+EuroLeague API), `col_basic_fill` (1 = basic college line from Wikipedia), `bio_measure_src` (0 none / 1 combine / 2 pro day / 3 team / 4 reported / 5 measured later).
- **Measurements**: six research agents searched published pre-draft wingspan, standing reach, no-shoes height and weight for the 142 drafted players missing them (NBA.com profiles, G League Elite Camp tables, Hoop Summit, USA Basketball camps, team releases, ESPN/SI/Athletic reports). Nothing estimated; every value has a source and a context flag. Found: 75 wingspans, 34 reaches. Older classes (2010–2014) had the thinnest yield.
- **College basic lines** filled for 16 players with no block (Wikipedia college tables; advanced stats stay NaN; flagged).
- Backup of the previous data is `data_v1/` on the box; vault re-sealed; tripwire holds.

## Remaining holes (for the data side)

- French LNB seasons still depend on Wikipedia coverage (French pages helped for 11 players). An LNB source would close the gap for Coulibaly, Rupert, Salaün, Traoré, Penda, Diawara and the 2026 French prospects.
- Overtime Elite (Thompson twins), NBB (Gui Santos), prep-only players (Maker, Simons, KJ Martin) still have no stat line.
- `col_team_strength` is all zeros in the source build.
- Standing reach remains missing for most non-combine players; wingspan sources disagree by 0.5–1.5 inches for many players (recorded values and alternates are in the agents' notes in the scratchpad results).

## Result: frozen model (same genome), old data vs rebuilt data

| Metric | v1 data | v3 data |
|---|---|---|
| Walk-forward mean IC, 2014–18 folds | +0.381 | +0.369 |
| Blind edge over the NBA draft, 2019–25 | +0.197 | +0.179 |
| Seasons at +0.2 or better | 4 of 7 | 4 of 7 |
| Value boards, model mean (league 71.3%) | 77.2% | 76.7% |
| 2023 board value / Wembanyama rank | 69.7% / 12th | 71.6% / **4th** |
| 2024 blind edge | +0.384 | +0.467 |

Walk-forward-only ablations showed the small loss is spread across the international corrections, the college fills and the new columns (each recovers about a third when removed), concentrated in the 2015 fold, and is at the level of fold noise. The rows were wrong before, so v3 stays as the canonical dataset; the frozen genome was tuned on v1 and the search has been restarted on v3 to re-tune. Blind differences of 0.02 are inside the noise floor (per-season SE about 0.12; the ledger now holds 682 blind scorings, so blind numbers cannot rank models anyway).
