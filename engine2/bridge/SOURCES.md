# SOURCES.md — what `bridge/rebuild_colin_data.py` downloads, and under what terms

Every source below is public and was fetched read-only, once, over plain HTTPS with a descriptive User-Agent
(`DraftDB-research/1.0 (contact: mike@alphax.inc)`). No login, paywall, JS challenge or rate-limited HTML page is
touched. Nothing under `/Users/kennakao/nba/datarebuild/` is modified.

**Deliberately NOT used** (Colin's pipeline reads them; this rebuild does not):
basketball-reference.com and sports-reference.com in any form (his draft pages, his per-season advanced tables, his
`bio_` biography family, and the international league pages `infra/builders/intl.py` scrapes), the live ESPN site,
stats.nba.com / stats.gleague.nba.com, tankathon.com, realgm.com, nikeeyb.com, and anything behind a login.
Our labels replace his, and our Wikipedia-rule blocks replace his `bio_`.

---

## 1. Bart Torvik — `getadvstats.php` CSV export (local cache, no new requests)

| | |
|:--|:--|
| URL | `https://barttorvik.com/getadvstats.php?year=YYYY&csv=1`, seasons 2008–2026 |
| Taken from | `/Users/kennakao/nba/datarebuild/novel/torvik_context/raw/adv_YYYY.csv.gz` (already cached 2026-09-08) |
| What was taken | all 90,739 D1 player-seasons, 67 unlabelled columns |
| Terms | `barttorvik.com/robots.txt` (read 2026-09-08) does **not** disallow `getadvstats.php`. The `User-agent: *` block sets `Crawl-Delay: 10`, honoured by the original fetch. Site has no separate terms-of-use page; no API key, no login. |
| Used for | `data/processed/torvik.parquet` (parsed with **his** `infra/download_data.py::TORVIK_COLS`, unmodified), which feeds `torvik_context` (`t_*`), `shrink` (`sh_*`), `transfers` (`tr_*`), `game_features` (`g_*`) and his `torvik_trajectory`. |

The rebuild only gunzips the cached files into `data/raw/torvik/advstats_YYYY.csv`; it issues **no** request to
barttorvik.com. Column 45 (`pick`) is post-draft and is used by his `match_torvik` solely to identify a row, exactly
as in his own code; no model feature is derived from it.

## 2. sportsdataverse / hoopR-mbb-data — ESPN men's college basketball, 2003–2026

| | |
|:--|:--|
| URLs | `https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/mbb/{player_box,team_box,schedules,player_core}/parquet/…parquet` |
| What was taken | `player_box_YYYY` and `mbb_schedule_YYYY` for 2003–2026; `team_box_YYYY` and `player_core_YYYY` for 2007–2026. 88 files, **118 MB**. Deleted-after-use is not applied here because `game_features.py` re-reads them across seasons; see RUNBOOK for the cleanup command. |
| Terms | The repo's `LICENSE` is an **unfilled MIT template** (`YEAR: 2026 / COPYRIGHT HOLDER: hoopR.mbb authors`); GitHub's licence detector therefore reports `NOASSERTION`. The `hoopR` package the repo serves is MIT-licensed. The data is ESPN-derived and redistributed by sportsdataverse; this is the **only permitted route** to ESPN game logs for this project — the live ESPN site is not used. Attribution: hoopR / sportsdataverse. |
| Used for | `h_*` (`college_sources.load_hoopr`), `g_*` (`game_features`), `rs_*` (`response`), and `data/processed/player_game.parquet`. |

`infra/builders/game_features.py::download()` already maps `schedules` → the repo's `mbb_schedule_*` filename, so his
code needed no change; the rebuild's own downloader writes the same filenames his loaders expect.

## 3. AyushBatra01 / NBADraft — `data/draft_players.csv` + `data/draft_players24.csv`

| | |
|:--|:--|
| URLs | `…/main/data/draft_players.csv` (251 KB) and `…/main/data/draft_players24.csv` (16 KB) |
| What was taken | 1,264 draftees, classes **2004–2024** (1,195 from the first file, 69 from the 2024 file, identical 44-column schema, concatenated so his one-file loader is unchanged): name, year, pick, team, position, height, **wingspan, weight, draft age, birthdate**, and a sports-reference final-college-season line (G, MP, PTS, TS%, eFG%, 3PAr, FTAr, USG%, AST/USG, AST/TO, PER, OWS/40, DWS/40, WS/40, OBPM, DBPM, BPM). |
| Terms | **No LICENSE file** — the repository is public but carries no grant, so default copyright applies. Used read-only, non-commercially, for research; the extracted values are factual sports statistics. If redistribution is ever contemplated, ask the author first. |
| Used for | the `a_*` family and `phys` (wingspan / weight / draft age), through **his** `infra/external.py::ayush`, unmodified. Also `game_features._external_dobs()` reads its `Birthdate` column. |

Note: this file's `Team` column is *not* always a college (it holds `SACA` for Dwight Howard, `Illawarra (NBL)` for
LaMelo Ball), so the rebuild only accepts it as a `college` value when the string resolves to a Bart Torvik D1 team.

## 4. JasonG7234 / NBA-Draft-Model — `data/draft_db.csv`

| | |
|:--|:--|
| URL | `https://raw.githubusercontent.com/JasonG7234/NBA-Draft-Model/master/data/draft_db.csv` (1.2 MB) |
| What was taken | 1,292 prospect-seasons, `2008-09` … `2024-25` (draft classes 2009–2025): **RSCI** recruiting rank, team **SOS** / wins / losses, height / weight / draft-day age, a per-40 and per-100 college line, hoop-math **shot-location** columns (% shots and FG% at rim / mid / three, % assisted, dunk counts and rates), an **AAU** line (GP, MIN, PTS, FG%, 3P%, FT%, TRB, AST, STL, BLK, TOV) and a **showcase-event** line (GP, MIN, PTS, FG%, 3P%, TRB, AST, STL, BLK, placement). |
| Terms | **MIT License** (`LICENSE`, SPDX `MIT`). Attribution: JasonG7234/NBA-Draft-Model. |
| Used for | `rsci`, `sos`, the `j_*`, `aau_*` and `ev_*` families, through **his** `infra/external.py::jasong`, unmodified. Also `game_features._external_dobs()` reads its `Birthday` column. |

## 5. MichLitt / nba-draft-oracle-pro — `data/raw/combine_2000_2026_raw.csv`

| | |
|:--|:--|
| URL | `https://raw.githubusercontent.com/MichLitt/nba-draft-oracle-pro/main/data/raw/combine_2000_2026_raw.csv` (274 KB) |
| What was taken | NBA draft-combine rows, seasons `2000-01` … `2026-27`: height without shoes, height with shoes, weight, wingspan, standing reach, body fat, hand length / width, standing and max vertical, lane agility, modified lane agility, three-quarter sprint, bench press. |
| Terms | **No LICENSE file**; public repository, default copyright. Used read-only for research. The underlying values are the NBA's own published `draftcombinestats` endpoint; Colin's `college_sources.py` already documents this repo as the mirror he used because stats.nba.com refuses cloud hosts. |
| Used for | the `c_*` combine family via **his** `college_sources.load_combine` (which writes the endpoint's own `resultSets` JSON layout into `data/external/nba_combine/`). |

⚠ The mirror carries anthropometrics and drills but **no shooting-drill columns** (`SPOT_*`, `OFF_DRIB_*`,
`ON_MOVE_*`), so `c_shoot_pct` cannot be rebuilt and is empty. This is the one `# BRIDGE:`-marked change in his code
(`college_sources.py`, three lines): without it the loader raises `AttributeError` on the scalar accumulator.

## 6. ianstack / NBA-Draft-Combine-Analysis — `data/College_WS.csv`

| | |
|:--|:--|
| URL | `https://raw.githubusercontent.com/ianstack/NBA-Draft-Combine-Analysis/main/data/College_WS.csv` (300 KB) |
| What was taken | `is_college_ws` — final-college-season Win Shares (sports-reference-derived) for combine attendees 2000–2022. Only that one column; the repeated combine measurements are ignored because §5 already supplies them. |
| Terms | **No LICENSE file**; public repository, default copyright. Read-only research use. |
| Used for | the `is_*` family via **his** `college_sources.load_ianstack`, unmodified. |

## 7. fivethirtyeight / data — `nba-raptor/historical_RAPTOR_by_player.csv`

| | |
|:--|:--|
| URL | `https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-raptor/historical_RAPTOR_by_player.csv` (3.1 MB) |
| What was taken | downloaded for completeness of his `data/raw/` layout. **Not used**: our own staging per-season outcomes supply every label, so `infra/war.py::season_war` is never called. |
| Terms | **CC BY 4.0** (`LICENSE`, SPDX `CC-BY-4.0`). Attribution: FiveThirtyEight. |

---

## Sources his code expects that this rebuild does **not** fetch

| his source | why not |
|:--|:--|
| basketball-reference draft pages / advanced season tables | not permitted here. Substituted by our identity file (`drafts.parquet`) and our staging labels (`season_war.parquet`, `target.parquet`). |
| basketball-reference biography (`bio.py`, `bio_*`) | not permitted. Our `wt_` / `misc_` Wikipedia-rule blocks already fill his `person` family through `bridge/build_table.py`. |
| `infra/builders/intl.py` (bbref international league pages, stats.gleague.nba.com, adidas Eurocamp, FIBA GDAP, EuroLeague NGT API) | the two largest stages are bbref and gleague, both blocked; and our own `intl_` / `fy_` / `eur_` blocks (from `datarebuild/novel/euroleague/` and `datarebuild/novel/fiba_youth/`) already fill `intl_pro` / `intl_fiba` / `intl_z`. The rebuild writes an **empty, correctly-typed** `data/external/intl_prospects.parquet` so his `infra/external.py::intl` and `infra/dataset.py::build_table` run unmodified. |
| `mocks.py` / `momentum.py` / `scouting.py` / `scouting_text.py` (Wayback captures of mock-draft boards and NBADraft.net) | our own `datarebuild/novel/{boards,nbadraftnet}/` blocks already fill `mock`, `momentum`, `scouting` and `comp` through the bridge; re-scraping Wayback would take 20–40 min per builder for columns we already have. |
| `odds.py`, `coaches.py`, `development.py`, `gleague.py`, `population.py`, `hoopexplorer.py` | live/paywalled or stats.nba.com-backed sources; none is in the winner's feature list. |
| Kaggle "College Basketball Players 2009-2021" (`load_kaggle_college`) and Emlembow march-madness (`load_marchmadness`) | both are `huggingface_hub.snapshot_download` calls. The Kaggle file is a documented Torvik snapshot that "returns only the keys of the player-seasons it holds" — it adds no column. March Madness would add 11 `mm_*` columns; it is not in the winner's feature list and needs an extra dependency. Skipped; see "could not be rebuilt". |
| SCORE network `nba_draft.csv` (`load_score`) | his own docstring: "there is no pre-draft column to expose: returns keys only". Skipped. |

---

## Could not be rebuilt

The full list — families skipped because their source is off-limits, loaders skipped because they add nothing,
columns the permitted source simply does not carry, and the first/last usable draft class of every rebuilt family
— is **`bridge/RUNBOOK.md` section 8**. Exact commands, run times and disk are **section 7** of the same file.

## Attribution, in one block

> College player-seasons: Bart Torvik (`barttorvik.com`). Game logs and schedules: hoopR / sportsdataverse
> (`hoopR-mbb-data`), ESPN-derived. Draft measurements and a second college line: AyushBatra01/NBADraft.
> Recruiting, strength of schedule, shot locations, AAU and event lines: JasonG7234/NBA-Draft-Model (MIT).
> NBA draft-combine measurements: MichLitt/nba-draft-oracle-pro, mirroring the NBA's `draftcombinestats` endpoint.
> Final-college-season win shares: ianstack/NBA-Draft-Combine-Analysis. RAPTOR: FiveThirtyEight (CC BY 4.0).
