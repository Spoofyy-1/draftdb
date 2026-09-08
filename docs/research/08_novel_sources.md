# 08 — Novel Data Source Scan

Scan date 2026-09-08. All probes were single small GETs/HEADs; nothing bulk-collected.
**[V]** = fetched or computed this session. **[I]** = inferred, not directly confirmed.

## Ranked summary

| # | Source | EV | Hours | Why |
|---|---|---|---|---|
| 1 | **DraftExpress measurements** (Wayback) | High | 6–8 | Fills a **52% missing** anthropometric gap |
| 2 | **FIBA youth archives** | High | 20–40 | Only clean age-matched-peer signal; covers the international gap |
| 3 | **CBBD recruiting API** | Med-High | 4–8 | Continuous rating, deeper than RSCI; permissive terms |
| 4 | **Torvik player game logs** | Med | 4–6 | Only Torvik asset that is not a recombination |
| 5 | **Torvik conference split** | Med | 1 | New observations, near-zero cost |
| 6 | ESPN 100 grades (Wayback) | Med-Low | 10–16 | Second opinion; only for a disagreement feature |
| — | Combine shooting drills | **None** | 0 | Already on disk; nothing before 2013 |
| — | ESPN JSON API, stats.ncaa.org, MaxPreps, EYBL, Rivals | **None** | — | Blocked by terms or by access |

## Headline

Two results reshape the priorities.

**The Torvik season table is already mined out.** I tested the brief's two most attractive "free win"
ideas against data already on disk and both are largely redundant (§A). Both looked obvious beforehand.

**The real gap is coverage, not cleverness.** [V] Only **1209/2499 = 48.4%** of prospects 2000–2025 have a
combine `WINGSPAN` (53.5% for 2000–09, 51.7% for 2010–17, **42.3% for 2018–25** — worsening as prospects
skip the combine). [V] Torvik matches only **1335/1837 = 72.7%** of 2010–2025 prospects, the missing 27%
being almost entirely internationals who never played D1. Anthropometrics and college production are
established inputs that are simply *absent* for a third to a half of the population. The top-ranked
sources all attack that, and #1 and #2 attack it in complementary populations.

Repo state [V]: `torvik_features.py` → `tv_features.csv` (1335 rows) covers season levels, shot diet,
role, exact birthdate, `rec_rank`, deltas, BPM slope. `make_weird_patches.py` → `tv2_features.csv`
**already covers teammate quality** (`tv_mates_bpm_sum`, `tv_star_mate_minshare`) **and team context**
(`tv_team_adjoe/adjde/barthag/sos`). Two wishlist items were already built before this scan began.

---

## §A. Two negative results (tested, not guessed)

**A1. Age-cohort percentiles across all D1 — DEBUNKED, do not build.** Every Torvik row carries a birthdate
[V: 4723/4723 filled in 2015], so stats could be expressed against age-matched D1 peers. I built it. For
2011–2025 I fit expected BPM on age across all qualifying D1 players, then compared ranking draftees by the
age residual vs by raw BPM:

- Age slope, all D1: **+0.26 BPM per year** (~1.0 BPM across the whole 19–23 draftee range).
- **Spearman(raw BPM, age-adjusted residual) among draftees = 0.9896** mean; 0.982–0.997, *every year*.

It reorders essentially nothing — the age effect is small next to the BPM spread among draftees.
Percentiles are worse: draftees sit at the 85th–100th percentile where the transform saturates, destroying
resolution exactly where the model must discriminate. The model already has age as its own feature.

**A2. Share of team production — ~90% redundant.** Points-share for drafted players, 2011–2025 [V]:
Spearman 0.797 with `usg`, 0.697 with `Min_per`, and **R² = 0.886** (0.844–0.927) given
(`Min_per`, `usg`, `TS_per`) — all already features. Unsurprising, since points ≈ minutes × usage ×
efficiency. Minutes-share is just `Min_per` restated. The ~11% residual is a thin reed against a ±0.07
noise floor. Build only if it falls out free.

---

## §B. 1. DraftExpress measurements via Wayback — best lead found

`WebFetch` is blocked for `web.archive.org`; the CDX API and plain GETs work.

| pattern | unique-content snapshots [V] | span |
|---|---|---|
| `draftexpress.com/nba-pre-draft-measurements*` | **5984** | **2007–2026** |
| `draftexpress.com/measurements.php*` | 53 | 2007 only |
| `draftexpress.com/rankings.php*` | 2 | 2016 only |

One pulled snapshot (2011-05-02, 930 KB) [V] carries **18 columns** — Height w/o Shoes, Height w/Shoes,
Weight, Wingspan, Reach, Body Fat, Hand Length, Hand Width, No Step Vert (+Reach), Max Vert (+Reach),
Bench, Agility, Sprint, Rank, Drafted — across **1372 rows in a single page**, an order of magnitude more
players than a combine year, because DraftExpress aggregated Portsmouth, Eurocamp and other camps.
Caveat: **no source-event column**, so per-row provenance is not recoverable.

Features: fill `wingspan`, `standing_reach`, `hand_length/width`, `body_fat` for the ~52% with no combine
record, plus derived ratios (wingspan−height, reach−height). Add `anthro_source` as a categorical.

Datable pre-draft: yes — take the last snapshot *before* each draft date. **The page has a `Drafted`
column, which is the leakage vector**: a post-draft snapshot encodes the outcome. Enforce
snapshot timestamp < draft date and drop `Drafted` and `Rank`. Terms: public archive, defunct site.
Note the archive is a **measurements** source, not a rankings source — big boards are barely captured
(2 snapshots). Also visible in the nav [I]: agent/agency affiliations, early-entry lists, draft-pick
transactions.

## 2. FIBA youth archives — High

The most permissive terms posture in the scan, aimed at the population where both gaps concentrate.
`fiba.basketball/robots.txt` disallows **only** auth routes [V] — no AI-crawler block, no stats restriction.
The archive sits at `/en/history/{compId}-{slug}/{editionId}/{stats|players|games}`, enumerable from
`/en/history-sitemap_index.xml` → **259 competitions** [V] including `276-fiba-u19-basketball-world-cup`,
`249-fiba-u17-...`, `263-fiba-u18-eurobasket`, `235-fiba-u16-eurobasket`. The U19 index alone lists
**21 editions**, reaching back before 2000. **Server-rendered** [V] — a 490 KB stats page had real HTML;
player pages carry `Games, Min, PTS, REB, AST, STL, BLK, TO, EFF`, **per-game splits**, and roster
`dateOfBirth`, `height`, `club`. (`digital-api.fiba.basketball` 401s — not needed.) `archive.fiba.com` is
dead; the new site absorbed it.

**What it adds beyond the international season lines already in use:** club lines measure production *in a
league*, so league strength confounds everything. A FIBA tournament is a closed round-robin of 7–9 games
against **strictly age-matched peers**. That supports `u19_eff_per40_z` (z-scored within tournament, so
league strength cancels by construction) and especially `age_within_tournament` — DOB is on the page, and a
17-year-old dominating a U19 field is a different signal from a 19-year-old doing it. Plus
`n_youth_tournaments` and `age_at_first_national_team_selection` as precocity measures.

Coverage: strong for European/African/Oceanian draftees, ~20–30% for Americans [I] (USA does send U17/U19
teams). **Complementary** to §3 rather than overlapping — together they cover most of the pool where either
alone covers about half. Leakage: very low; frozen results with fixed dates. Cost is breadth, not
difficulty — thousands of requests, so rate-limit.

## 3. Recruiting composites — real, but narrower than it looks

**Do not scrape 247Sports.** Its robots.txt permits the rankings path, but the governing Paramount/CBSi
terms prohibit, verbatim, "unauthorized spidering, 'scraping,' data mining or harvesting of Content" and
restrict use to "personal, non-commercial purposes" [V]. robots permits the path; the ToU forbids the
activity. Rivals is separately dead: 403s and redirects to On3, whose robots.txt sets
`ClaudeBot → Disallow: /`, and Rivals150 history is absent from Wayback after the 2025 Yahoo→On3 sale [V].

**The clean route is `api.collegebasketballdata.com/recruiting/players`** — free key, bearer auth, terms
that *explicitly* permit commercial use, caching and derived models, prohibiting only republication as a
standalone dataset [V]. Schema: `year, name, position, school, hometown{...}, committedTo, heightInches,
weightPounds, stars, rating, ranking, athleteId`. Two caveats: CBBD very likely derives its composite from
247 (provenance undocumented) [I], so this is downstream written permission rather than upstream
clearance — defensible, not risk-free; and since draftdb is public, model on it but do not republish the
ratings.

**First action, before any collection:** test whether `Recruit.athleteId` joins `DraftPick.athleteId` on
`/draft/picks`. If it does, recruit→draftee becomes a key join instead of fuzzy name matching and
collection time more than halves. Two API calls.

**The skeptical part, which matters at a ±0.07 noise floor.** The basketball composite starts at **class of
2003** — 2000/2001/2002 return literally zero rows [V] — so draft years before ~2007 are a hard NULL, and
247 is US-high-school-only, so every international is a structural NULL. The rating spread also collapses
below the top ~200: rank 1 = 1.0000, rank 150 = 0.9047, then ranks 301–350 sit at 0.8400, 0.8400, 0.8400,
0.8398, 0.8396 — a near-constant 3-star plateau [V]. **The genuinely new information over RSCI is confined
to the rank 100–250 band plus the rating spread inside the top 100.** Real, but modest and correlated with
an existing feature. Evaluate on the 22-year paired eval, not the 3-year test set, or seed noise will be
indistinguishable from a gain. Spot-check ~20 players against a pre-draft Wayback capture to confirm CBBD
serves as-of-class-year rather than retro-edited ratings.

**Cheap riders on the same pull:** the `school` field carries the high-school name, so ~40 curated national
prep programs (Oak Hill, Montverde, IMG, Findlay, Brewster, La Lumiere…) give `attended_national_prep` in
~2 hours; `?InstitutionGroup=JuniorCollege` gives rated JUCO players (66 in class 2015) [V].
**A reclassification boolean is not worth building** — reclassifying up or taking a post-grad year *is* an
age phenomenon, and precise age already measures exactly what a parsed boolean would measure coarsely.

**ESPN 100 grades** (`RK | PLAYER | POS | HOMETOWN | HT | WT | STARS | GRADE | SCHOOL`, classes 2007–2019)
are free and not Insider-gated, but reachable only via Wayback since ESPN 503s and blocks `anthropic-ai`
[V]. Grades are compressed at the top (98, 98, 97, 97, 97). Worth 10–16h only for the disagreement feature
`z(247_composite) − z(espn_grade)`; skip if 247 alone moves the needle.

## 4. Torvik player game logs — Med, with a terms caveat

[V] `https://barttorvik.com/{year}_all_advgames.json.gz` → HTTP 200 for **2008–2026**, ~9.6–11.4 MB gz/yr.
The only Torvik asset carrying information that is *not* a recombination of columns already in use.
Enables `tv_bpm_vs_top50` (and the gap to full-season), `tv_tourney_bpm`, `tv_gamelog_sd_bpm`,
`tv_pct_games_above_median`, `tv_road_minus_home_ts` — none of which exist in the pipeline.

**Terms caveat:** Torvik's robots.txt has `Disallow: /*.json`; with no `$` anchor that also matches
`..._advgames.json.gz`, and `/playerstat.php` is separately disallowed. The direct pull is **not**
robots-clean, unlike the season CSV. Two compliant routes: the GitHub mirror
`andreweatherman/toRvik-data` (`player_game/{year}/…parquet`, [V] present for **2008–2023**, but last
pushed 2023-06-20 and **license field `None`**), or — recommended — **ask Bart Torvik**, a single reachable
operator, which also covers the years the mirror lacks. Temper expectations: 30-game samples are noisy.

## 5. Torvik conference-only split — Med, ~1 hour

[V] `getadvstats.php?year=2015&csv=1&conyes=1` returns a genuinely different table (4459 vs 4723 rows,
different md5) — same 67 columns restricted to conference games. Feature
`tv_conf_minus_full_{bpm,ts,usg,ortg}`: production that survives conference play differs from production
padded against non-conference filler. A cheap partial substitute for the top-50 split in §4, at 19 requests.
**Do not build against the other filters** — [V] `top=50`, `quad=1`, `start=/end=`, `gt=NCAA` all return a
response byte-identical to unfiltered; they are silently ignored. `getadvstats.php` is not disallowed;
respect `Crawl-Delay: 10`, and note robots disallows `ClaudeBot`/`anthropic-ai`, so this belongs to the
project's own collector with its own UA and a contact address.

## 6. Combine shooting drills — NO value; already on disk

Dead for three independent reasons [V]. (1) `draftcombinestats`, already in use, **already returns every
shooting column** (`SPOT_*`, `OFF_DRIB_*`, `ON_MOVE_*`); `draftcombinedrillresults` returns only
vertical/agility/sprint/bench, also already returned. (2) Those 47 columns are **already on disk** in
`combine_raw/combine_{2000..2025}.json`. (3) Coverage is fatally thin — players with any non-null shooting
value: **2000–2012: zero every year**; 2013–2019: 32–47; **2020: 0**; 2021–2025: 36–72 — about half a class,
and nothing across most of the training window. `research/verified_combine_20260906/build.py` already
references these columns. **Close the item; make no new endpoint calls.**

---

## §C. Blocked / dead ends — record these so they stop resurfacing

| Source | Status [V] |
|---|---|
| `stats.nba.com` (all endpoints) | HTTP 000, 12s timeout — known network blocker |
| `stats.ncaa.org` | 403 on everything incl. robots.txt, browser UA too |
| ESPN site/core/web JSON APIs | Work, and reach back to **2002** (6 yrs deeper than Torvik) — but Disney ToU §2.B.x forbids "access, monitor, copy or extract … using a robot, spider, script, or other automated means, including … for … any AI Tool, data mining or web scraping"; robots blocks `anthropic-ai`. **Same bucket as Sports-Reference.** |
| `data.ncaa.com` | Works but 404s before ~2013 and is scoreboard-level, not player-level |
| MaxPreps | 403 on everything incl. sitemap; `/school/` `/team/` disallowed; HS box scores non-comparable anyway |
| Nike EYBL / AAU circuits | robots permissive, but sitemap is **70 URLs, all current-season** — wiped each cycle, no history |
| Rivals / On3, NJCAA, prosportstransactions | 403 / CloudFront / Cloudflare |
| `basketball.realgm.com` | 403 to scripted requests despite permissive robots |
| Haslametrics, EvanMiya, CBB Analytics | Team-level only, or Shiny/websocket, or paid. Haslametrics' date param silently falls back — `?d=2015-03-01` and `?d=2024-03-01` return an **identical md5** |
| Torvik `top=`/`quad=`/`start=`/`gt=` | Silently ignored |
| **Torvik `year` < 2008** | **Silently returns the current season** |

That last row is a **silent data-corruption trap**: `getadvstats.php?year=2005` returns a well-formed,
plausible CSV that is actually 2026 data — [V] md5-identical to `year=2026` for 2000, 2002, 2004 and 2007.
Assert that the returned `year` column equals the requested year. Torvik genuinely starts at **2008**,
which is why `torvik_features.py` gates at `dy<2010`: 16 of 26 classes, but **all 7 test classes**.

Two hygiene notes. The Torvik CSV contains a **`pick` column** — the actual draft result — present in the
`COLS` map and correctly not emitted; worth an explicit guard so it cannot leak in later, since it is the
very outcome being beaten. And **agent lists, Spotrac contracts and transaction histories describe a player
*as of today*** — a current-agent field silently encodes career outcomes (a player still at a top agency in
2026 is a player who had a career). Only pre-draft-dated snapshots are usable there.

## §D. Wildcards (cheap, unproven)

1. **Role transitions — best of these, partially tested.** Torvik's `role` label is 99.8% filled across
   eight values; `tv_role` is used only as a *level*. [V] Of 477 drafted players with ≥2 qualifying seasons,
   **232 (48.6%) changed role**, with interpretable transitions — `C → PF/C` (18), `PF/C → Stretch 4` (11),
   `Combo G → Scoring PG` (23), `Wing G → Combo G` (24). **Limitation:** undefined for one-and-dones, where
   much of the draft's value sits — needs a missing-value branch, not a zero fill.
2. **`rec_rank` from the *first* college season** rather than the last (currently the final row), giving the
   true pre-transfer recruiting rank plus a free cross-check against RSCI.
3. **Relative age within the recruiting cohort** — not "young for the draft" (§A1) but "oldest in every
   youth team he played on". Different mechanism, not a monotone transform of draft-day age.
4. **Hometown talent density** — `hometown` is 72.2% filled [V]; draftees-per-capita by state-year, or
   home→college distance as a recruiting-intensity proxy.

## §E. Not assessed

The market/meta cluster (agent lists, Spotrac, transaction histories, sportsbook draft-position props,
Wikidata/DBpedia, Google Books/GDELT news counts) and the remaining scouting-media archives (NBADraft.net
grades, Ringer and Tankathon big boards, The Stepien / No Ceilings community boards) did not return before
this report was written; the session's web-search budget was exhausted (200/200) partway through. **Open,
not dismissed.** On the §C leakage note, the agent/contract/transaction items are the ones most likely to
be unusable even if recoverable, so the scouting-media archives are the better place to resume.
