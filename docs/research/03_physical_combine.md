# Slice 03 — Physical / athletic / biomechanical evidence for the redraft model

Research date 2026-09-08. Ranked by expected value. Constraint check applied to every item:
knowable pre-draft, datable, obtainable 2000–2025, free/public, no LLM-scored text.

## Bottom line from the literature

The peer-reviewed record is consistent and unflattering for raw athleticism:

- **Teramoto et al. 2018**, *J Strength Cond Res* 32(2):396–408. N=234 combine participants 2010–2015;
  PCA → three subscales (**length-size, power-quickness, upper-body strength**); robust PC regression on
  Yr-1 (n=148) and Yr-3 (n=127) WS/BPM/**VORP**. Only **length-size** was consistently significant
  (p≤0.05), then upper-body strength. Anthropometrics (height, reach, weight, wingspan, hand length)
  correlated with defensive metrics at **r = 0.313–0.545**. Power-quickness added little.
  https://pubmed.ncbi.nlm.nih.gov/28135222/ · https://oasis.library.unlv.edu/som_fac_articles/211/
- **Moxley & Towne 2015**, *Psychol Sport Exerc* — 300+ combine invitees 2001–2006. Size/athleticism
  predicted **draft order**; only **college quality, college production and youth** predicted NBA success.
  https://www.sciencedirect.com/science/article/abs/pii/S1469029214000983
- **Berger & Daumann 2021**, *Sport Bus Manag* 11(5):515. Pre/post-draft data **2000–2019**, PCA +
  regression: better combine athleticism → **earlier pick**, with *no* justification in post-draft
  performance after controlling position, age and pre-draft production. Explicit "athleticism-induced
  decision-quality-lowering bias." https://www.emerald.com/sbm/article/11/5/515/347620/
- **Key Anthropometric and Physical Determinants… (PMC6820507)** — **3,610 combine participants
  2000–2018** (1,160 later drafted). Drafted-vs-undrafted Cohen's d **0.26–0.87**; discriminant
  r_c² **0.13–0.19**; reclassification 68.3% PG / 66.4% SG / **59.9% SF (n.s., p=0.398)** / 68.4% PF /
  71.4% C. **Bench press never significant; hand measures not predictive.** Guards discriminated by
  length + vertical; PF/C additionally by lane agility and ¾ sprint → a real **position interaction**.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6820507/
- **Citius, Altius, Fortius (PMC7601684)** — 723 drafted players 1999–2018. Combine largely does *not*
  predict draft slot except **hand width + height in frontcourt** and **both verticals in backcourt**;
  wingspan↔blocks (frontcourt) r=0.368, height↔assists (backcourt) r=−0.416.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC7601684/
- **Power and agility testing within the NBA pre-draft combine** (480 players 2010–2017): lane agility ↔
  shuttle r=0.45, lane agility ↔ sprint r=0.45, vertical ↔ sprint r=−0.51 → the running/jumping battery is
  **largely redundant**; expect little from adding more of it.
  http://www.scielo.br/j/rbcdh/a/Dtb3nwNWDVhVRtJ3Zv4RKtc/?lang=en
- Contrary (weak) evidence: an NCAA-D1 study reports WS/40 ↔ max vertical ρ=0.589 (35% of variance),
  61.4% with body composition. Tiny N, college-level, near-certainly overfit — treat as a hypothesis only.
  https://www.researchgate.net/publication/375027616_

**Implication:** we already have the measures that matter (length). Extra athleticism variables are
low-yield. The exploitable edge is that *the market* overweights athleticism (Berger & Daumann).

---

## 1. Athleticism-overvaluation residual (de-bias the consensus). EV: highest. ~3–4 h.

Documented, replicated market error, and it uses **only data we already hold** — dense for every class
2000–2025, which matters because our eval window is 2019–2025.

- **Feature.** Fit, on training classes only, `mock_consensus_rank ~ f(power-quickness PCs, length PCs,
  position, age)`. Emit `athleticism_premium = predicted_rank − actual_consensus_rank` (how much of a
  player's stock is explained by testing) plus a `length_residual` twin. Fade high athleticism_premium,
  back players ranked on production despite mediocre testing. Add `position × power-quickness` explicitly
  (PMC6820507 shows PF/C agility matters, SF nothing does).
- **Dating.** Combine is ~5 weeks pre-draft (mid-May); consensus mock must be the post-combine snapshot.
- **Leakage risk.** Low, but the *choice* of shrinkage must be fit in-fold; do not fit the residual model
  on test years.

## 2. Youth→draft growth trajectory from the archived DraftExpress measurement DB. EV: high. ~15–30 h.

The only item in this slice that is genuinely **new information** rather than a transform.

- **Evidence/coverage (verified).** draftexpress.com is dead (`/nba-pre-draft-measurements/` → 404) but the
  DB is in the Wayback Machine: **2,535 archived measurement URLs**, global listing paginated to **page
  155** (~15,500 measurement rows). The 2017 snapshot exposes filters for **30 sources** — Hoop Summit,
  Nike Skills Academy, Nike Elite 100, Nike Basketball Academy, NBA Top 100 Camp, USA Basketball, Eurocamp,
  Reebok Breakout, UA All-American Camp, Elite 24, LeBron James / Kevin Durant / Paul Pierce / Deron
  Williams / Amaré / Vince Carter camps, PG & Big Man Skills Academies, Biosteel All-Canadian, Portsmouth,
  D-League Elite Camp, NBA Pre-Draft Camp, NBA Draft Combine, Official College Team — and **years
  1987–2017**. Columns: height no-shoes / with-shoes, wingspan, standing reach, no-step & max vertical
  (+ reaches), weight, body fat, hand length/width, bench, agility, sprint.
  Snapshot used: `https://web.archive.org/web/2017/http://www.draftexpress.com/nba-pre-draft-measurements/`
- **Dating method.** Iterate `/{year}/{source}/{pos}/{draft}/{page}/{sort}/{dir}`; the (source, year) pair
  dates each row. Age at measurement = event date − DOB.
- **Gap.** DX stops in **2017**, i.e. exactly our eval window is uncovered. Supplements needed for
  2018–2025: annual Nike Hoop Summit measurement releases (NBADraft.net, CBS), USA Basketball U16/U17/U19
  training-camp measurement releases, FIBA/RealGM youth-tournament rosters with dated height/weight
  (https://basketball.realgm.com/national/tournament/11/U17-World-Cup/386/rosters), and Wayback snapshots
  of recruiting-site profile heights. Budget most of the 30 h here.
- **Features.** Δheight, Δwingspan, Δweight from earliest youth measurement to combine; annualised growth
  rate; `combine_height − predicted_from_age16_height` (late-grower residual); "still growing" flag
  (Δheight ≥ 0.5″ in final 24 months); weight-gain rate as a frame-filling proxy; wingspan/height at 16 vs
  at 20 (wingspan matures earlier, so a 16-y-o with adult wingspan and sub-adult height is a growth bet).
- **Leakage.** None (all strictly pre-draft), but **missingness is informative** — DX measured hyped
  prospects. Ship a `has_youth_measurement` indicator and impute within (class, consensus-rank bucket).

## 3. Combine shooting drills. EV: medium-high but coverage-crippled. ~6–10 h.

Standardised, ~100+ shots per player — real skill signal that college 3P% measures noisily for low-volume
shooters. **Endpoint details (all `LeagueID=00`, `SeasonYear=2019` style):**

- `https://stats.nba.com/stats/draftcombinespotshooting` — 54 cols:
  `{FIFTEEN|COLLEGE|NBA}_{CORNER_LEFT|BREAK_LEFT|TOP_KEY|BREAK_RIGHT|CORNER_RIGHT}_{MADE|ATTEMPT|PCT}`.
- `https://stats.nba.com/stats/draftcombinenonstationaryshooting` — 30 cols: off-dribble 15-ft and
  college-range (break L/R, top key) and on-the-move 15-ft / college.
- `https://stats.nba.com/stats/draftcombinedrillresults` — STANDING_VERTICAL_LEAP, MAX_VERTICAL_LEAP,
  LANE_AGILITY_TIME, **MODIFIED_LANE_AGILITY_TIME**, THREE_QUARTER_SPRINT, BENCH_PRESS.
  Docs: https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/draftcombinespotshooting.md
- **Years.** Shooting drills exist only **2014-15 → 2019-20** on those endpoints
  (https://www.nba.com/stats/draft/combine-spot-up) and **2021 → 2026** on a *different* page,
  https://www.nba.com/stats/draft/combine-shooting-drills, for which **nba_api has no endpoint** — capture
  its XHR on the box. 2020 was the virtual/regional combine (HomeCourt-recorded shooting), so treat
  separately. Net: ~5 training classes (2014–18) and ~5 eval classes.
- **Feature.** NBA-range make% pooled over 25 spot shots + off-dribble make%, empirical-Bayes shrunk to
  position mean; plus `drill3P% − expected(college 3P%, FT%, volume)` residual; hard missing-indicator.
- **Leakage.** None; but the drills feed post-combine mocks, so orthogonalise against the mock feature.

## 4. Era-correct handling of drills we already use (data hygiene). EV: medium. ~2–3 h.

Published result availability is **not** uniform (https://www.topendsports.com/sport/basketball/testing-draft-results.htm):
max vertical, ¾ sprint, lane agility **2000–2025**; **no-step vertical only 2006+**; **reactive shuttle
(= MODIFIED_LANE_AGILITY_TIME) only 2013+**, with a *changed protocol in 2020*; **bench press only
through 2015**. If the model currently feeds raw bench press or standing vertical it is learning an era
indicator. Fix: within-class z-scores, era masks, and drop bench press from any post-2015 inference.
Reactive-shuttle protocol: start mid-key, react to a signal, touch sideline, cross 16 ft, return through
start; best of three (https://www.topendsports.com/testing/tests/shuttle-reactive.htm).

## 5. Participation / "declined testing", and repeat-combine deltas. EV: medium-low. ~2–4 h.

- **Declined testing.** Voluntary through 2023 (Wiggins/Parker/Embiid skipped 2014); **mandatory from
  2024** — invited players must complete medicals, biomechanical/functional testing, strength & agility,
  shooting drills, anthropometrics or be draft-ineligible, with narrow exceptions (active FIBA club,
  injury, family). 5-on-5 remains optional.
  https://www.espn.com/nba/story/_/id/36224467/nba-nbpa-memo-outlines-new-nba-draft-selection-requirements
  Feature: `anthro_present & drills_absent` (derivable from endpoint nulls, 2000–2023) — but it mostly
  restates "expected lottery pick", so use it only as a **residual vs mock rank**; near-constant 2024+.
- **Repeat-combine deltas.** Withdraw-and-return players are measured twice (Zach Edey 2023→2024: +0.5″
  height, +0.25″ wingspan; Udoka Azubuike likewise). Free — group existing rows by player × SeasonYear.
  Gives Δheight/Δweight/Δvertical/Δsprint plus a "returned to school" flag. Small N, clean, zero leakage.
  https://www.nba.com/news/nba-draft-combine-largest-wingspans

## 6. Listed-vs-measured height gap. EV: medium-low. ~5–8 h.

The NBA only forced true no-shoes heights in **2019**; Zion was listed 6-7 at Duke, measured 6-6.
Feature `listed_height − measured_height_with_shoes` is a clean **program-hype** proxy, and its
international analogue catches stale listings (our known Wembanyama-class data bug).
Source: archived college/FIBA roster listings (dated), joined to combine anthro.
https://www.cbssports.com/nba/news/six-nba-players-whose-listed-heights-could-drastically-drop-when-league-reveals-true-measurements/

## 7. G League Elite Camp anthro + testing. EV: low-medium. ~8–12 h (OCR).

Official anthropometry, strength & agility and shooting-drill sheets published as **images** on
gleague.nba.com for **2021–2025** (~45 prospects/yr), plus D-League Elite Mini Camp measurements in the DX
archive back to ~2017. Value is (a) filling anthro for fringe prospects who never reach the main combine
and (b) the dated status signal "promoted from Elite Camp to the Combine". Eval-window only, so weak for
training. https://gleague.nba.com/news/2025-g-league-elite-camp-official-testing-results

## 8. 5-on-5 scrimmage box scores. EV: low. ~10–20 h.

No official structured feed found; the NBA broadcasts drills on the app and media transcribe box scores
(On3, ESPN, SI). Two games, ~20 minutes per player, no opponent control — the sample is far too small to
beat college production, and collection is manual per year. Recommend **skip**.

## 9. Injury-risk anthropometrics. EV: near-zero. Skip.

NBA studies find **age, height, weight and experience uncorrelated with injury incidence or surgical
intervention**; games missed is driven by minutes and usage. Size only moderated *post-injury*
performance decline. No public draft-class force-plate data.
https://www.sciencedirect.com/science/article/pii/S2666061X2200102X ·
https://www.tandfonline.com/doi/full/10.2147/OAJSM.S442750

## 10. Relative age effect / birth month. EV: near-zero — use as a negative control.

Selection-only effect, no performance effect: 1,738 drafted 1990–2019, Q1 28.2% vs 24.1% expected
(OR 1.236), **no career-performance differences** (https://pmc.ncbi.nlm.nih.gov/articles/PMC8019932/);
2024-season replication finds no skew at all (χ²=0.95, p=.81, Cramér's V 0.03–0.13; height R²=0.0075)
(https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2026.1787778/full).
We already use exact age. A birth-month feature that "works" is a leakage alarm — perfect negative control.

## 11. P3 / force plates / DEXA. Not obtainable. Skip.

P3 assesses ~68% of the NBA and now sits at the combine; the 2024+ combine adds biomechanical assessment,
isometric strength and **DEXA** — but all of it is team-confidential. P3's public output is marketing
write-ups ("six of the next ten knee injuries"), no player-level data.
https://www.p3.md/research · https://www.nba.com/news/nba-draft-combine-overview-2026

---

### Recommended order of work
(1) athleticism-overvaluation residual → (4) era masks → (5) repeat-combine deltas → (3) shooting drills
→ (2) youth growth trajectory (the big build) → (6) listed-vs-measured. Skip 8–11.
