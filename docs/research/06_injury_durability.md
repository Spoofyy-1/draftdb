# 06 — Injury History, Durability & Availability Predictors

Ranked by expected Spearman gain per hour. Target rewards availability directly (WAR/VORP over first seasons), and our current feature set has **no explicit availability channel** — college GP is used as a *volume* input, not as a *durability* input. That gap is the opportunity.

Base-rate warning up front: documented pre-draft ACLR is **21 of 1092 combine participants 2000–2015 (1.9%)** ([Mehran et al., OJSM 2016](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4887878/)). Any single binary injury flag is too sparse to move a rank correlation. Value comes from (a) a *continuous* availability ratio and (b) a *typed, graded* flag pooled across injury classes.

---

## 1. College availability ratio (games played / team games) — HIGHEST VALUE

**Feature.** Per college season *s*: `avail_s = player_G_s / team_G_s`. Ship three columns: `avail_final` (draft-year season), `avail_career` (Σ player G / Σ team G), `avail_min` (worst season). Plus `n_absence_blocks` = count of runs of ≥3 consecutive team games missed by a player who averaged ≥15 MPG in the 5 games before and after the gap (separates injury from coach's-decision DNPs and from redshirts, which show as a whole missing season).

**Evidence.** Pre-draft time-loss is the single best-replicated predictor of pro availability across the analogous NFL literature: [Athletes With Musculoskeletal Injuries Identified at the NFL Combine, systematic review, AJSM/OJSM 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6293380/) — prior cervical spine 42.1 vs 55.6 games (p=.01) and 3.7 vs 4.6 yr career (p=.01); lumbar spine 46.5 vs 50.8 games and 4.0 vs 4.3 yr (both p<.01); navicular 29% vs 70% reached ≥2 yr (p=.02) and 57% vs 31% undrafted (p=.001); Lisfranc 16.9 vs 23.3 games in first two seasons (p<.01). In basketball, [Sonnenfeld et al., PNAS Nexus 2022](https://academic.oup.com/pnasnexus/article/1/4/pgac176/6691726) — of 285 NBA players with severe ankle/knee/hip injury 2008–19, only 69% played a further season, 45% a second, and season minutes fell to ~0.5× pre-injury (p<.001). Descriptive corroboration that top picks increasingly miss college time: [CBS Sports, "James Wiseman is latest in a troubling pattern"](https://www.cbssports.com/college-basketball/news/the-court-report-james-wiseman-is-latest-in-troubling-pattern-of-top-nba-draft-picks-missing-games-in-college) (Noel 24/33, Embiid 28/35, Fultz 25/31, Tatum 29/38, Garland 5/~33).

**Source & dating.** Datable by construction — the season ends before the draft. 2008–2025: [Bart Torvik](https://barttorvik.com/) player game logs via the free `toRvik` API ([torvik.sportsdataverse.org](https://torvik.sportsdataverse.org/), `bart_player_game()`, no subscription); team G from the same schedule endpoint. 2002–2025 alternative: ESPN college box scores via `hoopR`. 2000–2007: [Sports-Reference CBB](https://www.sports-reference.com/cbb/) season rows give player G; team G from the school-season page (one request per school-season). **ToS caution:** SR limits to 20 req/min and prohibits building a database that substitutes for their service ([bot policy](https://www.sports-reference.com/bot-traffic.html), [data use](https://www.sports-reference.com/data_use.html)) — a derived per-player ratio is fine; a mirrored box-score archive is not. Prefer Torvik/NCAA.com where coverage allows.

**Coverage.** ~95% of US-college draftees 2000–2025. Missing: preps-to-pros (2000–05), internationals (see §7), G League Ignite / Overtime Elite (2021+, ~1–3 per class — hand-fill from league schedules).

**Hours.** 6–10. Torvik pull ~1 h; 2000–07 SR backfill ~2,800 requests ≈ 2.5 h wall-clock at 20/min; absence-block logic + validation 3–5 h.

**Leakage.** Low. Only risk is accidentally including postseason games played *after* a player declared, or a summer-league/pre-draft-workout artifact — clamp everything to the college season end date.

---

## 2. Height × availability prior (bigs are less available)

**Feature.** `exp_avail_by_height` — a monotone lookup (or spline) mapping measured height-with-shoes to expected NBA availability, and its interaction with §1. Our model already has height as a *skill/positional* term; it does not have it as an *availability discount*.

**Evidence.** Lottery picks since 2000: ≥6'9" (n=97) missed **18%** of potential games; ≤6'8" (n=95) missed **13.5%**; ≥7'0" ~**24%** (FiveThirtyEight, "Tall Players Like Joel Embiid Are More Prone To Injury" — original URL now redirects to abcnews.com; figures quoted via [PSU SiOWfa15 summary](https://sites.psu.edu/siowfa15/2015/09/16/is-it-easier-to-get-injured-if-you-are-tall-nba-edition/)). Corroborating peer-reviewed: taller players and centers were significantly less likely to return after lumbar disc herniation (p=.002–.005) — [Anakwenze et al., OJSM 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4702156/); center position OR **1.64** (95% CI 1.2–2.24, p=.002) for lower-extremity injury requiring surgery — [Multivariate Analysis of Risk Factors…NBA Athletes, OAJSM 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10859044/). Contrary evidence exists (some reviews find no height effect), so treat the magnitude as ~4.5 pp of games, not more.

**Source/dating/coverage.** Already in our combine anthropometrics. **Hours: 1–2.** **Leakage: none.**

---

## 3. Typed, graded pre-draft injury flags from cached Wikipedia revisions

**Feature.** Rule-based extraction over the pre-draft revision, producing *severity-weighted, body-part-typed* counts rather than one binary flag:

- `sev3` (career-relevant): `torn ACL`, `ACL reconstruction`, `Achilles`, `microfracture`, `navicular`, `Lisfranc`, `stress fracture` + `(back|lumbar|pars|spine|navicular|tibia)`, `herniated disc`, `spondylolysis`, `discectomy`, `hip labral`.
- `sev2`: `surgery`, `underwent .* procedure`, `meniscus`, `fractured`, `out for the season`, `season-ending`, `missed the (rest|remainder) of the season`, `redshirt` + `injur`.
- `sev1`: `sprain`, `concussion`, `strain`, `sore`, `bone bruise`, `missed .* games`.
- `n_surgeries` = count of distinct `surger|surgical|operat` sentences; `recency` = seasons since most recent sev≥2 mention.

**Evidence for *typing* (this is the whole point).** The NFL review above shows the effect is entirely injury-class dependent: **spine, navicular, Lisfranc, full-thickness chondral lesions, rotator cuff** → materially fewer games and shorter careers; **Jones/5th-metatarsal fracture, shoulder labral repair, hip arthroscopy, isolated MCL, athletic pubalgia** → *no* significant difference vs controls. Basketball agrees on the foot split: NBA Jones fractures return at high rates with no performance decrement ([Begly et al., Sports Health 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4922517/)), while navicular RTP is 33.3% and tibial stress 69.2% ([Khan et al., AJSM 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC5857731/)). Generic prior-injury OR from meta-analysis is ~2–3 (prior ankle sprain OR 2.74; prior concussion OR 1.93 for any subsequent injury, 3.06 for another concussion — [BMC SSMR 2025](https://link.springer.com/article/10.1186/s13102-025-01485-9)).

**Source & dating.** Our existing pre-draft Wikipedia revision cache — dating is already solved and audited, which is exactly what killed the quarantined medical columns. Backstop for thin articles: Wayback Machine captures of DraftExpress player profiles (shut down 2017, archived) and RealGM/RotoWire college injury reports, both timestamped.

**Coverage.** Wikipedia pre-draft revisions exist for most first-rounders 2003+; sparse for 2000–02 second-rounders and for internationals. Expect ~70–85% non-null; encode "no article / no mention" as its own level, never as zero.

**Hours.** 8–12 (regex authoring, negation handling — "avoided surgery", "no structural damage", "did not require" — plus a 60-player hand-audit).

**Leakage risk: MEDIUM-HIGH — the main hazard in this slice.** Wikipedia prose in a pre-draft revision can still describe a *future-looking* consensus, and article length itself correlates with fame → with draft position → with outcome. Mitigations: (a) never use article length or edit count as a feature; (b) require the injury sentence to contain a date or season string that precedes draft night; (c) run the negative control — shuffle the flag within draft-slot deciles and confirm the gain vanishes.

---

## 4. High-school multisport participation flag

**Feature.** `multisport_hs` = 1 if the pre-draft article's high-school section names a second varsity sport (`football|track|soccer|baseball|volleyball|American football`) in a high-school context.

**Evidence.** [Effects of Early Sport Specialization…NBA Players, OJSM 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11700406/) — 318 first-round picks 2013–23, multisport (n=87) missed **13.5%** of games vs **16.9%** single-sport (p<.001); award achievement **40.2% vs 19.0%** (p<.001); and workload correlated with injury only for single-sport athletes (ρ=0.37, p<.001 vs ρ=0.14, p=.20). That award gap is unusually large for a free, one-regex feature.

**Coverage.** ~60–75% resolvable. **Hours: 2–3.** **Leakage:** low-medium — same fame-correlates-with-detail hazard; code absence as unknown, not as single-sport.

---

## 5. Pre-draft workload / load markers

**Feature.** `mpg_final`, `total_college_minutes`, `minutes_per_team_game`, and for internationals `pro_seasons_before_20`.

**Evidence.** In-NBA, minutes per game is the dominant modifiable risk factor: OR **1.13** per unit MPG for lower-extremity injury requiring surgery and usage-rate OR 1.02 (p<.001) ([OAJSM 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10859044/)); MPG is also the primary season-ending-injury risk factor ([Arthroscopy 2024](https://www.sciencedirect.com/science/article/abs/pii/S0749806324000628)). Extrapolation from NBA load to *pre-draft* load is an assumption, not a finding — expect a small effect and possibly the wrong sign (high college MPG also proxies quality). **Hours: 1.** **Leakage: none** (already-owned columns).

---

## 6. Combine athletic tests as injury-risk markers (counterintuitive, nearly free)

Higher jumpers were **more** likely to later need lower-limb surgery: standing vertical 76.00 vs 73.86 cm (p=.005), max vertical 89.31 vs 86.89 cm (p=.009); lane agility, sprint, bench, height, weight, body-fat all null ([Patel et al., J Exp Orthop 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12231045/), 130 surgical vs 1,220 controls). Also: prior ACLR does **not** depress combine test scores (Mehran 2016) — so combine athleticism cannot be used as a proxy for medical clearance. Add `vertical × height` as an interaction only; expected gain small. **Hours: 1. Leakage: none.**

---

## 7. International durability — use with caution

RealGM international season pages ([basketball.realgm.com/international](https://basketball.realgm.com/international/)) give GP per league-season 2000–2025, so §1's ratio is mechanically computable. **But the construct breaks:** a 18-year-old on a Euroleague roster plays few games *by coach's choice*, not injury, and multi-competition schedules (domestic + Euroleague + national team) make "team games" ambiguous. Use only `dnp_full_season` (zero games in a league-season the player was rostered for) and cross-check against the Wikipedia flags in §3. I found **no** peer-reviewed comparison of international-vs-college prospect durability. **Hours: 6–8 for modest, noisy coverage — do this last.**

---

## 8. Do teams over- or under-react to medical flags?

No clean NBA study exists. The best available read: in the NFL, the injuries that *cause* draft slides (ACL p=.019, chondral p<.001, Lisfranc p=.04, navicular 57% undrafted) are largely the same ones that *do* predict worse participation — the market is directionally right, so a naive "buy the medical faller" strategy is unsupported. The exploitable residual is the mis-typing: Jones fractures, shoulder labral repairs, hip scopes and isolated MCLs generate real draft-day fear with no measured outcome penalty. Anecdotally the NBA shows the same pattern (Sullinger's back → out of round 1 and a short career, *consistent* with the spine literature; [ESPN 2012](https://africa.espn.com/nba/draft2012/story/_/id/8069495/2012-nba-draft-docs-medically-red-flag-jared-sullinger-sources-say)). **Concrete test we can run in-house:** regress realized 4-yr VORP on (mock-consensus rank residual × sev3 flag). If the interaction is positive, the market over-punishes; that is a direct, cheap, publishable check on our own data. **Hours: 2.**

---

## 9. Public injury databases — mostly unusable here

[ProSportsTransactions](https://www.prosportstransactions.com/basketball/) and the derived [Kaggle NBA Injury Stats 1951–2023](https://www.kaggle.com/datasets/loganlauton/nba-injury-stats-1951-2023) are **NBA-only** — post-draft, therefore pure leakage for a redraft model. Use them only to *build the label* (games-missed component of realized value), never as a feature. The NCAA Injury Surveillance Program publishes aggregate rates only, no player-level records. **There is no public player-level pre-NBA injury database.** §1 + §3 are the substitute.

---

## Cross-cutting cautions

- **Healthy-worker survivor bias:** availability-conditioned NBA studies (rookie-injury → longevity became null after adjustment, [OJSM 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8491104/)) overstate durability effects. Fit on all draftees including zero-game players; do not condition on having played.
- **Combine non-participation is NOT a clean medical signal** pre-2024: Wiggins, Embiid and Parker all skipped 2014 strategically ([ESPN](https://www.espn.com/nba/draft2014/story/_/id/10918949/2014-nba-draft-andrew-wiggins-joel-embiid-jabari-parker-skip-combine)); participation only became mandatory under the 2023 CBA. Skip this feature or restrict it to 2024–25.
- **Order of work:** §1 → §2 → §4 → §3 → §8 (test) → §5/§6 → §7. §1+§2+§4 are ~10–15 hours, near-zero leakage risk, and cover the durability channel the target actually pays for.
