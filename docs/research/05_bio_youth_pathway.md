# 05 — Biographical, Developmental & Youth-Pathway Predictors

Ranked by expected Spearman gain per hour, subject to the pre-draft/datable/free/no-LLM constraints.
Baseline: real draft ≈ 0.26, current model 0.45, target 0.55.

---

## 1. Age relative to high-school class cohort (= reclassification, mechanised)  ★ top pick

**Why it is not what we already have.** We have precise draft age. Age is *not* the same as
age-within-cohort: two prospects drafted at 19.4 differ enormously if one reclassified **up**
(young for grade, skipped a year) versus one who reclassified **down** or repeated a prep year.
Given birthdate + RSCI class year (both already in our data), this is pure arithmetic.

**Evidence.** ESPN's reclassification piece quotes an NBA exec on analytics valuing early
entry — "your best players will come out younger" — and states being older for one's high-school
class hurts draft stock, i.e. teams already price *level* but the residual is where alpha sits.
Reclass-up names (Barrett, Edwards, Murray, Flagg) versus reclass-down busts (Bates, G.G. Jackson,
Houstan) suggests the sign is real and unpriced in both directions.
https://www.espn.com/mens-college-basketball/story/_/id/24174879/reclassification-fast-tracks-prospects-college-nba-draft

**Feature.** `cohort_offset_days = birthdate − (Sept 1 of the year the player's RSCI/HS class began 9th grade)`,
plus `reclassified_up = 1[cohort_offset < −120d]`, `reclassified_down = 1[cohort_offset > +250d]`,
plus `years_since_hs_class = draft_year − hs_class_year`.
**Source/dating:** internal (RSCI class year, birthdate). Fully pre-draft.
**Coverage:** ~all US draftees 2000–2025 who appear in RSCI; ~85–90% of US picks. N/A for internationals (use §2).
**Hours:** 2–4. **Leakage:** none. **Overlap:** partial with age; test as a residual on top of age.

---

## 2. Relative age with the *correct* cutoff per pathway (Sept 1 US / Jan 1 FIBA)

**Evidence — mixed, and the mix is the point.** For NBA *draftees* 1990–2019 (n=1,738), USA-born Q1
(Jan–Mar) were over-represented, OR 1.236 (95% CI 1.044–1.465), χ² p=.005; Europeans Q2 32.1%, p=.028.
**But the same study found no career-performance differences across quartiles (p = .127–.924).**
https://pmc.ncbi.nlm.nih.gov/articles/PMC8019932/
A 2026 study of the 537-player 2024 NBA roster tested **both** Jan 1 and Sept 1 cutoffs and found
nothing: χ²(3)=0.95, p=.81, Cramér's V=0.03; Q1-vs-Q4 OR 1.07 (0.76–1.51); no draft-outcome or
anthropometric skew. https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2026.1787778/full
The countervailing case is the **underdog/reversal** literature: NHL elite born in Q4 score ~9 more
points/season and earn ~50% more (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0182827),
and in French youth basketball the "Resilient" cluster — shorter, lower jump scores, **later birthdates** —
produced the highest elite rate (χ²=11.6, Cramér's V=0.21; overall RAE p=.0005 yet elites skew late-born).
https://pmc.ncbi.nlm.nih.gov/articles/PMC12463884/

**Read:** selection-side RAE is well documented; performance-side RAE in the NBA is a **null or weak
reversal**. Expected effect is small — but the feature is nearly free, and the reversal direction is
exactly the kind of mispricing a redraft model monetises. US youth uses an **Aug 31/Sept 1** cutoff
(AAU age divisions and school grade), FIBA uses **Jan 1**
(https://aausports.org/boys-basketball/2026-aau-boys-basketball-eligibility/) — using one global cutoff,
as most papers do, is likely why effects wash out.
**Feature:** `rel_age_frac` = fraction through cohort year using pathway-specific cutoff, plus quartile dummies.
**Hours:** 1–2 (birthdates already held). **Leakage:** none. **Coverage:** 100%.

---

## 3. Multi-sport high-school background  ★ largest published effect in this slice

**Evidence.** 318 first-round picks 2013–2023; 87 (27.4%) multisport. First 3 NBA seasons:
PER **12.8 ± 11.6 vs 10.5 ± 5.1** (p<.05); games played **148.9 vs 125.8** (p<.01); games missed to
injury **13.5% vs 16.9%** (p<.001); end-of-season awards **40.2% vs 19.0%** (p<.001).
https://pmc.ncbi.nlm.nih.gov/articles/PMC11700406/
That award gap is a >2× ratio on a value-proxy close to our label. This is the single best effect size
I found on a variable we do not have.

**The catch — and how to beat it.** The authors classified multisport from "player interviews,
biographies, newspaper articles, MaxPreps, Athletic.net." That is **documentation-biased**: famous
prospects have more written about them, so "evidence of a second sport" partly measures fame, which
correlates with our label. We need a **uniform-coverage** source, not a search. **MaxPreps athlete
pages** list every sport an athlete has a stat line for in one structured block, including for players
nobody wrote a feature about; Wayback has `maxpreps.com/athlete/*` archived densely (confirmed via CDX,
2016–2025). Rule-based extraction: count distinct `gendersport/` sections. No LLM.

**Feature:** `n_hs_sports` (int), `played_football`, `played_track`, `played_baseball` binaries.
**Coverage:** MaxPreps is thin before ~2004 and weak for internationals. Realistically ~2006–2025 US
draftees, maybe 65–80%; pre-2006 will be missing-heavy — encode missingness explicitly.
**Dating:** HS sport participation is fixed pre-draft; still prefer the last Wayback snapshot before draft night.
**ToS:** MaxPreps robots.txt disallows `/school/` and `/team/` but **not** `/athlete/`; `Exabot` is fully
blocked. Pulling from Wayback snapshots sidesteps the live-site question entirely.
**Hours:** 25–40 (name→athlete-ID matching is the cost). **Leakage:** low; the risk is fame-bias, so
validate that `n_hs_sports` is uncorrelated with RSCI rank within pick bands before trusting it.
**Overlap:** none with current features.

---

## 4. Hometown geography, done properly (refines the in-progress birthplace work)

**Evidence.** Côté's birthplace effect: pros over-represented from cities <500k, under-represented ≥500k
(https://pubmed.ncbi.nlm.nih.gov/17115521/). NBA-specific replication: over-representation at
50k–99,999 (OR 1.33) and 2.5M–5M (OR 2.41); density band 2,500–4,999/km² OR 1.71
(https://pmc.ncbi.nlm.nih.gov/articles/PMC9582327/). The 1990–2019 draftee study gives USA ORs by band
(<50k OR 1.471; 50–100k OR 2.109; >5M OR 0.024) — note the non-monotonicity, so **use bands or a spline,
not linear population**.

**Hard caveat:** every one of these studies predicts *reaching* the league, not *performing* in it, and
the one study that tested performance found none (p = .127–.924). Treat this as **low-expectation** and
prefer versions plausibly causal about development quality: anchor on **high-school city, not birth
city**; add county/metro income or ADI (45% of NBA players' AAU teams and 42% of their high schools sit
in the top-3 ADI deciles = higher SES, p<.01; mean HS↔AAU distance 170 ± 309 miles,
https://pmc.ncbi.nlm.nih.gov/articles/PMC11335979/); and **per-capita NBA production of the county**,
computed leave-one-out on train years only.

**Source/dating:** Basketball-Reference `friv/birthplaces.fcgi` and `friv/high_schools.fcgi` (structured,
browsable by state); US Census/ACS for population, density, income. All static and pre-draft-safe.
**Leakage — the real one:** county talent-density must be built from **2000–2018 draftees only** and
exclude the focal player, or it leaks the label straight through geography.
**Hours:** 8–15 on top of work already in progress. **Overlap:** high — this is a refinement, not a new feature.

---

## 5. EYBL / AAU box scores — best signal, worst access

**Evidence.** ESPN states Pelton's projections "translate performance in NCAA Division I, **the Nike EYBL
AAU competition** and top professional leagues to an NBA equivalent," and that EYBL has "a significant
impact for freshmen, who often played nearly as many minutes in the EYBL as in the NCAA."
https://www.espn.com/nba/story/_/id/44888875/nba-draft-2025-projecting-30-best-prospects
A working professional model already pays for this data, which is the strongest signal-existence proof
in this report. RealGM publishes per-player EYBL/UAA/adidas Gauntlet lines including advanced splits.

**Why I still rank it 5th — access is genuinely blocked:**
- `basketball.realgm.com/robots.txt` names **`anthropic-ai`** and **`Claude-Web`** with `Disallow: /`
  (also CCBot, Bytespider); generic `*` gets only `crawl-delay: 2`, but the AI-agent block is explicit.
  The site also sits behind a **Cloudflare JS challenge** (WebFetch 403, curl returns the interstitial).
- `nikeeyb.com/robots.txt` blocks **`anthropic-ai`, `ClaudeBot`, `GPTBot`, `CCBot`**.
- **Wayback coverage is sparse**: CDX returns only scattered EYBL snapshots, essentially 2018–2019.
  EYBL began in **2010** (https://en.wikipedia.org/wiki/Nike_Elite_Youth_Basketball_League), capping
  coverage at draft classes ≥2011 regardless.

**Recommendation:** do not scrape. Either (a) ask the user to obtain permission / a licensed feed, or
(b) treat EYBL as out of scope and capture the same construct cheaply via §7.
**If it were available:** `eybl_ts_pct`, `eybl_usg`, `eybl_ast_pct`, `eybl_stl+blk` at 17U, age-adjusted.
**Hours if permitted:** 30–60. **Coverage ceiling:** ~2011–2025, US only, ~55–70% of US picks.

---

## 6. Pathway string: schools, JUCO, prep/post-grad, transfers

**Feature:** `n_institutions`, `jc_path` (JUCO before D1), `prep_or_postgrad_year`, `transferred_up`
(mid-major→high-major by KenPom conference tier), `transferred_down`.
**Source:** derivable almost entirely from the **college/JUCO season lines we already hold**; RealGM
publishes JUCO/NJCAA/CCCAA sitemaps if a gap-fill is needed. Wikipedia pre-draft revisions
(MediaWiki API with `rvend` = draft date) fill prep/post-grad, extracted by regex on "postgraduate",
"prep", school-name changes — rule-based, no LLM.
**Evidence:** anecdotal only (Butler, Knecht, Mogbo via JUCO). No published effect size — this is a
**plausibly novel** feature, not a proven one.
**Hours:** 6–10. **Leakage:** none if Wikipedia revisions are date-clamped. **Coverage:** ~95% of US picks.

---

## 7. Nike Hoop Summit World Team + youth-FIBA age-relative performance (internationals)

For international prospects there is no RSCI, so we have a weaker prior than for US players. Hoop Summit
World Select selection is a clean, dated, free binary: **150 of 20 editions' participants became
first-rounders** (https://www.fiba.basketball/en/news/nike-hoop-summit-world-team-loaded-with-fiba-bwb-experience).
Pair it with a derived feature we can build for free from data already held:
`age_at_first_pro_game` and `youth_fiba_prod_per36_age_z` (production z-scored **within tournament × age**,
so a 16-year-old at U19 is credited correctly) — the talent-ID literature's consistent finding is that
age-adjusted youth performance, not raw youth performance, carries the signal
(https://pmc.ncbi.nlm.nih.gov/articles/PMC8193982/, tactical/sprint predictors d ≥ 0.61, Nagelkerke R² 8–13%).
**Hours:** 4 (derived) + 6 (Hoop Summit rosters from Wikipedia/USAB). **Overlap:** moderate with existing
FIBA youth lines — the *age-relative normalisation* is the new part.

---

## 8–11. Ranked lower, with reasons

| Feature | Best evidence | Verdict |
|---|---|---|
| **Family athletes beyond NBA / siblings** | WSJ: **48.8%** of NBA players related to an elite athlete (pro/NCAA/national-team) vs 17.5% NFL, 14.5% MLB — https://www.si.com/nba/2016/05/25/players-study-athletic-bloodlines-lineage | Big base rate, but **heavy overlap** with the NBA-relatives work in progress. Cheap extension: regex Wikipedia pre-draft revisions for "son of"/"brother of" + sport token. 6–8 hrs. Parent *height* is not public at usable coverage — skip. |
| **Left-handedness** | Prevalence **5.1%** in pro basketball vs ~11% population; lefties showed better career points/rebounds/blocks and **significantly longer careers** — https://pubmed.ncbi.nlm.nih.gov/22403927/ | Direction is right (under-selected + outperforms = classic mispricing) but **no structured public handedness field exists**; extraction would be per-player and error-prone. Park it. |
| **McDonald's All-American / Jordan Brand Classic** | 1977–2004 MDAA: 51.3% drafted, 36.8% first round; 1993–2012: **42% played <25 NBA games**, only 43.1% >175 games — https://www.espn.com/espn/story/_/id/22941694/mcdonald-all-american-selections-not-guaranteed-nba-success | **Almost fully subsumed by RSCI top-100 rank**, which is finer-grained. Skip for US; only worth it as a fallback where RSCI is missing. |
| **Socioeconomic status of family** | Dubrow & Adams: low-income Black child **37% lower odds** of reaching NBA; low-income white **75% lower** — https://www.espn.com/espn/story/_/id/6777581/importance-athlete-background-making-nba | Predicts *reaching* the league, not performing once drafted. Individual-level SES is not publicly datable, and race-correlated proxies are ethically and legally inadvisable as model inputs. **Use only the geographic/ADI form in §4.** |
| **Position change during development** | Only anecdotal (combine height gains, e.g. Flagg +~1" in one year); "tweener score" work is proprietary | No rule-based, dated, high-coverage source. **Skip.** |

---

## Recommended sequence

1. **§1 + §2 (3–6 hrs total, zero new data)** — cohort-relative age, reclassification flags,
   pathway-specific cutoff. Ship first; if it moves nothing, we have lost half a day.
2. **§6 (6–10 hrs)** — pathway string from data already held + date-clamped Wikipedia revisions.
3. **§7 (10 hrs)** — age-relative youth-FIBA normalisation + Hoop Summit binary; targets the
   international sub-population where our prior is weakest.
4. **§3 (25–40 hrs)** — MaxPreps-via-Wayback multi-sport count. Highest published effect size in
   this slice; gate it on the fame-bias check before it enters the model.
5. **§4** as a refinement of the birthplace work already under way, with the leave-one-out /
   train-years-only construction for county talent density.
6. **§5** only if the user can secure permitted access to EYBL data.

**Two cross-cutting warnings.**
- Nearly every published effect in this literature is measured on *selection* (making the NBA, being
  drafted), and the one paper that tested *performance conditional on being drafted* found nothing
  (p = .127–.924). Our target is the second thing. Expect these features to be small; size the
  validation accordingly and remember the ±0.07 noise floor on 3 test years.
- Any feature built by *searching* for a fact (multi-sport, relatives, handedness) is documentation-
  biased toward famous players. Only uniform-coverage structured fields (a MaxPreps sport list, a
  Wikipedia infobox slot, a birthdate) are safe; free-text presence/absence is not.
