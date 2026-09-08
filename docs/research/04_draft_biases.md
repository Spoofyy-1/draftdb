# Documented NBA Draft Inefficiencies — Evidence and Exploitable Features

Ranked by expected contribution to Spearman vs. the real draft order. Every feature below is knowable before draft night and datable.

**Framing note that reorganizes this whole literature:** tournament performance splits into an *expected* component (team quality, seed, Final Four run) that teams **overweight**, and an *unexpected* component (performance above seed/season baseline) that teams **underweight**. Most popular writing conflates them and gets the sign wrong. Decomposing them is the single highest-value idea here.

---

## 1. March Madness *unexpected* performance — teams UNDER-weight it (highest value)

**Evidence.** Ichniowski & Preston, NBER WP 17928 / *JEBO* 142 (2017) 105-119. Hand-assembled data, 1997–2010, N=4,417 tournament players (415 drafted). Two-level Tobits on inverted draft order.
- PASE (tournament wins above seed expectation) coefficient **5.42–5.67***; survives controlling for **pre-tournament mock draft rank** (Heisler, NBA.com, DraftExpress): PASE **4.06***, PP×Δpoints **1.10***, PP×Δassists **3.17**.
- Headline magnitude: one extra tournament win + scoring 4 more points than season average ⇒ **+4.7 draft slots**.
- Crucially, the NBA-outcome models (Tables 9–12: made an NBA team, games played, minutes/points/assists, career WS/48) find **no evidence of overweighting**: "the unexpected performance in the MM tournament deserves more weight than it gets." Players with positive MM bumps are **more** likely to become superstars.
- **Asymmetry (exploitable):** PP×Δpoints is significant, PN×Δpoints is not — teams reward good tournaments but do not punish bad ones.

**Direction:** BOOST unexpected tournament performance, especially the *negative* tail that the market ignores (a prospect who played badly in March is not discounted enough — that is a short signal).

**Features.**
- `pase` = tournament wins − expected wins given seed (seed→wins table estimated **only on prior years**).
- `d_pts`, `d_ast`, `d_reb`, `d_stl`, `d_tov` = tournament per-game minus regular-season per-game.
- `pp_d_pts`, `pn_d_pts` = the above split by sign of PASE (replicates their spec).
- `tourney_games` (exposure control; median team plays ~2).

**Obtainability:** Sports-Reference CBB game logs + tournament brackets; free, covers 2000–2025 for all NCAA draftees (~65–70% of the pool). **Hours: 8–12.**
**Leakage risk: LOW** — tournament ends early April, draft is late June. *One real trap:* estimate the seed→wins expectation on prior seasons only, or PASE absorbs future information.

---

## 2. Scoring overvaluation; rebounds/turnovers ignored

**Evidence.** Berri, Brook & Fenn, "From College to the Pros," *J. Productivity Analysis* 35 (2011) 25-35 (sample 1995–2009). Per-SD effects on draft position: **points/40 ≈ 4 slots**, assists ≈ 1.5, blocks ≈ 2; **rebounds and turnovers per minute are NOT significant**. Berri's *Stumbling on Wins* generalizes: scoring dominates NBA evaluation (draft, salary, All-Rookie voting) while efficiency, rebounds and turnovers are discounted. Independently, *Applied Economics Letters* 25(5) (2018), drafted 2006–2013: **college turnover rate does not predict draft position or NBA minutes, despite turnover-prone college players being less effective in the NBA**; accounting for teammate effects could raise draft-pick offensive value by **20.3%**.

**Direction:** DISCOUNT raw points; BOOST low turnover rate, rebounding, and efficiency (TS%, FT%).

**Features.** `pts_share_of_production` (points/40 ÷ composite production); `tov_rate`; `usage_adjusted_efficiency` = TS% residualized on usage; `pts_per40_resid_on_mock` (points orthogonalized against dated mock consensus — isolates the part of scoring the market has already paid for). Teammate-effect control: `teammate_quality` = minutes-weighted mean of teammates' RSCI/production.

**Obtainability:** already in our college-stats pull; teammate features need team rosters. **Hours: 3–6.** **Leakage: NONE** (season complete before draft).

---

## 3. Final Four / deep-run halo — teams OVER-weight it

**Evidence.** Berri, Brook & Fenn (as cited in Ichniowski & Preston, p.10): a **Final Four appearance improves a player's draft position by ~12 slots**, and winning the championship by **~8 more**. Their draft models contain only conference dummies at team level, so these are plausibly proxies for team quality rather than player information.

**Direction:** DISCOUNT the raw team-success dummies — this is the *expected/pedigree* component, opposite in sign to §1's unexpected component. It also aligns with our own "busts have pedigree" finding.

**Features.** `final_four_flag`, `champion_flag`, `team_seed`, `team_srs`, plus the residualizer `final_four_flag × mock_rank` so the model can subtract the halo already priced into consensus. Include `pase` (§1) alongside so the two components compete.

**Obtainability:** trivial (Sports-Reference / bracket data). **Hours: 1–2.** **Leakage: NONE.**

---

## 4. Combine athleticism — the clearest documented "pay-for-nothing"

**Evidence.** Berger & Daumann, "Jumping to conclusions," *Sport, Business and Management* 11(5) (2021) 515-, pre/post-draft data **2000–2019**, PCA + regression. Players with better combine results are **drafted earlier**, but "controlling for position, age and pre-draft performance there seems to be no proper justification based on post-draft performance." Corroborated by the SI/combine-history observation that three of the top-6 vertical leapers in combine history went **undrafted**.

**Direction:** DISCOUNT athletic testing (vertical, lane agility, sprint) as a *value* signal while keeping it as a *draft-position* predictor. Keep length/anthropometrics separately — they are cheap and less clearly mispriced.

**Features.** `combine_athleticism_pc1` (PCA over max vert, no-step vert, 3/4 sprint, lane agility, bench); `combine_pc1_resid_on_mock`; keep `height`, `wingspan`, `standing_reach`, `wingspan_minus_height` as a separate block. Add `combine_participation_flag` (non-participation is itself informative and non-random).

**Obtainability:** NBA.com draft-combine pages (free, 2000/2001-present); coverage ~60–75% of drafted players, missing most internationals and top prospects who skip drills — model the missingness, don't impute silently. **Hours: 4–8.** **Leakage: NONE** — combine is mid-May.

---

## 5. RSCI recruiting-rank anchoring

**Evidence.** Berger & Daumann, "Anchoring bias in the evaluation of basketball players," *Managerial and Decision Economics* 42(5) (2021) 1248-1262. Front offices anchor heavily on the public RSCI top-100; actual selections deviate minimally from consensus. High-school rank is an important draft-decision factor but **does not predict NBA success once college performance is controlled** — "a source for systematic draft errors." Berger's dissertation (Univ. Jena, 2022) notes **upperclassmen with low RSCI are systematically undervalued** because managers prefer younger options.

**Direction:** DISCOUNT RSCI as a value signal; RETAIN it as a market-expectation predictor. This is a direct match to our own "steals are older, unranked recruits."

**Features.** `rsci_rank` (unranked = explicit missing category, not 101); `production_above_recruiting_expectation` = college production residualized on RSCI; and the key interaction `senior_flag × rsci_unranked × production_percentile`.

**Obtainability:** Basketball-Reference RSCI pages, 1998–present, free. **Hours: 2–3.** **Leakage: NONE** (high-school data).

---

## 6. Age / upside — an interaction, not a main effect

**Evidence.** Genuinely mixed; do not encode a blanket "boost old." Dartmouth Sports Analytics (2022; classes 2007, 2011, 2016, 2017) finds WS, WS/48 and VORP outcomes **statistically indistinguishable across draft ages 19–22**, with a real drop-off at 23+. FiveThirtyEight and others document the strong league-wide tilt to under-21 lottery picks. Berger's undervalued group is specifically **low-RSCI upperclassmen**, not all older players.

**Direction:** Boost the *joint* profile (older × unranked × productive); keep the age-23+ penalty. Do not linearize age.

**Features.** `age_at_draft` (continuous, days-precise from birthdate), `age_ge_23_flag`, `class_year`, and `age × rsci × production` tensor. Enforce a monotone penalty only above 23.
**Obtainability:** birthdates via Basketball-Reference. **Hours: 1.** **Leakage: NONE.**

---

## 7. International × draft-slot non-monotonicity

**Evidence.** ESPN Stats & Info: since 1998, **21% of American-college lottery picks made an All-Star team vs. 11% of Europeans drafted in the lottery** — internationals are *over*valued at the top. Harvard Sports Analysis (2016) finds Europeans look better in advanced metrics than counting stats and notes thin per-slot samples. Our own analysis finds second-round internationals are disproportionately steals. Both can be true: hype at the top, neglect in round 2.

**Direction:** Non-monotone — discount internationals in the lottery, boost them in the second round.

**Features.** `intl_flag × expected_slot_bucket` (from dated mock consensus, not actual slot), `intl_league_quality` (Euroleague/ACB/Adriatic tier), `intl_minutes_vs_age` percentile within league.
**Obtainability:** RealGM/Eurobasket league pages; harder and slower than NCAA. **Hours: 15–25.** **Leakage: LOW,** but international seasons run to June — cut stats at a fixed pre-draft date. *We already have a known bug here (stale international seasons); fix before trusting this.*

---

## 8. Mock-consensus construction (engineering gain, not a bias)

**Evidence.** Fisher & Montague, "Improving the Aggregation and Evaluation of NBA Mock Drafts," *JQAS* (arXiv:2310.16813); largest known mock database, **100+ authors, 2009–2021 drafts**. Rank-biased distance (RBD) is the right error metric (weights the top, tolerates different-length lists); consensus beats **any single author**; accuracy improves through the season and is **stable year-to-year in the final week**; ranked-choice aggregation ≥ Borda count.

**Direction:** Replace mean-rank Borda consensus with ranked-choice aggregation; weight authors by historical RBD; add `days_before_draft` as a confidence weight.
**Obtainability:** we already scrape these. **Hours: 4–6.** **Leakage: HIGH RISK** — every mock must carry a publication date and be cut at a fixed horizon; late-June mocks leak the actual order.

---

## 9. Pick-value curve — how to weight the objective

**Evidence.** Bewtra & Brill, "Exploring various NBA draft value curves" (Wharton, Nov 2024): value curves built on BPM/RAPTOR/WAR are "wildly different" by metric and aggregator; naive curves go negative or non-monotone, motivating their Gamma-regression salary mapping. Practical upshot: the curve is steep at the top and **flat and noisy through the middle**.

**Implication:** middle-of-class Spearman is mostly irreducible noise. Gains come from top-of-class ordering and from identifying second-round hits — consider reporting a top-weighted rank correlation alongside plain Spearman.

---

## 10–12. Lower-value / weak-evidence items

- **Conference / mid-major bias.** No clean effect-size study found; Berri et al. include conference only as dummies. Cheap to add (`mid_major_flag`, `conf_srs`, `pace/opponent-adjusted production`), low confidence. **Hours: 2.**
- **Team need / positional fit.** Fisher & Montague explicitly assume teams "draft the most talented player regardless of positional fit," and it fits their data well. Low expected value; also hard to date. **Skip.**
- **Promises, private workouts, agents.** Only journalistic sourcing (cancelled workouts as a promise tell, e.g. DaRon Holmes 2024). No rigorous study; unobtainable pre-~2010. **Skip** — fails the "obtainable for most players 2000–2025" constraint.
- **Loss aversion / risk-aversion.** Argued widely (GM tenure ⇒ safe picks) but I found no quantitative NBA test; the Berger anchoring result is the rigorous version of the same intuition. Treat as already captured by §5/§6.

---

## Suggested build order

1. §1 tournament decomposition (PASE + Δstats + PP/PN split) — best evidence, cleanly datable, and *contrarian to the usual "March Madness bias" story*.
2. §3 Final Four halo as its own discounted term (pairs with §1).
3. §2 scoring/turnover residualization — cheapest, data already in hand.
4. §5 + §6 the RSCI × age × production interaction — matches our own steals/busts profile.
5. §4 combine residual; §8 consensus re-aggregation; §7 international interaction last (data debt).

## Sources

- https://www.nber.org/papers/w17928 and https://www.nber.org/system/files/working_papers/w17928/w17928.pdf
- https://www.sciencedirect.com/science/article/abs/pii/S0167268117301932
- https://journalistsresource.org/home/march-madnessirrational-exuberance-nba-draft-decision-making/
- https://link.springer.com/article/10.1007/s11123-010-0187-x and https://ideas.repec.org/a/kap/jproda/v35y2011i1p25-35.html
- https://www.tandfonline.com/doi/abs/10.1080/13504851.2017.1319551
- https://onlinelibrary.wiley.com/doi/full/10.1002/mde.3305 (Berger & Daumann, anchoring)
- https://www.emerald.com/insight/content/doi/10.1108/sbm-11-2020-0117/full/html (Berger & Daumann, combine)
- https://www.db-thueringen.de/servlets/MCRFileNodeServlet/dbt_derivate_00057537/diss_TobiasBerger.pdf
- https://arxiv.org/pdf/2310.16813 (Fisher & Montague) and https://news.byu.edu/intellect/byu-stat-professors-research-could-give-nba-teams-an-edge-in-draft-night-predictions
- https://wsb.wharton.upenn.edu/wp-content/uploads/2024/12/NBA_draft_curves-6.pdf (Bewtra & Brill)
- https://journals.sagepub.com/doi/10.32731/IJSF.141.202019.04 (Greer, Price & Berri)
- https://sites.dartmouth.edu/sportsanalytics/2022/04/12/is-youth-a-determining-factor-of-potential-in-todays-nba-draft/
- https://www.espn.com/blog/statsinfo/post/_/id/106856/nba-draft-the-numbers-on-international-lottery-picks
- https://harvardsportsanalysis.org/2016/06/are-european-draft-picks-worse-value-than-americans/
- https://www.si.com/edge/2016/05/11/nba-draft-combine-results-measurements-vertical-jump-value
