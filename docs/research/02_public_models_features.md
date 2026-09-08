# Public NBA Draft Models: Inputs, Reported Importances, and What We're Missing

Scope: public/semi-public draft projection models, what each uses, what each reports about feature value,
and a ranked list of inputs used by multiple strong models that we do not currently use.
All sources free and public; no paywall bypass, no Sports-Reference scraping, no LLM text scoring.

---

## Part A — Model-by-model survey

**Kevin Pelton, WARP projections (ESPN).** Translates college/international lines to NBA-equivalent
performance across **14 core stats** (2P%, 3P%, FT%, ORB%, DRB%, AST/100 team plays, STL/100, PF/100,
BLK per 100 2PA, % of plays as 2PA / 3PA / FTA, TOV, usage), SOS-adjusted, then predicts average WARP over
five seasons. Multi-year weighting is explicit: most recent season ×2, prior ×3, two seasons back ×5
(inverted for internationals, who face older competition); projections regress to **positional** rookie means.
Age effect quantified: "each additional year of age tends to reduce a player's projected NBA value by about
0.5 WARP per season." Consensus version blends the stats model with ESPN top-100 rank (Vashro's "Humble"
idea). 2025 version adds G League and select European leagues. No published accuracy.
https://www.espn.com/espn/print?id=16235135 · https://www.espn.com/nba/story/_/id/44888875/nba-draft-2025-projecting-30-best-prospects

**Layne Vashro, EWP / Humble.** Quasi-Poisson regression to peak value, where value = 2-year average of Win
Shares and RAPM-wins. NCAA inputs: **age, experience (class), MPG, SOS, height, weight, per-possession box
stats**, fit on NCAA→NBA players since 1990; separate international model. Claim: out-of-sample retrodictions
beat actual front offices over the prior decade (no number published). Key qualitative finding: jump shooting
is the most *variable* skill in translation; shot-blocking the least.
https://fansided.com/2014/12/09/layne-vashros-draft-projections-tool/

**Ryan Davis, Model 284 (PNSP / Role Probability / Similarity Scores).** PNSP predicts each player's
**4th-year NBA per-36 box line** (11 components) with an ensemble, z-scores each component *within position*,
sums, then percentile-ranks vs historical draftees (0–100). Inputs: season box stats, **team-level college
stats (incl. SOS)**, physical measurements, **HS scouting rank**, position, years in college; trained on
1997–2016 entrants, US college only. Reported correlations of PNSP with realized NBA metrics: **BPM .69,
VORP .68, PER .87**. Role Probability model = same inputs → P(All-Star / starter / bench / out).
https://model284.com/peak-nba-statline-projection-model/ · https://model284.com/nba-role-probability-model/

**Model 284, scouting-text analysis — a negative result worth banking.** Bag-of-words + Bing Liu sentiment
lexicon on ~60 NBADraft.net reports: sentiment vs BPM **−0.19**, vs games played **−0.02**; rate-normalized
**−0.14 / −0.11**. Naive text sentiment has no usable signal. Confirms our rule-based-keyword-only stance.
https://model284.com/my-model-monday-nba-draft-scouting-text-analysis/

**Andrew Johnson — scouting-attribute regression (the single most actionable finding).** Regressed NBA
outcomes on twelve **numeric 1–10 grades published by NBADraft.net** (Athleticism, Quickness, Strength,
Defense, Leadership, Intangibles, …). Results: **Intangibles+Leadership explain ~11% of variance in NBA
performance (8–12% across sub-samples)**; **Quickness 3–10%**; the **Defense** grade had **no independent
effect** and rarely survived cross-validation. These are publisher-assigned integers, not LLM scoring.
https://fansided.com/2016/03/09/breaking-down-scouting-factors-in-the-nba-draft/

**ESPN Stats & Info draft model (Whitehead-era S&I / DraftBPM lineage).** Random-forest **survival** model →
SPM in NBA years 2–5, plus P(All-Star/starter/bench/bust). Inputs span **five categories: college performance,
international performance, scout rank, AAU/FIBA-juniors performance, combine measurables**, plus age, height,
weight, position. Importances stated in prose: **scout rank is by far the most important variable**; age,
height and **BMI** also significant; among the 14 college stats, **ORB% matters most for bigs and STL% for
guards**, 3PAr notable. Adding everything beyond scout rank **reduces SPM uncertainty ~10%**.
https://www.espn.com/blog/statsinfo/print?id=119567

**FiveThirtyEight CARMELO.** Nearest-neighbour career-arc model; rookie inputs = pace- and SOS-adjusted
college stats since 2001 (ESPN S&I), height, weight, age, position, **Chad Ford top-100 rank**, and
**draft slot weighted heavily**; target = SPM in years 2–5.
https://fivethirtyeight.com/features/how-were-predicting-nba-player-career/

**Nylon Calculus LUCARIO (2018).** Stacked ensemble (OLS + ElasticNet + GBDT + XGBoost, RSCI treated as a
fifth "model", random-forest meta-learner, 10-fold CV) over **60+ features**: position, height, **wingspan**,
age, per-36 offense/defense, advanced rates, **RSCI**, and explicit **interaction features — AST/TOV, "body
density" (mass/height³), assists×rebounds**. All variables transformed **relative to positional means**.
Target: **year-3 OBPM/DBPM** (rookie-contract value, not peak). Out-of-sample on 2017: mean error 0.6–1.5
draft slots better than the actual draft order.
https://fansided.com/2018/06/11/nylon-calculus-nba-draft-model-process-lucario/

**Nylon Calculus "Benefit of the Doubt" (2018).** BART (Bayesian additive regression trees) to peak
rookie-contract value (box + RAPM blend); paired pre-draft and post-draft variants to isolate the marginal
information in the draft slot itself. No numeric importances.
https://fansided.com/2018/06/29/nylon-calculus-benefit-doubt-draft-model/

**Jeremias Engelmann, "How to build an NBA draft model."** Argues for **probability-tier targets** (P(impact
<0), 0–1.5, 1.5–3, >3) over point estimates because it captures outcome variance. Importance findings:
**international status is a large positive**; **FT% and 2P% are the most important shooting inputs**;
**assists and height are consistently important**; **steals matter for clearing replacement level, blocks
gain importance at higher cutoffs**; draft pick number is extremely important and *grows* in relative
importance at higher cutoffs. Also names **NBA family connections, team quality, SOS**.
https://www.roycewebb.com/p/how-to-build-an-nba-draft-model

**Silver Bulletin PRISM (2025, Joseph George).** CatBoost **pairwise-ranking** classifier — given two
prospects, which has the better career — target = eWINS (EPM-based wins above replacement) summed over the
first seven seasons; trained on 2010–2021 ("the BartTorvik era"). Feature families: box production, **BPM/EPM
impact, BPM share and trajectory**, height/**wingspan/length/BMI**, **shot-creation and self-creation
metrics**, **play-type frequencies**, SOS, **role probability distributions (3 offensive × 3 defensive
archetypes)**, position-adjusted playstyles, STL/100, consensus scouting rank as prior. Beats consensus in
pairwise accuracy among players who got NBA minutes; no numeric metric published.
https://www.natesilver.net/p/how-our-prism-nba-draft-model-works

**Mike Gribanov, GRB / GRBU model (2025–26).** XGBoost over ~25,000 D1 player-career profiles since 2011,
no hard pool cut by recruit rank / draft slot / NBA minutes. Target = "GRB", a weighted blend of **DARKO +
EPM + LEBRON (+ xRAPM)** at prime age. **Predicts offense and defense separately and sums them — reported to
consistently beat predicting total directly.** Reported **Spearman ≈ 0.50** overall; **defense Spearman 0.59
vs offense 0.40**. ~18 features named in prose: per-100 shooting splits, playmaking, age, height, SOS,
recruit-rank percentile. This is the closest public benchmark to our 0.45→0.55 target.
https://grboard.netlify.app/ · https://www.patreon.com/posts/grb-draft-model-151940408

**Nick Restifo.** Ensemble (random forest + GB logistic + logistic + NN + CART); target = 2-year peak
RAPM/WS blend, LOESS-smoothed; NCAA only (international leagues too heterogeneous).

**"jsutthoops" model.** Target RAPM adjusted for availability/expected minutes; college stats + combine +
**consensus mock picks as prior**. Reports **~8% better than front offices** at ranking impact, and
**46.4% vs a 39.3% base rate** at flagging picks 16–30 who become top-15-impact players.
https://jsutthoops.substack.com/p/this-draft-model-beats-front-offices

**Others, briefly.** *Bart Torvik* — free 2008+ player data: PORPAGATU!, adjusted O/D ratings from PBP, shot
bins **dunks / close-2 / far-2 / 3**, **assisted %**, min%, plus arbitrary **opponent-quality game filters**;
this is the single best free college source now that hoop-math.com is dead (DNS gone, stopped updating).
*Evan Miyakawa BPR* — free, **2009-10 → present**: Box BPR prior + prior-informed RAPM from play-by-play,
split into **OBPR/DBPR**, teammate- and opponent-adjusted. *Hoop-Explorer* — free on/off + RAPM, but only
**2018+**. *Haslametrics* — team-level, little player value. *Cerebro Sports C-RAM* — paid; rule out.
*Jordan Sperber (Hoop Vision)* — qualitative/schematic, no model. *Sam Vecenie (The Athletic) / The Ringer /
Swish Theory / No Ceilings* — tiered human boards, useful only as extra mock-consensus publishers (already
used); Swish Theory additionally publishes a **13-analyst consensus with an explicit disagreement metric**.
*GitHub*: JasonG7234/NBA-Draft-Model is the most feature-rich open repo — scrapes Torvik + hoop-math +
RealGM + 247Sports + NBA combine and engineers **"% Assisted Overall", "Dunks per Minute Played"**, play style.
https://github.com/JasonG7234/NBA-Draft-Model · https://blog.evanmiya.com/p/bayesian-performance-rating

---

## Part B — Consolidated new inputs, ranked by expected value

Leakage note used below: "in-season" = fully determined by games played before the draft, no dating risk.

| # | Feature | Definition | Source / dating | Obtain | Hrs | Leakage |
|---|---|---|---|---|---|---|
| 1 | **Published numeric scouting grades** | NBADraft.net 1–10 grades: Athleticism, Quickness, Strength, Leadership, Intangibles, Potential (drop *Defense* — no independent effect). Use Intangibles+Leadership composite first. | nbadraft.net player pages; date via Wayback snapshot **before draft night** | Free; profiles exist back to early 2000s; needs archive crawl | 20–30 | **Medium**: live pages are edited post-draft. Must use pre-draft snapshots only. |
| 2 | **Self-creation / assisted-FG share** | %FG assisted overall and by bin (rim / 2 / 3); unassisted-rim rate = creation proxy | barttorvik.com player pages, 2008+ (toRvik/cbbdata) | Free, scriptable | 6–10 | None (in-season) |
| 3 | **Quality-of-opponent splits** | Same box/advanced stats restricted to **top-100 (and top-50) opponents**, plus the *delta* vs full season | Torvik `t100`/opponent-rank game filter + `bart_player_game`, 2008+ | Free | 10–15 | None |
| 4 | **Separate O/D targets + pairwise ranking objective** | Fit offense and defense value separately and sum; train with a pairwise/rank loss instead of regression | modeling change, no new data | n/a | 8–12 | None |
| 5 | **On/off-informed college impact (BPR)** | OBPR / DBPR — prior-informed RAPM, teammate- and opponent-adjusted; genuinely new signal vs box-BPM, especially on defense | evanmiya.com player ratings, **2009-10 → present** | Free leaderboards | 8–12 | None; but no coverage for 2000–2009 draftees (use missing-indicator) |
| 6 | **Dunk volume** | Dunks made and attempted per 40 / per 100; dunk share of rim attempts | Torvik shot bins, 2008+ | Free | 3–5 | None |
| 7 | **Team-context share features** | Min% of team minutes, share of team shots/possessions, **BPM share** (player BPM ÷ team total), team adjusted efficiency, teammate quality | Torvik team + player, 2008+ | Free | 4–6 | None |
| 8 | **Derived anthropometric indices** | BMI, wingspan−height, reach−height, mass/height³, length percentile within position — the "length-size" composite is what the combine literature finds predictive (r ≈ .31–.55 with defense), not individual drills | already-held combine data | n/a | 2–3 | None |
| 9 | **Game-level distribution features** | Mean/SD/skew of a per-game box-impact score, share of games above a threshold, NCAA-tournament-game splits | Torvik `bart_player_game`, 2008+ | Free | 8–12 | None |
| 10 | **Role-archetype soft memberships** | K-means/GMM on free shot-bin + assist + usage + rebound + block rates → 3 offensive × 3 defensive archetype probabilities; interact with anthropometrics | Derived from #2/#3/#6 | Free | 20–30 | None |
| 11 | **Positional-relative standardization** | Z-score every input within position or height decile before modeling (PNSP, LUCARIO, Pelton all do this) | transformation | n/a | 2 | None |
| 12 | **Tiered / threshold targets** | Model P(impact >0), >1.5, >3 separately — steals dominate the low cutoff, blocks the high one | modeling change | n/a | 4–6 | None |

**Do not pursue:** LLM-free sentiment scoring of scouting text (r ≈ −0.19/−0.02); NBADraft.net *Defense*
grade; individual combine drills in isolation (small effect sizes; use the length composite); Cerebro C-RAM
(paid); hoop-math.com (site dead — Torvik supersedes it); KenPom/Synergy (off-limits).

**Benchmarks to hold ourselves to:** GRB reports Spearman ≈ **0.50** overall (defense 0.59 / offense 0.40);
jsutthoops reports ~**8%** better ranking than front offices and **46.4% vs 39.3%** on late-first-round hits;
ESPN reports adding everything beyond scout rank cuts SPM uncertainty by only **~10%**. Our verified 0.45 is
already in the public state of the art's band; #1, #3 and #5 are the three families that no public model we
found combines at once.
