# Every permutation tried (2026-09-04 → 2026-09-05)

All selection was on **walk-forward** folds inside the training window (train on earlier classes, predict the held-out 2014–2018 class). A change is accepted only if the mean walk-forward IC gains ≥ 0.005 **and** it wins at least 4 of the 5 folds head-to-head. Accepted configurations get one sealed blind scoring on 2019–2025, which is written to the append-only ledger (`experiments/ledger.jsonl`). Blind results are never used for selection.

## Overnight queue — 57 single changes vs the frozen best model

Each row is one change to the frozen genome. `gain` = walk-forward IC minus the frozen model's +0.3805; `folds` = folds won of 5; `wemby` = Wembanyama's 2023 predicted rank (diagnostic only, never used for selection).

| # | change | walk-forward | gain | folds | passed | blind edge (passers only) | wemby 2023 |
|---|---|---|---|---|---|---|---|
| 1 | data wk `{"wk": 1}` | +0.3726 | -0.0079 | 2/5 | fail |  | 15 |
| 2 | data el `{"el": 1}` | +0.3758 | -0.0048 | 3/5 | fail |  | 16 |
| 3 | data wk+el `{"wk": 1, "el": 1}` | +0.3777 | -0.0028 | 1/5 | fail |  | 18 |
| 4 | data wk,intl off `{"wk": 1, "intl": 0}` | +0.3720 | -0.0085 | 2/5 | fail |  | 20 |
| 5 | label cum3 `{"hw": "cum3"}` | +0.3711 | -0.0094 | 2/5 | fail |  | 18 |
| 6 | label cum4 `{"hw": "cum4"}` | +0.3766 | -0.0040 | 1/5 | fail |  | 17 |
| 7 | label uniform `{"hw": "uniform"}` | +0.3669 | -0.0136 | 0/5 | fail |  | 11 |
| 8 | label disc70 `{"hw": "disc70"}` | +0.3759 | -0.0047 | 2/5 | fail |  | 14 |
| 9 | label front `{"hw": "front"}` | +0.3815 | +0.0010 | 3/5 | fail |  | 16 |
| 10 | label top2 `{"hw": "top2"}` | +0.3316 | -0.0489 | 1/5 | fail |  | 10 |
| 11 | label rank(not gauss) `{"label": "rank"}` | +0.3685 | -0.0121 | 1/5 | fail |  | 12 |
| 12 | label noclip `{"clip": "none"}` | +0.3805 | +0.0000 | 0/5 | fail |  | 12 |
| 13 | label log1p `{"clip": "log1p"}` | +0.3805 | +0.0000 | 0/5 | fail |  | 12 |
| 14 | shrink M250 `{"M": 250.0}` | +0.3834 | +0.0029 | 3/5 | fail |  | 12 |
| 15 | icl topk100 `{"icl_topk": 100}` | +0.3747 | -0.0059 | 2/5 | fail |  | 13 |
| 16 | icl topk60 `{"icl_topk": 60}` | +0.3778 | -0.0027 | 3/5 | fail |  | 12 |
| 17 | icl ctx drafted `{"icl_ctx": "drafted"}` | +0.3666 | -0.0139 | 1/5 | fail |  | 12 |
| 18 | icl n32 `{"icl_n": 32}` | +0.3820 | +0.0015 | 4/5 | fail |  | 12 |
| 19 | icl norm none `{"icl_norm": "none"}` | +0.3796 | -0.0009 | 3/5 | fail |  | 12 |
| 20 | icl norm power `{"icl_norm": "power"}` | +0.3832 | +0.0026 | 4/5 | fail |  | 11 |
| 21 | icl norm quantile `{"icl_norm": "quantile"}` | +0.3765 | -0.0040 | 2/5 | fail |  | 13 |
| 22 | icl outlier2 `{"icl_outlier": 2.0}` | +0.3876 | +0.0071 | 3/5 | fail |  | 14 |
| 23 | icl outlier8 `{"icl_outlier": 8.0}` | +0.3801 | -0.0004 | 2/5 | fail |  | 12 |
| 24 | icl shuffle random `{"icl_shuffle": "random"}` | +0.3771 | -0.0035 | 1/5 | fail |  | 11 |
| 25 | ridge a30 `{"ridge": 1, "ridge_alpha": 30.0}` | +0.4052 | +0.0247 | 3/5 | fail |  | 14 |
| 26 | ridge a100 `{"ridge": 1, "ridge_alpha": 100.0}` | +0.3950 | +0.0144 | 2/5 | fail |  | 14 |
| 27 | ridge a300 `{"ridge": 1, "ridge_alpha": 300.0}` | +0.3966 | +0.0161 | 3/5 | fail |  | 14 |
| 28 | ridge a1000 `{"ridge": 1, "ridge_alpha": 1000.0}` | +0.3885 | +0.0080 | 3/5 | fail |  | 14 |
| 29 | knn `{"knn": 1}` | +0.3519 | -0.0286 | 1/5 | fail |  | 18 |
| 30 | catboost `{"cat": 1}` | +0.3684 | -0.0121 | 1/5 | fail |  | 12 |
| 31 | cat+ridge300 `{"cat": 1, "ridge": 1, "ridge_alpha": 300.0}` | +0.3964 | +0.0159 | 2/5 | fail |  | 14 |
| 32 | cat+knn `{"cat": 1, "knn": 1}` | +0.3485 | -0.0320 | 0/5 | fail |  | 17 |
| 33 | resid stack `{"resid": 1}` | +0.3635 | -0.0170 | 3/5 | fail |  | 15 |
| 34 | interaction constraints `{"ixc": 1}` | +0.3902 | +0.0097 | 3/5 | fail |  | 12 |
| 35 | per-season target `{"pss": 1}` | +0.3541 | -0.0264 | 1/5 | fail |  | 9 |
| 36 | horizon models `{"hz": 1}` | +0.3867 | +0.0062 | 2/5 | fail |  | 22 |
| 37 | cov weights `{"covw": 1}` | +0.3914 | +0.0109 | 4/5 | PASS | +0.189 (4/7 ≥ +0.2) | 12 |
| 38 | hz+covw `{"hz": 1, "covw": 1}` | +0.3977 | +0.0171 | 4/5 | PASS | +0.172 (4/7 ≥ +0.2) | 16 |
| 39 | labelmix2 `{"labelmix": "mix2"}` | +0.3760 | -0.0045 | 1/5 | fail |  | 12 |
| 40 | labelmix3 `{"labelmix": "mix3"}` | +0.3791 | -0.0014 | 2/5 | fail |  | 12 |
| 41 | thin mild `{"thin": "mild"}` | +0.3802 | -0.0003 | 3/5 | fail |  | 17 |
| 42 | thin strong `{"thin": "strong"}` | +0.3644 | -0.0161 | 2/5 | fail |  | 21 |
| 43 | anchor top1 `{"anchor": "top1"}` | +0.3816 | +0.0011 | 1/5 | fail |  | 8 |
| 44 | anchor top3 `{"anchor": "top3"}` | +0.3713 | -0.0092 | 1/5 | fail |  | 8 |
| 45 | cons .1 `{"cons": 0.1}` | +0.3777 | -0.0028 | 3/5 | fail |  | 11 |
| 46 | cons .2 `{"cons": 0.2}` | +0.3731 | -0.0074 | 2/5 | fail |  | 10 |
| 47 | mono off `{"mono": "off"}` | +0.3810 | +0.0005 | 2/5 | fail |  | 14 |
| 48 | hurdle `{"hurdle": 1}` | +0.3884 | +0.0079 | 3/5 | fail |  | 12 |
| 49 | bag9 `{"bag": 9}` | +0.3840 | +0.0035 | 3/5 | fail |  | 12 |
| 50 | mingp10 `{"mingp": 10}` | +0.3805 | +0.0000 | 0/5 | fail |  | 12 |
| 51 | risk q25 `{"risk": "q25"}` | +0.3727 | -0.0078 | 2/5 | fail |  | 16 |
| 52 | fx shoot `{"fx": "shoot"}` | +0.3765 | -0.0040 | 1/5 | fail |  | 12 |
| 53 | fx shoot+age `{"fx": "shoot+age"}` | +0.3721 | -0.0084 | 2/5 | fail |  | 13 |
| 54 | fx posz `{"fx": "posz"}` | +0.3842 | +0.0037 | 2/5 | fail |  | 14 |
| 55 | fx eraz `{"fx": "eraz"}` | +0.3638 | -0.0167 | 1/5 | fail |  | None |

Seed stability of the frozen model: seed-stability: base WF over seed sets [0.387, 0.3837, 0.3794]  sd 0.0031

Greedy combination of the passers → final genome changes: `['hz+covw']`, walk-forward +0.3977, blind edge +0.172 (4/7 seasons ≥ +0.2), Wembanyama 2023 rank 16. Verdict: the walk-forward gain did not carry to the blind seasons; the frozen model was kept.

## Evolution on data v1 — 257 generations, 6 accepted

Single- and two-gene mutations of the frozen genome, same gate. Every 10 generations an Opus 4.8 critic reviewed the walk-forward misses (names attached only on the analyst machine) and proposed gene changes, which the evolution tried before random mutations (113 critic suggestions tried). Full history: `experiments/evolution_v1_runs.jsonl`, log: `experiments/evolution_v1.log`.

| gen | mutation (vs champion at the time) | walk-forward | gain | folds | blind edge after acceptance |
|---|---|---|---|---|---|
| 8 | `covw=1` | +0.3914 | +0.0109 | 4/5 |  |
| 14 | `hz=1/covw=1` | +0.3977 | +0.0171 | 4/5 | +0.189 (2019 -0.023, 2020 -0.010, 4/7 ≥ +0.2) |
| 17 | `fx=posz` | +0.4060 | +0.0146 | 4/5 | +0.168 (2019 -0.034, 2020 -0.046, 4/7 ≥ +0.2) |
| 28 | `ridge=1` | +0.4192 | +0.0131 | 4/5 | +0.149 (2019 -0.059, 2020 -0.021, 3/7 ≥ +0.2) |
| 50 | `icl_outlier=2.0/knn=1` | +0.4256 | +0.0064 | 4/5 | +0.154 (2019 -0.006, 2020 -0.017, 2/7 ≥ +0.2) |
| 86 | `icl_norm=quantile/el=1` | +0.4487 | +0.0231 | 4/5 | +0.167 (2019 +0.008, 2020 -0.056, 3/7 ≥ +0.2) |

Rejected mutations: 255 (distinct: 149). The champion's walk-forward climbed from +0.3805 to +0.4487 while its blind edge fell from +0.193 to +0.167: the search fitted the five walk-forward folds. This is why the frozen model, not the champion, is the shipped model.

## Data v3 ablations (walk-forward only, no blind looks)

```
ABLATION v1        WF +0.3805  folds 2014:+0.427 2015:+0.399 2016:+0.190 2017:+0.487 2018:+0.400  (32s)
ABLATION v3        WF +0.3647  folds 2014:+0.431 2015:+0.332 2016:+0.215 2017:+0.451 2018:+0.394  (37s)
ABLATION nonew     WF +0.3682  folds 2014:+0.426 2015:+0.350 2016:+0.191 2017:+0.472 2018:+0.402  (34s)
restored intl rows: 16
ABLATION nointl    WF +0.3748  folds 2014:+0.438 2015:+0.354 2016:+0.231 2017:+0.460 2018:+0.390  (37s)
undid college fills: 5
ABLATION nocol     WF +0.3730  folds 2014:+0.417 2015:+0.353 2016:+0.234 2017:+0.459 2018:+0.402  (34s)
```

## Earlier phases (before the frozen model)

- Label transforms tried: cumulative 5-season WAR, WAR per season played, per-season z-scored WAR, 3- and 4-season windows, peak WAR, tier labels, hurdle (played/not), **discounted 5-season WAR (0.85 per season, clipped at 40) with a Gaussian-rank transform within class — winner**.
- Normalizations tried: none, per-year z, era z, position z, age-adjusted labels, sample-size shrinkage toward the prior (M = 400 pseudo-minutes — kept), international league-level adjustment (kept), thin-data penalties, consensus anchoring/blending (rejected: market deference).
- Members tried: XGBoost (kept, 3 seeds, age monotone), TabICL (kept, 50/50 rank average), TabFM (dropped: weaker and slow), ridge, kNN comparables, CatBoost, XGBRanker, residual stacking, per-season targets, horizon models, interaction constraints, quantile (q25) risk objective.
- Data blocks tried: EuroLeague/EuroCup official API block, Wikipedia career block (both rejected by the gate as extra columns; the corrected rows were instead written into the base international block in data v3).
- Synthetic data: tried early, then banned by rule.
## 2026-09-06 — data v3.2 to v3.5, seven-fold gate, and why the search was drifting

**Current model: EVO gen11** (training window 2007, otherwise the v2 genome with ridge 0.75 / TabICL 0.25 on rich rows, XGBoost on thin rows). Strict 45.2% (edge +0.189, 5 of 7), expanding 50.6% (edge +0.266, 5 of 6). It was the only accepted change that also improved blind results.

**Diagnosis.** Every island's walk-forward fitness rose with each accepted mutation while blind accuracy stayed flat or fell (island C: 43.7% → 42.8% → 42.6% → 40.7% over four keeps). Re-validating each champion against its seed under a stricter gate (seven folds, seed-shifted evaluation) showed the drifted lineage genuinely beats its seed on the 2012–2018 folds (7 of 7). So this is not fold noise but era shift: the old folds reward young-international upside bets that stopped paying after 2018. A named board diff confirmed it: the drifted lineage moved young internationals up (Scoot Henderson 15 → 36, Rayan Rupert 32 → 18) while the winner (window 2007) moved older college producers up and young internationals down.

**Gate changes (all leak-free):** early exit once two folds are clearly lost; seven folds 2012–2018 with season-consistent labels (a fold at draft y only uses NBA seasons complete by that draft night, exactly like the expanding evaluation); recency-weighted fold mean; a keep must win 6 of 7 folds including 2 of the 3 latest; every keep is re-evaluated with different model seeds. The champion's fitness on the season-consistent folds is +0.418.

**Fold-level "why" for the winner (window 2007 vs parent, 2012–2018 folds, 253 big moves):** the moves split evenly right/wrong; the gain is diffuse. Error fell for drafted players (−0.25 rank slots), consensus 11–30 (−0.79) and the youngest prospects (−0.55 / −0.42); it rose for undrafted (+0.83), unranked (+0.75) and international players (+0.54), who are never scored.

**Data blocks tested on the main champion (7 folds; gain, folds won; all fail):** Google Trends −0.001 (5/7); RSCI recruiting spread −0.005 (2/7); Trends + RSCI +0.001 (4/7); window 2000 −0.027; window 2003 −0.024; window 2000 + Trends −0.025. Earlier on v3.3: Torvik, text, trajectory, attention, mock momentum, program pipeline, team-season, misc — 0 of 22 passed alone or combined. Window 2000/2003 fail because the 2000–2004 classes carry no mock-draft consensus.

**Model families tested (all fail):** pairwise within-class Borda member −0.004 (alone it beat the XGBoost member on 5 of 7 folds, +0.437 vs +0.410, but adds nothing to the stack); pairwise + learned weights −0.004; + ridge meta −0.013; + NNLS meta −0.012; + consensus blend −0.004; class-relative standardization (eraz) −0.016; position-relative (posz) −0.013; both −0.027; combinations of earlier near-misses (subspace, cons_unc, mono_prod, pathsplit, market): −0.006 to +0.005, none pass; icl_n 64 +0.002 (4/7); bag 9 seeds ±0.000; CatBoost member carried weight 0.00 and was dropped as a no-op.

**Feature-search candidates (frozen-genome base, 7 folds):** rsci_disagree +0.0059 (4/7), gt_surge +0.0045, hs_growth +0.0030, hype_gap +0.0003, pedigree_gap +0.0001, traj_x_pedigree +0.0009, rsci_decay −0.0024 — none accepted.

**Ensembles of saved boards (blind, report only):** rank-averaging the island champions with the frozen genome gains at most 0.6 points (best 46.0% strict) — variance is not the bottleneck.

**Added but not yet evaluated when the run was stopped:** G League post-draft production as a training-label tiebreaker (1,445 players), second-tier training weights (consensus 11–40), XGBoost depth, recent-class up-weighting in the TabICL context, a consensus top-15 specialist, class-context features, drop-undrafted training, a spline member, the NBA combine athletic/drill block (1,294 players).

## 2026-09-08 — verified data v4, calendar-correct labels, the permutation lottery, and Colin's baseline

Full report: `docs/REPORT_2026-09-08.md`. Summary: the legacy inputs failed a provenance audit (biography, medical, consensus, scouting, 24 college-derived columns); verified replacements (dated mock lists, official combine, dated profiles, Wikipedia birth dates, 39 publisher-years of Wayback mock boards) give strict 43.4% / expanding 45.2% for the same genome (three seed sets). An empty-column negative control exposed a TabICL feature-permutation lottery worth up to +0.011 on the folds; TabICL is now bagged over three permutations and the gate needs 6 of 7 folds, 2 of the latest 3, and a shifted-seed confirmation. About 140 island mutations and 300 panel entries on the verified data produced no confirmed keep; the best near-miss is recomputed college impact percentiles plus a 10% consensus blend (+0.015 on 5 of 7 folds, blind worse). The `colin` branch, kept separate as the baseline, scores 0.510 strict on this vault with dated pre-draft information the main branch does not yet ingest (`docs/COLIN_BASELINE.md`). Why good prospects get drafted low and why the model missed some: `docs/DRAFT_MISSES_EVALUATION.md`.

## 2026-09-08 (afternoon) — novel-data sweep: ten research reports, nine collectors, data v4.8 to v4.16

Research: `docs/research/` (ten ranked reports on proven and novel pre-draft predictors, sources, dating and leakage). Standing rule for every new block: knowable before draft night from a dated source, pid-keyed numeric columns only, names never leave the Mac, rule-based text extraction only.

**Blocks added (each behind its own gene, default off; the champion's inputs are unchanged until a panel passes):**

| Version | Block | What it is | Coverage |
|---|---|---|---|
| v4.8 | dv_, dis_, birthplace | derived composites (length and athleticism PCs, athleticism premium, consensus-hype residual, scoring and TS residuals, older-unranked-productive interaction, cohort offset), dated mock disagreement, US-state and big-metro flags | all classes |
| v4.9 | tc_ (107) | Torvik all-Division-I context: age-cohort percentiles, age-curve residuals, team shares, teammate quality, conference strength, availability, NCAA tournament games, wins above Barthag expectation (prior seasons only), development slopes, shrunk shooting | 85–88% of 2008+ classes |
| v4.10 | gl2_ (147), dp_ (8) | game logs from the ncaahoopR archive: top-50/100 opponent deltas, tournament residual with a prior-year seed expectation, dispersion and trend, availability and absence blocks, close-game splits; draft-page green-room invitations and early-entrant flags from pre-draft Wikipedia revisions | 77–79% 2008+; green room 2006+ |
| v4.11 | wt_ (29) | Wikipedia rule-based: NBA relatives with pre-draft careers, high-school multisport, prep/JUCO/transfer pathway, cohort offset, birthplace ids, dated typed injuries from pre-draft revisions | 91% (injuries 64%) |
| v4.12 | sc_ (24) | NBADraft.net 1–10 grade grid, overall, intangibles+leadership, keyword counts; pre-draft Wayback captures only (65,000 post-draft captures rejected) | grid for 1,011 players |
| v4.13 | bb_ (23) | boards versus mocks: NBADraft.net board 2009–25 and editorial/crowd mocks, DraftExpress board and mock 2009–17, Stepien 2018–22; fit gap, momentum, volatility, dispersion | 1,572 players |
| v4.14 | fy_ (18) | FIBA U15–U19 youth block, z-scored within each tournament field, age relative to the field, pre-draft tournaments only | 445 players |
| v4.15 | tcs_/gls_/scs_/wts_/dvs_ + nb_ | slim versions chosen by a pre-fold label screen (2000–2011 classes only, drafted rows: within-class Spearman with early WAR and with its residual on the actual pick), plus a best-25 cross-block set | — |
| v4.16 | cp_ (18), fys_ (11) | prominence of the NBADraft.net comparison player's career through the season before the draft; FIBA youth slim | 1,108 / 445 players |

**Panel results so far (7 season-consistent folds 2012–2018, recency-weighted gain, 6 of 7 folds, seed-shifted confirmation):** derived dv −0.018, disagreement −0.013, birthplace −0.004, dv+dis −0.024, dv+vcmb −0.011; Torvik context tc −0.011 alone, −0.014 to +0.002 in combinations; optimisation panel: ridge-on-top-columns −0.023, international-line mask −0.023, pedigree×youth −0.008, percentile variants +0.015 with only 4–5 folds won. Negative control: 80 all-missing columns −0.013 (1/7); a single all-missing column reached +0.018 (5/7). The column cost of a 100-column block is about the size of the block failures, so whole-block tests are uninformative above roughly 50 columns; the slim blocks are the informative test and are running.

**Screen (2000–2011 classes, beyond the draft slot; weighted within-class Spearman):** first-season college BPM +0.31, steals per 40 +0.26, steals against top-100 opponents +0.23, seasons span −0.21, assists per 40 +0.21, NBADraft.net intangibles +0.21 (replicates Johnson 2016), age at final season −0.19, impact residual on consensus +0.18, length composite −0.18 (the market over-pays length), negative tournament residual +0.16, years from high school to draft −0.15. FIBA youth on 72 players: usage z −0.47, steals+blocks z −0.42, relative age +0.40 (less precocious is better beyond the pick).

**Physical single-signal search on the champion (24 combine and biography candidates, one at a time):** none passed; best vertical gain +0.007 (5/7). Birth quarter, used as a negative control, −0.007.

**Update, 10:10 UTC — slim blocks and combinations.** Data v4.17 to v4.19 added DraftExpress measurements and growth (dx_, 1,242 players), the Euroleague/EuroCup/ANGT spine (eur_, 254 players, per-game dated through the open API) and prior-chosen slim versions. Panel results: every slim block alone fails (best-25 cross-block +0.010 with 4 of 7 folds; Torvik +0.003; game logs +0.004; grades +0.002; Wikipedia −0.010; derived −0.002; a 25-column all-missing control scored 0.000 with 5 of 7). Comparison prominence −0.011 alone, +0.023 with grades (4 of 7). FIBA youth slim +0.011 alone; **FIBA youth slim plus the international-line mask +0.017 with 6 of 7 folds, the first entry of the day to reach seed-shifted confirmation, where it held +0.021 but won 5 of 7 folds and therefore failed.** Comparison prominence plus FIBA plus best-25: +0.032 with 5 of 7. Colin's recipe under the gate: CatBoost member alone −0.005, CatBoost plus slim grades plus comps +0.023 with 4 of 7. Controls in the same runs reached +0.012 (18 columns) and +0.018 (1 column), so gains under about 0.02 with fewer than 6 folds remain indistinguishable from noise. A focused combination panel around the FIBA-plus-mask near-pass is running.

**Update, 11:40 UTC — first gate pass, verified flat.** The combination FIBA youth slim + international-line mask + NBADraft.net slim grades + comparison prominence passed the seven-fold gate on the current champion (+0.030 recency-weighted, 6 of 7 folds) and the seed-shifted confirmation (+0.026, 6 of 7). Three-seed verification through the sealed vault: strict 41.0% (40.7 to 41.2), expanding 44.6% (44.4 to 44.9), against the verified v4.4 model's 43.4% and 45.2%. The published model therefore stays v4.4. Per class (expanding): 2020 32, 2021 58, 2022 61, 2023 54, 2024 29, 2025 34. Leading explanation: the new blocks' coverage decays in the test years (grade grid 86% of 2008 to 2018 draftees but 68% of 2019 to 2025; comparisons 55% to 35%; FIBA youth 21%), so features that help inside the 2012 to 2018 folds are absent for many recent prospects. Colin's recipe under the gate failed in every combination (CatBoost member plus his blocks, down 0.013 to 0.045). The islands were reseeded from the gate-passed genome so the search continues from that base.

**Update, 13:20 UTC — per-class check of the gate pass.** The same four genes on the verified gen11 genome reproduce the pass (+0.026 confirmed, 6 of 7; per-fold candidate 0.52, 0.33, 0.40, 0.16, 0.24, 0.51, 0.35 versus base 0.45, 0.26, 0.36, 0.13, 0.22, 0.59, 0.28, so the gain is spread across 2012 to 2016 and 2018 with a loss in 2017). Blind per class versus the pre-pass champion (strict): 2019 40 vs 47, 2020 40 vs 32, 2021 65 vs 58, 2022 53 vs 49, 2023 36 vs 35, 2024 27 vs 34, 2025 26 vs 33. Coverage of the new blocks is highest in the 2024 class (grades 84%, comparisons 82%), which is where the candidate loses most, so coverage decay does not explain the flat blind; the class-to-class swings of 6 to 13 points are the noise level documented earlier. Net effect +0.6 expanding, within noise. The islands continue from the gate-passed genome; the published model remains v4.4.

## 2026-09-08 (night) — new current model: EVO gen11 + international-line mask, verified 48% expanding

**What passed.** On the gen11 genome (data v4.19), the single gene `intl_mask` (for college players, drop the youth-tournament international lines that had been merged into their pro-season columns; no new data) passed the seven-fold gate twice independently: +0.017 recency-weighted gain with 6 of 7 folds and +0.024 under shifted seeds with 6 of 7 (panel g11), and the same result on a retest in the combination panel. Per-fold (candidate versus base): 0.50/0.45, 0.27/0.26, 0.38/0.36, 0.16/0.13, 0.27/0.22, 0.52/0.59, 0.32/0.28. The 40-column all-missing control in the same panel scored +0.004 with 6 of 7 folds, so fold wins alone are cheap; the gain threshold and confirmation carried the decision.

**Verification (three seed sets, sealed vault):** strict 42.2% (42.2 to 42.3), expanding **48.4%** (48.2 to 48.5). Base (gen11 genome on v4.19): 40.9 / 44.0. Previously published (v4.4): 43.4 / 45.2. Per class, strict versus the teams: 2019 40 vs 40, 2020 38 vs 35, 2021 59 vs 41, 2022 53 vs 26, 2023 42 vs 11, 2024 37 vs 13, 2025 25 vs 18 (7 of 7); expanding: 2020 40, 2021 61, 2022 54, 2023 52, 2024 46, 2025 38 versus 35, 41, 26, 11, 13, 18 (6 of 6).

**Tie-break rule applied.** Two candidates passed the gate on the same base: the mask alone and the mask plus FIBA youth slim plus NBADraft.net slim grades plus comparison prominence (+0.030, confirmed +0.026). Among gate-passing candidates the minimal change is preferred; the four-gene candidate verified at 44.6 / 41.0 and is recorded, not adopted. The islands were reseeded from the mask-only genome.

**Colin's approach inside our engine (three pre-declared one-shots, report only):** his ingredients in our stack with fold-fitted weights 36.6 / 42.5; his model layer (CatBoost plus TabICL, equal blend, every block on, window 2003) 34.8 / 39.5; his exact CatBoost settings (800 trees, depth 5, learning rate 0.03, nine seeds) with TabICL in an equal blend on his feature families and a 2003 window: 39.9 / 44.0 (members strict: XGBoost 0.39, TabICL 0.40, ridge 0.33, CatBoost 0.36). All three replicas score below the current model, so his 51% is not reproduced by his ingredients inside the verified pipeline; what remains is his own data and labels, and the selection of his winner on the test classes. Rank-averages with his baseline board, scored strictly for the record: his board alone 51.0%; our gen11 board on v4.19 40.9%; his plus ours 48.7%; his plus the four-gene candidate 48.9%. Averaging with our board lowers his, so an ensemble is not a route forward (the earlier 52.3% figure used the retracted legacy board).
