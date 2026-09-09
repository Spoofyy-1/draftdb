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

**Update, 2026-09-09 04:00 local — blocks on the new champion, and block members.** On the gen11 + mask champion: game logs −0.002 (4 of 7), draft page −0.011, game logs + Torvik −0.004, game logs + percentiles +0.018 (5 of 7), 80-column control +0.006; Wikipedia rules −0.023, with Torvik +0.002, with percentiles +0.007, 29-column control −0.016. All fail. A block-compression member (an extra XGBoost trained on the core columns plus one block, weight learned on the folds) changed nothing for any block, because the champion's learned stack weights already give XGBoost zero (TabICL 0.75, ridge 0.25 on rich rows; TabICL alone on thin rows). Ridge- and TabICL-based block members are being tested next with the same 40-column controls.

## 2026-09-09 (morning) — new current model: mask + NBADraft.net grades + Torvik context, verified 49% expanding

**What passed.** On the gen11 + mask champion, the pair `sc` (NBADraft.net 1–10 grade grid and keyword counts, pre-draft Wayback captures) + `tc` (Torvik all-Division-I context: age-cohort percentiles, team shares, teammate quality, conference strength, availability, tournament residual) passed the gate: +0.033 recency-weighted with 6 of 7 folds (per fold 0.52, 0.40, 0.44, 0.13, 0.33, 0.53, 0.34 against 0.50, 0.27, 0.38, 0.16, 0.27, 0.52, 0.32), confirmed at +0.013 under shifted seeds. Grades alone scored −0.001 and grades with game logs +0.019 (5 of 7); the 25-column control +0.002 with 6 of 7 folds.

**Verification (three seed sets):** expanding **49.3%** (49.1 to 49.5), strict 42.1% (41.9 to 42.2). Previous model (mask only): 48.4 / 42.2, seed ranges non-overlapping on the expanding metric. Per class, expanding: 2020 33 (teams 35), 2021 67, 2022 53, 2023 58, 2024 47, 2025 37 (teams 41, 26, 11, 13, 18); strict: 2019 47, 2020 32, 2021 68, 2022 52, 2023 43, 2024 28, 2025 26 versus 40, 35, 41, 26, 11, 13, 18. Stack weights: rich rows TabICL 0.75 / ridge 0.25, thin rows XGBoost 0.5 / TabICL 0.5.

**Also verified and not adopted:** island B's accepted mutation (trajectory + Torvik own-line blocks, +0.046 on folds with 7 of 7) scored 42.5 / 38.6 blind. Ridge-based block members changed nothing for any block (−0.005 to +0.001), and TabICL-based ones failed too (FIBA −0.006, grades −0.001, comparisons +0.012 with 4 of 7, the three together +0.010 with 5 of 7, control −0.002), so the block-member idea is closed. The islands were reseeded from the new champion.

**Update, 2026-09-09 11:30 local — additions on the 49.3% champion, and the switch to engine2.** Every further block on the grades + Torvik context champion fails on the folds (FIBA youth −0.026, comparisons −0.017, game logs −0.053, percentiles −0.011, boards −0.008, draft page −0.011, their combinations −0.018 to −0.044, the 25-column control −0.016), and the islands have not accepted a mutation since being reseeded from it. Per the owner's instruction, Colin's pipeline is now the base: it is copied to `engine2/` on main (his branch untouched), a bridge builds his draft table from our verified data (his labels rebuilt from our per-season outcomes under both his `full` cutoff and the strict `causal` one), and his exact winner configuration is being run on the box under both cutoffs; its holdout predictions are scored once through the sealed vault.

**Update, 2026-09-09 11:40 local — Colin's winner on our data.** His exact configuration (CatBoost from 2003, 800 trees, ten seeds; TabICL from 2010, eight seeds; equal rank-average; his feature families filled from our verified blocks) run through his own layer-1 code on our table and scored once through the sealed vault: **39.9% strict** with strict labels (2019 34, 2020 40, 2021 60, 2022 37, 2023 42, 2024 35, 2025 33) and 40.1% with his `full` label cutoff. On his own data the same configuration scores 51.0%; our engine on our data scores 42.1% strict and 49.3% expanding. The difference is therefore in the data, not the model layer. What the port lacks: named Torvik features filled for only 58% of the 2003–2018 context rows (our own college lines were not used as fallbacks), thin context before 2010, 575 features where his had 324, board momentum from one publisher family, and no comparison-player WAR. Those are the next fixes; every change is selected on the 2012–2018 walk-forward window with strict labels and scored on the holdout once.

**Update, 2026-09-09 11:45 local — engine2, first change.** Starting CatBoost's context in 2008 instead of 2003 (our Torvik coverage begins with the 2008 season) raises his walk-forward score on the 2012 to 2018 window from 0.320 to 0.331 and the holdout from 39.9% to 42.1% strict (one look). The bridge is being extended with our own college lines as fallbacks for the named Torvik features, the international-line mask, a trimmed game-log family and our extra dated families; each variant is chosen on the walk-forward window before its holdout is scored.

**Update, 2026-09-09 12:30 local — engine2 variants on the walk-forward window.** With CatBoost's context from 2008 and strict labels, the blend's 2012–2018 mean / recency-weighted mean: bridge v2 defaults 0.360 / 0.350; game-log family trimmed to his width of 11 columns 0.369 / 0.360; international-line mask 0.368 / 0.352; **mask plus trim 0.386 / 0.374**, beating the defaults in all seven classes. The extra families (FIBA youth, Euroleague, draft page) are not yet referenced by the winner's configurations, so that variant was identical. The mask-plus-trim variant goes to seed-shifted confirmation, then a single strict holdout look and the expanding holdout.

**Update, 2026-09-09 13:00 local — engine2 pick confirmed and scored.** Mask plus trimmed game logs (CatBoost context from 2008, strict labels) holds under shifted seeds (+0.011 recency-weighted, 5 of 7 classes) and scores **41.2% strict** and **43.8% expanding** through the sealed vault (expanding: earlier test classes join the context with only the seasons completed by each draft night, from the vault's calendar-correct rows). Against his winner as first ported (39.9 / 42.1) that is progress; against our own engine's published 42.1 / 49.3 it is not. On this data his model layer stays below ours.

**Update, 2026-09-09 13:30 local — our families inside his model.** Adding the dated blocks he never had as feature groups (FIBA youth age-relative, the Euroleague spine, draft-page facts) plus a source flag lifts the walk-forward blend from 0.386 / 0.375 to 0.399 / 0.388 (mean / recency-weighted), winning 6 of 7 classes, with both of his members improving (CatBoost 0.35 to 0.36, TabICL 0.38 to 0.39). It is in seed-shifted confirmation, then a single strict holdout look and the expanding holdout. Separately, since his built data does not exist anywhere reachable, his data is being rebuilt from the permitted public sources his code names (Torvik exports, the sportsdataverse hoopR archive, the JasonG and AyushBatra draft datasets, a public combine table, RAPTOR), with our sealed labels and Wikipedia-rule biography standing in for his basketball-reference scrapes.

**Update, 2026-09-09 14:00 local — engine2 with our families, confirmed and scored.** The families pick holds under shifted seeds (+0.022 recency-weighted over the previous pick, 6 of 7 classes) and scores **41.7% strict / 44.2% expanding** through the sealed vault (strict per class 39, 43, 57, 45, 36, 34, 38; expanding 41, 58, 49, 43, 43, 38 for 2020 to 2025). That is the best engine2 result so far, still below our own engine's published 42.1 / 49.3. Next in the queue: the same configurations on the table that carries his rebuilt families (game-level, hoopR, shrunk rates, transfers, AyushBatra and JasonG lines), judged on the walk-forward window first.

**Update, 2026-09-09 14:35 local — his rebuilt data, judged.** With his families rebuilt from the permitted public sources and placed on the same tables, the walk-forward blend scores 0.357 (his defaults), 0.360 (mask plus trim), 0.375 (plus our families) and 0.372 (plus his game-level, hoopR, shrunk-rate and transfer families), against 0.388 for our families alone. His data as reconstructable does not improve his model on our labels, so the 51% he reported is not reachable from permitted sources; what remains of it is the choice of winner on the test classes, the basketball-reference-derived pieces, and the label protocol. Engine2's best stays 41.7 / 44.2. One pre-declared combination is scored next: the equal rank-average of our published champion and engine2's best, two different learners on the same data.

**Update, 2026-09-09 14:45 local — the two-learner ensemble, one look.** The pre-declared equal rank-average of our published champion and engine2's best scores 43.9% strict (up from 42.1) but 48.6% expanding (down from 49.3), so it is recorded and not adopted; the headline metric did not improve. Where the day ends: verified expanding accuracy 45.2% → 49.3% (mask, then grades plus Torvik context); every other line tried today, including Colin's model layer on our data, his rebuilt data, block members and further blocks, is closed with numbers in this log.

**Update, 2026-09-09 15:35 local — conference-only split, seeds.** A new dated block from Torvik's conference-only export (the conference regular-season line and its deltas against the full season, 53 columns, 1,694 players) fails in both engines: on our champion −0.021 (slim) and −0.023 (full) against an 8-column control at +0.011; inside engine2's best configuration 0.387 versus 0.388. Doubling engine2's seeds (twenty CatBoost and sixteen TabICL) changes nothing (0.387 versus 0.388), so seed variance is not the limit. Both lines closed.
