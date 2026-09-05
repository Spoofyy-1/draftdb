# Methods — every calculation behind the numbers

## 1. What is being predicted

For a drafted player, the outcome is his **WAR (wins above replacement) over the first five NBA seasons**, taken from the labels in `data/answers/answers_YYYY.csv` (`y_s1_war` … `y_s5_war`; see `LABELS.md` for how WAR/RAPTOR-equivalent impact is built). A class drafted in year Y has completed `k = min(5, 2025 − Y + 1)` seasons, so it is scored on the WAR it has actually produced so far: 2019–2021 on five seasons, 2022 on four, 2023 on three, 2024 on two, 2025 on one. 2026 has no outcome yet and is only boarded.

## 2. Order accuracy (the percentage on the front page)

For one class, let `s_i` be the model's score for drafted player i and `w_i` his k-season WAR. The order accuracy is the Spearman rank correlation

    ρ = corr( rank(s_i), rank(w_i) )

shown as `100·ρ` percent: 100% is a perfect order, 0% no relationship, negative would be anti-correlated. The **NBA scouts** number is the same statistic for the real draft order, `ρ_draft = corr( rank(−pick_i), rank(w_i) )`. **Drafts the AI won** counts classes with `ρ_model > ρ_draft`. The headline numbers are the plain means over the seven test classes. Only actually drafted players are in the pool (55–60 per class); undrafted players who later made the NBA are in the test files but not in the scoring pool, because the question is how to order a draft.

Noise: with n ≈ 57 the standard error of one class's ρ is about `1/√(n−3) ≈ 0.14`; the seven-class mean has SE ≈ 0.05. A model that is +0.18 above the draft on average is well clear of zero, but two models 0.02 apart cannot be separated on this test set.

## 3. Value of the order (the second metric)

Rank the drafted players by k-season WAR: the **perfect** order. Give slot j the weight `w_j = n − j + 1` (pick 1 counts n, the last pick counts 1) and define the value of an order as `V = Σ_j w_j · WAR(player at slot j)`. Then

    value % = 100 · (V − V_worst) / (V_perfect − V_worst)

so the perfect order is 100%, the reverse order 0%, and a random order 50% in expectation. This rewards getting the top of the board right more than the bottom, which the correlation does not.

## 4. The training label

Raw WAR is skewed and its scale drifts across eras, so the model is trained on a transformed label:

1. **Discounted window:** `L = Σ_{i=1..5} 0.85^(i−1) · WAR_i`, i.e. season 1 counts fully, season 5 counts 0.52. Earlier seasons are weighted more because they are what the drafting team actually receives before the second contract.
2. **Clip** at 40 (removes the few extreme careers that otherwise dominate a squared loss).
3. **Gaussian rank within the class:** `z = Φ⁻¹( (rank(L) − 0.5) / n_class )`. The model learns each prospect's position inside his own class on a normal scale; classes of different strength become comparable, and the ranking objective matches how the model is evaluated.

The XGBoost member is fit with squared error on `z`; TabICL is fit as a regressor on `z`. Both use 2010–2018 rows only (770 rows: every drafted player plus every undrafted player who reached the NBA, who carries the label he earned).

## 5. Sample-size shrinkage of rate stats

For every rate feature `x` (per-36 stats, shooting percentages, usage, rebound/assist/steal/block rates) with a minutes count `m`, the model sees

    x_shrunk = w·x + (1 − w)·x̄_train,   w = m / (m + 400)

where `x̄_train` is the training-set mean, estimated inside each walk-forward fold from the training rows only. A 227-minute line (Wembanyama's old ASVEL season) keeps 36% of its own value; a 1,400-minute line keeps 78%. The raw value and `w` are kept as features too, so the trees can learn how much to trust thin samples.

## 6. International production

International per-36 stats are multiplied by a league-level factor (EuroLeague 1.0 down to FIBA youth) before shrinkage; data v3 adds `intl_lg_strength`, a finer 0–1 map keyed on the actual league name (ACB 0.9, EuroCup 0.85, LNB 0.75, ABA/NBL 0.72, CBA 0.55, …), `intl_lg_adj_pts36 = pts36 · strength`, a minutes-weighted last-two-seasons block, and `intl_season_gap` (how many years before the draft the last season was).

## 7. The ensemble

Each member produces a score per prospect in the class. Scores are converted to within-class ranks and averaged:

    board = 0.5 · rank(XGB) + 0.5 · rank(TabICL)

XGBoost: three seeds averaged, learning rate 0.03, depth 3–4, a monotone constraint that older age at draft can never raise the prediction, per-feature weights from `BEST_MODEL.json` (used as column-sampling weights). TabICL: 8 in-context estimators, default normalization. Weights and members were selected on walk-forward folds; the final stack weights are 50/50.

## 8. Walk-forward selection and the acceptance gate

For each fold year y in 2014…2018: train on classes 2010…y−1, predict class y, score it at horizons 1…5 seasons, average. The fitness of a configuration is the mean over the five folds. A change is accepted only if fitness gains at least +0.005 **and** it beats the incumbent in at least 4 of the 5 folds. Seed-to-seed noise of the fitness is ±0.003.

## 9. Blind scoring, the vault and the ledger

Test inputs are stripped of every outcome, pick and year column before they reach the model (`vault.blind_inputs()`); the scoring call takes predictions and returns `{ρ_model, ρ_draft, n}` for one season and nothing else; every call appends a hash-chained record (config name, season, k, ρ) to `experiments/ledger.jsonl`. A SHA-256 manifest locks the data files; if any byte changes, the vault refuses to score until it is re-sealed. `tripwire.py` runs 18 cheat attempts (peeking at answers, training on a test class, using the pick as a feature, moving the training window, tampering with a file) and must report that all are blocked before any run. The ledger stood at 703 records when this was written: the blind seasons have been looked at that many times across the whole project, so blind numbers are not a clean holdout for choosing between models anymore — which is why selection is walk-forward only.

## 10. Where the model's edge comes from

Across the seven test classes the model's gain over the real draft is concentrated in the bottom two-thirds of each class — it avoids busts and finds second-round value — while the very top of the board (the consensus top 3) is where the teams are usually right and the model adds little. Its known weaknesses, from the walk-forward misses: thin-data international prospects, older productive college players (under-rated), and young athletic guards with poor efficiency (over-rated).
