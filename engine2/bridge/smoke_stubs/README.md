# bridge/smoke_stubs — CPU plumbing harness, never for real numbers

This Mac has only the system Python 3.9 with pandas / numpy / scipy / pyarrow. CatBoost, TabICL, XGBoost, LightGBM
and scikit-learn are not installed and the task forbids network access, so the real learners cannot run here.

These modules stand in for `catboost` and `tabicl` **only when `bridge/smoke_stubs` is put first on `PYTHONPATH`**
(`python -m bridge.run_winner --stub`). They implement the same constructor keywords and `fit` / `predict` API and
return a genuine (if weak) prediction — ridge regression on median-imputed, standardised features, with a
seed-dependent bootstrap so seed averaging still does something — so the smoke test proves that

* the bridge table satisfies `tournament.contract.validate`,
* `tournament.layer1` resolves the winner's feature groups, builds `disc85_gaussrank` / `match3` labels from our
  per-season WAR rows under both label cutoffs, walks forward correctly and writes predictions,
* `tournament.layer2.wide` + `rule_blend` reproduce the archived rank-average and the CSVs come out.

They prove **nothing** about the model's accuracy. On the GPU box the real packages are installed, `--stub` must be
omitted, and `PYTHONPATH` must not contain this directory.
