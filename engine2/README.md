# engine2 — Colin's pipeline as the base (copied from the `colin` branch on 2026-09-09)

This directory is a copy of the model pipeline from the separate `colin` branch (kept untouched as the baseline). From here on it is the
base we build on with what the verified track learned: strict pre-draft dating of every source, the seven-fold walk-forward gate with
seed-shifted confirmation for every change, verification on three seed sets through the sealed vault before anything is published,
and the international-line mask. Sources his builders used that this repository cannot use (basketball-reference for labels and
biography, ESPN for game logs) are replaced by the verified blocks collected on 2026-09-08 (see `docs/research/` and `docs/EXPERIMENTS.md`).

Original documentation: `README_colin.md`, `handoff_colin.md`, `tournament/README.md`, `winners/README.md`.

## How changes are judged here (the discipline carried over from the verified track)

1. **Selection only on the walk-forward context window.** `python -m bridge.run_winner --window walkforward --cutoff causal ...` scores classes 2012–2018, each predicted by a model that saw only earlier classes with labels containing only seasons completed by that draft night. A change is a candidate when it improves the recency-weighted mean of those seven classes by at least 0.008 and wins at least 6 of 7, and holds that under a different seed set.
2. **Holdout is scored once per candidate, through the sealed vault on the box** (`score_external.py`), never used to pick among variants. Three seed sets before anything is published; the published number is the strict/causal protocol, not the `full` label cutoff (which lets context labels contain seasons after the scored draft night; see `handoff_colin.md`).
3. **Names never reach the box.** `bridge/build_table.py` reads only `actual_pick` from the identity file; every artefact is pid-keyed.
4. **Closed lines, do not retry blindly:** whole feature blocks bolted onto the old engine, block-compression members, ensembles with the baseline board, the three Colin-recipe replicas inside the old engine (`docs/EXPERIMENTS.md`).

## Planned extensions (in order)

- Reproduce the winner on our data (both cutoffs) and record the vault score.
- Context start 2008 for CatBoost (our Torvik coverage begins with the 2008 season), and a trimmed `response` family (147 columns here versus his 11).
- The international-line mask (for college players, drop youth-tournament lines merged into pro-season columns) as a `build_table` option.
- Our extra dated families as first-class groups: FIBA youth age-relative (`fy_`), DraftExpress growth (`dx_`), Euroleague spine (`eur_`), Wikipedia rules (`wt_`), boards-versus-mocks (`bb_`).
- More seeds per member once the configuration is fixed.
