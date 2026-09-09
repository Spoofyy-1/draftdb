# engine2 — Colin's pipeline as the base (copied from the `colin` branch on 2026-09-09)

This directory is a copy of the model pipeline from the separate `colin` branch (kept untouched as the baseline). From here on it is the
base we build on with what the verified track learned: strict pre-draft dating of every source, the seven-fold walk-forward gate with
seed-shifted confirmation for every change, verification on three seed sets through the sealed vault before anything is published,
and the international-line mask. Sources his builders used that this repository cannot use (basketball-reference for labels and
biography, ESPN for game logs) are replaced by the verified blocks collected on 2026-09-08 (see `docs/research/` and `docs/EXPERIMENTS.md`).

Original documentation: `README_colin.md`, `handoff_colin.md`, `tournament/README.md`, `winners/README.md`.
