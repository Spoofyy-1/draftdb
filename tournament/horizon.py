"""Does matching the training-label horizon to the evaluation horizon help? Answered on context years only.

A holdout class judged today carries however many NBA seasons it has actually played: 2019 has five, 2025 has one.
The models are trained on five-season labels regardless, so for the young classes they rank something the scorer
cannot measure. This asks whether that mismatch costs anything, using mature context classes scored against a
deliberately truncated label -- no holdout year is read.

For each eval year, each training horizon gets one fit; each fit is then scored against every evaluation horizon.
The diagonal is horizon-matched, the H'=5 row is what the tournament does today.

Usage: python -m tournament.horizon [--model xgb] [--device cuda:0] [--years 2013,2014,2015,2016,2017]
"""

import argparse
import itertools

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from infra import config as C
from infra.models import TARGET, predict
from pipeline.run import context_years
from tournament import contract as F
from tournament.layer1 import horizon_target, select_features, transform_labels

CURATED = ["all_torvik", "-cat", "traj", "phys", "intl_pro", "intl_fiba", "combine", "tctx", "intl_z", "mock"]
LEAN = ["core", "rate", "traj", "phys", "intl_fiba", "intl_z", "tctx"]
FEATURE_SETS = {"curated": CURATED, "lean": LEAN}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xgb")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--years", default="2011,2012,2013,2014,2015,2016,2017,2018")
    ap.add_argument("--horizons", default="1,2,3,4,5")
    ap.add_argument("--features", default="curated", choices=list(FEATURE_SETS))
    ap.add_argument("--n-estimators", type=int, default=800)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", help="write the matrix to this csv")
    a = ap.parse_args()
    years = [int(y) for y in a.years.split(",")]
    horizons = tuple(int(h) for h in a.horizons.split(","))
    feats_spec = FEATURE_SETS[a.features]
    assert all(y in C.CONTEXT_YEARS for y in years), "this experiment reads context years only"

    table = pd.read_parquet(C.PROC / "draft_table.parquet")
    seasons = pd.read_parquet(C.PROC / "season_war.parquet")
    G = F.validate(table, verbose=False)
    labelled = sorted(set(table.loc[table.modelled & table.labelled, "draft_year"]))
    feats_all = F.resolve(feats_spec, G, table.columns)

    cells = {}
    for year in years:
        cy = [c for c in context_years(year, "causal", labelled) if c >= 2003]
        ctx_rows = table[table.modelled & table.labelled & table.draft_year.isin(cy)]
        pool = table[table.draft_year == year]
        # Evaluation labels: the same class judged as if only H seasons had been played.
        evals = {h: horizon_target(pool, seasons, h, C.LAST_SEASON) for h in horizons}
        for h_train in horizons:
            ctx = ctx_rows.assign(**{TARGET: horizon_target(ctx_rows, seasons, h_train, C.LAST_SEASON)})
            ctx = transform_labels(ctx, "disc85_gaussrank", seasons, C.LAST_SEASON, h_train)
            feats = select_features(ctx, feats_all)
            outs = [predict(a.model, ctx, pool, feats, a.device, s, n_estimators=a.n_estimators)[0] for s in range(a.seeds)]
            score = np.mean(outs, axis=0)
            for h_eval in horizons:
                keep = np.isfinite(evals[h_eval])
                cells.setdefault((h_train, h_eval), []).append(spearmanr(score[keep], evals[h_eval][keep]).correlation)
        print(f"  {year} done", flush=True)

    m = pd.DataFrame(
        [[np.mean(cells[(ht, he)]) for he in horizons] for ht in horizons],
        index=[f"train h={h}" for h in horizons], columns=[f"eval h={h}" for h in horizons],
    )
    print(f"\nmean Spearman over {years}, model={a.model}, {a.features} features, causal context from 2003\n")
    print(m.round(4).to_string())
    print("\nbest training horizon per evaluation horizon, and its gain over train h=5:")
    for he in horizons:
        ht = max(horizons, key=lambda t: m.loc[f"train h={t}", f"eval h={he}"])
        gain = m.loc[f"train h={ht}", f"eval h={he}"] - m.loc["train h=5", f"eval h={he}"]
        print(f"  eval h={he}: train h={ht}  {m.loc[f'train h={ht}', f'eval h={he}']:.4f}  ({gain:+.4f})")
    if a.out:
        m.assign(model=a.model, features=a.features).to_csv(a.out)
        print("written:", a.out)


if __name__ == "__main__":
    main()
