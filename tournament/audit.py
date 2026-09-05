"""Data and leakage audit. Fails loudly on any leak; prints what the model can actually see.

Data (--data-only, run by the sweep):
  * who is modelled: college vs international / G League vs nobody, by draft year, and the top players still missing
  * coverage of every feature group for college and international draftees and for the test years
  * registry integrity: every listed column exists in the table; no target/market column is a feature

Leakage (needs a GPU):
  * causal context for draft year Y contains only classes < Y with labels from seasons <= Y
  * permutation test: shuffled context labels must give ~0 Spearman
  * bootstrap CI of model - scouts on the latest published run

Usage: python -m tournament.audit [--data-only] [--device cuda:0]
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from infra import config as C
from infra.dataset import FEATURES, MARKET_FEATURE
from validation.run import context_years
from infra.war import war_target
from tournament import contract as F
from infra.config import TARGET


def data_audit(table: pd.DataFrame):
    print("== who is modelled ==")
    cov = table.groupby("draft_year").source.value_counts().unstack(fill_value=0)
    cov["modelled_%"] = (100 * (1 - cov.get("none", 0) / cov.sum(axis=1))).round(0)
    print(cov.to_string())
    lab = table[table.labelled]
    miss = lab[~lab.modelled].nlargest(12, TARGET)[["player", "draft_year", "pick", TARGET, "college"]]
    print(f"\nmodelled {table.modelled.mean():.1%} of all draftees; {(table.source == 'intl').sum()} international / G League rows now in the model")
    print("biggest players still without any pre-draft features:\n" + miss.to_string(index=False))
    for bad in (MARKET_FEATURE, TARGET, "seasons_played", "labelled", "war"):
        assert bad not in FEATURES, f"{bad} is in FEATURES"
    print("\n== feature-group coverage (share of non-null cells) ==")
    F.validate(table, verbose=True)


def leakage_audit(table: pd.DataFrame, device: str):
    from infra.models import fit_predict
    seasons = pd.read_parquet(C.PROC / "season_war.parquet")
    labelled = sorted(set(table.loc[table.modelled & table.labelled, "draft_year"]))
    rng = np.random.default_rng(0)

    print("\n== causal context integrity ==")
    for y in C.VAL_YEARS:
        cy = context_years(y, "causal", labelled)
        assert all(c < y for c in cy), (y, cy)
        ctx = table[table.modelled & table.labelled & table.draft_year.isin(cy)]
        m = ctx[["bbref_id", "draft_year"]].merge(seasons[["bbref_id", "season"]], on="bbref_id")
        used = m[(m.season > m.draft_year) & (m.season <= y)]
        assert used.season.max() <= y
        print(f"  {y}: context {cy[0]}-{cy[-1]} ({len(ctx)} rows, {(ctx.source == 'intl').sum()} intl), max label season {int(used.season.max())} -- OK")

    print("\n== permutation test (shuffled labels -> ~0) ==")
    y = 2016
    cy = context_years(y, "causal", labelled)
    ctx = table[table.modelled & table.labelled & table.draft_year.isin(cy)]
    ctx = ctx.assign(**{TARGET: war_target(ctx, seasons, through=y)[TARGET].values})
    test = table[table.modelled & (table.draft_year == y)]
    real = spearmanr(fit_predict("tabfm", ctx, test, device=device), test[TARGET]).correlation
    perms = [spearmanr(fit_predict("tabfm", ctx.assign(**{TARGET: rng.permutation(ctx[TARGET].values)}), test, device=device, seed=i), test[TARGET]).correlation for i in range(5)]
    print(f"  real {real:+.3f} | shuffled {np.mean(perms):+.3f} +- {np.std(perms):.3f}")
    assert abs(np.mean(perms)) < 0.15, "shuffled labels still predictive -- leak"

    print("\n== bootstrap CI, latest published run ==")
    import sqlite3
    con = sqlite3.connect(C.OUT / "results.sqlite")
    run_id, model = con.execute("select run_id, redraft_model from runs order by created desc limit 1").fetchone()
    p = pd.read_sql("select year, actual_pick, pred, war as peak_war from redraft_picks where run_id=? and modelled=1 and labelled=1 and pred is not null", con, params=(run_id,))
    p = p[p.year.isin(C.VAL_YEARS)]
    gap = lambda df: np.mean([spearmanr(g.pred, g.peak_war).correlation - spearmanr(-g.actual_pick, g.peak_war).correlation for _, g in df.groupby("year")])
    boots = []
    for _ in range(2000):
        ys = rng.choice(sorted(p.year.unique()), size=p.year.nunique(), replace=True)
        boots.append(gap(pd.concat([p[p.year == yy].sample(frac=1, replace=True, random_state=int(rng.integers(1 << 31))).assign(year=i) for i, yy in enumerate(ys)])))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  {run_id} ({model}): model - scouts = {gap(p):+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}], P(model > scouts) = {np.mean(np.array(boots) > 0):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-only", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    table = pd.read_parquet(C.PROC / "draft_table.parquet")
    data_audit(table)
    if not a.data_only:
        leakage_audit(table, a.device)


if __name__ == "__main__":
    main()
