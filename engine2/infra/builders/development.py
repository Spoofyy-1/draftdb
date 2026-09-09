"""Population development priors (columns dv_*): what a prospect's season-to-season improvement *should* have been.

Torvik holds ~90k D1 player-seasons, almost none of them draftees. Pairs of consecutive seasons of the same player teach
a model of college development -- how much a player with this age, class, role, usage and production profile improves
(BPM, offensive rating, true shooting, usage) the following year. The 1,300-row draftee table cannot learn those
interactions; the 40k-pair population can. Every draftee then gets:

  dv_exp_dbpm, dv_exp_dortg, dv_exp_dts, dv_exp_dusg   expected next-season change given his final-season profile
  dv_exp_bpm_next                                       final-season BPM plus the expected change (projected level)
  dv_surprise_bpm                                       his actual last change minus what was expected from the season
                                                        before: development the population model did not predict
  dv_exp_dbpm2                                          expected two-season change (upside horizon)

Causal by construction: the model scoring draft class Y is fit only on season pairs whose later season is <= Y, so
nothing after draft night Y is in the fit. Draftees whose final season predates Torvik (2008) get NaN.

    from infra.builders.development import load_development
    python -m infra.builders.development     # rebuild + coverage / sanity report
"""

import numpy as np
import pandas as pd

from infra import config as C

KEYS = ["key", "draft_year"]
PROFILE = ["GP", "Min_per", "ORtg", "usg", "eFG", "TS_per", "ORB_per", "DRB_per", "AST_per", "TO_per", "FT_per", "twoP_per", "TP_per",
           "blk_per", "stl_per", "ftr", "porpag", "adjoe", "pfr", "ast_tov", "rim_pct", "mid_pct", "dunk_made", "drtg", "adrtg", "dporpag",
           "stops", "bpm", "obpm", "dbpm", "gbpm", "mpg", "oreb", "dreb", "treb", "ast", "stl", "blk", "pts", "height_in", "class_year", "age"]
TARGETS = {"dbpm": "bpm", "dortg": "ORtg", "dts": "TS_per", "dusg": "usg"}
MIN_GP, MIN_MIN = 10, 20


def _seasons() -> pd.DataFrame:
    tv = pd.read_parquet(C.PROC / "torvik.parquet").reset_index(drop=True)
    tv["tidx"] = tv.index
    end = pd.to_datetime(tv.year.astype(str) + "-04-01")
    tv["age"] = ((end - tv.birthdate) / np.timedelta64(365, "D")).astype(float)
    return tv


def _pairs(tv: pd.DataFrame, gap: int) -> pd.DataFrame:
    a = tv[(tv.GP >= MIN_GP) & (tv.Min_per >= MIN_MIN)]
    nxt = a[["pid", "year", *TARGETS.values()]].rename(columns={c: f"{c}_next" for c in TARGETS.values()})
    nxt["year"] = nxt.year - gap
    p = a.merge(nxt, on=["pid", "year"], how="inner")
    for k, c in TARGETS.items():
        p[k] = p[f"{c}_next"] - p[c]
    return p


def _fit(pairs: pd.DataFrame, target: str, seed: int = 0):
    import lightgbm as lgb
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, reg_lambda=5.0, random_state=seed, verbose=-1)
    return m.fit(pairs[PROFILE].astype(float), pairs[target].astype(float))


def build() -> pd.DataFrame:
    tv = _seasons()
    table = pd.read_parquet(C.PROC / "draft_table.parquet", columns=KEYS + ["torvik_idx"])
    table = table[table.torvik_idx.notna()].copy()
    table["tidx"] = table.torvik_idx.astype(int)
    rows = table.merge(tv, on="tidx", how="left", suffixes=("", "_tv"))
    pairs1, pairs2 = _pairs(tv, 1), _pairs(tv, 2)
    prev = tv[["pid", "year", *PROFILE]].assign(year=lambda d: d.year + 1)  # the season before each row's season
    out = []
    for y, g in rows.groupby("draft_year"):
        fit1 = pairs1[pairs1.year + 1 <= y]
        fit2 = pairs2[pairs2.year + 2 <= y]
        if len(fit1) < 2000:
            continue
        rec = g[KEYS].copy()
        X = g[PROFILE].astype(float)
        for k in TARGETS:
            rec[f"dv_exp_{k}"] = _fit(fit1, k).predict(X)
        rec["dv_exp_bpm_next"] = g.bpm.values + rec["dv_exp_dbpm"].values
        if len(fit2) >= 2000:
            rec["dv_exp_dbpm2"] = _fit(fit2, "dbpm").predict(X)
        # surprise: actual last change vs what was expected from the previous season
        pv = g[["pid", "year"]].merge(prev, on=["pid", "year"], how="left")
        has = pv.bpm.notna().values
        exp_prev = np.full(len(g), np.nan)
        if has.any():
            exp_prev[has] = _fit(fit1, "dbpm").predict(pv.loc[has, PROFILE].astype(float))
        rec["dv_surprise_bpm"] = (g.bpm.values - pv.bpm.values) - exp_prev
        out.append(rec)
        print(f"{y}: {len(g)} draftees, fit on {len(fit1)} pairs", flush=True)
    return pd.concat(out, ignore_index=True).drop_duplicates(KEYS)


def load_development() -> pd.DataFrame:
    return build()


if __name__ == "__main__":
    f = build()
    t = pd.read_parquet(C.PROC / "draft_table.parquet", columns=KEYS + ["source", "war5", "pick"]).merge(f, on=KEYS, how="left")
    col = t[t.source == "college"]
    print("coverage (college rows) by year:", col.groupby("draft_year").dv_exp_dbpm.apply(lambda s: s.notna().mean()).round(2).to_dict())
    print(f[[c for c in f.columns if c.startswith("dv_")]].describe().T.round(3).to_string())
    from scipy.stats import spearmanr
    for c in [c for c in f.columns if c.startswith("dv_")]:
        ok = col[c].notna() & (col.draft_year >= 2013)
        print(f"{c:18s} rho with war5 (2013+): {spearmanr(col.loc[ok, c], col.loc[ok, 'war5']).correlation:+.3f}")
