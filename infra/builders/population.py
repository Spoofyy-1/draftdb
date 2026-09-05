"""Population NBA-outcome priors (columns pop_*): what happened to *every* D1 player with this profile, drafted or not.

The draftee table is 60 rows a year. Torvik has every D1 final college season (~4,000 a year, 2008-), and
basketball-reference tells which of those players ever reached the NBA and what they did in their first three seasons
after leaving college (season_war). A model fit on that whole population -- not on the 60 who happened to be drafted --
gives each draftee a prior that does not inherit the draft's own selection:

  pop_p_nba       P(>= 500 NBA minutes in the three seasons after his final college season)
  pop_pred_war3   expected WAR over those three seasons (never played = 0)
  pop_pred_mp3    expected log(1 + minutes) over those three seasons

Causal: the model scoring class Y is fit on final seasons <= Y - 3, whose three-season outcomes are complete by draft
night Y. Undrafted players are matched to basketball-reference by normalised name with a debut within four years of the
final college season; players never seen in the NBA count as zero.

    from infra.builders.population import load_population
    python -m infra.builders.population       # rebuild + report
"""

import numpy as np
import pandas as pd

from infra import config as C
from infra.builders.development import PROFILE, _seasons
from infra.dataset import norm_name
from infra.download_data import _cell, _rows

KEYS = ["key", "draft_year"]
FEATS = PROFILE + ["rec_rank"]
LOOK = 3


def _nba_names() -> dict:
    names = {}
    for s in C.NBA_SEASONS:
        path = C.RAW / "bbref" / f"advanced_{s}.html"
        if not path.exists():
            continue
        for tr in _rows(path, "advanced"):
            nm = _cell(tr, "name_display")
            if nm is not None and nm.get("data-append-csv"):
                names.setdefault(nm.get("data-append-csv"), nm.text_content().strip())
    return names


def _outcomes(rows: pd.DataFrame, seasons: pd.DataFrame) -> pd.DataFrame:
    """first-LOOK-season NBA minutes and WAR after `year` for rows with a bbref_id (others -> 0)."""
    r = rows[["bbref_id", "year"]].reset_index(drop=True); r["_i"] = np.arange(len(r))
    p = r.dropna(subset=["bbref_id"]).merge(seasons[["bbref_id", "season", "mp", "war"]], on="bbref_id", how="inner")
    p = p[(p.season > p.year) & (p.season <= p.year + LOOK)]
    a = p.groupby("_i").agg(mp=("mp", "sum"), war=("war", "sum"))
    out = pd.DataFrame({"mp3": 0.0, "war3": 0.0}, index=np.arange(len(r)))
    out.loc[a.index, "mp3"] = a.mp.values; out.loc[a.index, "war3"] = a.war.values
    return out


def build() -> pd.DataFrame:
    tv = _seasons()
    tv["key"] = tv.player_name.map(norm_name)
    final = tv.sort_values("year").groupby("pid").tail(1)
    final = final[(final.GP >= 10) & (final.Min_per >= 20) & (final.year <= C.LAST_SEASON - 1)].copy()
    seasons = pd.read_parquet(C.PROC / "season_war.parquet")
    drafts = pd.read_parquet(C.PROC / "drafts.parquet")
    table = pd.read_parquet(C.PROC / "draft_table.parquet", columns=KEYS + ["bbref_id", "torvik_idx"])
    # bbref ids: draftees through the table's torvik match; everyone else by name + debut window
    by_tidx = table.dropna(subset=["torvik_idx"]).assign(tidx=lambda d: d.torvik_idx.astype(int)).set_index("tidx").bbref_id
    final["bbref_id"] = final.tidx.map(by_tidx)
    names = _nba_names()
    debut = seasons.groupby("bbref_id").season.min()
    undrafted = pd.DataFrame({"bbref_id": [b for b in debut.index if b not in set(drafts.bbref_id)]})
    undrafted["key"] = undrafted.bbref_id.map(names).fillna("").map(norm_name)
    undrafted["debut"] = undrafted.bbref_id.map(debut)
    m = final[final.bbref_id.isna()].reset_index().merge(undrafted, on="key", how="inner", suffixes=("", "_u"))
    m = m[(m.debut > m.year) & (m.debut <= m.year + 4)].sort_values("debut").drop_duplicates("index")
    final.loc[m["index"].values, "bbref_id"] = m.bbref_id_u.values
    oc = _outcomes(final, seasons)
    final["mp3"], final["war3"] = oc.mp3.values, oc.war3.values
    final["nba"] = (final.mp3 >= 500).astype(int)
    print(f"population: {len(final)} final seasons {final.year.min()}-{final.year.max()}, {final.bbref_id.notna().sum()} reached the NBA, "
          f"{final.nba.sum()} with >= 500 minutes in {LOOK} seasons")

    import lightgbm as lgb
    rows = table.dropna(subset=["torvik_idx"]).assign(tidx=lambda d: d.torvik_idx.astype(int)).merge(tv[["tidx", *FEATS]], on="tidx", how="left")
    out = []
    for y, g in rows.groupby("draft_year"):
        fit = final[final.year <= y - LOOK]
        if fit.nba.sum() < 100:
            continue
        X = fit[FEATS].astype(float); Xq = g[FEATS].astype(float)
        rec = g[KEYS].copy()
        clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.8, reg_lambda=5.0, random_state=0, verbose=-1).fit(X, fit.nba)
        rec["pop_p_nba"] = clf.predict_proba(Xq)[:, 1]
        for col, tgt in [("pop_pred_war3", fit.war3), ("pop_pred_mp3", np.log1p(fit.mp3))]:
            reg = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
                                    colsample_bytree=0.8, reg_lambda=5.0, random_state=0, verbose=-1).fit(X, tgt)
            rec[col] = reg.predict(Xq)
        out.append(rec)
        print(f"{y}: {len(g)} draftees, fit on {len(fit)} population rows ({fit.nba.sum()} NBA)", flush=True)
    return pd.concat(out, ignore_index=True).drop_duplicates(KEYS)


def load_population() -> pd.DataFrame:
    return build()


if __name__ == "__main__":
    f = build()
    t = pd.read_parquet(C.PROC / "draft_table.parquet", columns=KEYS + ["source", "war5"]).merge(f, on=KEYS, how="left")
    col = t[t.source == "college"]
    print("coverage (college rows) by year:", col.groupby("draft_year").pop_p_nba.apply(lambda s: s.notna().mean()).round(2).to_dict())
    from scipy.stats import spearmanr
    for c in [c for c in f.columns if c.startswith("pop_")]:
        ok = col[c].notna() & (col.draft_year >= 2013)
        print(f"{c:14s} rho with war5 (2013+): {spearmanr(col.loc[ok, c], col.loc[ok, 'war5']).correlation:+.3f}")
