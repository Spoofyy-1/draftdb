"""G League translation priors (columns gl_*): how a college profile like this one has produced as a professional.

Every season several hundred college players go straight to the G League and post a full pro stat line
(basketball-reference, data/external/gleague/gleague_<season>_advanced.html). Pairing each one's final Torvik season
with his first G League season gives thousands of college -> pro translations -- an order of magnitude more than the
draftee table -- for a model of "given this college profile, how efficient and how productive is he as a pro?". Each
draftee is then scored by that model on his own final college season:

  gl_pred_ws48   expected first-pro-season Win Shares per 48 minutes
  gl_pred_per    expected PER
  gl_pred_ts     expected true-shooting %
  gl_pred_usg    expected usage %
  gl_pred_mp     expected minutes (log) -- how much a pro coach plays this profile

Causal: the model scoring class Y is fit only on G League seasons <= Y (the G League season ends in spring, before the
June draft). The draftees themselves are never their own training rows: only players not drafted in the pairing year enter.

    from infra.builders.gleague import load_gleague
    python -m infra.builders.gleague          # rebuild + report
"""

import re

import numpy as np
import pandas as pd
from lxml import html

from infra import config as C
from infra.builders.development import PROFILE, _seasons
from infra.dataset import norm_name

EXT = C.ROOT / "data" / "external"
KEYS = ["key", "draft_year"]
STATS = {"per": "g_per", "ts_pct": "g_ts", "usg_pct": "g_usg", "ws_per_48": "g_ws48", "mp": "mp", "g": "g_g", "age": "g_age"}
MIN_MP = 200


def parse_season(year: int) -> pd.DataFrame:
    path = EXT / "gleague" / f"gleague_{year}_advanced.html"
    if not path.exists() or path.stat().st_size < 10000:
        return pd.DataFrame()
    text = path.read_text().replace("<!--", "").replace("-->", "")  # tables ship inside HTML comments
    doc = html.fromstring(text)
    rows = []
    for tr in doc.xpath('//tbody/tr[th[@data-stat="player"]]'):
        cell = {td.get("data-stat"): td.text_content().strip() for td in tr.xpath("./th|./td")}
        rows.append({"season": year, "player": cell.get("player", ""), "team": cell.get("team_id", ""),
                     **{v: pd.to_numeric(cell.get(k), errors="coerce") for k, v in STATS.items()}})
    d = pd.DataFrame(rows)
    d = d[d.player.ne("")]
    # multi-team rows: keep the total line (max minutes) per player-season
    return d.sort_values("mp", ascending=False).drop_duplicates(["season", "player"]).reset_index(drop=True)


def _fit(X, y, seed=0):
    import lightgbm as lgb
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, reg_lambda=5.0, random_state=seed, verbose=-1)
    return m.fit(X.astype(float), y.astype(float))


def build() -> pd.DataFrame:
    tv = _seasons()
    tv["key"] = tv.player_name.map(norm_name)
    final = tv.sort_values("year").groupby("pid").tail(1)
    final = final[(final.GP >= 10) & (final.Min_per >= 20)]
    gl = pd.concat([parse_season(y) for y in range(2009, C.LAST_SEASON + 1)], ignore_index=True)
    gl = gl[gl.mp >= MIN_MP].copy()
    gl["key"] = gl.player.map(norm_name)
    first = gl.sort_values("season").drop_duplicates("key")  # first G League season with real minutes
    # pair: final college season 1-3 years before the first G League season, unique name in that window
    cand = first.merge(final[["key", "year", "tidx", "pick", *PROFILE]], on="key", how="inner")
    cand = cand[(cand.year <= cand.season - 1) & (cand.year >= cand.season - 3)]
    cand = cand[cand.pick.isna()]  # draftees are scored, never trained on
    cand = cand.sort_values("year").drop_duplicates("key", keep="last")
    cand["log_mp"] = np.log1p(cand.mp)
    print(f"college -> G League pairs: {len(cand)} ({cand.season.min()}-{cand.season.max()})")

    table = pd.read_parquet(C.PROC / "draft_table.parquet", columns=KEYS + ["torvik_idx"])
    table = table[table.torvik_idx.notna()].copy(); table["tidx"] = table.torvik_idx.astype(int)
    rows = table.merge(tv[["tidx", *PROFILE]], on="tidx", how="left")
    targets = {"gl_pred_ws48": "g_ws48", "gl_pred_per": "g_per", "gl_pred_ts": "g_ts", "gl_pred_usg": "g_usg", "gl_pred_mp": "log_mp"}
    out = []
    for y, g in rows.groupby("draft_year"):
        fit = cand[cand.season <= y]
        if len(fit) < 300:
            continue
        rec = g[KEYS].copy()
        for col, tgt in targets.items():
            ok = fit[tgt].notna()
            rec[col] = _fit(fit.loc[ok, PROFILE], fit.loc[ok, tgt]).predict(g[PROFILE].astype(float))
        out.append(rec)
        print(f"{y}: {len(g)} draftees scored, fit on {len(fit)} pairs", flush=True)
    res = pd.concat(out, ignore_index=True)
    # a draftee's OWN pre-draft G League line (Ignite, Select Team, veterans): the season ends before draft night
    tab = pd.read_parquet(C.PROC / "draft_table.parquet", columns=KEYS)
    own = tab.merge(gl, on="key", how="inner")
    own = own[(own.season <= own.draft_year) & (own.season >= own.draft_year - 1)].sort_values("season").drop_duplicates(KEYS, keep="last")
    own = own[KEYS].assign(gl_own_per=own.g_per.values, gl_own_ts=own.g_ts.values, gl_own_ws48=own.g_ws48.values,
                           gl_own_mp=own.mp.values, gl_own_age=own.g_age.values)
    print(f"draftees with their own pre-draft G League season: {len(own)}")
    res = res.merge(own, on=KEYS, how="outer")
    return res.drop_duplicates(KEYS)


def load_gleague() -> pd.DataFrame:
    return build()


if __name__ == "__main__":
    f = build()
    t = pd.read_parquet(C.PROC / "draft_table.parquet", columns=KEYS + ["source", "war5"]).merge(f, on=KEYS, how="left")
    col = t[t.source == "college"]
    print("coverage (college rows) by year:", col.groupby("draft_year").gl_pred_ws48.apply(lambda s: s.notna().mean()).round(2).to_dict())
    from scipy.stats import spearmanr
    for c in [c for c in f.columns if c.startswith("gl_")]:
        ok = col[c].notna() & (col.draft_year >= 2013)
        print(f"{c:14s} rho with war5 (2013+): {spearmanr(col.loc[ok, c], col.loc[ok, 'war5']).correlation:+.3f}")
