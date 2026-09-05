"""Challenge -> response features (columns rs_*) from the prospect's final-season game log (hoopR player boxes).

Everybody measures how good a prospect was. These measure what happened to him when the game got harder, all of it
on film before draft night:

  rs_rematch_gs / rs_rematch_ts / rs_rematch_usg   second and third meetings with the same opponent minus the first
                                                    meeting (game score per 36, true shooting, usage): did he solve a
                                                    defence that had prepared for him? Shrunk toward 0 by n / (n + 3)
  rs_bounce_gs                                      game score in the game after a bottom-quartile game minus his season
                                                    mean: response to failure
  rs_tov_persist                                    lag-1 autocorrelation of turnover rate across games: do mistakes
                                                    cluster (repeated errors) or get corrected at once?
  rs_close_gs_delta                                 game score in close games (final margin <= 6) minus the rest
  rs_close_min_ratio                                minutes per close game / minutes per other game: coach trust when
                                                    it matters
  rs_gs_worst_q                                     mean game score of his worst quartile of games: the floor
  rs_min_trend                                      slope of minutes over the season (per 10 games): earned trust
  rs_n_games                                        games used

Population-relative versions are unnecessary: every column is a within-player contrast.

    from infra.builders.response import load_response
    python -m infra.builders.response
"""

import numpy as np
import pandas as pd

from infra import config as C

KEYS = ["key", "draft_year"]
SHRINK = 3.0


def _player(g: pd.DataFrame) -> dict:
    g = g.sort_values("game_date")
    g = g[g.played & (g.minutes >= 8)]
    out = {"rs_n_games": len(g)}
    if len(g) < 12:
        return out
    gs, ts, usg, mins = g.gs36.values, g.ts.values, g.usg.values, g.minutes.values
    # rematches: later meetings with the same opponent vs the first meeting that season
    first = g.drop_duplicates("opp_team_id", keep="first").set_index("opp_team_id")
    later = g[g.duplicated("opp_team_id", keep="first")]
    if len(later):
        d_gs = later.gs36.values - first.gs36.reindex(later.opp_team_id).values
        d_ts = later.ts.values - first.ts.reindex(later.opp_team_id).values
        d_usg = later.usg.values - first.usg.reindex(later.opp_team_id).values
        n = len(later)
        out["rs_rematch_gs"] = np.nanmean(d_gs) * n / (n + SHRINK)
        out["rs_rematch_ts"] = np.nanmean(d_ts) * n / (n + SHRINK)
        out["rs_rematch_usg"] = np.nanmean(d_usg) * n / (n + SHRINK)
        out["rs_rematch_n"] = n
    # bounce-back after a bad game
    q25 = np.nanquantile(gs, 0.25)
    after_bad = [gs[i + 1] for i in range(len(gs) - 1) if gs[i] <= q25]
    if len(after_bad) >= 3:
        out["rs_bounce_gs"] = np.nanmean(after_bad) - np.nanmean(gs)
    # repeated errors: lag-1 autocorrelation of turnover rate
    t = pd.Series(g.tov_pct.values)
    if t.notna().sum() >= 12 and t.std() > 0:
        out["rs_tov_persist"] = t.autocorr(lag=1)
    # close games: performance and coach trust
    close = g.margin.abs() <= 6
    if close.sum() >= 3 and (~close).sum() >= 3:
        out["rs_close_gs_delta"] = np.nanmean(gs[close.values]) - np.nanmean(gs[~close.values])
        out["rs_close_min_ratio"] = np.nanmean(mins[close.values]) / max(np.nanmean(mins[~close.values]), 1.0)
    out["rs_gs_worst_q"] = np.nanmean(np.sort(gs)[: max(3, len(gs) // 4)])
    x = np.arange(len(mins))
    out["rs_min_trend"] = np.polyfit(x, mins, 1)[0] * 10 if len(mins) >= 12 else np.nan
    return out


def build() -> pd.DataFrame:
    xw = pd.read_parquet(C.PROC / "crosswalk_hoopr.parquet")
    xw = xw[xw.athlete_id.notna() & (xw.confidence >= 0.8)]
    pg = pd.read_parquet(C.PROC / "player_game.parquet",
                         columns=["athlete_id", "season", "season_type", "game_date", "opp_team_id", "played", "minutes", "gs36", "ts", "usg", "tov_pct", "margin"])
    pg = pg[pg.athlete_id.isin(xw.athlete_id) & pg.season_type.isin([2, 3])]
    rows = []
    for r in xw.itertuples():
        g = pg[(pg.athlete_id == r.athlete_id) & (pg.season == r.final_season)]
        if g.empty:
            continue
        rows.append({"key": r.key, "draft_year": int(r.draft_year), **_player(g)})
    out = pd.DataFrame(rows).drop_duplicates(KEYS)
    print(f"response features for {len(out)} draftees")
    return out


def load_response() -> pd.DataFrame:
    return build()


if __name__ == "__main__":
    f = build()
    t = pd.read_parquet(C.PROC / "draft_table.parquet", columns=KEYS + ["source", "war5"]).merge(f, on=KEYS, how="left")
    col = t[t.source == "college"]
    print("coverage (college rows) by year:", col.groupby("draft_year").rs_n_games.apply(lambda s: s.notna().mean()).round(2).to_dict())
    from scipy.stats import spearmanr
    for c in [c for c in f.columns if c.startswith("rs_")]:
        ok = col[c].notna() & (col.draft_year >= 2013)
        print(f"{c:20s} n={ok.sum():4d} rho with war5 (2013+): {spearmanr(col.loc[ok, c], col.loc[ok, 'war5']).correlation:+.3f}")
