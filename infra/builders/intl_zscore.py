"""International / G League production standardised within the player's own league-season (prefix iz_).

A raw per-36 line is not comparable across leagues; an in-context model with ~10 internationals per draft class
cannot learn the translation on its own. Here every stat of a draftee's last pre-draft pro season is z-scored against
all players of that league-season with >= MIN_MP minutes (basketball-reference international totals and G League
season totals cached by infra/builders/intl.py). "How dominant was he relative to his peers, at his age" -- the same
idea as a college BPM being relative to D1. League strength itself stays a separate feature (i_league_level).

Leakage: standardisation uses only that season's league population; the season ended before the draft. No NBA data.
"""

import numpy as np
import pandas as pd

from infra.config import PROC, ROOT, TARGET
from infra.dataset import norm_name

EXT = ROOT / "data" / "external"
RAW = EXT / "intl_raw"
MIN_MP = 200
STATS = ["pts", "trb", "ast", "stl", "blk", "tov", "fga", "fg3a", "fta"]


def _population() -> pd.DataFrame:
    b = pd.read_parquet(RAW / "bbref_intl_totals.parquet")
    g = pd.read_parquet(RAW / "gleague_seasons.parquet")
    g["league"] = "GLG"
    p = pd.concat([b, g], ignore_index=True)
    for c in STATS + ["g", "mp", "fg", "ft", "age"]:
        p[c] = pd.to_numeric(p.get(c), errors="coerce")
    p = p[p.mp >= MIN_MP].copy()
    per36 = 36.0 / p.mp
    for c in STATS:
        p[f"{c}_36"] = p[c] * per36
    p["mpg"] = p.mp / p.g
    p["ts"] = p.pts / (2 * (p.fga + 0.44 * p.fta)).replace(0, np.nan)
    p["usg_36"] = (p.fga + 0.44 * p.fta + p.tov) * per36
    p["eff_36"] = (p.pts + p.trb + p.ast + p.stl + p.blk - p.tov - (p.fga - p.fg) - (p.fta - p.ft)) * per36
    p["key"] = p.player.map(norm_name)
    return p


Z_COLS = ["pts_36", "trb_36", "ast_36", "stl_36", "blk_36", "tov_36", "fg3a_36", "fta_36", "ts", "usg_36", "eff_36", "mpg", "age"]


def load_intl_zscore() -> pd.DataFrame:
    """One row per (key, draft_year) for every draftee whose last pre-draft pro season is in the population."""
    i = pd.read_parquet(EXT / "intl_prospects.parquet")
    i = i[i.i_has_pro.fillna(0) > 0][["key", "draft_year", "i_league", "i_season", "i_age_season"]].copy()
    i["i_season"] = pd.to_numeric(i.i_season, errors="coerce")
    p = _population()
    grp = p.groupby(["league", "season"])
    z = p[["league", "season", "key"] + Z_COLS].copy()
    for c in Z_COLS:
        mu, sd = grp[c].transform("mean"), grp[c].transform("std").replace(0, np.nan)
        z[f"iz_{c}"] = (p[c] - mu) / sd
    z["iz_pct_eff"] = grp["eff_36"].rank(pct=True)                     # percentile of efficiency in his league-season
    z["iz_n_league"] = grp["eff_36"].transform("size")
    z = z.drop(columns=Z_COLS).rename(columns={"league": "i_league", "season": "i_season"})
    z = z.sort_values("iz_n_league", ascending=False).drop_duplicates(["i_league", "i_season", "key"])
    out = i.merge(z, on=["key", "i_league", "i_season"], how="left")
    # young-and-productive: production above league average, weighted by how far under 22 he was
    young = np.clip(22 - out.i_age_season, 0, None)
    out["iz_young_x_eff"] = out.iz_eff_36 * young
    out["iz_young_x_mpg"] = out.iz_mpg * young
    out["iz_young_x_usg"] = out.iz_usg_36 * young
    keep = ["key", "draft_year"] + [c for c in out.columns if c.startswith("iz_")]
    return out[keep].drop_duplicates(["key", "draft_year"])


if __name__ == "__main__":
    d = load_intl_zscore()
    print(d.shape, "| matched with z:", d.iz_eff_36.notna().sum())
    t = pd.read_parquet(PROC / "draft_table.parquet")[["key", "draft_year", "player", TARGET]]
    m = t.merge(d, on=["key", "draft_year"]).dropna(subset=["iz_eff_36"])
    print(m.sort_values("iz_young_x_eff", ascending=False).head(12)[["player", "draft_year", "iz_eff_36", "iz_mpg", "iz_age", "iz_young_x_eff", TARGET]].round(2).to_string(index=False))
