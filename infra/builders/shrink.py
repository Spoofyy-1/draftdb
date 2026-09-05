"""Minutes-shrunk production (prefix sh_): rate stats pulled toward the D1 average in proportion to how little a player
actually played, so 3 games of monster numbers (James Wiseman, 2020: 3 GP, BPM 13.8) do not read like a season.

    sh_stat = (minutes * stat + K * mean_stat) / (minutes + K)      K = 400 minutes

mean_stat is the minutes-weighted D1 mean of that stat in the SAME season (the season is over by draft night) -- no
outcome data, no fitting. Also emits sh_minutes and sh_weight = minutes / (minutes + K), the share of the shrunk value
that is the player's own evidence, so a model can see reliability explicitly.
"""

import pandas as pd

from infra.config import PROC, TARGET
from infra.dataset import match_torvik, norm_name

K = 400.0
STATS = ["bpm", "obpm", "dbpm", "usg", "TS_per", "eFG", "ORtg", "porpag", "adjoe", "adrtg", "dporpag", "AST_per", "TO_per", "ORB_per", "DRB_per",
         "stl_per", "blk_per", "TP_per", "FT_per", "ftr", "stops", "gbpm"]


def load_shrunk() -> pd.DataFrame:
    drafts = pd.read_parquet(PROC / "drafts.parquet")
    torvik = pd.read_parquet(PROC / "torvik.parquet")
    d = match_torvik(drafts, torvik)
    cur = torvik.reindex(d.torvik_idx.values).reset_index(drop=True)
    minutes = (cur.GP * cur.mpg).astype(float)
    # same-season D1 means, minutes-weighted, players with a real role (Min_per >= 20)
    pop = torvik[torvik.Min_per >= 20].assign(_m=lambda x: x.GP * x.mpg)
    means = pop.groupby("year").apply(lambda g: pd.Series({s: (g[s] * g._m).sum() / g._m.sum() for s in STATS if s in g}), include_groups=False)
    mu = means.reindex(cur.year.values).reset_index(drop=True)
    out = pd.DataFrame({"key": drafts.player.map(norm_name), "draft_year": drafts.draft_year, "sh_minutes": minutes.values})
    w = minutes / (minutes + K)
    out["sh_weight"] = w.values
    for s in STATS:
        if s in cur:
            out[f"sh_{s}"] = (w * cur[s] + (1 - w) * mu[s]).values
    return out[out.sh_minutes.notna()].drop_duplicates(["key", "draft_year"])


if __name__ == "__main__":
    s = load_shrunk()
    t = pd.read_parquet(PROC / "draft_table.parquet")[["key", "draft_year", "player", "bpm", "GP", TARGET]]
    m = t.merge(s, on=["key", "draft_year"])
    print(m[m.player.isin(["James Wiseman", "Anthony Edwards", "Onyeka Okongwu", "Zion Williamson", "Chet Holmgren"])][["player", "draft_year", "GP", "bpm", "sh_bpm", "sh_weight", TARGET]].round(2).to_string(index=False))
