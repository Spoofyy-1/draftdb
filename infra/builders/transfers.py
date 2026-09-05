"""College transfers inferred from the Torvik player-season table (pure pandas, no download).

load_transfers() -> frame keyed by (key, draft_year) for every draftee with a Torvik pid, prefix tr_:
  tr_transferred            1 if his team changed between two consecutive Torvik seasons up to the draft year, else 0
  tr_n_transfers            number of such changes
  tr_up_level               1 if the last transfer went to a stronger conference (see below), else 0
  tr_conf_strength_delta    strength(new conference) - strength(old conference)
  tr_bpm_delta, tr_usg_delta, tr_ts_delta, tr_mpg_delta
                            first season at the new team minus last season at the old team (last transfer)
  tr_seasons_at_final_team  consecutive seasons at the team of his final pre-draft season
Players who never transferred: tr_transferred = 0, everything else NaN (tr_seasons_at_final_team = all his seasons).

Conference strength = mean of (adjoe - adrtg) over that conference's player-seasons in the season *before* the first season
at the new team -- both conferences measured in the same, already completed season, so it is what was knowable when the
transfer happened. A season gap between the two teams (redshirt / sit-out year) still counts as a transfer. Only seasons
<= draft_year are used; Torvik's own draft-pick tag is used by infra.dataset.match_torvik purely to identify the player.
"""

import numpy as np
import pandas as pd

from infra.config import PROC
from infra.dataset import match_torvik, norm_name

KEYS = ["key", "draft_year"]
STATS = {"bpm": "bpm", "usg": "usg", "TS_per": "ts", "mpg": "mpg"}


def _pids() -> pd.DataFrame:
    """key, draft_year, pid for every draftee with a Torvik row (draft_table's torvik_idx when present, else match_torvik)."""
    torvik = pd.read_parquet(PROC / "torvik.parquet", columns=["pid"])
    if (PROC / "draft_table.parquet").exists():
        t = pd.read_parquet(PROC / "draft_table.parquet", columns=KEYS + ["torvik_idx"])
    else:
        d = pd.read_parquet(PROC / "drafts.parquet")
        t = match_torvik(d, pd.read_parquet(PROC / "torvik.parquet")).assign(key=d.player.map(norm_name))[KEYS + ["torvik_idx"]]
    t = t.dropna(subset=["torvik_idx"])
    t["pid"] = torvik.pid.reindex(t.torvik_idx.astype(int).values).values
    return t.dropna(subset=["pid"]).drop_duplicates(KEYS)[KEYS + ["pid"]]


def load_transfers() -> pd.DataFrame:
    tv = pd.read_parquet(PROC / "torvik.parquet", columns=["pid", "year", "team", "conf", "adjoe", "adrtg"] + list(STATS))
    strength = (tv.adjoe - tv.adrtg).groupby([tv.conf, tv.year]).mean()  # (conf, season) -> mean adjoe - adrtg
    ids = _pids()
    seasons = tv[tv.pid.isin(ids.pid)].sort_values(["pid", "year"])
    rows = []
    for r in ids.itertuples(index=False):
        s = seasons[(seasons.pid == r.pid) & (seasons.year <= r.draft_year)]
        if s.empty:
            continue
        team = s.team.map(norm_name).values
        change = np.flatnonzero(team[1:] != team[:-1]) + 1  # positions of the first season at a new team
        f = {"key": r.key, "draft_year": r.draft_year, "tr_transferred": int(len(change) > 0), "tr_n_transfers": len(change),
             "tr_seasons_at_final_team": len(team) - (change[-1] if len(change) else 0)}
        if len(change):
            new, old = s.iloc[change[-1]], s.iloc[change[-1] - 1]
            ref = new.year - 1  # the completed season before he arrived
            d = strength.get((new.conf, ref), np.nan) - strength.get((old.conf, ref), np.nan)
            f["tr_conf_strength_delta"] = d
            f["tr_up_level"] = float(d > 0) if pd.notna(d) else np.nan
            for src, dst in STATS.items():
                f[f"tr_{dst}_delta"] = new[src] - old[src]
        rows.append(f)
    cols = KEYS + ["tr_transferred", "tr_n_transfers", "tr_up_level", "tr_conf_strength_delta"] + [f"tr_{v}_delta" for v in STATS.values()] + ["tr_seasons_at_final_team"]
    return pd.DataFrame(rows).reindex(columns=cols)


if __name__ == "__main__":
    out = load_transfers()
    print(out.shape, out.tr_transferred.value_counts().to_dict())
    print(out.describe().T.round(2).to_string())
