"""Lineup-based impact from Hoop Explorer (columns he_*): RAPM, on/off, play-type profile, and the same numbers against
top-100 opponents. Seasons 2018-19 onward (draft classes 2019+), so the walk-forward models learn the columns only from
the classes that already carry them; earlier context rows are NaN.

Source: the public static bundle of hundredguaranteed/college-hoop-explorer-site (data/vendor/college_hoop_players_parts),
cached as data/external/hoopexplorer/part-*.js and parsed to players.parquet. Every number is computed from the college
season's play-by-play, all of it finished before draft night. Two views per player-season: All games and vs T100 opponents.

  he_rapm / he_orapm / he_drapm            regularised adjusted plus-minus (total, offence, defence; defence negative = good)
  he_rapm_prod / he_orapm_prod             production-weighted RAPM
  he_net_o / he_net_d, he_wowy_o / he_wowy_d  net points model and its with-or-without-you components
  he_onoff_net, he_onoff_raw               team net rating with him on court (adjusted / raw)
  he_usage, he_off_poss_pct, he_def_poss_pct, he_adj_opp_o, he_adj_opp_d, he_conf, he_gp
  he_style_<type>_pct / _adjpts            play-type share of usage and adjusted points for the main styles
  he_t100_*                                the same core impact numbers against top-100 opponents
  he_rapm_drop, he_orapm_drop              All minus vs-T100: how much the impact shrinks against real competition

    from infra.builders.hoopexplorer import load_hoopexplorer
    python -m infra.builders.hoopexplorer
"""

import glob
import io
import re

import numpy as np
import pandas as pd

from infra import config as C
from infra.dataset import _same_school, norm_name

EXT = C.ROOT / "data" / "external"
KEYS = ["key", "draft_year"]
CORE = {"rapm": "he_rapm", "off_adj_rapm": "he_orapm", "def_adj_rapm": "he_drapm", "rapm_prod": "he_rapm_prod", "off_adj_rapm_prod": "he_orapm_prod",
        "def_adj_rapm_prod": "he_drapm_prod", "net_pts_o": "he_net_o", "net_pts_d": "he_net_d", "net_pts_o_wowy": "he_wowy_o", "net_pts_d_wowy": "he_wowy_d",
        "on_off_net": "he_onoff_net", "on_off_raw_net": "he_onoff_raw", "off_usage": "he_usage", "off_team_poss_pct": "he_off_poss_pct",
        "def_team_poss_pct": "he_def_poss_pct", "off_adj_opp": "he_adj_opp_o", "def_adj_opp": "he_adj_opp_d", "gp": "he_gp",
        "off_adj_rtg": "he_off_adj_rtg", "def_adj_rtg": "he_def_adj_rtg", "on_off_adj_ppp": "he_on_off_ppp", "on_def_adj_ppp": "he_on_def_ppp"}
T100 = {"rapm": "he_t100_rapm", "off_adj_rapm": "he_t100_orapm", "def_adj_rapm": "he_t100_drapm", "net_pts_o": "he_t100_net_o", "net_pts_d": "he_t100_net_d",
        "off_usage": "he_t100_usage", "on_off_net": "he_t100_onoff_net"}
STYLES = ["transition", "attack_and_kick", "dribble_jumper", "pick_and_pop", "post_up", "rim_attack", "perimeter_sniper", "high_low", "backdoor_cut",
          "big_cut_and_roll", "put_back", "hits_cutter", "dribble_pull_up", "pnr_passer"]


def parse_bundle() -> pd.DataFrame:
    cache = EXT / "hoopexplorer" / "players.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    chunks = []
    for p in sorted(glob.glob(str(EXT / "hoopexplorer" / "part-*.js"))):
        for m in re.finditer(r'window\.COLLEGE_HOOP_PLAYERS_CSV\.push\("((?:[^"\\]|\\.)*)"\);', open(p).read()):
            chunks.append(bytes(m.group(1), "utf-8").decode("unicode_escape"))
    df = pd.read_csv(io.StringIO("".join(chunks)), low_memory=False)
    df.to_parquet(cache, index=False)
    return df


def build() -> pd.DataFrame:
    he = parse_bundle()
    he = he[he.player_name.notna()].copy()
    he["key"] = he.player_name.map(norm_name)
    he["draft_year"] = he.source_year.astype(int) + 1  # season 2024/25 (source_year 2024) is the 2025 class's final season
    he["he_conf"] = he.confidence.map({"Low": 0.0, "Medium": 1.0, "High": 2.0})
    style_cols = {}
    for s in STYLES:
        for suffix, dst in (("poss_pct_usg", "pct"), ("adj_pts", "adjpts")):
            c = f"style_{s}_{suffix}"
            if c in he.columns:
                style_cols[c] = f"he_style_{s}_{dst}"
    table = pd.read_parquet(C.PROC / "draft_table.parquet", columns=KEYS + ["college", "college_team"])
    school = table.college_team.where(table.college_team.notna(), table.college)
    table = table.assign(school=school)

    def pick(view: str, cols: dict) -> pd.DataFrame:
        v = he[he.source_view == view]
        m = table.merge(v, on=KEYS, how="inner")
        # same name twice in a season: keep the row whose team matches the draftee's school
        m["_ok"] = [_same_school(a, b) if pd.notna(a) else True for a, b in zip(m.school, m.team_name)]
        m = m.sort_values(["_ok", "gp"], ascending=False).drop_duplicates(KEYS)
        out = m[KEYS].copy()
        for src, dst in cols.items():
            out[dst] = pd.to_numeric(m[src], errors="coerce").values if src in m.columns else np.nan
        return out

    all_view = pick("All", {**CORE, **style_cols})
    all_view = all_view.merge(pick("All", {"he_conf": "he_conf"}) if "he_conf" in he.columns else all_view[KEYS], on=KEYS, how="left")
    t100 = pick("vs T100", T100)
    out = all_view.merge(t100, on=KEYS, how="left")
    out["he_rapm_drop"] = out.he_rapm - out.he_t100_rapm
    out["he_orapm_drop"] = out.he_orapm - out.he_t100_orapm
    print(f"hoop-explorer impact for {len(out)} draftees, classes {out.draft_year.min()}-{out.draft_year.max()}")
    return out.drop_duplicates(KEYS)


def load_hoopexplorer() -> pd.DataFrame:
    return build()


if __name__ == "__main__":
    f = build()
    t = pd.read_parquet(C.PROC / "draft_table.parquet", columns=KEYS + ["source", "war5", "player"]).merge(f, on=KEYS, how="left")
    col = t[t.source == "college"]
    print("coverage (college rows) by year:", col.groupby("draft_year").he_rapm.apply(lambda s: s.notna().mean()).round(2).to_dict())
    from scipy.stats import spearmanr
    h = col[col.draft_year.between(2019, 2025)]
    for c in [c for c in f.columns if c.startswith("he_") and not c.startswith("he_style")]:
        ok = h[c].notna()
        if ok.sum() > 50:
            print(f"{c:20s} n={ok.sum():4d} rho with war5 (2019-25): {spearmanr(h.loc[ok, c], h.loc[ok, 'war5']).correlation:+.3f}")
