"""Extra pre-draft features merged onto the draft table from data/external.

Every column is knowable on draft night: the player's own college trajectory, his measured physicals, and a second,
independently sourced college stat line. Sources are joined on (normalised name, draft year). Nothing here is a
model output, a mock-draft rank, or an NBA statistic.

  torvik_trajectory  the season before the final college season, and the change into it (Torvik, already downloaded)
  ayush              height, wingspan, weight, draft age + sports-reference college line (AyushBatra01/NBADraft)
  jasong             RSCI, SOS, hoop-math shot location, AAU and showcase-event lines (JasonG7234/NBA-Draft-Model)
"""

import numpy as np
import pandas as pd

from infra.config import RAW

EXT = RAW.parent / "external"

TRAJ_STATS = ["bpm", "obpm", "dbpm", "usg", "TS_per", "porpag", "mpg"]
TRAJECTORY_FEATURES = [f"prev_{s}" for s in TRAJ_STATS] + [f"d_{s}" for s in TRAJ_STATS] + ["career_bpm_mean", "career_bpm_max"]

PHYSICAL_FEATURES = ["wingspan_in", "weight_lb", "wing_minus_height", "draft_age_x"]
AYUSH_BOX = {"G": "a_G", "MP": "a_MP", "PTS": "a_PTS", "TS%": "a_TS", "eFG%": "a_eFG", "3PAr": "a_3PAr", "FTAr": "a_FTAr", "USG%": "a_USG",
             "AST/USG": "a_AST_USG", "AST/TO": "a_AST_TO", "PER": "a_PER", "OWS/40": "a_OWS40", "DWS/40": "a_DWS40", "WS/40": "a_WS40",
             "OBPM": "a_OBPM", "DBPM": "a_DBPM", "BPM": "a_BPM"}

JASONG_NUM = {"RSCI": "rsci", "SOS": "sos", "Height": "j_height", "Weight": "j_weight", "Draft Day Age": "j_age", "Wins": "j_wins", "Losses": "j_losses",
              "PER": "j_PER", "TS%": "j_TS", "3PAr": "j_3PAr", "FTr": "j_FTr", "ORB%": "j_ORB", "DRB%": "j_DRB", "AST%": "j_AST", "STL%": "j_STL",
              "BLK%": "j_BLK", "Stock%": "j_stock", "TOV%": "j_TOV", "USG%": "j_USG", "Offensive Load": "j_offload", "WS/40": "j_WS40",
              "OBPM": "j_OBPM", "DBPM": "j_DBPM", "BPM": "j_BPM", "OFF RTG": "j_ORTG", "DEF RTG": "j_DRTG", "Hands-On Buckets": "j_hob",
              "Pure Point Rating": "j_ppr", "3 Point Proficiency": "j_3pprof", "3 Point Confidence": "j_3pconf",
              "Adj OFF +/-": "j_adj_off", "Adj DEF +/-": "j_adj_def",
              "Dunks per Minute Played": "dunks_per_min", "# Dunks": "j_dunks", "Dunk vs Rim Shot Percentage": "j_dunk_vs_rim",
              "% Dunks Unassisted": "j_dunks_unast", "Unassisted Shots @ Rim /100Poss": "j_rim_unast_100", "% Shots @ Rim": "pct_rim",
              "FG% @ Rim": "j_fg_rim", "%Astd @ Rim": "j_astd_rim", "% Shots @ Mid": "j_pct_mid", "FG% @ Mid": "j_fg_mid", "%Astd @ Mid": "j_astd_mid",
              "% Shots @ 3": "j_pct_3", "%Astd @ 3": "j_astd_3", "% Assisted": "pct_astd",
              "AAU GP": "aau_gp", "AAU MIN": "aau_min", "AAU PTS": "aau_pts", "AAU FG%": "aau_fg", "AAU 3P%": "aau_3p", "AAU FT%": "aau_ft",
              "AAU TRB": "aau_trb", "AAU AST": "aau_ast", "AAU STL": "aau_stl", "AAU BLK": "aau_blk", "AAU TOV": "aau_tov",
              "Event GP": "ev_gp", "Event MIN": "ev_min", "Event PTS": "ev_pts", "Event FG%": "ev_fg", "Event 3P%": "ev_3p", "Event TRB": "ev_trb",
              "Event AST": "ev_ast", "Event STL": "ev_stl", "Event BLK": "ev_blk", "Event Placement": "ev_placement"}

# International / non-college pathway: the last pre-draft pro season (per-36 line, league level) and FIBA youth
# national-team tournaments (which many US college players also have). Built by experiments/intl_build.py from
# basketball-reference international, G League, FIBA archive and EuroLeague NGT pages -- all seasons end before the draft.
INTL_PRO = ["i_has_pro", "i_league_level", "i_top_level", "i_n_comps", "i_age_season", "i_gp", "i_mpg", "i_pts_36", "i_trb_36", "i_orb_36", "i_ast_36",
            "i_stl_36", "i_blk_36", "i_tov_36", "i_pir_36", "i_fga_36", "i_fg3a_36", "i_fta_36", "i_fg_pct", "i_fg3_pct", "i_ft_pct", "i_fg3a_rate",
            "i_fta_rate", "i_ts", "i_efg", "i_ast_tov", "i_top_mpg", "i_top_pir_36"]
INTL_FIBA = ["i_fiba_youth_events", "i_fiba_max_level", "i_fiba_u16", "i_fiba_u17wc", "i_fiba_u18", "i_fiba_u19wc", "i_fiba_u20", "i_fiba_best_rank",
             "i_fiba_last_age", "i_fiba_gp", "i_fiba_mpg", "i_fiba_pts_36", "i_fiba_trb_36", "i_fiba_ast_36", "i_fiba_stl_36", "i_fiba_blk_36",
             "i_fiba_pir_36", "i_fiba_ts", "i_fiba_fg3a_rate", "i_fiba_ft_pct", "i_fiba_last_pts_36", "i_fiba_last_eff_36", "i_fiba_best_effpg",
             "i_ngt", "i_ngt_last_age", "i_ngt_pir_36", "i_ngt_pts_36"]
INTL_FEATURES = INTL_PRO + INTL_FIBA

# Further sources are materialised by experiments/materialize.py as data/external/feat_<name>.parquet, keyed by
# (key, draft_year), every column carrying that source's prefix. Whatever is present is merged; the feature registry in
# experiments/features.py groups them by prefix. Prefixes: c_ combine, k_ kaggle college, s_ SCORE, is_ ianstack,
# h_ hoopR season boxes, mm_ march madness, t_ torvik context, g_ game-level context.
OPTIONAL_PREFIXES = ["c_", "k_", "s_", "is_", "h_", "mm_", "t_", "g_"]

EXTERNAL_FEATURES = TRAJECTORY_FEATURES + PHYSICAL_FEATURES + list(AYUSH_BOX.values()) + list(JASONG_NUM.values()) + INTL_FEATURES


def optional_sources() -> list[pd.DataFrame]:
    """Every materialised feat_*.parquet in data/external, numeric columns only, keyed by (key, draft_year)."""
    out = []
    for p in sorted(EXT.glob("feat_*.parquet")):
        f = pd.read_parquet(p)
        num = [c for c in f.columns if c not in ("key", "draft_year") and pd.api.types.is_numeric_dtype(f[c])]
        out.append(f[["key", "draft_year"] + num].drop_duplicates(["key", "draft_year"]))
    return out


MIN_COVERAGE = 0.02  # a column observed for under 2% of the ~1,200 modelled draftees is cost without signal


def optional_columns(min_coverage: float = MIN_COVERAGE) -> list[str]:
    """Column names the optional sources contribute (pre-draft only: the mock-draft source is consensus and stays out;
    near-empty columns are dropped)."""
    cols = []
    for p in sorted(EXT.glob("feat_*.parquet")):
        if p.stem == "feat_mock":
            continue
        f = pd.read_parquet(p)
        cols += [c for c in f.columns if c not in ("key", "draft_year") and pd.api.types.is_numeric_dtype(f[c]) and f[c].notna().sum() >= min_coverage * 1200]
    return list(dict.fromkeys(cols))

_num = lambda s: pd.to_numeric(s, errors="coerce")


def torvik_trajectory(drafts: pd.DataFrame, torvik: pd.DataFrame, torvik_idx: pd.Series) -> pd.DataFrame:
    """Previous-season stats, the change into the final season, and career BPM, aligned to `drafts` row order."""
    cur = torvik.reindex(torvik_idx.values).reset_index(drop=True)
    tv = torvik.drop_duplicates(["pid", "year"]).set_index(["pid", "year"])
    prev = tv.reindex(pd.MultiIndex.from_arrays([cur.pid, cur.year - 1])).reset_index(drop=True)
    career = torvik.merge(pd.DataFrame({"pid": cur.pid, "last": cur.year}).dropna(), on="pid", how="inner")
    career = career[career.year <= career["last"]].groupby("pid").bpm.agg(["mean", "max"]).reindex(cur.pid.values)
    out = pd.DataFrame(index=drafts.index)
    for s in TRAJ_STATS:
        out[f"prev_{s}"] = prev[s].values
        out[f"d_{s}"] = (cur[s] - prev[s]).values
    out["career_bpm_mean"], out["career_bpm_max"] = career["mean"].values, career["max"].values
    return out


def ayush(norm_name) -> pd.DataFrame:
    a = pd.read_csv(EXT / "ayush_draft_players.csv")
    out = pd.DataFrame({"key": a.Name.map(norm_name), "draft_year": a.Year, "wingspan_in": a.Wingspan, "weight_lb": a.Weight, "draft_age_x": a["Draft Age"]})
    out["wing_minus_height"] = a.Wingspan - a.Height
    for src, dst in AYUSH_BOX.items():
        out[dst] = _num(a[src])
    return out.drop_duplicates(["key", "draft_year"])


def intl() -> pd.DataFrame:
    i = pd.read_parquet(EXT / "intl_prospects.parquet")
    out = i[["key", "draft_year"] + [c for c in INTL_FEATURES if c in i.columns]].copy()
    for c in INTL_FEATURES:
        out[c] = _num(out[c]) if c in out.columns else np.nan
    out["i_has_pro"] = out["i_has_pro"].fillna(0)
    return out.drop_duplicates(["key", "draft_year"])


def jasong(norm_name) -> pd.DataFrame:
    j = pd.read_csv(EXT / "jasong_draft_db.csv", low_memory=False)
    out = pd.DataFrame({"key": j.Name.map(norm_name), "draft_year": _num(j.Season.astype(str).str[:4]) + 1})
    for src, dst in JASONG_NUM.items():
        out[dst] = _num(j[src]) if src in j.columns else np.nan
    return out.dropna(subset=["draft_year"]).drop_duplicates(["key", "draft_year"])
