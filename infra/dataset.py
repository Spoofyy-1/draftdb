"""Join draftees to their final college season (Torvik), to extra pre-draft sources, and to the WAR target.

Only information available on draft night is used as a feature. Players without a Torvik row or a hoopR season line
(international / non-D1) are kept in the table with `modelled=False` unless they have a pre-draft pro season.
"""

import re
import unicodedata

import numpy as np
import pandas as pd

from infra import external
from infra.config import DRAFT_DAY, PROC

NUMERIC_FEATURES = [
    "GP", "Min_per", "ORtg", "usg", "eFG", "TS_per", "ORB_per", "DRB_per", "AST_per", "TO_per",
    "FT_per", "twoP_per", "TP_per", "blk_per", "stl_per", "ftr", "porpag", "adjoe", "pfr", "rec_rank",
    "ast_tov", "rim_pct", "mid_pct", "dunk_made", "drtg", "adrtg", "dporpag", "stops", "bpm", "obpm",
    "dbpm", "gbpm", "mpg", "ogbpm", "dgbpm", "oreb", "dreb", "treb", "ast", "stl", "blk", "pts",
    "TPA_pg", "FTA_pg", "height_in", "class_year", "n_college_seasons", "age_at_draft",
]
CATEGORICAL_FEATURES = ["conf", "role"]
# Published pure feature set ("ALL" in the tournament). Mock history is kept as an explicitly labelled pre-draft
# variant; sportsbook odds and scouting grades did not improve the 2011-2018 validation score.
_OPTIONAL = external.optional_columns()
FEATURES = NUMERIC_FEATURES + external.EXTERNAL_FEATURES + [c for c in _OPTIONAL if not c.startswith(("mo_", "odds_", "sc_"))]
# This reproduces the tournament's validated "all old groups + momentum" configuration, including sparse legacy columns.
_ALL_OPTIONAL = external.optional_columns(min_coverage=0)
_NONNUMERIC_AFTER_MERGE = {"g_dob_imputed", "g_first_game_censored", "g_hoopr_matched"}
_JASONG = list(external.JASONG_NUM.values())
_J_BIO = ["rsci", "j_height", "j_weight", "j_age"]
_J_TEAM = ["sos", "j_wins", "j_losses"]
_J_BOX = [v for v in _JASONG if v.startswith("j_") and v not in ("j_height", "j_weight", "j_age", "j_wins", "j_losses")
          and not any(v.startswith(p) for p in ("j_dunk", "j_rim", "j_fg_", "j_astd", "j_pct"))]
_J_SHOT = ["dunks_per_min", "j_dunks", "j_dunk_vs_rim", "j_dunks_unast", "j_rim_unast_100", "pct_rim", "j_fg_rim",
           "j_astd_rim", "j_pct_mid", "j_fg_mid", "j_astd_mid", "j_pct_3", "j_astd_3", "pct_astd"]
_J_AAU = [v for v in _JASONG if v.startswith("aau_")]
_J_EVENT = [v for v in _JASONG if v.startswith("ev_")]
_ORDERED_PREFIXES = ("iz_", "c_", "is_", "h_", "mm_", "t_", "g_", "ec_", "bwb_", "acad_", "tr_", "sh_", "mo_")
MOMENTUM_FEATURES = list(dict.fromkeys(
    NUMERIC_FEATURES + external.TRAJECTORY_FEATURES + external.PHYSICAL_FEATURES + list(external.AYUSH_BOX.values())
    + _J_BIO + _J_TEAM + _J_BOX + _J_SHOT + _J_AAU + _J_EVENT + external.INTL_PRO + external.INTL_FIBA
    + [c for p in _ORDERED_PREFIXES for c in _ALL_OPTIONAL if c.startswith(p) and c not in _NONNUMERIC_AFTER_MERGE]
))
# Narrow set for the classification member of the stack: production, trajectory, physicals, FIBA youth, standardised
# international line and Torvik context (validated on 2013-2017 as the best classifier feature set).
LEAN_FEATURES = ["bpm", "obpm", "dbpm", "porpag", "dporpag", "adjoe", "adrtg", "usg", "TS_per", "age_at_draft", "height_in", "rec_rank", "class_year",
                 "eFG", "ORB_per", "DRB_per", "AST_per", "TO_per", "FT_per", "twoP_per", "TP_per", "blk_per", "stl_per", "ftr", "ast_tov",
                 *external.TRAJECTORY_FEATURES, *external.PHYSICAL_FEATURES, *external.INTL_FIBA,
                 *[c for c in external.optional_columns() if c.startswith(("iz_", "t_"))]]
MARKET_FEATURE = "pick"  # actual draft slot; only used by the explicit "+market" model variants


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s.lower())
    return re.sub(r"[^a-z]", "", s)


def _same_school(a, b) -> bool:
    a, b = norm_name(a).replace("state", "st"), norm_name(b).replace("state", "st")
    return bool(a) and bool(b) and (a[:5] in b or b[:5] in a)


def _last_name(s: str) -> str:
    """Last real name token: suffixes like Jr. / II normalise to '' and would otherwise match any other Jr."""
    toks = [t for t in s.split() if norm_name(t)]
    return norm_name(toks[-1]) if toks else ""


def match_torvik(drafts: pd.DataFrame, torvik: pd.DataFrame) -> pd.DataFrame:
    """Return drafts with a `torvik_idx` column (index into torvik) or NaN.

    1) Torvik tags each player-season with his eventual draft pick: match on
       (draft_year == season, pick) + same last name. Handles nicknames (Bam Adebayo, Wes Johnson).
    2) Otherwise same normalised full name, season within [draft_year-2, draft_year]; same-name
       collisions (two Ryan Andersons) are resolved by college name, then most recent season.
    """
    t = torvik.assign(key=torvik.player_name.map(norm_name))
    d = drafts.assign(key=drafts.player.map(norm_name))
    idx = pd.Series(np.nan, index=d.index)

    tp = t[t.pick.notna()].reset_index()
    cand = d.reset_index().merge(tp, left_on=["draft_year", "pick"], right_on=["year", "pick"], suffixes=("", "_t"))
    cand = cand[[_last_name(a) == _last_name(b) for a, b in zip(cand.player, cand.player_name)]]
    idx.loc[cand["index"].values] = cand["index_t"].values

    todo = d[idx.isna()].reset_index()
    cand = todo.merge(t.reset_index(), on="key", suffixes=("", "_t"))
    cand = cand[(cand.year <= cand.draft_year) & (cand.year >= cand.draft_year - 2)]
    cand["score"] = 2 * np.array([_same_school(a, b) for a, b in zip(cand.college.fillna(""), cand.team)]) + (cand.year - cand.draft_year) * 0.1
    best = cand.sort_values("score").groupby("index")["index_t"].last()
    idx.loc[best.index] = best.values
    return drafts.assign(torvik_idx=idx)


def build_table() -> pd.DataFrame:
    drafts = pd.read_parquet(PROC / "drafts.parquet")
    torvik = pd.read_parquet(PROC / "torvik.parquet")
    target = pd.read_parquet(PROC / "target.parquet")

    d = match_torvik(drafts, torvik)
    feat = torvik.reindex(d.torvik_idx.values).reset_index(drop=True)
    feat["TPA_pg"] = feat.TPA / feat.GP
    feat["FTA_pg"] = feat.FTA / feat.GP
    draft_date = pd.to_datetime(d.draft_year.astype(str) + "-" + DRAFT_DAY).values
    feat["age_at_draft"] = (draft_date - feat.birthdate.values) / np.timedelta64(365, "D") if len(feat) else np.nan
    feat["age_at_draft"] = feat["age_at_draft"].astype(float)
    traj = external.torvik_trajectory(d, torvik, d.torvik_idx).reset_index(drop=True)

    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["player_name", "team"]
    table = pd.concat([d.reset_index(drop=True), feat[cols].rename(columns={"team": "college_team"}), traj], axis=1)
    table["key"] = table.player.map(norm_name)
    for src in [external.ayush(norm_name), external.jasong(norm_name), external.intl(), *external.optional_sources()]:
        table = table.merge(src, on=["key", "draft_year"], how="left")
    # A draftee is modelled if he has a final college season (Torvik, or -- for the pre-Torvik 2003-2007 classes and Torvik
    # misses -- a hoopR/ESPN season line, feat_hoopr h_gp) or a pre-draft pro season abroad / in the G League.
    # a hoopR line counts as a college season only with at least 5 games (the 2003-04 ESPN feed has 1-game stubs)
    has_college = d.torvik_idx.notna().values | ((table["h_gp"] >= 5).values if "h_gp" in table else False)
    table["source"] = np.where(has_college, "college", np.where(table.i_has_pro.fillna(0) > 0, "intl", "none"))
    table["modelled"] = table.source != "none"
    table = table.merge(target, on=["bbref_id", "draft_year"], how="left")

    cov = table.groupby("draft_year").agg(picks=("pick", "size"), college=("source", lambda s: (s == "college").sum()),
                                          intl=("source", lambda s: (s == "intl").sum()), modelled=("modelled", "sum"))
    print(cov.T.to_string())
    print(f"modelled {table.modelled.sum()} of {len(table)} draftees ({table.modelled.mean():.1%}): "
          f"{(table.source == 'college').sum()} college, {(table.source == 'intl').sum()} international / G League")
    print("unmodelled:", table.loc[~table.modelled, "player"].tolist())
    return table


if __name__ == "__main__":
    tbl = build_table()
    tbl.to_parquet(PROC / "draft_table.parquet", index=False)
