"""Join draftees to their final college season (Torvik) and to the WAR target.

Only information available on draft night is used as a feature. Players without a
Torvik row (international / non-D1) are kept in the table with `modelled=False`.
"""

import re
import unicodedata

import numpy as np
import pandas as pd

from nbadraft.config import DRAFT_DAY, PROC

NUMERIC_FEATURES = [
    "GP", "Min_per", "ORtg", "usg", "eFG", "TS_per", "ORB_per", "DRB_per", "AST_per", "TO_per",
    "FT_per", "twoP_per", "TP_per", "blk_per", "stl_per", "ftr", "porpag", "adjoe", "pfr", "rec_rank",
    "ast_tov", "rim_pct", "mid_pct", "dunk_made", "drtg", "adrtg", "dporpag", "stops", "bpm", "obpm",
    "dbpm", "gbpm", "mpg", "ogbpm", "dgbpm", "oreb", "dreb", "treb", "ast", "stl", "blk", "pts",
    "TPA_pg", "FTA_pg", "height_in", "class_year", "n_college_seasons", "age_at_draft",
]
CATEGORICAL_FEATURES = ["conf", "role"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MARKET_FEATURE = "pick"  # actual draft slot; only used by the explicit "+market" model variants


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s.lower())
    return re.sub(r"[^a-z]", "", s)


def _same_school(a, b) -> bool:
    a, b = norm_name(a).replace("state", "st"), norm_name(b).replace("state", "st")
    return bool(a) and bool(b) and (a[:5] in b or b[:5] in a)


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
    cand = cand[[norm_name(a.split()[-1]) == norm_name(b.split()[-1]) for a, b in zip(cand.player, cand.player_name)]]
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

    table = pd.concat([d.reset_index(drop=True), feat[FEATURES + ["player_name", "team"]].rename(columns={"team": "college_team"})], axis=1)
    table["modelled"] = d.torvik_idx.notna().values
    table = table.merge(target, on=["bbref_id", "draft_year"], how="left")

    cov = table.groupby("draft_year").agg(picks=("pick", "size"), college=("college", "count"), matched=("modelled", "sum"))
    print(cov.T.to_string())
    print(f"matched {table.modelled.sum()} of {table.college.notna().sum()} college draftees ({table.modelled.mean():.1%} of all picks)")
    unmatched = table[table.college.notna() & ~table.modelled]
    print("unmatched college players:", unmatched.player.tolist())
    return table


if __name__ == "__main__":
    tbl = build_table()
    tbl.to_parquet(PROC / "draft_table.parquet", index=False)
