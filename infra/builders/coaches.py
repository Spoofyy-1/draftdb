"""Head coach and program context (columns co_*) for every college draftee, keyed by (key, draft_year).

Source: sports-reference's coaches-by-season table, one page per season (data/external/coaches/coaches_<YYYY>.html,
season YYYY = the campaign that ends in draft year YYYY). Everything is public before draft night: the coach's tenure
and career record, his tournament history, the team's season record and AP rankings. On top of that, two causal track
records are computed from earlier draft classes only, with each earlier class's WAR counted through season Y (what was
knowable on draft night Y): how the coach's and the program's previous draftees fared in the NBA relative to their class
and relative to where they were picked.

  co_tenure_yrs       seasons the coach has been at the school (inclusive)
  co_car_games/wl     coach career games and win share to date; co_car_ncaa/sw16/ff/champ tournament history to date
  co_cur_wl/ncaa      same at the current school
  co_seas_wl          the draftee's final season record; co_seas_ap_pre/post AP rank (26 = unranked); co_seas_ncaa (made it)
  co_coach_prior_n, co_coach_prior_rank, co_coach_prior_vs_pick     coach's earlier draftees (2003..Y-1): count, mean
                       within-class WAR percentile (through season Y), mean (WAR percentile - pick percentile), both shrunk
                       toward 0.5 / 0 by n / (n + 3)
  co_prog_prior_n, co_prog_prior_rank, co_prog_prior_vs_pick        the same for the program

    from infra.builders.coaches import load_coaches      # builds from the cached pages
    python -m infra.builders.coaches                      # rebuild + coverage report
"""

import re

import numpy as np
import pandas as pd
from lxml import html

from infra import config as C
from infra.dataset import norm_name

EXT = C.ROOT / "data" / "external"
KEYS = ["key", "draft_year"]
SHRINK = 3.0


def _school_key(s: str) -> str:
    s = re.sub(r"\b(university|of|at|the|college)\b", "", str(s).lower())
    s = s.replace("state", "st").replace("&", "and")
    return re.sub(r"[^a-z]", "", s)


def parse_season(year: int) -> pd.DataFrame:
    path = EXT / "coaches" / f"coaches_{year}.html"
    if not path.exists():
        return pd.DataFrame()
    doc = html.parse(str(path))
    rows = []
    for tr in doc.xpath('//table[@id="coaches"]//tbody/tr[not(contains(@class,"thead"))]'):
        cell = {td.get("data-stat"): td.text_content().strip() for td in tr.xpath("./th|./td")}
        href = tr.xpath('.//*[@data-stat="coach"]//a/@href')
        if not cell.get("school"):
            continue
        rows.append({"season": year, "coach_id": href[0].rsplit("/", 1)[-1].replace(".html", "") if href else cell.get("coach"),
                     "school": cell["school"], **{k: v for k, v in cell.items() if k not in ("coach", "school", "DUMMY")}})
    d = pd.DataFrame(rows)
    num = ["wins_seas", "losses_seas", "ap_pre", "ap_post", "wins_cur", "losses_cur", "ncaa_cur", "wins_car", "losses_car", "ncaa_car", "sw16_car", "ff_car", "champ_car"]
    for c in num:
        d[c] = pd.to_numeric(d.get(c), errors="coerce")
    d["since_year"] = pd.to_numeric(d.get("since", pd.Series(dtype=str)).astype(str).str[:4], errors="coerce") + 1
    d["school_key"] = d.school.map(_school_key)
    return d


def _seasons() -> pd.DataFrame:
    return pd.concat([parse_season(y) for y in range(2003, C.LAST_SEASON + 1)], ignore_index=True)


def _match_school(college: str, keys: set[str]) -> str | None:
    k = _school_key(college)
    if k in keys:
        return k
    for cand in keys:  # prefix agreement (Utah St University -> utahst)
        if len(k) >= 5 and len(cand) >= 5 and (k.startswith(cand) or cand.startswith(k)):
            return cand
    return None


def build() -> pd.DataFrame:
    from tournament.layer1 import horizon_target

    table = pd.read_parquet(C.PROC / "draft_table.parquet", columns=["key", "draft_year", "bbref_id", "pick", "college", "college_team", "source"])
    seasons_war = pd.read_parquet(C.PROC / "season_war.parquet")
    co = _seasons()
    keys = set(co.school_key)
    school = table.college.where(table.college.notna(), table.college_team)
    table = table.assign(school_key=[_match_school(s, keys) if pd.notna(s) else None for s in school])
    m = table.merge(co, left_on=["school_key", "draft_year"], right_on=["school_key", "season"], how="left")

    out = m[KEYS].copy()
    out["co_tenure_yrs"] = m.draft_year - m.since_year + 1
    out["co_car_games"] = m.wins_car + m.losses_car
    out["co_car_wl"] = m.wins_car / out.co_car_games
    for c in ["ncaa_car", "sw16_car", "ff_car", "champ_car", "ncaa_cur"]:
        out[f"co_{c}"] = m[c].fillna(0).where(m.season.notna())
    out["co_cur_wl"] = m.wins_cur / (m.wins_cur + m.losses_cur)
    out["co_seas_wl"] = m.wins_seas / (m.wins_seas + m.losses_seas)
    out["co_seas_ap_pre"] = m.ap_pre.fillna(26).where(m.season.notna())
    out["co_seas_ap_post"] = m.ap_post.fillna(26).where(m.season.notna())
    out["co_seas_ncaa"] = m.get("ncaa_seas", pd.Series("", index=m.index)).fillna("").astype(str).str.len().gt(0).astype(float).where(m.season.notna())

    # causal NBA track records: earlier classes, WAR through the draft-night season
    hist = m[m.bbref_id.notna()][["bbref_id", "draft_year", "pick", "coach_id", "school_key"]].reset_index(drop=True)
    for col, prefix in [("coach_id", "co_coach_prior"), ("school_key", "co_prog_prior")]:
        n_col, rank_col, vs_col = f"{prefix}_n", f"{prefix}_rank", f"{prefix}_vs_pick"
        out[n_col], out[rank_col], out[vs_col] = np.nan, np.nan, np.nan
        for y in sorted(m.draft_year.unique()):
            prior = hist[(hist.draft_year < y) & (hist.draft_year >= 2003)].copy()
            if prior.empty:
                continue
            prior["war"] = horizon_target(prior, seasons_war, C.TARGET_SEASONS, through=y)
            prior["r_true"] = prior.groupby("draft_year").war.rank(pct=True)
            prior["r_pick"] = prior.groupby("draft_year")["pick"].rank(pct=True, ascending=False)
            g = prior.dropna(subset=[col]).groupby(col).agg(n=("war", "size"), r=("r_true", "mean"), v=("r_true", lambda s: (s - prior.loc[s.index, "r_pick"]).mean()))
            idx = m.index[(m.draft_year == y) & m[col].notna()]
            j = m.loc[idx, col].map(g.n).fillna(0)
            out.loc[idx, n_col] = j
            out.loc[idx, rank_col] = (m.loc[idx, col].map(g.r).fillna(0.5) * j + 0.5 * SHRINK) / (j + SHRINK)
            out.loc[idx, vs_col] = (m.loc[idx, col].map(g.v).fillna(0.0) * j) / (j + SHRINK)
    out.loc[m.season.isna(), [c for c in out.columns if c.startswith("co_")]] = np.nan
    return out.drop_duplicates(KEYS)


def load_coaches() -> pd.DataFrame:
    return build()


if __name__ == "__main__":
    f = build()
    t = pd.read_parquet(C.PROC / "draft_table.parquet", columns=KEYS + ["source", "college", "pick", "war5"]).merge(f, on=KEYS, how="left")
    col = t[t.source == "college"]
    print("coach matched for college draftees by year:")
    print(col.groupby("draft_year").co_car_wl.apply(lambda s: s.notna().mean()).round(2).to_string())
    print("unmatched colleges:", sorted(col[col.co_car_wl.isna()].college.dropna().unique())[:40])
    print(f[[c for c in f.columns if c.startswith("co_")]].describe().T.round(3).to_string())
