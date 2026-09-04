"""Download raw sources and parse them into tidy tables.

Sources (all pre-existing public data, downloaded once, idempotent):
  * barttorvik.com   -- every D1 college player-season since 2008 (pre-draft features)
  * basketball-reference.com -- draft results 2007-2026, per-season advanced stats 2008-2026
  * FiveThirtyEight RAPTOR -- per-season RAPTOR ratings and WAR, 1977-2022
"""

import concurrent.futures as cf
import re
import time

import numpy as np
import pandas as pd
import requests
from lxml import html

from nbadraft.config import DRAFT_YEARS, NBA_SEASONS, PROC, RAW, TORVIK_YEARS

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}
RAPTOR_URL = "https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-raptor/historical_RAPTOR_by_player.csv"

TORVIK_COLS = [
    "player_name", "team", "conf", "GP", "Min_per", "ORtg", "usg", "eFG", "TS_per", "ORB_per",
    "DRB_per", "AST_per", "TO_per", "FTM", "FTA", "FT_per", "twoPM", "twoPA", "twoP_per", "TPM",
    "TPA", "TP_per", "blk_per", "stl_per", "ftr", "yr", "ht", "num", "porpag", "adjoe", "pfr",
    "year", "pid", "hometown", "rec_rank", "ast_tov", "rim_made", "rim_att", "mid_made", "mid_att",
    "rim_pct", "mid_pct", "dunk_made", "dunk_att", "dunk_pct", "pick", "drtg", "adrtg", "dporpag",
    "stops", "bpm", "obpm", "dbpm", "gbpm", "mpg", "ogbpm", "dgbpm", "oreb", "dreb", "treb", "ast",
    "stl", "blk", "pts", "role", "unk65", "birthdate",
]


def _get(url, path, sleep=0.0):
    if path.exists() and path.stat().st_size > 0:
        return
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.content)
    time.sleep(sleep)


def download_all():
    with cf.ThreadPoolExecutor(8) as ex:
        for y in TORVIK_YEARS:
            ex.submit(_get, f"https://barttorvik.com/getadvstats.php?year={y}&csv=1", RAW / "torvik" / f"advstats_{y}.csv")
        ex.submit(_get, RAPTOR_URL, RAW / "raptor" / "historical_RAPTOR_by_player.csv")
    # bbref rate-limits (~20 req/min): sequential + polite sleep
    for y in DRAFT_YEARS:
        _get(f"https://www.basketball-reference.com/draft/NBA_{y}.html", RAW / "bbref" / f"draft_{y}.html", sleep=3.5)
    for y in NBA_SEASONS:
        _get(f"https://www.basketball-reference.com/leagues/NBA_{y}_advanced.html", RAW / "bbref" / f"advanced_{y}.html", sleep=3.5)


# --------------------------------------------------------------------------- parsers

def parse_torvik() -> pd.DataFrame:
    frames = [pd.read_csv(RAW / "torvik" / f"advstats_{y}.csv", header=None, names=TORVIK_COLS) for y in TORVIK_YEARS]
    t = pd.concat(frames, ignore_index=True)
    str_cols = {"player_name", "team", "conf", "yr", "ht", "hometown", "role", "birthdate"}
    for c in t.columns.difference(str_cols):
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t["birthdate"] = pd.to_datetime(t["birthdate"], errors="coerce")
    ht = t["ht"].astype(str).str.extract(r"(\d+)-(\d+)").astype(float)
    t["height_in"] = ht[0] * 12 + ht[1]
    t["class_year"] = t["yr"].map({"Fr": 1, "So": 2, "Jr": 3, "Sr": 4})
    t["n_college_seasons"] = t.groupby("pid")["year"].rank(method="first")
    return t


def _rows(path, table_id):
    doc = html.fromstring(path.read_bytes())
    return doc.xpath(f'//table[@id="{table_id}"]/tbody/tr[not(contains(@class,"thead"))]')


def _cell(tr, stat):
    c = tr.xpath(f'.//*[@data-stat="{stat}"]')
    return c[0] if c else None


def parse_drafts() -> pd.DataFrame:
    recs = []
    for y in DRAFT_YEARS:
        for tr in _rows(RAW / "bbref" / f"draft_{y}.html", "stats"):
            p = _cell(tr, "player")
            link = p.xpath(".//a/@href") if p is not None else []
            if not link:  # forfeited pick
                continue
            recs.append({
                "draft_year": y,
                "pick": int(_cell(tr, "pick_overall").text_content()),
                "team": _cell(tr, "team_id").text_content().strip(),
                "player": p.text_content().strip(),
                "bbref_id": re.search(r"/players/\w/(\w+)\.html", link[0]).group(1),
                "college": _cell(tr, "college_name").text_content().strip() or None,
            })
    return pd.DataFrame(recs)


def parse_nba_seasons() -> pd.DataFrame:
    recs = []
    for s in NBA_SEASONS:
        for tr in _rows(RAW / "bbref" / f"advanced_{s}.html", "advanced"):
            name = _cell(tr, "name_display")
            if name is None or not name.get("data-append-csv"):
                continue
            rec = {"season": s, "bbref_id": name.get("data-append-csv"), "team": _cell(tr, "team_name_abbr").text_content()}
            for k in ("age", "games", "mp", "bpm", "vorp", "ws"):
                rec[k] = pd.to_numeric(_cell(tr, k).text_content(), errors="coerce")
            recs.append(rec)
    d = pd.DataFrame(recs)
    # players traded mid-season have one row per team plus a total row; keep the total (max minutes)
    d = d.sort_values("mp", ascending=False).drop_duplicates(["season", "bbref_id"]).sort_values(["season", "bbref_id"])
    return d.reset_index(drop=True)


def parse_raptor() -> pd.DataFrame:
    r = pd.read_csv(RAW / "raptor" / "historical_RAPTOR_by_player.csv")
    r = r.rename(columns={"player_id": "bbref_id"})[["bbref_id", "season", "poss", "mp", "raptor_total", "war_total", "war_reg_season"]]
    return r


def build_tidy():
    PROC.mkdir(parents=True, exist_ok=True)
    for name, fn in [("torvik", parse_torvik), ("drafts", parse_drafts), ("nba_seasons", parse_nba_seasons), ("raptor", parse_raptor)]:
        df = fn()
        df.to_parquet(PROC / f"{name}.parquet", index=False)
        print(f"{name:12s} {len(df):7d} rows")


if __name__ == "__main__":
    download_all()
    build_tidy()
