"""Scouting text (columns sct_*): what the pre-draft Strengths / Weaknesses paragraphs say, and who the scout compared him to.

Reads the NBADraft.net profile captures that infra.builders.scouting cached (every one taken before the player's own draft
night). Two kinds of feature, both walk-forward:

  sct_pred, sct_pred_strengths, sct_pred_weaknesses   a TF-IDF (1-2 grams) ridge regression from the paragraph text to the
                                                       within-class Gaussian-ranked WAR, fit only on classes BEFORE the one
                                                       being scored (labels as known by that draft night) -- the scout's
                                                       words scored by what similar words meant for earlier prospects
  sct_words_total                                      length of the write-up
  sct_comp_war, sct_comp_n                             the "NBA comparison" player's mean WAR per season through the season
                                                       before the draft (a comparison to Jrue Holiday is a different statement
                                                       from a comparison to Jake Layman); NaN when the comp is not an NBA player

Only the write-up and the comp name are read; no grade, pick or outcome of the prospect himself enters a feature.

    from infra.builders.scouting_text import load_scouting_text
    python -m infra.builders.scouting_text
"""

import json
import re

import numpy as np
import pandas as pd
from scipy.stats import norm

from infra import config as C
from infra.builders.scouting import NDN_DIR, parse_profile
from infra.dataset import norm_name

KEYS = ["key", "draft_year"]
MIN_DOCS = 150


def _profiles() -> pd.DataFrame:
    rows = []
    for status in sorted(NDN_DIR.glob("*/_status.json")):
        year = int(status.parent.name)
        for key, rec in json.loads(status.read_text()).items():
            if rec.get("status") != "ok":
                continue
            page = status.parent / f"{rec['slug']}.html"
            if not page.exists():
                continue
            p = parse_profile(page.read_text())
            rows.append({"key": key, "draft_year": year, "strengths": p.get("strengths", "") or "", "weaknesses": p.get("weaknesses", "") or "",
                         "compares_to": p.get("compares_to", "") or ""})
    return pd.DataFrame(rows).drop_duplicates(KEYS)


def _labels(table: pd.DataFrame, seasons: pd.DataFrame, through: int) -> pd.Series:
    from tournament.layer1 import horizon_target
    rows = table[table.draft_year < through]
    war = horizon_target(rows, seasons, C.TARGET_SEASONS, through)
    s = pd.Series(war, index=rows.index)
    q = ((s.groupby(rows.draft_year).rank() - 0.5) / s.groupby(rows.draft_year).transform("count")).clip(0.01, 0.99)
    return pd.Series(norm.ppf(q), index=rows.index)


def _fit_predict(train_text, y, test_text, seed=0):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    m = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True, stop_words="english"), Ridge(alpha=3.0))
    return m.fit(train_text, y).predict(test_text)


def build() -> pd.DataFrame:
    prof = _profiles()
    table = pd.read_parquet(C.PROC / "draft_table.parquet", columns=KEYS + ["bbref_id", "player"])
    seasons = pd.read_parquet(C.PROC / "season_war.parquet")
    d = table.merge(prof, on=KEYS, how="inner").reset_index(drop=True)
    d["text"] = (d.strengths + " " + d.weaknesses).str.strip()
    out = d[KEYS].copy()
    out["sct_words_total"] = d.text.str.split().str.len().astype(float)
    for c in ["sct_pred", "sct_pred_strengths", "sct_pred_weaknesses"]:
        out[c] = np.nan
    for y in sorted(d.draft_year.unique()):
        lab = _labels(d, seasons, y)
        tr = d.loc[lab.index]
        tr = tr[tr.text.str.len() > 40]
        te = d[d.draft_year == y]
        if len(tr) < MIN_DOCS or te.empty:
            continue
        yv = lab.loc[tr.index].values
        out.loc[te.index, "sct_pred"] = _fit_predict(tr.text, yv, te.text)
        out.loc[te.index, "sct_pred_strengths"] = _fit_predict(tr.strengths, yv, te.strengths)
        out.loc[te.index, "sct_pred_weaknesses"] = _fit_predict(tr.weaknesses, yv, te.weaknesses)
        print(f"{y}: {len(te)} profiles scored from {len(tr)} earlier write-ups", flush=True)
    # comparison player's realised value, known on draft night: WAR per season through the season before the draft.
    # Names come from basketball-reference (2004+) and FiveThirtyEight's historical RAPTOR file (1977-2022), so comps to
    # players who retired before 2004 (Shawn Kemp, Mark Price ...) are valued too.
    from infra.builders.population import _nba_names
    names = pd.Series(_nba_names())
    raptor_raw = pd.read_csv(C.RAW / "raptor" / "historical_RAPTOR_by_player.csv", usecols=["player_name", "player_id", "season", "war_reg_season"])
    names = pd.concat([names, raptor_raw.drop_duplicates("player_id").set_index("player_id").player_name]).groupby(level=0).first()
    by_key = names.map(norm_name).reset_index().rename(columns={"index": "bbref_id", 0: "ckey"})
    by_key = by_key.drop_duplicates("ckey", keep=False)  # ambiguous names dropped
    war = pd.concat([raptor_raw.rename(columns={"player_id": "bbref_id", "war_reg_season": "war"})[["bbref_id", "season", "war"]],
                     seasons.loc[seasons.season > 2022, ["bbref_id", "season", "war"]]], ignore_index=True)
    comp = (d.compares_to.fillna("").str.replace(r"\(.*?\)", "", regex=True).str.replace(r"\bNico Van den Bogaerd.*$", "", regex=True)
            .str.replace(r"^(?:a |the )?(?:poor|rich) man'?s ", "", regex=True, case=False)
            .str.split(r"/|,| or | and |;", regex=True).str[0].str.strip().map(norm_name))
    m = pd.DataFrame({"ckey": comp, "draft_year": d.draft_year}).merge(by_key, on="ckey", how="left")
    vals, ns = [], []
    for cid, y in zip(m.bbref_id, m.draft_year):
        if pd.isna(cid):
            vals.append(np.nan); ns.append(np.nan); continue
        s = war[(war.bbref_id == cid) & (war.season <= y)]
        vals.append(s.war.mean() if len(s) else np.nan); ns.append(float(len(s)))
    out["sct_comp_war"], out["sct_comp_n"] = vals, ns
    print(f"scouting text for {len(out)} draftees; comp matched for {out.sct_comp_war.notna().sum()}")
    # The TF-IDF predictions carry no out-of-sample signal (rho 0.01-0.05 with WAR on 2013-2025) and only add noise for the
    # in-context models, so the published group keeps the write-up length and the comparison player's value.
    return out.drop(columns=["sct_pred", "sct_pred_strengths", "sct_pred_weaknesses"]).drop_duplicates(KEYS)


def load_scouting_text() -> pd.DataFrame:
    return build()


if __name__ == "__main__":
    f = build()
    t = pd.read_parquet(C.PROC / "draft_table.parquet", columns=KEYS + ["source", "war5"]).merge(f, on=KEYS, how="left")
    from scipy.stats import spearmanr
    h = t[t.draft_year.between(2013, 2025)]
    for c in [c for c in f.columns if c.startswith("sct_")]:
        ok = h[c].notna()
        print(f"{c:22s} n={ok.sum():4d} rho with war5 (2013+): {spearmanr(h.loc[ok, c], h.loc[ok, 'war5']).correlation:+.3f}")
