"""Biography features (columns bio_*) from basketball-reference player pages (data/external/bbref_players/<bbref_id>.html).

Facts about the person that exist long before draft night and that no box score carries:

  bio_nba_father        father listed as a relative in the basketball-reference database (a pro)
  bio_nba_relative_n    relatives listed (father, brother, uncle, cousin ...)
  bio_shoots_left       shooting hand
  bio_born_us           born in the United States
  bio_n_high_schools    high schools listed: prep-school hopping / reclassification
  bio_prep_academy      attended a national prep academy (Montverde, IMG, Oak Hill, Findlay, La Lumiere, Sunrise Christian, Prolific,
                        Huntington, Brewster, Wasatch, Link, AZ Compass, Hillcrest, Sierra Canyon, Overtime Elite ...)

Only the meta block of each page is read: draft line, NBA debut and career tables are never parsed.

    from infra.builders.bio import load_bio
    python -m infra.builders.bio
"""

import re

import pandas as pd
from lxml import html

from infra import config as C

EXT = C.ROOT / "data" / "external"
KEYS = ["key", "draft_year"]
ACADEMIES = ["montverde", "img academy", "oak hill", "findlay", "la lumiere", "sunrise christian", "prolific prep", "huntington prep",
             "brewster", "wasatch", "link academy", "compass prep", "hillcrest", "sierra canyon", "overtime elite", "nba academy",
             "napa", "bishop gorman", "dematha", "st. benedict", "mater dei", "rancho christian", "combine academy", "spire"]


def _meta(path) -> str:
    doc = html.parse(str(path))
    m = doc.xpath('//div[@id="meta"]')
    return re.sub(r"\s+", " ", m[0].text_content()) if m else ""


def parse(path) -> dict:
    t = _meta(path)
    out = {}
    rel = re.search(r"Relatives:\s*(.*?)(?: College:| High School| Recruiting| Draft:| NBA Debut| Died:|$)", t)
    rel_txt = rel.group(1) if rel else ""
    out["bio_nba_relative_n"] = float(len([r for r in rel_txt.split(";") if r.strip()])) if rel_txt else 0.0
    out["bio_nba_father"] = float(bool(re.search(r"\bFather\b", rel_txt)))
    shoots = re.search(r"Shoots:\s*(Left|Right)", t)
    out["bio_shoots_left"] = float(shoots.group(1) == "Left") if shoots else float("nan")
    born = re.search(r"Born:.*? in (.*?)(?: Relatives:| College:| High School| Recruiting| Draft:| NBA Debut| Died:|$)", t)
    out["bio_born_us"] = float(bool(re.search(r"\bus\b", born.group(1)))) if born else float("nan")
    hs = re.search(r"High Schools?:\s*(.*?)(?: Recruiting| Draft:| NBA Debut| Died:|$)", t)
    hs_txt = hs.group(1) if hs else ""
    out["bio_n_high_schools"] = float(len(re.findall(r" in [A-Z][^,]*, [A-Z][A-Za-z .]+", hs_txt))) if hs_txt else float("nan")
    out["bio_prep_academy"] = float(any(a in hs_txt.lower() for a in ACADEMIES)) if hs_txt else float("nan")
    return out


def build() -> pd.DataFrame:
    d = pd.read_parquet(C.PROC / "draft_table.parquet", columns=KEYS + ["bbref_id"])
    rows = []
    for r in d.itertuples():
        p = EXT / "bbref_players" / f"{r.bbref_id}.html"
        if p.exists() and p.stat().st_size > 1000:
            rows.append({"key": r.key, "draft_year": int(r.draft_year), **parse(p)})
    out = pd.DataFrame(rows).drop_duplicates(KEYS)
    print(f"bio features for {len(out)} draftees")
    return out


def load_bio() -> pd.DataFrame:
    return build()


if __name__ == "__main__":
    f = build()
    t = pd.read_parquet(C.PROC / "draft_table.parquet", columns=KEYS + ["source", "war5", "player"]).merge(f, on=KEYS, how="left")
    print(f[[c for c in f.columns if c.startswith("bio_")]].describe().T.round(3).to_string())
    print("fathers:", t[t.bio_nba_father == 1][["draft_year", "player"]].sort_values("draft_year").player.tolist()[-25:])
    from scipy.stats import spearmanr
    h = t[t.draft_year.between(2013, 2025)]
    for c in [c for c in f.columns if c.startswith("bio_")]:
        ok = h[c].notna()
        print(f"{c:20s} n={ok.sum():4d} mean={h.loc[ok, c].mean():.3f} rho with war5: {spearmanr(h.loc[ok, c], h.loc[ok, 'war5']).correlation:+.3f}")
