"""Pre-draft youth-camp sources for the international pathway: adidas Eurocamp, Basketball Without Borders Global, NBA Academy.

Every loader returns a frame keyed by (key, draft_year) -- key = infra.dataset.norm_name(player) -- whose other columns are
numeric and carry the source prefix. Only what a scout knew on draft night: camp rosters and measurements taken before the
draft, Academy Games editions that ended before the draft, roster snapshots archived before the draft. No NBA data.
Missing = not observed. The 0/1 flags are 0 only when the player's plausible camp years are covered by the archive and he is
not on any roster; they are NaN when nothing is archived for those years. Raw downloads are cached under data/external/<src>/.

  load_eurocamp()  ec_    grassroots.adidas.com/eurocamp/archive rosters (2012-14, 2016-17, 2023-25: name, nationality, year of
                          birth, height, team) + BAM measurement sheets: 2012 (hoopsfix.com PDF; europeanprospects.com table with
                          2010-2012 anthros, 3/4 sprint and verticals) and 2013 (NBC Sports PDFs via the Wayback Machine).
                          Units follow the NBA combine c_ columns: inches, pounds, seconds.
  load_bwb()       bwb_   BWB Global camp boys rosters 2015-2020, 2023-2026 (no camp in 2021-22): NBA.com roster PDFs 2023-25
                          (DOB, height, weight), pr.nba.com / nba.com press releases and youthbasket.com lists otherwise.
  load_academy()   acad_  NBA Academy membership from the archived nbaacademy.nba.com "Current Roster" tables (Africa, Australia,
                          India, Latin America, China; Wayback snapshots 2017-2026: age, height) and the NBA Academy Games database
                          (academy-games-database.floorandceiling.workers.dev: per-player box scores 2017, 2019, 2022-26, birth
                          dates, event-roster heights). Academy Games take place in late June / July: an edition counts for a
                          draft only if its last game was played before that draft night.

Run `python -m infra.builders.intl_extras` to download everything and print the coverage against draft_table.parquet.
"""

import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from lxml import html

from infra.builders.intl import ALIASES, DRAFT_DATES
from infra.config import PROC, ROOT
from infra.dataset import norm_name

EXT = ROOT / "data" / "external"
KEYS = ["key", "draft_year"]
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}
WAYBACK = "http://web.archive.org/web"


_wayback_failures = 0  # consecutive failed Wayback downloads in this process; the archive goes offline for minutes at a time


def _get(url: str, path: Path, sleep: float = 0.0, retries: int = 3, timeout: int = 90) -> bool:
    """Download url to path unless cached. Failures are not cached (retried next run). Returns True if path holds content.
    After 5 consecutive Wayback failures the remaining Wayback downloads of the run are skipped (whatever is cached is used)."""
    global _wayback_failures
    if path.exists() and path.stat().st_size > 0:
        return True
    wayback = "web.archive.org" in url
    if wayback and _wayback_failures >= 5:
        return False
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            if r.status_code == 404:
                return False
            r.raise_for_status()
            if b"Internet Archive: Temporarily Offline" in r.content[:2000]:
                raise requests.RequestException("wayback offline")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(r.content)
            time.sleep(sleep)
            if wayback:
                _wayback_failures = 0
            return True
        except requests.RequestException as e:
            print(f"retry {attempt + 1}/{retries} {url}: {str(e)[:80]}", flush=True)
            time.sleep(5 * (attempt + 1))
    if wayback:
        _wayback_failures += 1
        if _wayback_failures == 5:
            print("wayback unreachable: skipping the remaining archive downloads this run", flush=True)
    return False


def _wayback_snapshots(url: str, path: Path) -> list[str]:
    """Timestamps of archived 200 responses for url (monthly collapse), cached."""
    if _get(f"{WAYBACK[:-4]}/cdx/search/cdx?url={url}&output=txt&fl=timestamp,statuscode&filter=statuscode:200&collapse=timestamp:6", path):
        return [ln.split()[0] for ln in path.read_text().splitlines() if ln.strip() and ln[0].isdigit()]
    return []


def _ascii(s) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()


def _tokens(name) -> list[str]:
    """Lower-case ascii name tokens without suffixes / annotations ('Maker Thon (MVP)' -> ['maker', 'thon'])."""
    s = re.sub(r"\(.*?\)|\*|\bmvp\b|\s-\s.*$", " ", _ascii(name))
    return [t for t in re.findall(r"[a-z]+", s) if t not in ("jr", "sr", "ii", "iii", "iv")]


def _tokkey(name) -> str:
    return "".join(sorted(_tokens(name)))


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _ft_in(s) -> float:
    """"6'10\"" / "6'10.5" / "6´6\"" -> inches; NaN if unparsable."""
    m = re.match(r"^\s*(\d)\s*['’´]\s*(\d{1,2}(?:\.\d+)?)?", str(s))
    return float(m.group(1)) * 12 + float(m.group(2) or 0) if m else np.nan


# --------------------------------------------------------------------------- draftees and name matching

def _draft_date(y: int) -> pd.Timestamp:
    return pd.Timestamp(f"{y}-{DRAFT_DATES.get(int(y), '06-25')}")


def _draftees() -> pd.DataFrame:
    """One row per draftee: key, draft_year, player, source, draft_date, draft_age (Torvik birth date, else the international
    build's bbref/FIBA birth date, else the ayush / jasong draft ages) and the implied decimal birth year."""
    t = pd.read_parquet(PROC / "draft_table.parquet")
    d = t[KEYS + ["player", "source"] + [c for c in ("age_at_draft", "draft_age_x", "j_age") if c in t.columns]].copy()
    if (EXT / "intl_prospects.parquet").exists():
        d = d.merge(pd.read_parquet(EXT / "intl_prospects.parquet", columns=KEYS + ["i_age_at_draft"]), on=KEYS, how="left")
    age = pd.Series(np.nan, index=d.index)
    for c in ("age_at_draft", "i_age_at_draft", "draft_age_x", "j_age"):
        if c in d.columns:
            age = age.fillna(_num(d[c]))
    d["draft_age"] = age
    d["draft_date"] = d.draft_year.map(_draft_date)
    d["birth_dec"] = d.draft_date.dt.year + d.draft_date.dt.dayofyear / 365.25 - d.draft_age
    return d[KEYS + ["player", "source", "draft_date", "draft_age", "birth_dec"]].drop_duplicates(KEYS).reset_index(drop=True)


def _draft_keys(player: str) -> set[str]:
    toks = _tokens(player)
    keys = {norm_name(player), "".join(toks)}
    if len(toks) > 2:
        keys.add(toks[0] + toks[-1])
    for a in ALIASES.get(player, []):
        keys.add(norm_name(a))
    return {k for k in keys if k}


def _match(draft: pd.DataFrame, ev: pd.DataFrame, years_back: int, age_lo: float, age_hi: float) -> pd.DataFrame:
    """Pair draftees with event rows (columns: name, year, date, yob) by name, restricted to events dated before the draft night
    and at most `years_back` seasons earlier, and to a plausible age: the roster's year of birth within a year of the draftee's,
    or -- when the roster has none -- an age at the event inside [age_lo, age_hi]. Names match on the full normalised name, either
    token order, first + last token or the aliases; failing that on the last name plus a 3-letter first-name prefix when that
    identifies a single person. Returns (idx = draft row, ev_idx = event row)."""
    ev = ev.reset_index(drop=True)
    toks = ev.name.map(_tokens)
    by_key: dict[str, set] = {}
    by_last: dict[str, set] = {}
    for i, tk in toks.items():
        if not tk:
            continue
        for k in {"".join(tk), "".join(tk[::-1]), tk[0] + tk[-1], tk[-1] + tk[0]}:
            by_key.setdefault(k, set()).add(i)
        for t in tk:  # either token may be the family name
            by_last.setdefault(t, set()).add(i)
    pairs = []
    for i, r in draft.iterrows():
        cand = set().union(*[by_key.get(k, set()) for k in _draft_keys(r.player)])
        if not cand:
            dt = _tokens(r.player)
            if len(dt) >= 2:
                for j in by_last.get(dt[-1], set()):
                    other = [t for t in toks[j] if t != dt[-1]]
                    if any(t[:3] == dt[0][:3] for t in other):
                        cand.add(j)
                if len({_tokkey(ev.name[j]) for j in cand}) > 1:  # ambiguous: two different people
                    cand = set()
        if not cand:
            continue
        c = ev.loc[sorted(cand)]
        c = c[(c.date < r.draft_date) & (c.year >= r.draft_year - years_back)]
        if pd.notna(r.birth_dec):
            has_yob = c.yob.notna()
            ok_yob = (c.yob + 0.5 - r.birth_dec).abs() <= 1.5
            age_ev = r.draft_age - (r.draft_date - c.date).dt.days / 365.25
            c = c[(has_yob & ok_yob) | (~has_yob & (age_ev >= age_lo) & (age_ev <= age_hi))]
        for j in c.index:
            pairs.append((i, j))
    return pd.DataFrame(pairs, columns=["idx", "ev_idx"])


def _flag(draft: pd.DataFrame, matched_idx, covered_years: set, years_back: int) -> pd.Series:
    """1 for matched draftees, 0 when at least one covered camp year falls inside his window, NaN otherwise."""
    out = pd.Series(np.nan, index=draft.index)
    win = [bool(set(range(y - years_back, y + 1)) & covered_years) for y in draft.draft_year]
    out[win] = 0.0
    out[list(matched_idx)] = 1.0
    return out


def _age_at(draft_row, date: pd.Timestamp, yob) -> float:
    """Age on `date` from the draftee's draft age, else from the roster's year of birth (mid-year birthday assumed)."""
    if pd.notna(draft_row.draft_age):
        return draft_row.draft_age - (draft_row.draft_date - date).days / 365.25
    if pd.notna(yob):
        return date.year + date.dayofyear / 365.25 - (yob + 0.5)
    return np.nan


# --------------------------------------------------------------------------- 1. adidas Eurocamp
EC = EXT / "eurocamp"
EC_YEARS = [2012, 2013, 2014, 2016, 2017, 2023, 2024, 2025]  # archive pages with a roster
EC_MEAS = {  # measurement sheets: file name -> url (Wayback id_ form serves the original bytes)
    "meas_2012_hoopsfix.pdf": "https://www.hoopsfix.com/wp-content/uploads/2012/06/adidas-EUROCAMP-2012-Measurements.pdf",
    "meas_2012_europeanprospects.html": f"{WAYBACK}/20120715030557id_/http://www.europeanprospects.com/adidas-eurocamp-2012-vitals-measurments-analysed/",
    "meas_2013_anthros.pdf": f"{WAYBACK}/20240629130518id_/https://nbc-sports.go-vip.net/wp-content/uploads/sites/12/2013/06/adidas-eurocamp-2013-bam-test-anthros.pdf",
    "meas_2013_results.pdf": f"{WAYBACK}/20240630120017id_/https://nbc-sports.go-vip.net/wp-content/uploads/sites/12/2013/06/adidas-eurocamp-2013-bam-test-results.pdf",
}
EC_COLS = ["h_noshoes", "weight", "wingspan", "reach", "vert_max", "sprint", "agility"]


def eurocamp_download():
    for y in EC_YEARS:
        _get(f"https://grassroots.adidas.com/eurocamp/archive/{y}", EC / f"archive_{y}.html", sleep=1.0)
    for f, url in EC_MEAS.items():
        _get(url, EC / f)


def _ec_camp_date(y: int) -> pd.Timestamp:
    return pd.Timestamp(f"{y}-06-08")  # the camp runs over a June weekend, ~2 weeks before the draft


def parse_eurocamp_rosters() -> pd.DataFrame:
    """One row per camper x year from the archive pages' Next.js payload ({"teamName": ..., "rosters": [{...}]}; the rendered
    cards of 2012 / 2023 carry jersey numbers only). Squads brought in as opposition (USA Select, 3SSB Select, national U20
    teams) are dropped; the Next Gen (U18), Team World / Asia and NBL Next Stars invitees are campers. The sheets' "NBA Draft"
    column (the outcome) is never read."""
    recs = []
    for y in EC_YEARS:
        p = EC / f"archive_{y}.html"
        if not p.exists():
            continue
        raw = p.read_bytes().decode("utf-8", "ignore")
        payload = "".join(json.loads(f'"{c}"') for c in re.findall(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)</script>', raw, flags=re.S))
        for team, arr in re.findall(r'"teamName":"([^"]*)","rosters":(\[.*?\])', payload):
            if re.search(r"usa|select|national|3ssb", team, re.I):
                continue
            for o in json.loads(arr):
                f = {k.lower(): str(v).strip() for k, v in o.items() if v not in (None, "")}
                get = lambda *labels: next((f[k] for k in f if any(k.startswith(l) for l in labels)), None)
                name = get("name") or " ".join(x for x in (get("first name"), get("last name")) if x)
                if not name:
                    continue
                recs.append({"year": y, "team": team, "name": re.sub(r"\s*\(.*?\)", "", name), "nationality": get("nationality", "country", "nation"),
                             "position": get("position"), "height_cm": _num(re.sub(r"[^\d.]", "", get("height (cm)", "centimeters") or "")),
                             "yob": _num(get("year of birth")), "club": get("last team", "team")})
    df = pd.DataFrame(recs).drop_duplicates(["year", "name"])
    df["date"] = df.year.map(_ec_camp_date)
    return df


def _pdf_lines(path: Path) -> list[str]:
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        return [ln for pg in pdf.pages for ln in (pg.extract_text() or "").splitlines()]


def parse_eurocamp_measurements() -> pd.DataFrame:
    """BAM sheets -> one row per player x year with h_noshoes / wingspan / reach / vert_max (inches), weight (lb), sprint and
    lane agility (s). 2012 anthros from the hoopsfix PDF, 2010-2012 anthros + sprint / verticals from europeanprospects (cm, kg),
    2013 anthros + athletic tests from the NBC PDFs."""
    recs = []
    p = EC / "meas_2012_hoopsfix.pdf"
    if p.exists():  # "Last First team h_shoes_cm h_shoes_in h_noshoes_cm h_noshoes_in wing_cm wing_in reach_cm reach_in kg lb"
        for ln in _pdf_lines(p):
            m = re.match(r"^([A-Za-z' -]+?) (\d) (\d{3}) \S+ (\d{3}) \S+ (\d{3}) [\d.]+ (\d{3}) [\d.]+ ([\d.]+) (\d+)$", ln)
            if m:
                recs.append({"year": 2012, "name": m.group(1), "h_noshoes": int(m.group(4)) / 2.54, "wingspan": int(m.group(5)) / 2.54,
                             "reach": int(m.group(6)) / 2.54, "weight": int(m.group(8))})
    p = EC / "meas_2012_europeanprospects.html"
    if p.exists():  # Name Year Height Height-with-shoes Reach Weight Wingspan 3/4-sprint Vert-standing Vert-free (cm, kg, s)
        for tr in html.fromstring(p.read_bytes()).xpath("//table//tr")[1:]:
            c = [td.text_content().strip().replace(",", ".") for td in tr.xpath("./td")]
            if len(c) >= 10 and c[1].isdigit():
                recs.append({"year": int(c[1]), "name": c[0], "h_noshoes": _num(c[2]) / 2.54, "reach": _num(c[4]) / 2.54,
                             "weight": _num(c[5]) * 2.2046, "wingspan": _num(c[6]) / 2.54, "sprint": _num(c[7]), "vert_max": _num(c[9]) / 2.54})
    p = EC / "meas_2013_anthros.pdf"
    if p.exists():  # "First Last reach_in h_shoes_in h_shoes_cm h_noshoes_in h_noshoes_cm wing_in lb kg"
        for ln in _pdf_lines(p):
            m = re.match(r"^([A-Za-z' -]+?) ([\d.]+) [\d.]+ \d+ ([\d.]+) \d+ ([\d.]+) (\d+) \d+$", ln)
            if m:
                recs.append({"year": 2013, "name": m.group(1), "reach": float(m.group(2)), "h_noshoes": float(m.group(3)),
                             "wingspan": float(m.group(4)), "weight": int(m.group(5))})
    p = EC / "meas_2013_results.pdf"
    if p.exists():  # "Last First sprint lane_agility reaction_shuttle vert_standing vert_approach"
        for ln in _pdf_lines(p):
            m = re.match(r"^([A-Za-z' -]+?) ([\d.]+) ([\d.]+) [\d.]+ [\d.]+ ([\d.]+)$", ln)
            if m:
                recs.append({"year": 2013, "name": m.group(1), "sprint": float(m.group(2)), "agility": float(m.group(3)), "vert_max": float(m.group(4))})
    df = pd.DataFrame(recs)
    df["tk"] = df.name.map(_tokkey)
    df = df.groupby(["year", "tk"], as_index=False).agg({"name": "first", **{c: "mean" for c in EC_COLS if c in df.columns}})
    for c in EC_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df["date"] = df.year.map(_ec_camp_date)
    return df


def load_eurocamp() -> pd.DataFrame:
    eurocamp_download()
    ros, meas = parse_eurocamp_rosters(), parse_eurocamp_measurements()
    ev = pd.concat([ros.assign(src="roster"), meas.assign(src="meas", yob=np.nan)], ignore_index=True)
    ev["tk"] = ev.name.map(_tokkey)
    draft = _draftees()
    pairs = _match(draft, ev, years_back=6, age_lo=15, age_hi=24)
    rows = []
    for i, g in pairs.groupby("idx"):
        e = ev.loc[g.ev_idx].sort_values("year")
        r = draft.loc[i]
        first = e.iloc[0]
        f = {"idx": i, "ec_first_year": int(first.year), "ec_n_camps": int(e.year.nunique()),
             "ec_age_first": _age_at(r, first.date, e.yob.dropna().iloc[0] if e.yob.notna().any() else np.nan)}
        m = meas[meas.tk.isin(e.tk) & (meas.year <= r.draft_year)].sort_values("year")
        for c in EC_COLS:
            v = m[c].dropna()
            f[f"ec_{c}"] = v.iloc[-1] if len(v) else np.nan
        rows.append(f)
    feats = pd.DataFrame(rows).set_index("idx") if rows else pd.DataFrame()
    out = draft[KEYS].copy()
    out["ec_invited"] = _flag(draft, feats.index, set(ros.year), years_back=6)  # a 0 needs a full roster, not the odd repeat camper's sheet
    out = out.join(feats).rename(columns={"ec_h_noshoes": "ec_height_noshoes"})
    out["ec_wing_minus_height"] = out.ec_wingspan - out.ec_height_noshoes
    return out[out.ec_invited.notna()].reset_index(drop=True)


# --------------------------------------------------------------------------- 2. Basketball Without Borders Global
BWB = EXT / "bwb"
BWB_SRC = {  # camp year -> list of (file name, url); the first source with a parsable roster wins per player, later ones add campers
    2015: [("roster_2015_youthbasket.html", "https://youthbasket.com/Basketball-Without-Border/2015-Global-Camp-New-York")],
    2016: [("roster_2016_youthbasket.html", "https://youthbasket.com/Basketball-Without-Border/2016-Global-Camp-Toronto")],
    2017: [("roster_2017_pr.html", "https://pr.nba.com/bwb-global-camp-nba-all-star-2017/"),
           ("roster_2017_youthbasket.html", "https://youthbasket.com/Basketball-Without-Border/2017-Global-Camp-New-Orleans")],
    2018: [("roster_2018_pr.html", "https://pr.nba.com/basketball-without-borders-global-camp-nba-all-star-2018/"),
           ("roster_2018_youthbasket.html", "https://youthbasket.com/Basketball-Without-Border/2018-Global-Camp-Los-Angeles")],
    2019: [("roster_2019_youthbasket.html", "https://youthbasket.com/Basketball-Without-Border/2019-Global-Camp-Charlotte"),
           ("roster_2019_pr.html", "https://pr.nba.com/bwb-global-camp-2019-all-star/")],
    2020: [("roster_2020_pr.html", "https://pr.nba.com/2020-basketball-without-borders-global-camp/")],
    2023: [("roster_2023.pdf", "https://cms.nba.com/elite-academy/wp-content/uploads/sites/86/2023/02/2023-BWB-Global-Boys-Roster-By-Jersey-Number-Country-Teams.pdf")],
    2024: [("roster_2024.pdf", "https://cdn.flowpage.com/images/fdcc6fa9-51bc-44f4-bffe-edc8b560dc0f-pdf?m=1708134606"),
           ("roster_2024_pr.html", "https://pr.nba.com/basketball-without-borders-global-camp-nba-all-star-2024/")],
    2025: [("roster_2025.pdf", "https://cdn.flowpage.com/images/44329065-f612-49c1-86fd-2ded326b2425-pdf?m=1739638059")],
    2026: [("roster_2026_nba.html", "https://www.nba.com/news/basketball-without-borders-all-star-camp-2026")],
}
BWB_MULTI = r"(?:Dominican Rep(?:ublic)?|Dominica Republic|Central Africa(?:n)? Rep(?:ublic)?|New Zealand|South Sudan|South Korea|North Macedonia" \
            r"|Puerto Rico|Chinese Taipei|Czech Republic|Ivory Coast|Trinidad and Tobago|Cape Verde|Great Britain|United Kingdom|Virgin Islands)"
# Late additions absent from the archived announcement lists, confirmed as campers by later NBA / FIBA releases.
BWB_EXTRA = {2015: [("Lauri Markkanen", "Finland")]}


def bwb_download():
    for y, srcs in BWB_SRC.items():
        for f, url in srcs:
            _get(url, BWB / f, sleep=1.0)


def _bwb_camp_date(y: int) -> pd.Timestamp:
    return pd.Timestamp(f"{y}-02-15")  # All-Star weekend


def _parse_bwb_pdf(path: Path, year: int) -> list[dict]:
    """'# First Last Country M/D/YYYY 6'11" 220 PF' lines; the roster repeats by team / number / country, so dedupe by name."""
    recs = []
    for ln in _pdf_lines(path):
        m = re.match(rf"^\d+ (.+?) ({BWB_MULTI}|\S+) (\d{{1,2}}/\d{{1,2}}/\d{{4}}) (\d[’'´]\d{{1,2}}\"?) (\d{{2,3}}) (\S+)$", ln)
        if not m:
            continue
        mo, da, yr = m.group(3).split("/")
        yr = int(yr) - 1000 if int(yr) > 2100 else int(yr)  # '3007' typos in the 2025 sheet
        recs.append({"year": year, "name": m.group(1), "country": m.group(2), "dob": pd.Timestamp(year=yr, month=int(mo), day=int(da)),
                     "height_in": _ft_in(m.group(4)), "weight_lb": float(m.group(5)), "position": m.group(6)})
    return recs


def _parse_bwb_html(path: Path, year: int) -> list[dict]:
    """Press-release tables (Last | First | Country, or Name | Country) below the 'BOYS ROSTER' heading, or the youthbasket
    country-headed list ('Name (6'9''-F/C-01), club')."""
    raw = path.read_bytes().decode("utf-8", "ignore")
    doc = html.fromstring(raw)
    recs = []
    box = doc.xpath('//div[@id="bwbcontent"]')
    if box:  # flag image = country header; one <br>-separated chunk per player
        segs = re.split(r'<img[^>]*IconsFlags/([^."]+)\.gif[^>]*>', html.tostring(box[0], encoding="unicode"))
        for country, seg in zip(segs[1::2], segs[2::2]):
            for chunk in re.split(r"<br[^>]*>", seg):
                if not chunk.strip() or re.search(r"<b[\s>]", chunk):
                    continue
                s = re.sub(r"\s+", " ", html.fromstring(f"<div>{chunk}</div>").text_content().replace("\xa0", " ")).strip(" ,")
                s = re.sub(r"\(MVP\)|\s+-\s*MVP", "", s).strip()
                m = re.match(r"^(.+?)\s*\((\d['’]\d{1,2})''-([A-Z/]+)(?:-(\d\d))?\)", s)
                if m:
                    recs.append({"year": year, "name": m.group(1), "country": country.replace("-", " "), "height_in": _ft_in(m.group(2)),
                                 "position": m.group(3), "yob": 2000 + int(m.group(4)) if m.group(4) else np.nan})
                elif s and len(s.split()) <= 5 and s[0].isupper() and not re.search(r"[(:\d]", s):
                    recs.append({"year": year, "name": s, "country": country.replace("-", " ")})
        return recs
    boys = raw.upper().find("BOYS ROSTER")
    boys_line = raw[:boys].count("\n") + 1 if boys >= 0 else 0
    for tbl in doc.xpath("//table"):
        if (tbl.sourceline or 0) < boys_line:  # the girls' table precedes the BOYS ROSTER heading
            continue
        rows = [[c.text_content().strip() for c in tr.xpath("./th|./td")] for tr in tbl.xpath(".//tr")]
        rows = [r for r in rows if len(r) >= 2 and r[0]]
        if not rows:
            continue
        head = [h.lower() for h in rows[0]]
        body = rows[1:] if any(h in ("last name", "name", "first name") for h in head) else rows
        for r in body:
            name = f"{r[1]} {r[0]}" if len(r) >= 3 and "last name" in head else r[0]
            recs.append({"year": year, "name": name.replace("*", "").strip(), "country": r[-1]})
    return recs


def parse_bwb() -> pd.DataFrame:
    frames = []
    for y, srcs in BWB_SRC.items():
        for f, _ in srcs:
            p = BWB / f
            if not p.exists():
                continue
            recs = _parse_bwb_pdf(p, y) if p.suffix == ".pdf" else _parse_bwb_html(p, y)
            frames.append(pd.DataFrame(recs))
    frames.append(pd.DataFrame([{"year": y, "name": n, "country": c} for y, ps in BWB_EXTRA.items() for n, c in ps]))
    df = pd.concat(frames, ignore_index=True)
    for c in ("dob", "yob", "height_in", "weight_lb"):
        if c not in df.columns:
            df[c] = np.nan
    df["tk"] = df.name.map(_tokkey)
    df["yob"] = df.yob.fillna(pd.to_datetime(df.dob, errors="coerce").dt.year)
    df = df.sort_values(["year", "dob"]).groupby(["year", "tk"], as_index=False).agg(
        {"name": "first", "country": "first", "dob": "first", "yob": "first", "height_in": "first", "weight_lb": "first"})
    df["date"] = df.year.map(_bwb_camp_date)
    return df


def load_bwb() -> pd.DataFrame:
    bwb_download()
    ev = parse_bwb()
    draft = _draftees()
    pairs = _match(draft, ev, years_back=8, age_lo=14, age_hi=19.5)
    rows = []
    for i, g in pairs.groupby("idx"):
        e = ev.loc[g.ev_idx].sort_values("year")
        r, first = draft.loc[i], e.iloc[0]
        dob = e.dob.dropna()
        age = (first.date - dob.iloc[0]).days / 365.25 if len(dob) else _age_at(r, first.date, first.yob)
        h, w = e.height_in.dropna(), e.weight_lb.dropna()
        rows.append({"idx": i, "bwb_year": int(first.year), "bwb_age": age, "bwb_height_cm": h.iloc[-1] * 2.54 if len(h) else np.nan,
                     "bwb_weight_kg": w.iloc[-1] / 2.2046 if len(w) else np.nan})
    feats = pd.DataFrame(rows).set_index("idx") if rows else pd.DataFrame()
    out = draft[KEYS].copy()
    out["bwb_selected"] = _flag(draft, feats.index, set(ev.year), years_back=8)
    out = out.join(feats)
    return out[out.bwb_selected.notna()].reset_index(drop=True)


# --------------------------------------------------------------------------- 3. NBA Academy
ACAD = EXT / "academy"
ACAD_DB_URL = "https://academy-games-database.floorandceiling.workers.dev/"
ACAD_LOCATIONS = ["africa", "australia", "india", "nba-academy-latin-america", "zhejiang", "shandong", "xinjiang"]
# Academy squads at the Academy Games. "NBA Academy Select" teams are outside invitees; Basketball Australia's Centre of Excellence
# hosts NBA Global Academy and the NBA lists its players as Academy alumni.
ACAD_TEAM = re.compile(r"^NBA (Global )?Academy(?! Select)|^Centre of Excellence")


def academy_download(per_year: int = 2):
    _get(ACAD_DB_URL, ACAD / "games_db.html")
    for loc in ACAD_LOCATIONS:
        snaps = _wayback_snapshots(f"nbaacademy.nba.com/location/{loc}/", ACAD / f"cdx_{loc}.txt")
        keep = []
        for y, g in pd.Series(snaps).groupby(pd.Series(snaps).str[:4]):
            g = g.sort_values()
            step = max(len(g) // per_year, 1)
            keep += list(g.iloc[::step][:per_year])
        for ts in keep:
            _get(f"{WAYBACK}/{ts}id_/https://nbaacademy.nba.com/location/{loc}/", ACAD / f"roster_{loc}_{ts}.html", sleep=0.5)


def _js_const(raw: str, name: str):
    i = raw.find(f"const {name}=")
    return json.JSONDecoder().raw_decode(raw[i + len(name) + 7:])[0] if i >= 0 else None


def parse_academy_games() -> tuple[pd.DataFrame, pd.Series]:
    """Games database: one row per player x edition (team, totals, birth date, event-roster height in inches); plus the last game
    date of every edition."""
    raw = (ACAD / "games_db.html").read_bytes().decode("utf-8", "ignore")
    db, meas = _js_const(raw, "DB"), _js_const(raw, "MEASUREMENT_DATA") or {}
    ends = pd.Series({int(e): pd.Timestamp(d) for e, d in
                      pd.DataFrame([{"e": g["id"][:4], "d": g.get("date")} for g in db["games"] if g.get("date")]).groupby("e").d.max().items()})
    rows = meas.get("rows", {})
    recs = []
    for p in db["players"]:
        m = rows.get(p["id"], {})
        t = p.get("totals") or {}
        listed = m.get("heightIn") if str(m.get("heightSource", "")).startswith(("nba", "user_", "official", "event")) else None
        recs.append({"edition": int(p["edition"]), "pid": p["id"], "name": p.get("fullName") or p["name"], "team": p["team"],
                     "dob": pd.to_datetime(p.get("dob"), errors="coerce") if p.get("dobStatus") == "exact" else pd.NaT,
                     "gp": t.get("gp"), "pts": t.get("pts"), "reb": t.get("reb"), "ast": t.get("ast"), "height_in": listed,
                     "academy": bool(ACAD_TEAM.match(p["team"]))})
    df = pd.DataFrame(recs)
    df["date"] = df.edition.map(ends)
    return df, ends


def parse_academy_rosters() -> pd.DataFrame:
    """Archived 'Current Roster' tables: one row per player x snapshot (location, age, listed height in inches)."""
    recs = []
    for p in sorted(ACAD.glob("roster_*.html")):
        loc, ts = re.match(r"roster_(.+)_(\d{14})$", p.stem).groups()
        date = pd.Timestamp(ts[:8])
        try:
            doc = html.fromstring(p.read_bytes())
        except Exception:
            continue
        for tr in doc.xpath('//table[contains(@class,"roster")]//tr | //table//tr[td]'):
            c = [td.text_content().strip() for td in tr.xpath("./td")]
            if len(c) >= 5 and c[0] and _num(c[3]) == _num(c[3]) and 12 <= float(_num(c[3])) <= 25:
                recs.append({"location": loc, "date": date, "name": c[0].title() if c[0].isupper() else c[0], "country": c[2],
                             "age": float(c[3]), "height_in": _ft_in(c[4])})
    df = pd.DataFrame(recs).drop_duplicates(["location", "date", "name"])
    df["year"] = df.date.dt.year
    df["yob"] = (df.date.dt.year + df.date.dt.dayofyear / 365.25 - df.age - 0.5).round()  # age is an integer: birthday ~ half a year back
    return df


def load_academy() -> pd.DataFrame:
    academy_download()
    games, _ = parse_academy_games()
    ros = parse_academy_rosters()
    ev = pd.concat([
        games.assign(year=games.edition, yob=games.dob.dt.year, src="games"),
        ros.assign(src="roster", academy=True),
    ], ignore_index=True)
    ev["tk"] = ev.name.map(_tokkey)
    draft = _draftees()
    pairs = _match(draft, ev, years_back=8, age_lo=13, age_hi=21)
    rows = []
    for i, g in pairs.groupby("idx"):
        e = ev.loc[g.ev_idx].sort_values("date")
        r = draft.loc[i]
        mem = e[e.academy]
        if mem.empty:  # only seen on a guest / Select team at the Academy Games: not an Academy student
            continue
        first = mem.iloc[0]
        dob = e.dob.dropna()
        age = (first.date - dob.iloc[0]).days / 365.25 if len(dob) else _age_at(r, first.date, first.yob)
        h = e.height_in.dropna()
        f = {"idx": i, "acad_first_year": int(first.year), "acad_age_first": age, "acad_height_cm": h.iloc[-1] * 2.54 if len(h) else np.nan}
        gm = e[(e.src == "games") & (e.gp.fillna(0) > 0)]
        if len(gm):
            gp = gm.gp.sum()
            f.update({"acad_gp": gp, "acad_ppg": gm.pts.sum() / gp, "acad_rpg": gm.reb.sum() / gp, "acad_apg": gm.ast.sum() / gp})
        rows.append(f)
    feats = pd.DataFrame(rows).set_index("idx") if rows else pd.DataFrame()
    out = draft[KEYS].copy()
    out["acad_member"] = _flag(draft, feats.index, set(ev.year), years_back=8)
    out = out.join(feats)
    for c in ("acad_gp", "acad_ppg", "acad_rpg", "acad_apg"):
        if c not in out.columns:
            out[c] = np.nan
    return out[out.acad_member.notna()].reset_index(drop=True)


# --------------------------------------------------------------------------- coverage report

def coverage():
    t = pd.read_parquet(PROC / "draft_table.parquet", columns=KEYS + ["player", "source"])
    core = t.draft_year.between(2007, 2026)
    for fn in (load_eurocamp, load_bwb, load_academy):
        ex = fn()
        flag = [c for c in ex.columns if c.endswith(("_invited", "_selected", "_member"))][0]
        m = t.merge(ex, on=KEYS, how="left")
        hit = m[flag] == 1
        print(f"{fn.__name__:14s} {ex.shape[1] - 2:2d} cols {len(ex):5d} rows | matched {hit.sum()} ({hit[core].sum()} of 2007-26; "
              f"intl {hit[m.source == 'intl'].sum()}, college {hit[m.source == 'college'].sum()}, none {hit[m.source == 'none'].sum()}) | "
              f"known-0 {(m[flag] == 0).sum()}")
        print("   " + ", ".join(f"{p} {y}" for p, y in zip(m.player[hit], m.draft_year[hit])))


if __name__ == "__main__":
    coverage()
