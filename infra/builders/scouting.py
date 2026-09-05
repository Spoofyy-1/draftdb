"""Pre-draft NBADraft.net scouting grades (columns sc_*) for the 2008-2026 draft classes, keyed by (key, draft_year).

NBADraft.net keeps editing a prospect's profile for years after he is drafted (NBA comparison, grades, text), so the live
page is leakage. Every profile used here is a Wayback Machine capture taken BEFORE the player's draft night (same cutoff
as mocks.py: 22:00 UTC on the first night of his draft, before the first pick). Per player the latest such capture is
used, so the grades are the site's final pre-draft opinion; a player whose profile was only captured after his draft is
left missing. Captures at or after the cutoff are counted as "rejected" in the status files.

Three layouts: 2008 (ASP site, nbadraft.net/admincp/profiles/<name>.html), 2008-11 -> 2019 (Drupal, /players/<slug>) and
2019-11 onward (WordPress, /players/<slug>/). All carry the 1-10 grid -- Athleticism, Size, Defense, Strength, Quickness,
Leadership, Jump Shot, NBA Ready, Potential, Intangibles, plus Ball Handling + Passing (guards/wings) or Rebounding + Post
Skills (bigs) -- an Overall grade, an NBA comparison and Strengths / Weaknesses paragraphs. On the two older layouts the
grid labels are an image picked by position (attribute_banner_<POS>.gif / parameters.gif); the column order used below
was read off those images. Text flags are keyword counts: injury / upside / ready over Strengths + Weaknesses, motor /
character / shooting_concern over the Weaknesses paragraph only ("great motor" in Strengths is not a concern).

Cache (never touched by load_scouting): data/external/scouting/nbadraftnet/cdx_index.csv (every capture of every profile
URL), <year>/<slug>.html + .json (the capture used per draftee) and <year>/_status.json (per draftee: ok, or why not).
data/external/scouting/provenance.csv lists the capture used per player.

    from infra.builders.scouting import load_scouting   # cache only, seconds
    python -m infra.builders.scouting                    # download missing profiles, rebuild, coverage report
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from lxml import html

from infra import config as C
from infra.dataset import norm_name

EXT = C.ROOT / "data" / "external"
SC_DIR = EXT / "scouting"
NDN_DIR = SC_DIR / "nbadraftnet"
KEYS = ["key", "draft_year"]
YEARS = range(2008, 2027)

# first night of each draft (YYYYMMDD), as in mocks.py; a capture is pre-draft if its timestamp < that day 22:00 UTC
DRAFT_NIGHT = {2008: "20080626", 2009: "20090625", 2010: "20100624", 2011: "20110623", 2012: "20120628", 2013: "20130627",
               2014: "20140626", 2015: "20150625", 2016: "20160623", 2017: "20170622", 2018: "20180621", 2019: "20190620",
               2020: "20201118", 2021: "20210729", 2022: "20220623", 2023: "20230622", 2024: "20240626", 2025: "20250625",
               2026: "20260623"}
CUTOFF_HHMM = "2200"

WAYBACK = "https://web.archive.org"
_S = requests.Session()
_S.headers["User-Agent"] = "Mozilla/5.0 (nba-redraft research; pre-draft scouting archive)"

# --------------------------------------------------------------------------- wayback


def _get(url, params=None, tries=5, timeout=180):
    for i in range(tries):
        try:
            r = _S.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(5 * (i + 1))
    return None


_TS = re.compile(r"^\d{14} ")


def _cdx(params: dict, tries: int = 6) -> list[list[str]] | None:
    """Rows of a CDX query (fl=timestamp,original,statuscode). The API intermittently answers with an HTML 'Temporarily
    Offline' page at status 200, so anything that is not a capture listing is retried."""
    for i in range(tries):
        r = _get(f"{WAYBACK}/cdx/search/cdx", {"fl": "timestamp,original,statuscode", **params})
        if r is not None:
            text = r.text.strip()
            if not text:
                return []
            lines = text.splitlines()
            if all(_TS.match(line) for line in lines):
                return [line.split(" ", 2) for line in lines]
        time.sleep(15 * (i + 1))
    return None


def _decode(content: bytes) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("cp1252", errors="replace")  # the 2008 ASP pages


def _offline(text: str) -> bool:
    return "Internet Archive" in text[:400] and "Temporarily Offline" in text[:400]  # served with status 200


def _download(ts: str, original: str) -> str | None:
    """Raw capture (id_ flag); recent captures stored zstd/brotli-encoded fall back to the rewritten page (as in mocks.py)."""
    for i in range(4):
        r = _get(f"{WAYBACK}/web/{ts}id_/{original}", tries=3)
        if r is None:
            return None
        if r.headers.get("content-encoding", "").lower() in ("zstd", "br") or r.content[:4] == b"\x28\xb5\x2f\xfd":
            r = _get(f"{WAYBACK}/web/{ts}/{original}", tries=3)
            if r is None:
                return None
        text = _decode(r.content)
        if not _offline(text):
            return text
        time.sleep(15 * (i + 1))
    return None


# --------------------------------------------------------------------------- capture index

INDEX = NDN_DIR / "cdx_index.csv"
_PLAYERS_PATH = re.compile(r"^https?://(?:www\.)?nbadraft\.net(?::80)?/players/([a-z0-9][a-z0-9\-]*)/?$")
_ASP_PATH = re.compile(r"^https?://(?:www\.)?nbadraft\.net(?::80)?/admincp/profiles/([a-z0-9\-]+)\.html?$")


def _index_rows(rows) -> list[dict]:
    out = []
    for ts, url, status in rows:
        if status != "200":
            continue
        m = _PLAYERS_PATH.match(url)
        if m:
            out.append({"ts": ts, "url": url, "slug": m.group(1), "layout": "players"})
            continue
        m = _ASP_PATH.match(url.lower())
        if m:
            out.append({"ts": ts, "url": url, "slug": m.group(1), "layout": "asp"})
    return out


def build_index(force: bool = False) -> pd.DataFrame:
    """Every 200-OK capture of every NBADraft.net profile URL (cached). The /players/* listing is queried one calendar
    year at a time (<= 15k rows each): the API's own paging returns inconsistent page boundaries between calls."""
    if INDEX.exists() and not force:
        return pd.read_csv(INDEX, dtype=str)
    rows = []
    for y in range(2006, max(YEARS) + 2):
        page = _cdx({"url": "nbadraft.net/players/*", "from": str(y), "to": str(y), "limit": 100000})
        assert page is not None, f"CDX listing for {y} failed"
        rows += _index_rows(page)
        print(f"cdx players {y}: {len(rows)} profile captures so far")
    asp = _cdx({"url": "nbadraft.net/admincp/profiles/*", "to": "20091231", "limit": 100000})
    assert asp is not None, "CDX admincp listing failed"
    rows += _index_rows(asp)
    df = pd.DataFrame(rows).drop_duplicates(["ts", "url"]).sort_values(["slug", "ts"])
    NDN_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(INDEX, index=False)
    return df


# --------------------------------------------------------------------------- slugs <-> draftees

_TRANSLIT = str.maketrans({"ı": "i", "İ": "I", "ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "ß": "ss", "ø": "o", "Ø": "O", "æ": "ae", "Æ": "Ae"})
ALIASES = {  # slug-derived name (norm_name) -> draft_table key, where normalisation alone does not resolve it
    "aleksandarpavlovic": "sashapavlovic", "eneskanter": "enesfreedom", "guillermohernangomez": "willyhernangomez",
    "timotheluwawu": "timotheluwawucabarrot", "hansenyang": "yanghansen", "bjboston": "brandonboston",
    "giannisadetokunbo": "giannisantetokounmpo", "giannisadetokoubo": "giannisantetokounmpo",
    "billwalker": "henrywalker", "bjmullens": "byronmullens", "jeffpendergraph": "jeffayres", "patrickbeverly": "patrickbeverley",
    "sergiigladyr": "sergiygladyr", "patrickmills": "pattymills", "moeharkless": "mauriceharkless", "jefferytaylor": "jefftaylor",
    "ricardoledo": "rickyledo", "waltertavares": "edytavares", "roydevynmarble": "devynmarble", "juanvaulet": "juanpablovaulet",
    "josephyoung": "joeyoung", "satnamsinghbhamara": "satnamsingh", "danieldiez": "danidiez", "juanhernangomez": "juanchohernangomez",
    "kahlilfelder": "kayfelder", "wesleyiwundu": "wesiwundu", "aleksandarvezenkov": "sashavezenkov", "mohamedbamba": "mobamba",
    "sviatoslavmykhailiuk": "svimykhailiuk", "raymondspalding": "rayspalding", "cameronreddish": "camreddish", "nicolasclaxton": "nicclaxton",
    "marcoslouzadasilva": "didilouzada", "nahshonhyland": "boneshyland", "cameronthomas": "camthomas", "alexandresarr": "alexsarr",
    "ronaldholland": "ronholland", "carltoncarrington": "bubcarrington", "lrmbahamoute": "lucmbahamoute",
}


def _slug_key(slug: str) -> str:
    return norm_name(slug.replace("-", " "))


def _alt(s: str) -> str:
    return norm_name(str(s).translate(_TRANSLIT).replace(".", ""))


def _draftees() -> pd.DataFrame:
    d = pd.read_parquet(C.PROC / "draft_table.parquet", columns=KEYS + ["player", "pick"])
    return d[d.draft_year.isin(YEARS)].drop_duplicates(KEYS)


def _mock_slugs(year: int) -> dict[str, str]:
    """slug -> normalised link name from the cached pre-draft NBADraft.net mock of that class (data/external/mocks/...)."""
    p = EXT / "mocks" / "nbadraftnet" / f"{year}.html"
    if not p.exists():
        return {}
    doc = html.fromstring(p.read_text(errors="replace"))
    out = {}
    for a in doc.iter("a"):
        href = (a.get("href") or "").lower()
        m = re.search(r"/players/([a-z0-9\-]+)/?$", href) or re.search(r"/admincp/profiles/([a-z0-9\-]+)\.html?$", href)
        if m:
            name = re.sub(r"\s+\d-\d{1,2}\b.*$", "", " ".join(a.text_content().split()))  # "Name 6-8 220 SF ..." -> Name
            out[m.group(1)] = norm_name(name)
    return out


def _candidates(pool: pd.DataFrame, index: pd.DataFrame, year: int) -> dict[str, list[str]]:
    """key -> candidate slugs (mock-linked slug first, then every indexed slug whose name normalises to the key)."""
    by_key, slugs = {}, set(index.slug)
    for slug in slugs:
        k = _slug_key(slug)
        by_key.setdefault(ALIASES.get(k, k), []).append(slug)
    mock = _mock_slugs(year)
    out = {}
    for key, player in zip(pool.key, pool.player):
        cands = list(by_key.get(key, []))
        if _alt(player) != key:
            cands += [s for s in by_key.get(_alt(player), []) if s not in cands]
        linked = [s for s, n in mock.items() if n == key or ALIASES.get(n) == key or (s in slugs and _slug_key(s) in (key, _alt(player)))]
        for s in linked:
            if s in cands:
                cands.remove(s)
        out[key] = linked + cands
    return out


# --------------------------------------------------------------------------- parsers

GUARD_GRID = ["athleticism", "size", "defense", "strength", "quickness", "leadership", "jumpshot", "nba_ready", "handle", "potential", "passing", "intangibles"]
BIG_GRID = ["athleticism", "size", "defense", "strength", "quickness", "leadership", "jumpshot", "nba_ready", "rebounding", "potential", "post", "intangibles"]
LABELS = {"athleticism": "athleticism", "size": "size", "defense": "defense", "strength": "strength", "quickness": "quickness",
          "leadership": "leadership", "jump shot": "jumpshot", "nba ready": "nba_ready", "ball handling": "handle", "rebounding": "rebounding",
          "potential": "potential", "post skills": "post", "passing": "passing", "intangibles": "intangibles"}
GRID_COLS = ["athleticism", "size", "defense", "strength", "quickness", "leadership", "jumpshot", "nba_ready", "post", "passing",
             "intangibles", "handle", "potential", "rebounding"]
BIG_POS = ("center", "power forward", "small forward/power forward", "forward/center")


def _clean(s: str) -> str:
    return " ".join(s.split())


def _num(s: str) -> float:
    m = re.search(r"\d+(?:\.\d+)?", s or "")
    return float(m.group(0)) if m else np.nan


def _height_in(s: str) -> float:
    m = re.search(r"(\d)\s*[-'’′]\s*(\d{1,2}(?:\.\d+)?)", s or "")
    return int(m.group(1)) * 12 + float(m.group(2)) if m else np.nan


def _grid_group(text: str, position: str) -> list[str] | None:
    """Which unlabelled grid an ASP / Drupal page shows: the position banner image, else the position text."""
    m = re.search(r"attribute_banner_([A-Z]+)\.gif", text)
    if m:
        return BIG_GRID if m.group(1) in ("C", "PF", "SFPF", "PFC") else GUARD_GRID
    m = re.search(r"parameters(\d?)\.gif", text)
    if m:
        return GUARD_GRID if m.group(1) == "1" else BIG_GRID
    if position:
        return BIG_GRID if position.lower().strip() in BIG_POS else GUARD_GRID
    return None


_TEXT_END = (r"(?:Strengths?:|Weakness(?:es)?:|NBA Comparison:|Notes?:|Outlook:|Overall:|High School:|College:|Related|YouTube|Youtube|"
             r"[A-Z][a-z]+(?: [A-Z]\.)? [A-Z][A-Za-z']+(?: III| Jr\.?)?\s*[-–—]?\s*\d{1,2}/\d{1,2}/\d{2,4}|$)")  # next section / author signature
_STRENGTHS = re.compile(r"Strengths?:\s*(.*?)\s*(?=" + _TEXT_END + ")", re.S)
_WEAKNESSES = re.compile(r"Weakness(?:es)?:\s*(.*?)\s*(?=" + _TEXT_END + ")", re.S)
_COMPARISON = re.compile(r"NBA Comparison:\s*(.{1,80}?)\s*(?=" + _TEXT_END + ")", re.S)


def _flat(el) -> str:
    return _clean(" ".join(el.itertext()))  # a space between elements: "<h3>..Okeke</h3><p>Strengths:" must not run together


def parse_profile(text: str) -> dict:
    """{name, position, height_in, weight, grid: {col: score}, overall, compares_to, strengths, weaknesses} for any of the
    three NBADraft.net layouts; missing pieces are absent / NaN."""
    doc = html.fromstring(text)
    for bad in doc.xpath("//script|//style"):
        bad.drop_tree()
    out = {"grid": {}, "overall": np.nan, "height_in": np.nan, "weight": np.nan, "position": "", "name": ""}
    body = None
    if doc.xpath('//div[@id="p_details" or @id="nbap_p_details"]'):  # 2008 ASP layout (two id variants)
        out["name"] = _clean(" ".join(doc.xpath('//div[@id="name" or @id="nbap_name"]//text()')))
        for li in doc.xpath('//div[@id="detail_list"]//li'):
            title = _clean(" ".join(li.xpath('.//div[@class="item_title"]//text()'))).lower()
            val = _clean(" ".join(li.xpath('.//div[@class="item_content"]//text()')))
            if title.startswith("nba pos"):
                out["position"] = val
            elif title.startswith("ht"):
                out["height_in"] = _height_in(val)
            elif title.startswith("wt"):
                out["weight"] = _num(val)
        scores = []
        for block in doc.xpath('//div[@id="p_details" or @id="nbap_p_details"]/div[contains(@class,"block")]'):
            t = _clean(block.text_content())
            if "Overall" in t:
                out["overall"] = _num(t.replace("Overall", ""))
            else:
                scores.append(_num(t))
        group = _grid_group(text, out["position"])
        if group and len(scores) == len(group):
            out["grid"] = dict(zip(group, scores))
        body = doc.xpath('//div[@id="content_bottom"]')
    elif doc.xpath('//div[@id="nba_player_attrib_blocks"] | //div[@id="nba_player_stats"]'):  # Drupal layout
        out["name"] = re.sub(r"^\s*\d+\s*-\s*", "", _clean(" ".join(doc.xpath('//h2[@class="number"]//text()'))))
        for li in doc.xpath('//div[@id="nba_player_stats_middle"]//li'):
            label = _clean(" ".join(li.xpath('./span[@class="label"]//text()'))).lower()
            val = _clean(li.text_content().replace(_clean(" ".join(li.xpath('./span[@class="label"]//text()'))), "", 1))
            if label.startswith("nba pos"):
                out["position"] = val
            elif label.startswith("ht"):
                out["height_in"] = _height_in(val)
            elif label.startswith("wt"):
                out["weight"] = _num(val)
        scores = [_num(_clean(p.text_content())) for p in doc.xpath('//p[@class="nba_player_attrib_score"]')]
        tot = doc.xpath('//div[contains(@class,"attrib_total")]//p[@class="whitebox"]')
        if tot:
            out["overall"] = _num(_clean(tot[0].text_content()))
        group = _grid_group(text, out["position"])
        if group and len(scores) == len(group):
            out["grid"] = dict(zip(group, scores))
        body = doc.xpath('//div[@id="nbap_content_bottom"]')
    else:  # WordPress layout (2019-11 onward)
        h1 = doc.xpath('//h1[contains(@class,"player-name")]')
        if h1:
            for sp in h1[0].xpath('.//span[contains(@class,"player-number")]'):
                sp.drop_tree()
            out["name"] = _clean(h1[0].text_content())
        out["position"] = _clean(" ".join(doc.xpath('//span[contains(@class,"player-position")]//text()')))
        out["height_in"] = _height_in(_clean(" ".join(doc.xpath('//span[contains(@class,"player-height")]//text()'))))
        out["weight"] = _num(_clean(" ".join(doc.xpath('//span[contains(@class,"player-weight")]//text()'))))
        for row in doc.xpath('//div[contains(@class,"player-attributes")]//div[contains(@class,"div-table-row")]'):
            label = _clean(" ".join(row.xpath('.//div[contains(@class,"attribute-name")]//text()'))).lower()
            val = _num(_clean(" ".join(row.xpath('.//div[contains(@class,"attribute-value")]//text()'))))
            if label in LABELS and not np.isnan(val):
                out["grid"][LABELS[label]] = val
        tot = doc.xpath('//div[contains(@class,"attribute-overall")]//span[contains(@class,"value")]')
        if tot:
            out["overall"] = _num(_clean(tot[0].text_content()))
        body = doc.xpath('//h3[contains(., "NBA Comparison")]/.. | //p[strong[contains(., "Strength")]]/..')
    if not out["name"]:
        title = _clean(" ".join(doc.xpath("//title//text()")))
        out["name"] = re.sub(r"\s*[|\-]\s*NBADraft\.net\s*|NBADraft\.net\s*[-|]\s*", "", title).strip()
    flat = _flat(body[0]) if body else _flat(doc)
    m = _COMPARISON.search(flat)
    out["compares_to"] = m.group(1).strip(" .") if m and m.group(1).strip(" .").upper() not in ("N/A", "NA", "TBD", "?") else ""
    m = _STRENGTHS.search(flat)
    out["strengths"] = m.group(1) if m else ""
    m = _WEAKNESSES.search(flat)
    out["weaknesses"] = m.group(1) if m else ""
    return out


# --------------------------------------------------------------------------- features

_FLAGS = {  # column -> (regex, scope): counts of keyword hits
    "injury": (r"injur|surger|\bknee|\bankle|stress fracture|\btorn\b|\btear\b|concussion|achilles|meniscus|\bacl\b|"
               r"back (?:injur|issue|problem|surger|spasm)|bad back|foot (?:injur|issue|problem|surger)|out for the season|missed (?:the|most of the|the entire) season", "both"),
    "motor": (r"\bmotor\b|\beffort\b|\blazy\b|\bcoast(?:s|ed|ing)?\b|takes? plays off|\bpassive\b", "weaknesses"),
    "character": (r"character|matur|attitude|off[- ]?(?:the[- ])?court|disciplin|red flag|coachab|\bego\b|selfish|work ethic", "weaknesses"),
    "shooting_concern": (r"mechanic|inconsistent|poor shoot|free[- ]throw|\bft\b|streaky|shooting (?:form|stroke)|jump ?shot|jumper|"
                         r"three[- ]point|3[- ]?p(?:t|oint)|perimeter shot|shooting range", "weaknesses"),
    "upside": (r"upside|ceiling|\braw\b|\bproject\b|long[- ]term", "both"),
    "ready": (r"polished|nba[- ]ready|ready (?:to contribute|now|made)|day one|plug[- ]and[- ]play|contribute (?:immediately|right away)|"
              r"immediate(?:ly)? (?:impact|contribut)", "both"),
}
_WING = [re.compile(r"(\d)\s*[-'’′]\s*(\d{1,2}(?:\.\d+)?)?\s*[\"”″]?\s*(?:\+|plus)?\s*wingspan", re.I),
         re.compile(r"wingspan\s*(?:of|is|at|measured(?: at| in at)?|around|near|:)?\s*(?:a |an |about |over |nearly |almost |roughly |close to )?(\d)\s*[-'’′]\s*(\d{1,2}(?:\.\d+)?)?", re.I),
         re.compile(r"(\d)[- ]foot(?:[- ](\d{1,2}(?:\.\d+)?))?\s*(?:\+|plus)?\s*wingspan", re.I)]


def _wingspan(text: str) -> float:
    for pat in _WING:
        for m in pat.finditer(text):
            v = int(m.group(1)) * 12 + (float(m.group(2)) if m.group(2) else 0)
            if 70 <= v <= 100:
                return v
    return np.nan


def _words(s: str) -> int:
    return len(re.findall(r"[A-Za-z][A-Za-z'’]*", s))


def _grade(v: float) -> float:
    return v if v > 0 else np.nan  # the site shows 0 for a prospect it has not graded yet


def features(p: dict, snapshot: str, year: int) -> dict:
    f = {f"sc_{c}": _grade(p["grid"].get(c, np.nan)) for c in GRID_COLS}
    f["sc_overall"] = _grade(p["overall"])
    f["sc_height_in"] = p["height_in"]
    f["sc_weight"] = p["weight"]
    s, w = p["strengths"], p["weaknesses"]
    f["sc_wingspan_in"] = _wingspan(s + " " + w)
    for col, (pat, scope) in _FLAGS.items():
        f[f"sc_flag_{col}"] = len(re.findall(pat, (s + " " + w) if scope == "both" else w, re.I)) if (s or w) else np.nan
    ws, ww = _words(s), _words(w)
    f["sc_words_strengths"] = ws if s else np.nan
    f["sc_words_weaknesses"] = ww if w else np.nan
    f["sc_ratio_strength_weakness"] = ws / ww if ww else np.nan
    f["sc_snapshot_days_before_draft"] = (pd.Timestamp(DRAFT_NIGHT[year]) - pd.Timestamp(snapshot[:8])).days
    f["sc_position"] = p["position"] or None
    f["sc_compares_to"] = p["compares_to"] or None
    return f


# --------------------------------------------------------------------------- fetch + cache

MAX_TRIES = 3  # captures tried per candidate slug (a capture can be truncated or a wrong-person page)
NAME_MIN = 0.85


def _name_ok(page_name: str, key: str, player: str) -> bool:
    n = norm_name(page_name)
    if not n:
        return False
    if n == key or _alt(page_name) == key or ALIASES.get(n) == key:
        return True
    return SequenceMatcher(None, n, key).ratio() >= NAME_MIN or SequenceMatcher(None, _alt(page_name), _alt(player)).ratio() >= NAME_MIN


def _usable(p: dict) -> bool:
    return bool(p["grid"]) or bool(p["strengths"]) or bool(p["weaknesses"])


def fetch_profile(key: str, player: str, year: int, slugs: list[str], index: pd.DataFrame) -> dict:
    """Download + cache the latest pre-draft capture of the first candidate slug whose page is this player and parses.
    Returns the status record: {'status': 'ok', 'slug', 'url', 'snapshot', 'n_rejected'} or {'status': <reason>, ...}."""
    cutoff = DRAFT_NIGHT[year] + CUTOFF_HHMM
    ydir = NDN_DIR / str(year)
    if not slugs:
        return {"status": "no_slug"}
    reason = "no_predraft_capture"
    caps = index[index.slug.isin(slugs)]
    rejected = int((caps.ts >= cutoff).sum())
    caps = caps[caps.ts < cutoff]
    # freshest pre-draft capture first: a 2009 draftee's Drupal page beats his 2008 ASP page
    for slug in sorted(slugs, key=lambda s: caps.ts[caps.slug == s].max() if (caps.slug == s).any() else "", reverse=True):
        pre = caps[caps.slug == slug].sort_values("ts", ascending=False)
        if pre.empty:
            continue
        reason = "download_failed"
        for ts, url in list(zip(pre.ts, pre.url))[:MAX_TRIES]:
            text = _download(ts, url)
            if text is None or len(text) < 2000:
                continue
            try:
                p = parse_profile(text)
            except Exception:
                reason = "parse_failed"
                continue
            if not _name_ok(p["name"], key, player):
                reason = "name_mismatch"
                break  # the slug is another player; try the next slug
            if not _usable(p):
                reason = "no_content"
                continue
            ydir.mkdir(parents=True, exist_ok=True)
            (ydir / f"{slug}.html").write_text(text)
            meta = {"status": "ok", "key": key, "draft_year": year, "slug": slug, "url": url, "snapshot": ts, "n_rejected": rejected,
                    "page_name": p["name"]}
            (ydir / f"{slug}.json").write_text(json.dumps(meta))
            return meta
    return {"status": reason, "n_rejected": rejected, "slugs": slugs}


def _status_path(year: int) -> Path:
    return NDN_DIR / str(year) / "_status.json"


def download(years=YEARS, workers: int = 4, retry: bool = False, verbose: bool = True) -> None:
    """Fill the cache for every draftee of `years` who has no record yet (retry=True also re-tries recorded misses)."""
    index = build_index()
    d = _draftees()
    for y in years:
        pool = d[d.draft_year == y]
        sp = _status_path(y)
        status = json.loads(sp.read_text()) if sp.exists() else {}
        todo = [k for k in pool.key if k not in status or (retry and status[k]["status"] != "ok")]
        if not todo:
            continue
        cands = _candidates(pool, index, y)
        players = dict(zip(pool.key, pool.player))
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = ex.map(lambda k: (k, fetch_profile(k, players[k], y, cands[k], index)), todo)
            for k, rec in results:
                status[k] = rec
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(status, indent=0))
        if verbose:
            ok = sum(r["status"] == "ok" for r in status.values())
            print(f"{y}: {ok}/{len(pool)} profiles cached ({len(todo)} fetched in {time.time() - t0:.0f}s)")


# --------------------------------------------------------------------------- build / load

TEXT_COLS = ["sc_position", "sc_compares_to"]


def load_scouting(write_provenance: bool = False) -> pd.DataFrame:
    """Parse the cached pre-draft profiles into one sc_* row per draftee with a profile. Cache only, no network."""
    rows, prov = [], []
    for y in YEARS:
        sp = _status_path(y)
        if not sp.exists():
            continue
        for key, rec in json.loads(sp.read_text()).items():
            if rec.get("status") != "ok":
                continue
            hp = NDN_DIR / str(y) / f"{rec['slug']}.html"
            if not hp.exists():
                continue
            assert rec["snapshot"] < DRAFT_NIGHT[y] + CUTOFF_HHMM, (key, y, rec["snapshot"])  # pre-draft guard
            f = features(parse_profile(hp.read_text()), rec["snapshot"], y)
            rows.append({"key": key, "draft_year": y, **f})
            prov.append({"key": key, "draft_year": y, "source": "nbadraftnet", "url": rec["url"], "snapshot": rec["snapshot"],
                         "fields_parsed": int(sum(pd.notna(v) for k, v in f.items() if k not in TEXT_COLS))})
    out = pd.DataFrame(rows)
    if write_provenance:
        SC_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(prov).sort_values(KEYS).to_csv(SC_DIR / "provenance.csv", index=False)
    return out


def build(years=YEARS, retry: bool = False) -> pd.DataFrame:
    download(years, retry=retry)
    return load_scouting(write_provenance=True)


# --------------------------------------------------------------------------- report


def report(out: pd.DataFrame) -> None:
    d = _draftees()
    m = d.merge(out, on=KEYS, how="left")
    print("\nyear  draftees  profile  grid  text  median_days_before  min  max   misses (no_slug / no_predraft_capture / other)")
    for y, g in m.groupby("draft_year"):
        sp = _status_path(y)
        st = json.loads(sp.read_text()) if sp.exists() else {}
        reasons = pd.Series([r["status"] for r in st.values()]).value_counts()
        days = g.sc_snapshot_days_before_draft.dropna()
        other = int(sum(v for k, v in reasons.items() if k not in ("ok", "no_slug", "no_predraft_capture")))
        print(f"{y}  {len(g):8d}  {int(g.sc_overall.notna().sum() + g.sc_words_strengths.notna().sum() - (g.sc_overall.notna() & g.sc_words_strengths.notna()).sum()):7d}  "
              f"{int(g.sc_athleticism.notna().sum()):4d}  {int(g.sc_words_strengths.notna().sum()):4d}  "
              f"{days.median() if len(days) else float('nan'):18.0f}  {days.min() if len(days) else float('nan'):3.0f}  {days.max() if len(days) else float('nan'):4.0f}   "
              f"{int(reasons.get('no_slug', 0)):3d} / {int(reasons.get('no_predraft_capture', 0)):3d} / {other}")
    rejected = sum(r.get("n_rejected", 0) for y in YEARS if _status_path(y).exists() for r in json.loads(_status_path(y).read_text()).values())
    print(f"\n{len(out)} profiles; grid for {out.sc_athleticism.notna().sum()}; post-draft captures rejected: {rejected}")
    print("columns:", [c for c in out.columns if c not in KEYS])
    for name, y in [("zionwilliamson", 2019), ("anthonyedwards", 2020), ("jameswiseman", 2020), ("victorwembanyama", 2023), ("jalenbrunson", 2018)]:
        r = out[(out.key == name) & (out.draft_year == y)]
        if r.empty:
            print(name, y, "-- missing")
            continue
        r = r.iloc[0]
        print(f"{name} {y}: snapshot -{r.sc_snapshot_days_before_draft:.0f}d  ", {c.replace("sc_", ""): r[c] for c in out.columns if c.startswith("sc_") and pd.notna(r[c]) and c not in TEXT_COLS},
              r.sc_position, "| cf.", r.sc_compares_to)


if __name__ == "__main__":
    t0 = time.time()
    res = build()
    print(f"built {res.shape} in {time.time() - t0:.0f}s")
    report(res)
