"""Pre-draft consensus mock-draft ranks (columns mock_*) for the 2003-2026 draft classes, keyed by (key, draft_year).

Every board is a Wayback Machine snapshot captured BEFORE draft night (cutoff: 22:00 UTC on the day of the first round,
before the first pick), so nothing here can be derived from the actual selections. Live pages are never used: the sites
keep editing their mocks after the draft. Per (source, year) the code lists the CDX captures of the candidate URLs in
the draft year, takes the latest one before the cutoff, and caches it as data/external/mocks/<source>/<year>.html with a
<year>.json sidecar (url, snapshot timestamp, players parsed). data/external/mocks/provenance.csv summarises them.

  nbadraftnet   NBADraft.net two-round mock, every class 2003-2026: 2003-2008 the home page (index.asp) embedded the live
                board, 2009-2019 /<year>mock_draft, 2020+ /nba-mock-drafts/
  draftexpress  DraftExpress two-round mock 2006-2008, 2010-2017 (mock.php?y= / mymock.php / nba-mock-draft/<year>/). No
                board for 2005 (only the 2006 default page was captured) and 2009 (last pre-draft capture is February);
                the site stopped publishing in 2017.
  insidehoops   InsideHoops.com first-round mock (30 picks) 2003-2010; 2011-2012 captures do not parse
  tankathon     Tankathon mock 2015 (30 picks) and 2016-2026 (60 picks)
  espn          Chad Ford's / Givony's mocks were Insider (ESPN+) pages whose captures hold only the paywall teaser, so the
                column is NaN through 2024; the 2025-2026 finals are free two-round stories (URLs in ESPN_FINALS)
  ringer        The Ringer NBA Draft Guide big board (Kevin O'Connor) 2017-2026: server-rendered cards 2017-2020, the CMS
                JSON behind the Nuxt app 2021-2025 (2021: last capture June 22, 30 players), the Next.js RSC payload 2026
  cbs           CBS Sports mock-draft index 2016-2026: the featured expert's first round (Parrish's 60 in 2017)
  nbadraftroom  NBADraftRoom first-round mock 2016-2024 (no 2025-2026 capture)
  netscouts     NetScouts Basketball two-round mock 2019-2024 (2019 last captured June 4)
  draftsite     DraftSite two-round mock 2016 and 2019 (May 27); the other years' pages were not captured pre-draft
  bleacher      Jonathan Wasserman's final two-round mock 2017-2026 (BLEACHER_FINALS; the 2016 final was not captured)
  hoopshype     HoopsHype's staff first-round mock 2016-2018 (HOOPSHYPE_MOCKS)

A capture is accepted only if it parses >= 25 picks with at most 2 gaps and at least half of its top 30 are draftees of
that class (a site's default page can already show next year's board). Captures before May 1 of the draft year are not
used (they predate the early-entry deadline). Boards are truncated to their first 60 entries; a draftee missing from a
source's board gets rank 61 for that source (31 for a first-round-only board). mock_rank_consensus = mean over the boards
found for the year, mock_rank_std the sample std across them, mock_n_sources = boards that list the player,
mock_first_round = consensus <= 30. Names are matched to draft_table by normalised name, transliteration, ALIASES, a
unique surname, then a logged fuzzy match; unmatched names are undrafted players.

    from infra.builders.mocks import load_mocks   # builds from the cache, downloading what is missing
    python -m infra.builders.mocks                 # rebuild + coverage / provenance / Spearman-vs-pick sanity report
"""

import json
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from lxml import html

from infra import config as C
from infra.dataset import norm_name

EXT = C.ROOT / "data" / "external"
MOCK_DIR = EXT / "mocks"
KEYS = ["key", "draft_year"]
SOURCES = ["nbadraftnet", "draftexpress", "insidehoops", "tankathon", "espn", "ringer", "cbs", "nbadraftroom", "netscouts", "draftsite",
           "bleacher", "hoopshype"]
BOARD_LEN = 60
MISSING_RANK = 61  # draftee absent from a two-round board
MISSING_RANK_R1 = 31  # draftee absent from a first-round-only board (InsideHoops): "not a first-rounder", not "61st"

# first night of each draft (YYYYMMDD); every snapshot used is <= that day 22:00 UTC (18:00 ET, before the first pick)
DRAFT_NIGHT = {2003: "20030626", 2004: "20040624", 2005: "20050628", 2006: "20060628", 2007: "20070628", 2008: "20080626",
               2009: "20090625", 2010: "20100624", 2011: "20110623", 2012: "20120628", 2013: "20130627", 2014: "20140626",
               2015: "20150625", 2016: "20160623", 2017: "20170622", 2018: "20180621", 2019: "20190620", 2020: "20201118",
               2021: "20210729", 2022: "20220623", 2023: "20230622", 2024: "20240626", 2025: "20250625", 2026: "20260623"}
CUTOFF_HHMM = "2200"
EARLIEST_MMDD = "0501"

WAYBACK = "https://web.archive.org"
_S = requests.Session()
_S.headers["User-Agent"] = "Mozilla/5.0 (nba-redraft research; pre-draft mock archive)"


def _nbadraftnet_urls(y: int) -> list[str]:
    if y == 2003:
        return ["http://www.nbadraft.net/"]  # the 2003 mock was the home page
    if y <= 2008:  # the home page embedded the live two-round mock; the yearly ASP pages are the same table
        return ["http://www.nbadraft.net/index.asp", f"http://www.nbadraft.net/index.asp?content=mock{y}", f"http://nbadraft.net/mock{y}.asp"]
    if y <= 2019:
        return [f"http://www.nbadraft.net/{y}mock_draft"]
    return ["https://www.nbadraft.net/nba-mock-drafts/"]


def _draftexpress_urls(y: int) -> list[str]:
    if y <= 2004:
        return []  # the site launched in spring 2004 without a mock page
    if y == 2005:
        return ["http://www.draftexpress.com/mock.php"]
    if y == 2006:
        return ["http://www.draftexpress.com/mock.php?y=2006"]  # the bare mock.php already showed the 2007 board in June 2006
    if y == 2007:
        return ["http://www.draftexpress.com/mymock.php?page=official&year=2007", "http://www.draftexpress.com/mock.php?y=2007"]
    return [f"http://www.draftexpress.com/nba-mock-draft/{y}/", "http://www.draftexpress.com/nba-mock-draft/"]


def _espn_urls(y: int) -> list[str]:
    # Chad Ford's mock lived at insider.espn.go.com/nba/draft/mock/?season=Y&version=N and Givony's 2018-2024 finals at
    # /nba/insider/story/: the captures hold only the Insider/ESPN+ paywall teaser (checked 2014, 2022). The 2025+ finals
    # are free /nba/story/ pages with the full two rounds.
    return ESPN_FINALS.get(y, [])


ESPN_FINALS = {
    2025: ["https://www.espn.com/nba/story/_/id/45559441/2025-nba-mock-draft-latest-first-second-round-predictions-all-59-picks"],
    2026: ["https://www.espn.com/nba/story/_/id/48790115/2026-nba-mock-draft-projecting-60-picks-post-combine-peterson-dybantsa-boozer"],  # post-combine edition (May); the final's URL is not known
}

# Article-based mocks have a new URL every year (found by hand; the CDX API cannot enumerate a large site by keyword).
# Every URL is the outlet's final pre-draft edition; the capture used is still the latest one before the cutoff.
BLEACHER_FINALS = {  # Jonathan Wasserman's final two-round mock
    2016: ["https://bleacherreport.com/articles/2647672-2016-nba-mock-draft-jonathan-wassermans-final-2-round-predictions"],
    2017: ["https://bleacherreport.com/articles/2716889-2017-nba-mock-draft-final-2-round-predictions"],
    2018: ["https://bleacherreport.com/articles/2781931-2018-nba-mock-draft-final-2-round-predictions"],
    2019: ["https://bleacherreport.com/articles/2841103-bleacher-reports-final-2019-nba-mock-draft"],
    2020: ["https://bleacherreport.com/articles/2917825-bleacher-reports-final-2020-nba-mock-draft"],
    2021: ["https://bleacherreport.com/articles/2946238-brs-final-2021-nba-mock-draft-top-4-picks-coming-into-focus",
           "https://bleacherreport.com/articles/2945939-post-nba-finals-mock-draft-2-round-predictions-and-latest-buzz"],
    2022: ["https://bleacherreport.com/articles/10039256-brs-final-2022-nba-mock-draft"],
    2023: ["https://bleacherreport.com/articles/10080062-brs-final-2023-nba-mock-draft-full-2-round-predictions"],
    2024: ["https://bleacherreport.com/articles/10125940-brs-final-2024-nba-mock-draft-full-2-round-predictions"],
    2025: ["https://bleacherreport.com/articles/25211849-brs-final-2025-nba-mock-draft-including-full-2-round-projections-and-pro-comps"],
    2026: ["https://bleacherreport.com/articles/25262746-2026-nba-mock-draft"],
}
HOOPSHYPE_MOCKS = {  # HoopsHype's own staff mock (30 picks), not its aggregate of other outlets. 2019+ the site only ran the
    2016: ["http://hoopshype.com/2016/06/22/nba-mock-draft-2016/", "http://hoopshype.com/2016/06/20/nba-mock-draft-2016/"],  # aggregate (a consensus
    2017: ["http://hoopshype.com/2017/06/17/nba-mock-draft-2017/"],  # of boards already here), whose final pre-draft editions were not captured
    2018: ["https://hoopshype.com/2018/06/21/nba-mock-draft-2018/", "https://hoopshype.com/2018/06/15/nba-mock-draft-2018/"],
}


def _insidehoops_urls(y: int) -> list[str]:
    return ["http://www.insidehoops.com/nba-mock-draft.shtml"] if y <= 2012 else []


def _tankathon_urls(y: int) -> list[str]:
    return ["http://www.tankathon.com/mock_draft", "https://www.tankathon.com/mock_draft", "https://www.tankathon.com/mock-draft"] if y >= 2015 else []


# The Ringer's guide became a client-rendered Nuxt app in 2021; the board then lives in a CMS JSON whose config id changes
# per class (the trailing * makes the CDX query a prefix match over the cache-busting query strings). 2026+: new Next.js
# site, the board is embedded in the page's RSC payload.
_RINGER_CMS = "https://storage.googleapis.com/nbadraft-theringer-com-cms/hardrefresh/data/{cfg}/content.production.json.gz*"
_RINGER_CFG = {2021: "ringernba2021", 2022: "ringernba2022", 2023: "ringernbadraft2023", 2024: "ringernba2024", 2025: "ringernba2025"}


def _ringer_urls(y: int) -> list[str]:
    if y < 2017:
        return []  # the first Ringer draft guide was 2017
    if y <= 2020:
        return ["https://nbadraft.theringer.com/"]
    if y in _RINGER_CFG:
        return [_RINGER_CMS.format(cfg=_RINGER_CFG[y])]
    return [f"https://www.theringer.com/nba-draft/{y}/big-board"]


def _cbs_urls(y: int) -> list[str]:
    return ["https://www.cbssports.com/nba/draft/mock-draft/"] if y >= 2016 else []


def _nbadraftroom_urls(y: int) -> list[str]:
    if y < 2016:
        return []
    return [f"http://www.nbadraftroom.com/p/{y}-nba-mock-draft.html", f"https://nbadraftroom.com/p/{y}-nba-mock-draft/"]


def _netscouts_urls(y: int) -> list[str]:
    return [f"https://netscoutsbasketball.com/scouting/{y}-nba-mock-draft/"] if y >= 2016 else []


def _draftsite_urls(y: int) -> list[str]:
    return [f"http://www.draftsite.com/nba/mock-draft/{y}/"] if y >= 2016 else []


CANDIDATES = {"nbadraftnet": _nbadraftnet_urls, "draftexpress": _draftexpress_urls, "insidehoops": _insidehoops_urls,
              "tankathon": _tankathon_urls, "espn": _espn_urls, "ringer": _ringer_urls, "cbs": _cbs_urls,
              "nbadraftroom": _nbadraftroom_urls, "netscouts": _netscouts_urls, "draftsite": _draftsite_urls,
              "bleacher": lambda y: BLEACHER_FINALS.get(y, []), "hoopshype": lambda y: HOOPSHYPE_MOCKS.get(y, [])}


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


def _snapshots(url: str, year: int) -> list[tuple[str, str]]:
    """(timestamp, original) of the 200-OK captures of `url` between EARLIEST_MMDD of the draft year and the cutoff, oldest
    first. Boards older than that predate the early-entry deadline (late April): half their names never enter the draft."""
    p = {"url": url, "output": "json", "filter": "statuscode:200", "from": f"{year}{EARLIEST_MMDD}", "to": DRAFT_NIGHT[year] + CUTOFF_HHMM, "limit": 5000}
    r = _get(f"{WAYBACK}/cdx/search/cdx", p)
    if r is None or not r.text.strip().startswith("["):
        return []
    rows = json.loads(r.text)[1:]
    return sorted((row[1], row[2]) for row in rows if row[1] <= DRAFT_NIGHT[year] + CUTOFF_HHMM)


def _decode(content: bytes) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("cp1252", errors="replace")  # the 2003-2008 ASP pages


def _download(ts: str, original: str) -> str | None:
    """Raw capture (id_ flag). Recent captures are stored zstd/brotli-encoded, which requests cannot decode; for those the
    rewritten Wayback page is fetched instead (decoded server-side; only URLs and a toolbar differ, the tables are intact)."""
    r = _get(f"{WAYBACK}/web/{ts}id_/{original}", tries=3)
    if r is None:
        return None
    if r.headers.get("content-encoding", "").lower() in ("zstd", "br") or r.content[:4] == b"\x28\xb5\x2f\xfd":
        r = _get(f"{WAYBACK}/web/{ts}/{original}", tries=3)
        if r is None:
            return None
    return _decode(r.content)


# --------------------------------------------------------------------------- parsers -> [(rank, name)]

_HEIGHT = re.compile(r"\s+\d-\d{1,2}\b.*$")  # "Adam Morrison 6-8 220 SF Gonzaga Jr." -> "Adam Morrison"
_NAME_HEIGHT = re.compile(r"^([A-Z][A-Za-z'.\- ]{2,40}?)\s+\d-\d{1,2}\b")  # the same line without a profile link (old ASP pages)
_PICK = re.compile(r"^(\d{1,3})\.?$")


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _slug_name(href: str) -> str:
    """/players/michael-kidd-gilchrist or /profile/Karl-Towns-61831/ -> 'michael kidd gilchrist' (full name when the cell is truncated)."""
    slug = href.rstrip("/").split("/")[-1]
    slug = re.sub(r"-\d+$", "", slug)
    return slug.replace("-", " ")


def _walk_table_rows(doc, anchor_re: str, text_fallback: bool = False) -> list[tuple[int, str]]:
    """Generic mock table walker: a cell holding a pick number followed (within 3 cells) by a cell with a player anchor
    (or, on the old ASP pages, a plain 'Name 6-8 220 ...' cell). Handles one- and two-column layouts (the old NBADraft.net
    home page shows picks n and n+30 on one row)."""
    out = []
    for tr in doc.iter("tr"):
        cells = [c for c in tr if c.tag in ("td", "th")]
        i = 0
        while i < len(cells):
            m = _PICK.match(_clean(cells[i].text_content()))
            if m:
                for j in range(i + 1, min(i + 4, len(cells))):
                    a = [x for x in cells[j].iter("a") if x.get("href") and re.search(anchor_re, x.get("href"))]
                    name = ""
                    if a:
                        name = _HEIGHT.sub("", _clean(a[0].text_content()))
                        if (name.endswith("...") or not name) and "-" in a[0].get("href").rstrip("/").split("/")[-1]:
                            name = _slug_name(a[0].get("href"))  # slug URLs carry the full name; old ASP hrefs are stale, not used
                    if not name and text_fallback:
                        t = _NAME_HEIGHT.match(_clean(cells[j].text_content()))
                        name = t.group(1) if t else ""
                    if name:
                        out.append((int(m.group(1)), name))
                        i = j
                        break
            i += 1
    return out


def _dedupe(pairs: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """First occurrence per rank and per name in document order (a page may repeat the board or append an older one)."""
    seen, names, out = set(), set(), []
    for rank, name in pairs:
        if rank not in seen and name not in names:
            seen.add(rank)
            names.add(name)
            out.append((rank, name))
    return sorted(out)


def _contiguous(players: list[tuple[int, str]], max_gaps: int = 2) -> bool:
    """A board is usable only if it is (nearly) complete from pick 1: a gap would turn a listed draftee into 'missing'."""
    ranks = {r for r, _ in players}
    return len([r for r in range(1, max(ranks) + 1) if r not in ranks]) <= max_gaps


def parse_nbadraftnet(text: str) -> list[tuple[int, str]]:
    doc = html.fromstring(text)
    tabs = doc.xpath('//table[contains(@id,"nba_mock_consensus_table") or contains(@class,"nba_mock_consensus_table")]')
    if tabs:  # 2009+: one table per round, columns # / Team / Player / ...
        pairs = []
        for t in tabs:
            pairs += _walk_table_rows(t, r"/players?/")
        return _dedupe(pairs)
    return _dedupe(_walk_table_rows(doc, r"profiles/", text_fallback=True))  # 2003-2008 ASP layouts


def parse_draftexpress(text: str) -> list[tuple[int, str]]:
    doc = html.fromstring(text)
    anchor = r"/profile/|viewprofile|profile\.php"
    pairs = []
    for label, offset in (("First Round", 0), ("Second Round", 30)):  # 2007-2010 number each round 1-30 in its own table
        for t in doc.xpath(f'//table[(./tr|./tbody/tr)/th[contains(., "{label}")]]'):
            pairs += [(r + offset, n) for r, n in _walk_table_rows(t, anchor) if r <= 30]
    if not pairs:
        pairs = _walk_table_rows(doc, anchor)
    if len(pairs) < MIN_PARSED:  # 2016 layout: <div class='ranking-item'><div class='numero'>1.</div> ... <a href="/profile/...">
        pairs = []
        for item in doc.xpath('//div[contains(@class,"ranking-item")]'):
            num, a = item.xpath('.//div[contains(@class,"numero")]'), item.xpath('.//a[contains(@href,"/profile/")]')
            m = _PICK.match(_clean(num[0].text_content())) if num else None
            if m and a:
                pairs.append((int(m.group(1)), _clean(a[0].text_content())))
    return _dedupe(pairs)


_PK = r"(?<!\d)(\d{1,2})[).]\s+"  # pick number
_TEAM = r"(?:(?!\d{1,2}[).]\s)[^:]){2,50}?"  # team text up to the separator, never running into the next pick number
_IH_PATTERNS = [  # prose formats, one per era; group 1 = pick, group 2 = name
    # "13) Memphis - (from Hou)- Jarvis Hayes, 6-6, 220" (2003) / "7. Toronto - Danny Granger 6-8 230, SF)" (2005)
    re.compile(_PK + r"(?:[A-Z][^\d\-()]{1,30}?)?(?:\s*-?\s*\([^)]{1,40}\))?\s*-?\s*(?:\([^)]{1,40}\)\s*)?([A-Z][^\d,()]{2,40}?)\s*[,(]?\s*\d-\d{1,2}\b"),
    # "1) Los Angeles Clippers: Blake Griffin, PF/C, 6-10, 250" (2009)
    re.compile(_PK + _TEAM + r":\s*([A-Z][^,()\d]{2,40}?)\s*,\s*[A-Z/\-]{1,6}\s*,\s*\d-\d{1,2}\b"),
    # "2. Chicago (via New York) Tyrus Thomas (LSU, 6-9, 229, PF, Fr.)" (2006): the two capitalised words before the bracket
    re.compile(_PK + r"(?:[A-Z][^()]{1,40}?\s(?:\([^()]{1,30}\)\s)?)?([A-Z][\w'.\-]+\s+[A-Z][\w'.\-]+)\s*\((?:[^()\d]{1,30},\s*)?\d-\d{1,2}\b"),
    # "15. Phoenix (From Atl): Joe Alexander (F 6-8 230 West Virginia)" (2008)
    re.compile(_PK + _TEAM + r":\s*([A-Z][^,()\d]{2,40}?)\s*\((?:[A-Z/\-]{1,6}\s+)?\d-\d{1,2}\b"),
    # "2) Philadelphia 76ers: Evan Turner (SG/SF, Ohio State, Junior)" (2010, no height)
    re.compile(_PK + _TEAM + r":\s*([A-Z][^,()\d]{2,40}?)\s*\([A-Z/\-]{1,6},"),
    # "9. Chicago (from NY) -- Spencer Hawes (Washington) 6-11, 235, PF, Freshman" (2007)
    re.compile(_PK + _TEAM + r"--\s*([A-Z][^,()\d]{2,40}?)\s*\([^()]{1,40}\)\s*\d-\d{1,2}\b"),
]


def parse_insidehoops(text: str) -> list[tuple[int, str]]:
    """InsideHoops.com mock: a Pick / Team / Player table (2004) or prose lines in one of the _IH_PATTERNS formats. The
    pattern that yields the most picks wins; only the first list on the page counts (older versions are appended below)."""
    doc = html.fromstring(text)
    pairs = _walk_table_rows(doc, r"(?!)", text_fallback=True)
    if len(pairs) < MIN_PARSED:
        flat = _clean(html.tostring(doc, method="text", encoding="unicode"))
        cands = []
        for pat in _IH_PATTERNS:
            found, last, start = [], 0, len(flat)
            for m in pat.finditer(flat):
                rank = int(m.group(1))
                if rank <= last:  # pick numbers restart: an older list follows, stop
                    break
                found.append((rank, re.split(r"\s-\s*", m.group(2))[-1].strip()))  # drop 'LA Clippers - '
                last, start = rank, min(start, m.start())
            cands.append((found, start))
        # the current mock is the first list on the page: earliest-starting pattern with a full list wins, else the longest
        full = [c for c in cands if len(c[0]) >= 20]
        pairs = min(full, key=lambda c: c[1])[0] if full else max(cands, key=lambda c: len(c[0]))[0]
    return _dedupe(pairs)


def parse_tankathon(text: str) -> list[tuple[int, str]]:
    doc = html.fromstring(text)
    pairs = []
    for row in doc.xpath('//div[contains(concat(" ", normalize-space(@class), " "), " mock-row ")]'):
        num = row.xpath('.//div[contains(@class,"mock-row-pick-number")]/text()')
        name = row.xpath('.//div[contains(@class,"mock-row-name")]')
        m = _PICK.match(_clean(num[0])) if num else None
        if m and name:
            pairs.append((int(m.group(1)), _clean(name[0].text_content())))
    if not pairs:  # 2015-2018 layout: <tr class="mock-row"><td class="pick">1</td> ... <td class="player"><a href="/players/...">
        pairs = _walk_table_rows(doc, r"/players/")
    return _dedupe(pairs)


def parse_espn(text: str) -> list[tuple[int, str]]:
    doc = html.fromstring(text)
    pairs = _walk_table_rows(doc, r"/player/|/nba/draft/")
    if len(pairs) < 20:  # 2025+ story layout, first round: <h2>N. <a>Team</a></h2><p><strong><a href=".../player/...">Name</a>, SG, School
        pairs = []
        for h in doc.iter("h2", "h3"):
            m = re.match(r"^(\d{1,2})\.", _clean(h.text_content()))
            for sib in h.itersiblings() if m else ():
                if sib.tag in ("h2", "h3"):
                    break
                a = sib.xpath('.//a[contains(@href, "/player/")]')
                strong = sib.xpath(".//strong")
                if a or strong:
                    name = _clean(a[0].text_content()) if a else _clean(strong[0].text_content()).split(",")[0]
                    pairs.append((int(m.group(1)), name))
                    break
        for p in doc.iter("p"):  # second round: <p><b>N. <a>Team</a> (via X)</b><br/><a>Name</a>, PG, Club</p>; the name is what follows the <br>
            m = re.match(r"^(\d{1,2})\.\s", _clean(p.text_content()))
            br = p.find("br")
            if m and br is not None:
                after = (br.tail or "") + "".join(html.tostring(s, method="text", encoding="unicode", with_tail=True) for s in br.itersiblings())
                name = _clean(after).split(",")[0]
                if name:
                    pairs.append((int(m.group(1)), name))
    if len(pairs) < 20:  # teaser-only (paywalled) pages carry a few names in prose; not a board
        return []
    return _dedupe(pairs)


# "1. Philadelphia 76ers: Ben Simmons (LSU, PF, Freshman)" / "2. Sacramento: Marvin Bagley Power Forward / 6-11 / Duke":
# pick, team, colon, then the name up to a bracket, comma, pipe, dash or a position word
_PICK_LINE = re.compile(r"^(?:No\.\s*)?(\d{1,2})[.)]\s+[^:]{2,60}:\s*([A-Z][^(,|—–]{2,60}?)\s*(?:[(,|—–]|$)")
_POS_TAIL = re.compile(r"\s+(?:Point Guard|Shooting Guard|Small Forward|Power Forward|Center|Guard|Forward|Wing|Big|PG|SG|SF|PF|C|G|F)(?:/[A-Z]{1,2})?\b.*$")


def _pick_lines(doc) -> list[tuple[int, str]]:
    """Prose mocks: every heading / paragraph / list item / cell of the form 'N. Team: Name ...'."""
    pairs = []
    for el in doc.iter("h1", "h2", "h3", "h4", "p", "li", "div", "td", "strong", "b"):
        m = _PICK_LINE.match(_clean(el.text_content()))
        if m:
            name = _POS_TAIL.sub("", m.group(2)).strip(" .")
            if 2 <= len(name.split()) <= 5:
                pairs.append((int(m.group(1)), name))
    return _dedupe(pairs)


def parse_bleacher(text: str) -> list[tuple[int, str]]:
    """Wasserman's final mock: slide titles (h2) for the top picks, '<strong>N. Team: Name (...)</strong>' paragraphs for the rest."""
    return _pick_lines(html.fromstring(text))


def parse_hoopshype(text: str) -> list[tuple[int, str]]:
    """HoopsHype staff mock: '<div>N. <a>Team</a>: <a>Name</a></div>' cells with the position on the next line."""
    return _pick_lines(html.fromstring(text))


_RSC_PLAYER = re.compile(r'"title":"([^"]+)","playerSlug":"[^"]*"(?:(?!"playerSlug").)*?"rankings":\{"order":(\d+)', re.S)
UNRANKED = 100  # unranked players carry a placeholder order (9999) in the Ringer data


def parse_ringer(text: str) -> list[tuple[int, str]]:
    """The Ringer's big board. 2017-2020: server-rendered cards (<li|div|article class="card-item" data-id=rank>, name in
    the h3); 2021-2025: the CMS JSON (contents.players.content[], 0-based `order`); 2026+: Next.js RSC payload."""
    s = text.lstrip()
    if s.startswith("{"):
        players = json.loads(s)["contents"]["players"]["content"]
        return _dedupe([(int(p["order"]) + 1, _clean(p["title"])) for p in players
                        if p.get("order") is not None and int(p["order"]) < UNRANKED and not p.get("deleted") and p.get("title")])
    if "self.__next_f" in text:
        flat = text.replace('\\"', '"')
        return _dedupe([(int(m.group(2)), _clean(m.group(1))) for m in _RSC_PLAYER.finditer(flat) if int(m.group(2)) < UNRANKED])
    doc = html.fromstring(text)
    pairs = []
    for card in doc.xpath('//*[contains(concat(" ", normalize-space(@class), " "), " card-item ")]'):
        rank = card.get("data-id") or ""
        if not rank.isdigit():  # 2017: data-id is the player slug, the rank is printed as "01"
            rank = "".join(card.xpath('.//div[contains(concat(" ", normalize-space(@class), " "), " rank ")]//text()')).strip()
        name = _clean(" ".join(card.xpath(".//h3//text()")))
        if rank.isdigit() and name:
            pairs.append((int(rank), name))
    return _dedupe(pairs)


def parse_cbs(text: str) -> list[tuple[int, str]]:
    """CBS Sports mock-draft index. 2017+: one column per expert and round (td.cell-rank / td.cell-player-info rows, the
    name a profile link or a span.player-name for internationals), merged per expert; the longest board wins, ties go to
    the first (featured) expert. 2016: the two experts share one table side by side; the walker's dedupe keeps the first."""
    doc = html.fromstring(text)
    boards = {}
    for col in doc.xpath('//*[contains(@class, "experts-column") or contains(@class, "MockDraft-column")]'):
        author = _clean(" ".join(col.xpath('.//*[contains(@class, "author-name") or contains(@class, "MockDraft-author")]//text()'))) or str(len(boards))
        for tr in col.iter("tr"):
            rank = tr.xpath('./td[contains(@class, "cell-rank")]')
            cell = tr.xpath('./td[contains(@class, "cell-player-info")]')
            m = _PICK.match(_clean(rank[0].text_content())) if rank else None
            if m and cell:
                name = cell[0].xpath('.//a | .//span[contains(@class, "player-name")]')
                if name and _clean(name[0].text_content()):
                    boards.setdefault(author, []).append((int(m.group(1)), _clean(name[0].text_content())))
    if not boards:
        for a in doc.xpath("//a[@data-popurl]"):  # 2016 player links are href="#" with the profile in data-popurl
            a.set("href", a.get("data-popurl"))
        boards[""] = _walk_table_rows(doc, r"popover/player")
    boards = [_dedupe(b) for b in boards.values() if b]
    return max(boards, key=len) if boards else []


def parse_nbadraftroom(text: str) -> list[tuple[int, str]]:
    """# / Team / Player / Pos ... rows: the third cell is the player (usually a link to the scouting report, plain text
    when the site has none; the position cell also carries a link, so the generic anchor walker is not used)."""
    doc = html.fromstring(text)
    pairs = []
    for tr in doc.iter("tr"):
        cells = [c for c in tr if c.tag in ("td", "th")]
        m = _PICK.match(_clean(cells[0].text_content())) if len(cells) >= 3 else None
        if m:
            a = cells[2].xpath(".//a")
            name = _clean(a[0].text_content()) if a and _clean(a[0].text_content()) else _clean(cells[2].text_content())
            if 2 <= len(name.split()) <= 4:
                pairs.append((int(m.group(1)), name))
    return _dedupe(pairs)


_HT = re.compile(r"^\d[-–]\d{1,2}$")


def parse_netscouts(text: str) -> list[tuple[int, str]]:
    """Plain table rows: pick | team | player | height | weight | pos | school | class (no links): the name is the cell
    before the height."""
    doc = html.fromstring(text)
    pairs = []
    for tr in doc.iter("tr"):
        cells = [_clean(c.text_content()) for c in tr if c.tag in ("td", "th")]
        m = _PICK.match(cells[0]) if cells else None
        hs = [i for i, c in enumerate(cells) if _HT.match(c)]
        if m and hs and hs[0] >= 2 and cells[hs[0] - 1]:
            pairs.append((int(m.group(1)), cells[hs[0] - 1]))
    return _dedupe(pairs)


def parse_draftsite(text: str) -> list[tuple[int, str]]:
    doc = html.fromstring(text)
    return _dedupe(_walk_table_rows(doc, r"/player/"))


PARSERS = {"nbadraftnet": parse_nbadraftnet, "draftexpress": parse_draftexpress, "insidehoops": parse_insidehoops,
           "tankathon": parse_tankathon, "espn": parse_espn, "ringer": parse_ringer, "cbs": parse_cbs,
           "nbadraftroom": parse_nbadraftroom, "netscouts": parse_netscouts, "draftsite": parse_draftsite,
           "bleacher": parse_bleacher, "hoopshype": parse_hoopshype}
MIN_PARSED = 25  # a page that yields fewer picks is not the board (teaser, redirect, wrong page)
MAX_TRIES = 8  # captures downloaded per (source, year) before giving up


# --------------------------------------------------------------------------- fetch + cache

def _paths(source: str, year: int) -> tuple[Path, Path]:
    d = MOCK_DIR / source
    return d / f"{year}.html", d / f"{year}.json"


def _same_class(players: list[tuple[int, str]], pool: pd.DataFrame) -> bool:
    """A board is for this draft class if at least half of its top 30 names are draftees of the year (a site's default
    mock page can already show the following year's board in June)."""
    top = [n for _, n in sorted(players)[:30]]
    hits = sum(k is not None for k in _match_names(top, pool))
    return hits >= 0.5 * len(top)


def fetch_board(source: str, year: int, pool: pd.DataFrame, verbose: bool = True) -> dict | None:
    """Cached pre-draft board for (source, year): {'url','snapshot','players':[(rank,name),...]} or None when no
    pre-draft capture of any candidate URL parses as a board of this class. Re-runs never touch the network for a
    cached year (a miss is recorded in the json sidecar too)."""
    hp, jp = _paths(source, year)
    parse = PARSERS[source]
    if jp.exists():
        meta = json.loads(jp.read_text())
        if not hp.exists():  # recorded miss
            return None
        meta["players"] = parse(hp.read_text())  # re-parsed, so parser fixes apply without a download
        meta["n_players"] = len(meta["players"])
        return meta
    urls = CANDIDATES[source](year)
    if not urls:  # the source had no board that year (site not yet / no longer running one)
        return None
    snaps = sorted({s for url in urls for s in _snapshots(url, year)}, reverse=True)
    for ts, original in snaps[:MAX_TRIES]:  # latest pre-draft capture of any candidate URL first
        text = _download(ts, original)
        if text is None:
            continue
        try:
            players = parse(text)
        except Exception:  # empty or truncated capture
            continue
        if len(players) >= MIN_PARSED and _contiguous(players) and _same_class(players, pool):
            hp.parent.mkdir(parents=True, exist_ok=True)
            hp.write_text(text)
            meta = {"source": source, "year": year, "url": original, "snapshot": ts, "n_players": len(players), "players": players}
            jp.write_text(json.dumps(meta))
            if verbose:
                print(f"{source:12s} {year} {ts[:8]} {len(players):3d} players  {original}")
            return meta
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps({"source": source, "year": year, "url": None, "snapshot": None, "n_players": 0, "players": []}))
    if verbose:
        print(f"{source:12s} {year} -- no pre-draft board found")
    return None


# --------------------------------------------------------------------------- name matching

ALIASES = {  # mock spelling (norm_name) -> draft_table key, where neither normalisation nor surname matching resolves it
    "aleksandarpavlovic": "sashapavlovic", "marcusviniciusvieiradesouza": "marcusvinicius", "eneskanter": "enesfreedom",
    "guillermohernangomez": "willyhernangomez", "timotheluwawu": "timotheluwawucabarrot", "hansenyang": "yanghansen",
    "bjboston": "brandonboston", "justinjacksonmd": "justinjackson", "justinjacksonumd": "justinjackson",
    "giannisadetokunbo": "giannisantetokounmpo", "giannisadetokoubo": "giannisantetokounmpo", "borisdiawriffiod": "borisdiaw",
}
FUZZY_MIN = 0.8
FUZZY_MIN_SURNAME = 0.7  # the surnames alone must also be close (DeAndre Kane is not DeAndre Daniels)
FUZZY_LOG = []  # (mock name, key, similarity) of every last-resort match, printed by the report for review


def _draftees() -> pd.DataFrame:
    d = pd.read_parquet(C.PROC / "draft_table.parquet", columns=["key", "draft_year", "player"])
    return d.drop_duplicates(KEYS)


_TRANSLIT = str.maketrans({"ı": "i", "İ": "I", "ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "ß": "ss", "ø": "o", "Ø": "O", "æ": "ae", "Æ": "Ae"})


def _alt(s: str) -> str:
    """norm_name after transliterating the letters NFKD cannot decompose (Aşık -> asik, not ask); dots dropped (J.R. -> jr)."""
    return norm_name(str(s).translate(_TRANSLIT).replace(".", ""))


def _tokens(s: str) -> list[str]:
    return [t for t in re.split(r"[\s\-]+", str(s).replace(".", "")) if t]


def _match_names(names: list[str], pool: pd.DataFrame) -> list[str | None]:
    """Mock names -> draft_table keys of that class: exact normalised name, transliterated name, alias table, then a
    unique surname match (first name, else first initial, breaking ties). Unmatched -> None: non-draftees, or a spelling
    to add to ALIASES."""
    keys = set(pool.key)
    alt = {_alt(p): k for k, p in zip(pool.key, pool.player)}
    by_last = {}
    for k, p in zip(pool.key, pool.player):
        toks = _tokens(p)
        if toks:
            by_last.setdefault(toks[-1].lower(), []).append((k, toks[0].lower()))
    out = []
    for n in names:
        k = norm_name(n)
        if k in keys:
            out.append(k)
            continue
        if _alt(n) in alt:
            out.append(alt[_alt(n)])
            continue
        if k in ALIASES and ALIASES[k] in keys:
            out.append(ALIASES[k])
            continue
        toks = [t.lower() for t in _tokens(n)]
        cands = by_last.get(toks[-1], []) if toks else []
        if len(cands) > 1:
            same_first = [c for c in cands if c[1] == toks[0]]
            cands = same_first or [c for c in cands if c[1][:1] == toks[0][:1]]
        if len(cands) == 1:
            out.append(cands[0][0])
            continue
        # last resort, transliteration variants (Sergey Monya / Sergei Monia): one clearly closest draftee, same initial,
        # surnames close as well
        scores = sorted(((SequenceMatcher(None, _alt(n), a).ratio(), k) for a, k in alt.items() if a[:1] == _alt(n)[:1]), reverse=True)
        if scores and scores[0][0] >= FUZZY_MIN and (len(scores) == 1 or scores[1][0] < FUZZY_MIN):
            k = scores[0][1]
            last_k = [t.lower() for t in _tokens(pool.player[pool.key == k].iloc[0])][-1]
            if toks and SequenceMatcher(None, toks[-1], last_k).ratio() >= FUZZY_MIN_SURNAME:
                FUZZY_LOG.append((n, k, round(scores[0][0], 2)))
                out.append(k)
                continue
        out.append(None)
    return out


# --------------------------------------------------------------------------- build

def _board_frame(meta: dict, year: int, pool: pd.DataFrame) -> pd.DataFrame:
    players = sorted(meta["players"])[:BOARD_LEN]
    names = [n for _, n in players]
    keys = _match_names(names, pool)
    df = pd.DataFrame({"rank": [r for r, _ in players], "key": keys}).dropna(subset=["key"]).drop_duplicates("key")
    return df.assign(draft_year=year).rename(columns={"rank": f"mock_rank_{meta['source']}"})


def build(years=None, verbose: bool = True) -> pd.DataFrame:
    years = list(years or C.DRAFT_YEARS)
    d = _draftees()
    frames, prov = [], []
    for y in years:
        pool = d[d.draft_year == y]
        base = pool[KEYS].copy()
        ranks, missing = [], {}
        for s in SOURCES:
            meta = fetch_board(s, y, pool, verbose=verbose)
            if meta is None:
                continue
            assert f"{y}{EARLIEST_MMDD}" <= meta["snapshot"] <= DRAFT_NIGHT[y] + CUTOFF_HHMM, (s, y, meta["snapshot"])  # pre-draft guard
            bf = _board_frame(meta, y, pool)
            base = base.merge(bf, on=KEYS, how="left")
            ranks.append(f"mock_rank_{s}")
            missing[f"mock_rank_{s}"] = MISSING_RANK if meta["n_players"] > 30 else MISSING_RANK_R1  # first-round-only mocks
            prov.append({"year": y, "source": s, "url": meta["url"], "snapshot_date": pd.Timestamp(meta["snapshot"][:8]).date(),
                         "n_players_parsed": meta["n_players"], "n_draftees_matched": int(bf.key.isin(pool.key).sum())})
        for s in SOURCES:
            if f"mock_rank_{s}" not in base:
                base[f"mock_rank_{s}"] = np.nan
        if not ranks:  # no board for this class: no rows
            continue
        r = base[ranks].fillna(missing)
        base["mock_rank_consensus"] = r.mean(axis=1)
        base["mock_n_sources"] = base[ranks].notna().sum(axis=1)
        base["mock_rank_std"] = r.std(axis=1, ddof=1) if len(ranks) > 1 else np.nan
        base["mock_first_round"] = (base.mock_rank_consensus <= 30).astype(int)
        frames.append(base)
    out = pd.concat(frames, ignore_index=True)
    cols = KEYS + ["mock_rank_consensus"] + [f"mock_rank_{s}" for s in SOURCES] + ["mock_n_sources", "mock_rank_std", "mock_first_round"]
    out = out[cols]
    MOCK_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prov).to_csv(MOCK_DIR / "provenance.csv", index=False)
    return out


def load_mocks() -> pd.DataFrame:
    return build(verbose=False)


# --------------------------------------------------------------------------- report

def report(out: pd.DataFrame) -> None:
    d = pd.read_parquet(C.PROC / "draft_table.parquet", columns=KEYS + ["pick"]).drop_duplicates(KEYS)
    m = d.merge(out, on=KEYS, how="left")
    prov = pd.read_csv(MOCK_DIR / "provenance.csv")
    print("\nyear  draftees  ranked(any)  n_boards  spearman(consensus, pick)   boards")
    for y, g in m.groupby("draft_year"):
        p = prov[prov.year == y]
        has = g.mock_n_sources.fillna(0) > 0
        rho = g.loc[g.mock_rank_consensus.notna(), ["mock_rank_consensus", "pick"]].corr(method="spearman").iloc[0, 1] if has.any() else np.nan
        boards = ", ".join(f"{r.source}@{r.snapshot_date}({r.n_players_parsed}p/{r.n_draftees_matched}m)" for r in p.itertuples())
        print(f"{y}  {len(g):8d}  {int(has.sum()):11d}  {len(p):8d}  {rho:26.3f}   {boards}")
    print(f"\n{len(prov)} boards; {out.shape[0]} rows; consensus available for {m.mock_rank_consensus.notna().mean():.1%} of draftees; "
          f"on >= 2 boards: {(m.mock_n_sources >= 2).mean():.1%}; on no board: {(m.mock_n_sources == 0).sum()}")
    print("fuzzy name matches (mock spelling -> key):", sorted(set(FUZZY_LOG)))


if __name__ == "__main__":
    t0 = time.time()
    res = build()
    print(f"built {res.shape} in {time.time() - t0:.0f}s")
    report(res)
