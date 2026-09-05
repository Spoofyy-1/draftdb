"""Pre-draft sportsbook markets (columns odds_*) for the 2017-2026 draft classes, keyed by (key, draft_year).

Nevada first allowed NBA-draft wagering in 2017; since then offshore and (from 2019) US books have posted draft-position
over/unders ("Trae Young 7.5: over -135 / under -105"), top-5 / top-10 / lottery / first-round yes-no markets and exact-pick
odds for the top ~20-45 prospects. Nothing here is derived from the actual selections: every page used was either a Wayback
Machine capture taken before the first pick or a live article whose published timestamp (meta article:published_time /
datePublished, recorded in the json sidecar) precedes the scheduled start of the first round. Pages published after the first
pick or recapping results are rejected (see REJECTED). One page per (site, article) is cached as
data/external/odds/<site>/<year>/<slug>.html with a <slug>.json sidecar (url, wayback timestamp or published/modified time,
rows parsed); data/external/odds/provenance.csv summarises them and rows.csv lists every market row used (for review).

Markets are parsed per article (they are prose or small tables, one layout per site; see PARSERS) into rows
(player, book, market, line, over, under, yes, no). Prose that no regex fits is transcribed by hand in HAND and every hand row
is verified against the cached text (surname and every quoted number must appear within a few hundred characters).
Per (player, year) the rows are aggregated across books:

  odds_line            median over books of the final draft-position O/U line
  odds_over_price      median American price of the over (books that quote prices)
  odds_under_price     median American price of the under
  odds_expected_pick   median over books of the vig-free implied expected pick (see expected_pick)
  odds_prob_top5/10    P(pick <= 5 / 10 / 14 / 30): median over books of the vig-free probability from explicit yes/no
  odds_prob_lottery    markets ("to be a top-10 pick", "picks 1-5 / 6-plus", "1st-round pick yes/no") and from O/U lines set
  odds_prob_first_round  exactly at 5.5 / 10.5 / 14.5 / 30.5 (the under of "5.5" is the top-5 market); NaN when no book has one
  odds_n_books         books quoting a final O/U line
  odds_line_std        sample std of the final O/U line across books (NaN with one book)
  odds_line_move       final line minus the earliest line recorded for the same book (explicit opening lines, "last week was
                       23.5", or an earlier-dated capture), median over books; NaN when no movement was recorded
  odds_days_before_draft  days between the latest capture/publication used for the player and the first round

expected_pick: American prices -> implied probabilities, the vig removed multiplicatively (p_under = q_under/(q_over+q_under)); a
one-sided quote is used raw (the missing side is 1-p); no prices -> 0.5. The pick is modelled as a normal with centre m and
sd sigma(L) = 1 + 0.2 L (uncertainty grows down the board), discretised on picks 1..60 and renormalised; m is solved so that
P(pick < L) = p_under (p_under clipped to [0.02, 0.98]) and the feature is E[pick] = sum_j j p_j. With a 50/50 price the
expected pick is (about) the line itself; a -10000 favourite at 1.5 gives ~1.0. Median prices are medians in probability space.

    from infra.builders.odds import load_odds   # cache only, no network: pages missing from the cache are skipped
    python -m infra.builders.odds               # download missing pages, rebuild, coverage / Spearman report
"""

import json
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from lxml import html

from infra import config as C
from infra.dataset import norm_name

EXT = C.ROOT / "data" / "external"
ODDS_DIR = EXT / "odds"
KEYS = ["key", "draft_year"]
YEARS = range(2017, 2027)

# scheduled tip-off of the first round (UTC); every capture / publication used is earlier than this (the first pick follows
# 20-40 minutes later). 2026's first round was Tuesday June 23 (second round June 24).
FIRST_ROUND_UTC = {2017: "2017-06-22T23:00", 2018: "2018-06-21T23:00", 2019: "2019-06-20T23:00", 2020: "2020-11-19T00:00",
                   2021: "2021-07-30T00:00", 2022: "2022-06-24T00:00", 2023: "2023-06-23T00:00", 2024: "2024-06-27T00:00",
                   2025: "2025-06-26T00:00", 2026: "2026-06-24T00:00"}
DRAFT_DATE = {y: (pd.Timestamp(t) - pd.Timedelta(hours=5)).date() for y, t in FIRST_ROUND_UTC.items()}  # US calendar day of round 1

WAYBACK = "https://web.archive.org"
_S = requests.Session()
_S.headers["User-Agent"] = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"

TOPK = {5: "top5", 10: "top10", 14: "lottery", 30: "first_round"}


# --------------------------------------------------------------------------- odds arithmetic

def american_to_prob(a) -> float:
    """Implied (vigged) probability of an American price; 'EVEN'/'Even' = +100."""
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return np.nan
    if isinstance(a, str):
        a = 100 if a.strip().lower() in ("even", "ev", "pk") else int(a.replace("\u2212", "-").replace("+", ""))
    a = float(a)
    return -a / (-a + 100) if a < 0 else 100 / (a + 100)


def fair_prob(yes, no) -> float:
    """Vig-free P(yes) from the two sides (multiplicative), the raw implied probability with one side, NaN with none."""
    py, pn = american_to_prob(yes), american_to_prob(no)
    if np.isnan(py) and np.isnan(pn):
        return np.nan
    if np.isnan(pn):
        return py
    if np.isnan(py):
        return 1 - pn
    return py / (py + pn)


_PICKS = np.arange(1, 61, dtype=float)


def expected_pick(line: float, p_under: float) -> float:
    """E[pick] under a normal pick distribution (centre m, sd 1 + 0.2 L) discretised on picks 1..60 and renormalised, with m
    solved (bisection) so that P(pick < L) = p_under. Truncation keeps a -10000 favourite at ~1.0 rather than below 1."""
    p = 0.5 if np.isnan(p_under) else min(max(p_under, 0.02), 0.98)
    sigma = 1 + 0.2 * line

    def weights(m):
        w = np.exp(-0.5 * ((_PICKS - m) / sigma) ** 2)
        return w / w.sum()

    lo, hi = -80.0, 140.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if weights(mid)[_PICKS < line].sum() > p:  # too much mass below the line: the centre must move down the board
            lo = mid
        else:
            hi = mid
    return float((weights((lo + hi) / 2) * _PICKS).sum())


def median_price(prices) -> float:
    """Median of American prices taken in probability space (the median of +101 and -102 is not -0.5), back to American."""
    ps = [american_to_prob(x) for x in prices if pd.notna(x)]
    if not ps:
        return np.nan
    p = float(np.median(ps))
    return float(round(-100 * p / (1 - p))) if p >= 0.5 else float(round(100 * (1 - p) / p))


# --------------------------------------------------------------------------- html -> text

_BLOCK = {"p", "div", "li", "tr", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6", "br", "table", "ul", "ol", "section",
          "article", "header", "footer", "blockquote", "figcaption", "dt", "dd"}


def page_text(raw: bytes) -> str:
    """Visible text, one line per block element (table cells and paragraphs become lines), whitespace collapsed."""
    doc = html.fromstring(raw)
    for bad in doc.xpath("//script|//style|//noscript|//svg|//iframe"):
        bad.drop_tree()
    for el in doc.iter():
        if isinstance(el.tag, str) and el.tag.lower() in _BLOCK:
            el.tail = "\n" + (el.tail or "")
    t = doc.text_content().replace("\xa0", " ").replace("\u2212", "-").replace("\u2019", "'").replace("\u2018", "'")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r" *\n *", "\n", t)
    return re.sub(r"\n{2,}", "\n", t).strip()


def _meta_dates(raw: bytes) -> tuple[str | None, str | None]:
    t = raw.decode("utf-8", errors="replace")
    pub = re.findall(r'(?:article:published_time|datePublished)["\']?\s*(?:content=|:)\s*["\']([^"\']+)', t)
    mod = re.findall(r'(?:article:modified_time|dateModified)["\']?\s*(?:content=|:)\s*["\']([^"\']+)', t)
    return (pub[0] if pub else None), (mod[0] if mod else None)


# --------------------------------------------------------------------------- row helpers

NAME_RE = r"[A-Z][A-Za-z'.\-]+(?: (?:[A-Z][A-Za-z'.\-]+|da|de|van|Jr\.?|II|III|IV)){0,4}"
PRICE = r"[+-]\d{3,5}|EVEN|Even|EV"


def _row(player, line=None, over=None, under=None, k=None, yes=None, no=None, prob=None, book=None, stage="close", date=None):
    return {"player": player.strip(), "line": line, "over": over, "under": under, "k": k, "yes": yes, "no": no, "prob": prob,
            "book": book, "stage": stage, "date": date}


def _num(s):
    return None if s is None else float(s)


def _price(s):
    if s is None:
        return None
    s = s.strip().replace("\u2212", "-")
    if s.lower() in ("even", "ev", "pk"):
        return 100
    return int(s.replace("+", ""))


def _section(text: str, start: str, end: str | None = None) -> str:
    """Text between the first line containing `start` and the next line containing `end` ('' -> to the end)."""
    i = text.find(start)
    if i < 0:
        return ""
    j = text.find(end, i + len(start)) if end else -1
    return text[i:j] if j > 0 else text[i:]


# --------------------------------------------------------------------------- parsers (one per page layout) -> rows

def p_sportsinsights(t):
    """'Jayson Tatum 4.5' / 'Over (-195 to +240)' / 'Under (+160 to -300)': the June 16 opening and the June 22 line."""
    rows = []
    pat = re.compile(r"^(?P<name>[A-Z][^\n\d]{3,40}?) (?P<l1>\d+(?:\.5)?)(?: to (?P<l2>\d+(?:\.5)?))?\n"
                     r"Over \((?P<o1>[+-]\d+)(?: to (?P<o2>[+-]\d+))?\)\n"
                     r"Under \(\+?(?P<u1>[+-]?\d+)(?: to \+?(?P<u2>[+-]?\d+))?\)", re.M)
    for m in pat.finditer(t):
        u1 = m["u1"] if m["u1"].startswith(("+", "-")) else "+" + m["u1"]
        u2 = m["u2"] and (m["u2"] if m["u2"].startswith(("+", "-")) else "+" + m["u2"])
        rows.append(_row(m["name"], _num(m["l1"]), _price(m["o1"]), _price(u1), stage="open", date="2017-06-16"))
        rows.append(_row(m["name"], _num(m["l2"] or m["l1"]), _price(m["o2"] or m["o1"]), _price(u2 or u1)))
    return rows


def p_name_line_table(t, start, end):
    """Two-line table cells: 'Lonzo Ball' / '2.5' (no prices)."""
    sec = _section(t, start, end)
    return [_row(m["name"], _num(m["l"])) for m in re.finditer(rf"^(?P<name>{NAME_RE})\n(?P<l>\d+\.5)$", sec, re.M)]


def p_forbes(t):
    return p_name_line_table(t, "Draft Position Over/Under", "College Selection")


def p_sbd2018(t):
    rows = [_row(m["name"], _num(m["l"]), _price(m["o"]), _price(m["u"]))
            for m in re.finditer(rf"^(?P<name>{NAME_RE})\n(?P<l>\d+\.5)\n(?P<o>[+-]\d+)o/(?P<u>[+-]\d+)u$", t, re.M)]
    m = re.search(r"Will (?P<name>[A-Z][A-Za-z' ]+) be a 1st Round Pick\?\nNo\nYes\n(?P<no>[+-]\d+)\n(?P<yes>[+-]\d+)", t)
    if m:
        rows.append(_row(m["name"], k=30, yes=_price(m["yes"]), no=_price(m["no"])))
    return rows


def p_sn_betonline(t):
    """'3rd or Worse -220' / '2nd or Better +175' = over/under 2.5; 'Will X be a 1st-round pick? No -140 / Yes +110'."""
    rows = []
    pat = re.compile(rf"^(?P<name>{NAME_RE})\n(?P<a>\d+)(?:st|nd|rd|th) or (?P<ad>Worse|Better) (?P<ap>[+-]\d+)\n"
                     r"(?P<b>\d+)(?:st|nd|rd|th) or (?P<bd>Worse|Better) (?P<bp>[+-]\d+)$", re.M)
    for m in pat.finditer(t):
        sides = {}
        for n, d, p in ((m["a"], m["ad"], m["ap"]), (m["b"], m["bd"], m["bp"])):
            sides["over" if d == "Worse" else "under"] = (int(n) - 0.5 if d == "Worse" else int(n) + 0.5, _price(p))
        if len(sides) == 2 and sides["over"][0] == sides["under"][0]:
            rows.append(_row(m["name"], sides["over"][0], sides["over"][1], sides["under"][1]))
    for m in re.finditer(rf"Will (?P<name>{NAME_RE}) be a 1st-round pick\?\n(?P<s1>Yes|No) (?P<p1>[+-]\d+)\n(?P<s2>Yes|No) (?P<p2>[+-]\d+)", t):
        pr = {m["s1"]: _price(m["p1"]), m["s2"]: _price(m["p2"])}
        rows.append(_row(m["name"], k=30, yes=pr.get("Yes"), no=pr.get("No")))
    return rows


def p_ou_colon(t, head):
    """'<head pattern with (?P<name>)>' then 'Over 6.5: -120' / 'Under 6.5: -120' (Bovada, BetAnySports, RotoWire)."""
    pat = re.compile(head + r"\n?Over (?P<l>\d+\.5): (?P<o>" + PRICE + r")\n?Under \d+\.5: (?P<u>" + PRICE + r")", re.M)
    return [_row(m["name"], _num(m["l"]), _price(m["o"]), _price(m["u"])) for m in pat.finditer(t)]


def p_bovada2018(t):
    return p_ou_colon(t, rf"^Draft Position [\u2013\-] (?P<name>{NAME_RE})$")


def p_betdsi(t):
    pat = re.compile(rf"^(?P<name>{NAME_RE}) draft position\nOver (?P<l>\d+\.5) \((?P<o>[+-]\d+)\)\nUnder \d+\.5 \((?P<u>[+-]\d+)\)", re.M)
    return [_row(m["name"], _num(m["l"]), _price(m["o"]), _price(m["u"])) for m in pat.finditer(t)]


def p_pointsbet2019(t):
    sec = _section(t, "Draft Positions", "Duke Specials")
    return [_row(m["name"], _num(m["l"]), _price(m["o"]), _price(m["u"]))
            for m in re.finditer(rf"^(?P<name>{NAME_RE})\n(?P<l>\d+\.5)\n(?P<o>[+-]\d+)\n(?P<u>[+-]\d+)$", sec, re.M)]


def p_fanduel2019(t):
    """'Garland picks 1-5 (-550)/picks 6-plus (+350)': under 5.5 -550, over +350; 'Will Carsen Edwards be drafted in the first round?'"""
    rows = [_row(m["name"], int(m["k"]) + 0.5, _price(m["o"]), _price(m["u"]))
            for m in re.finditer(rf"(?P<name>{NAME_RE}) picks 1-(?P<k>\d+) \((?P<u>[+-]\d+)\)/(?:picks |plus )\d+-plus \((?P<o>[+-]\d+)\)", t)]
    m = re.search(rf"Will (?P<name>{NAME_RE}) be drafted in the first round\?\s*Yes \((?P<yes>[+-]\d+)\); No \((?P<no>[+-]\d+)\)", t)
    if m:
        rows.append(_row(m["name"], k=30, yes=_price(m["yes"]), no=_price(m["no"])))
    return rows


def p_table4(t, start=None, end=None):
    """'Aaron Henry' / '43.5' / '-120' / '-110' (SportsBettingDime's DraftKings tables)."""
    sec = _section(t, start, end) if start else t
    return [_row(m["name"], _num(m["l"]), _price(m["o"]), _price(m["u"]))
            for m in re.finditer(rf"^(?P<name>{NAME_RE})\n(?P<l>\d+\.5)\n(?P<o>[+-]\d+)\n(?P<u>[+-]\d+)$", sec, re.M)]


def p_sbd_table(t):
    return p_table4(t, "Draft Position", None)


def p_an_consensus(t):
    """'Anthony Edwards' / '1' (mock consensus) / '1.5 (+220/-313) [Bet Now]' (Action Network 2020, NBA.com 2021)."""
    return [_row(m["name"], _num(m["l"]), _price(m["o"]), _price(m["u"]))
            for m in re.finditer(rf"^(?P<name>{NAME_RE})\n\d{{1,2}}\n(?P<l>\d+\.5) \((?P<o>[+-]\d+)/(?P<u>[+-]\d+)\)", t, re.M)]


def p_thescore2020(t):
    rows = p_name_line_table(t, "Draft spot props", "Draft spot props open")
    for m in re.finditer(rf"(?P<name>{NAME_RE}) (?P<side>under|over) (?P<l>\d+\.5) \((?P<p>[+-]\d+)\)", t):
        rows.append(_row(m["name"], _num(m["l"]), **{m["side"]: _price(m["p"])}))
    return rows


def p_rotowire(t):
    return p_ou_colon(t, rf"^(?P<name>{NAME_RE})$")


def p_betus(t):
    """Rotation-number board: 'Anthony Edwards NBA Draft Position' / 'Moneyline' / '1301' / 'Over 1½ Draft Position' / '-135' ..."""
    pat = re.compile(rf"^(?P<name>{NAME_RE}) NBA Draft Position\nMoneyline\n\d+\nOver (?P<l>\d+)½ Draft Position\n(?P<o>[+-]\d+)\n"
                     r"\d+\nUnder \d+½ Draft Position(?:\n(?P<u>[+-]\d+))?", re.M)
    return [_row(m["name"], int(m["l"]) + 0.5, _price(m["o"]), _price(m["u"])) for m in pat.finditer(t)]


def p_cbs_wh2021(t):
    return [_row(m["name"], _num(m["l"])) for m in re.finditer(rf"(?P<name>{NAME_RE}) draft position\s*Prop: Over/under (?P<l>\d+\.5)", t)]


def p_side_paren(t, book_from_text=False):
    """'Keegan Murray UNDER 5.5 (-285)', 'Jarace Walker Under 6.5 (+100, DraftKings)', 'Draft Position: Dereck Lively U10.5 (+175)'."""
    rows = []
    pat = re.compile(rf"(?P<name>{NAME_RE}):? (?P<side>UNDER|OVER|Under|Over|U|O)\s?(?P<l>\d+\.5) \((?P<p>[+-]\d+)(?:, (?P<book>[A-Za-z0-9]+))?\)")
    for m in pat.finditer(t):
        side = "under" if m["side"].lower().startswith("u") else "over"
        rows.append(_row(m["name"], _num(m["l"]), **{side: _price(m["p"])}, book=m["book"] if book_from_text else None))
    return rows


def p_dkn(t):
    rows = p_side_paren(_section(t, "Draft Position:") or t)
    for m in re.finditer(rf"1st Round Pick: (?P<name>{NAME_RE}) \((?P<p>[+-]\d+)\)", t):
        rows.append(_row(m["name"], k=30, yes=_price(m["p"])))
    for m in re.finditer(rf"Drafted Top-5: (?P<name>{NAME_RE}) \((?P<p>[+-]\d+)\)", t):
        rows.append(_row(m["name"], k=5, yes=_price(m["p"])))
    return rows


def p_an2023(t):
    rows = p_side_paren(t, book_from_text=True)
    for m in re.finditer(rf"(?P<name>{NAME_RE}) Top-(?P<k>\d+) \((?P<p>[+-]\d+), (?P<book>[A-Za-z0-9]+)\)", t):
        if int(m["k"]) in TOPK:
            rows.append(_row(m["name"], k=int(m["k"]), yes=_price(m["p"]), book=m["book"]))
    return rows


def p_joeduffy(t):
    """'A.J. Griffin draft position' / 'Over/Under 11.5' / '(Last week was 23.5)' -- SportsBetting.ag, no prices."""
    rows = []
    pat = re.compile(rf"^(?P<name>{NAME_RE}) draft position\nOver/Under (?P<l>\d+(?:\.5)?)(?:\n\(Last week was (?P<prev>\d+(?:\.5)?)\))?", re.M)
    for m in pat.finditer(t):
        rows.append(_row(m["name"], _num(m["l"])))
        if m["prev"]:
            rows.append(_row(m["name"], _num(m["prev"]), stage="open", date="2022-06-14"))
    return rows


def p_covers2023(t):
    pat = re.compile(rf"^(?P<name>{NAME_RE})\nOver (?P<l>\d+\.5) \((?P<o>[+-]\d+)\)\nUnder \d+\.5 \((?P<u>[+-]\d+)\)", re.M)
    return [_row(m["name"], _num(m["l"]), _price(m["o"]), _price(m["u"])) for m in pat.finditer(t)]


def p_sn_ou_sections(t):
    """Sporting News 2022/2023: 'Keegan Murray UNDER 5.5 (-285)' plus a 'top 10' section 'Jalen Duren (-120)'."""
    rows = p_side_paren(t)
    sec = _section(t, "Best value to be picked top 10", "Basketball\n")
    for m in re.finditer(rf"(?P<name>{NAME_RE}) \((?P<p>[+-]\d+)\)", sec):
        rows.append(_row(m["name"], k=10, yes=_price(m["p"])))
    return rows


def p_stltoday(t):
    rows = [_row(m["name"], _num(m["l"])) for m in re.finditer(rf"(?P<name>{NAME_RE}): Over/Under (?P<l>\d+\.5)", t)]
    sec = _section(t, "Favorites to make Top 10", "Favorites to make Top 20")
    rows += [_row(m["name"], k=10, yes=_price(m["p"])) for m in re.finditer(rf"(?P<name>{NAME_RE}) \((?P<p>[+-]\d+) or better\)", sec)]
    return rows


def p_betanysports(t):
    return p_ou_colon(t, rf"^(?P<name>{NAME_RE}) NBA Draft Position$")


def p_covers2024(t):
    pat = re.compile(rf"^(?P<name>{NAME_RE}) Draft Position\nOver (?P<l>\d+\.5)\n(?P<o>[+-]\d+)\nUnder \d+\.5\n(?P<u>[+-]\d+)", re.M)
    return [_row(m["name"], _num(m["l"]), _price(m["o"]), _price(m["u"])) for m in pat.finditer(t)]


def p_sds2024(t):
    rows = []
    for m in re.finditer(rf"(?P<name>{NAME_RE}) draft position: (?P<side>Under|Over) (?P<l>\d+\.5) \((?P<p>[+-]\d+)\) at (?P<book>[A-Za-z0-9]+)", t):
        rows.append(_row(m["name"], _num(m["l"]), **{m["side"].lower(): _price(m["p"])}, book=m["book"]))
    m = re.search(r"At bet365, Dillingham's draft position over/under is (\d+\.5)", t)
    if m:
        rows.append(_row("Rob Dillingham", _num(m[1]), book="bet365"))
    return rows


def p_yes_list(sec):
    """'Kon Knueppel' / '-1000' or 'Kon Knueppel   -900' lists of one-sided yes prices."""
    return [(m["name"], _price(m["p"])) for m in re.finditer(rf"^(?P<name>{NAME_RE})\s*\n?\s*(?P<p>{PRICE})$", sec, re.M)]


def p_lvsb2025(t):
    """Bovada board: 'Ace Bailey - Draft Position' / 'Over 3.5 -850' / 'Under 3.5 +475'; then top-5/10/20 and 1st-round lists."""
    rows = []
    pat = re.compile(rf"^(?P<name>{NAME_RE}) [\u2013\-] Draft Position\nOver (?P<l>\d+\.5) (?P<o>{PRICE})(?:\nUnder \d+\.5 (?P<u>{PRICE}))?", re.M)
    for m in pat.finditer(t):
        rows.append(_row(m["name"], _num(m["l"]), _price(m["o"]), _price(m["u"])))
    for head, nxt, k in (("To be Drafted in the Top 5", "To be Drafted in the Top 10", 5), ("To be Drafted in the Top 10", "To be Drafted in the Top 20", 10),
                         ("1st Round Pick", "NBA Odds source", 30)):
        rows += [_row(n, k=k, yes=p) for n, p in p_yes_list(_section(t, head, nxt))]
    return rows


def p_thescore2025(t):
    rows = [_row(m["name"], _num(m["l"]), over=_price(m["o"]))
            for m in re.finditer(rf"^(?P<name>{NAME_RE})\n(?P<l>\d+\.5) \((?P<o>[+-]\d+)\)$", t, re.M)]
    for m in re.finditer(rf"(?P<name>[A-Z][a-z]+) (?P<side>over|under) (?P<l>\d+\.5) \((?P<p>[+-]\d+)\)", t):
        rows.append(_row(m["name"], _num(m["l"]), **{m["side"]: _price(m["p"])}))
    m = re.search(r"(?P<name>[A-Z][a-z]+) selected in 1st round \((?P<p>[+-]\d+)\)", t)
    if m:
        rows.append(_row(m["name"], k=30, yes=_price(m["p"])))
    return rows


def p_betonline2025(t):
    sec = _section(t, "NBA Draft Over/Unders", "Gambling News")
    return [_row(m["name"], _num(m["l"])) for m in re.finditer(rf"^(?P<name>{NAME_RE})\nOver/Under (?P<l>\d+\.5)$", sec, re.M)]


def p_cbs2025(t):
    rows = []
    sec = _section(t, "Player draft positions", "Who will be drafted first")
    pat = re.compile(rf"^(?P<name>{NAME_RE})\nOver (?P<l>\d+\.5) \((?P<o>[+-]\d+)\)\nUnder \d+\.5 \((?P<u>[+-]\d+)\)", re.M)
    rows += [_row(m["name"], _num(m["l"]), _price(m["o"]), _price(m["u"])) for m in pat.finditer(sec)]
    for head, nxt, k in (("To be a top-five pick", "To be a top-10 pick", 5), ("To be a top-10 pick", "To be a first-round pick", 10),
                         ("To be a first-round pick", "Player draft positions", 30)):
        rows += [_row(n, k=k, yes=p) for n, p in p_yes_list(_section(t, head, nxt))]
    return rows


def p_sn2026(t):
    rows = []
    for head, nxt, k in (("NBA Draft top 5 pick odds", "NBA Draft to be drafted in the top 5 best bet", 5),
                         ("NBA Draft top 10 pick odds", "NBA Draft to be drafted in the top 10 best bet", 10)):
        rows += [_row(n, k=k, yes=p) for n, p in p_yes_list(_section(t, head, nxt))]
    return rows


def p_rti2026(t):
    sec = _section(t, "Nate Ament to be drafted in the top 10", "DraftKings \u201cto be drafted in the top 10\u201d odds")
    return [_row(m["name"], k=10, yes=_price(m["p"]))
            for m in re.finditer(rf"^(?P<name>{NAME_RE})\n(?P<p>[+-]\d+)\n\$", sec, re.M)]


def p_hand(_t):
    return []


PARSERS = {"sportsinsights": p_sportsinsights, "forbes": p_forbes, "sbd2018": p_sbd2018, "sn_betonline": p_sn_betonline,
           "bovada2018": p_bovada2018, "betdsi": p_betdsi, "pointsbet2019": p_pointsbet2019, "fanduel2019": p_fanduel2019,
           "sbd_table": p_sbd_table, "an_consensus": p_an_consensus, "thescore2020": p_thescore2020, "rotowire": p_rotowire,
           "betus": p_betus, "cbs_wh2021": p_cbs_wh2021, "dkn": p_dkn, "an2023": p_an2023, "joeduffy": p_joeduffy,
           "covers2023": p_covers2023, "sn_ou": p_sn_ou_sections, "side_paren": p_side_paren, "stltoday": p_stltoday,
           "betanysports": p_betanysports, "covers2024": p_covers2024, "sds2024": p_sds2024, "lvsb2025": p_lvsb2025,
           "thescore2025": p_thescore2025, "betonline2025": p_betonline2025, "cbs2025": p_cbs2025, "sn2026": p_sn2026,
           "rti2026": p_rti2026, "hand": p_hand}


# --------------------------------------------------------------------------- hand-transcribed prose (verified against the page)

def H(player, line=None, over=None, under=None, k=None, yes=None, no=None, prob=None, book=None, stage="close", date=None):
    return _row(player, line, over, under, k, yes, no, prob, book, stage, date)


HAND = {
    # OddsShark consensus prose: "Luka Doncic is going off at +175 to be drafted second or better ... -220 to fall to third or lower"
    ("sbnation", 2018): [H("Luka Doncic", 2.5, -220, 175), H("Marvin Bagley", 3.5, 260, -350), H("Jaren Jackson Jr.", 3.5, -150, 120),
                         H("Michael Porter Jr.", 6.5, 120, -150), H("Trae Young", 7.5, 110, -140)],
    ("actionnetwork", 2018): [H("Jaren Jackson", 3.5, -230, 170)],
    ("actionnetwork_garland", 2019): [H("Darius Garland", 5.5, under=-340)],
    ("actionnetwork_cheat", 2019): [H("Coby White", 6.5, under=130), H("Darius Garland", 5.5, under=-340), H("De'Andre Hunter", 5.5, -130, 111)],
    ("lakersnation", 2019): [H("Zion Williamson", 1.5, 1400, -10000), H("Cam Reddish", 7.5, over=-190), H("Jarrett Culver", 5.5, under=-190),
                             H("DeAndre Hunter", 5.5, over=-130), H("Darius Garland", 5.5, under=-400)],
    # SI's DraftKings board is a screenshot (img-3736.jpg, cached next to the page): the ten visible lines + Stanley from the text
    ("si", 2020): [H("LaMelo Ball", 2.5, 190, -240), H("James Wiseman", 2.5, -110, -110), H("Anthony Edwards", 1.5, -115, -106),
                   H("Obi Toppin", 4.5, -250, 200), H("Deni Avdija", 4.5, 100, -121), H("Tyrese Haliburton", 7.5, 125, -155),
                   H("Killian Hayes", 7.5, -240, 190), H("Devin Vassell", 11.5, 105, -127), H("Cole Anthony", 20.5, -115, -106),
                   H("Isaac Okoro", 8.5, 118, -143), H("Cassius Stanley", 40.5)],
    ("williamhill", 2021): [H("James Bouknight", 7.5, 140, -170), H("Franz Wagner", 9.5, 125, -155), H("Josh Giddey", 9.5, -155, 125),
                            H("Josh Giddey", 10.5, under=-165, stage="open"), H("Sharife Cooper", 21.5), H("Sharife Cooper", 20.5, stage="open"),
                            H("Quentin Grimes", 31.5), H("Quentin Grimes", 32.5, stage="open"), H("Joel Ayayi", 41.5), H("Joel Ayayi", 40.5, stage="open"),
                            H("Davion Mitchell", 10.5), H("Moses Moody", 11.5), H("Corey Kispert", 13.5), H("JT Thor", 31.5),
                            H("Josh Christopher", 33.5), H("Luka Garza", 54.5)],
    ("actionnetwork", 2021): [H("Keon Johnson", 8.5, over=-150, book="BetMGM"), H("Davion Mitchell", 8.5, over=-105, book="BetMGM"),
                              H("Moses Moody", 11.5, under=-110, book="BetMGM"), H("Josh Giddey", 11.5), H("Alperen Sengun", 13.5, under=-180, book="PointsBet"),
                              H("Usman Garuba", 15.5)],
    ("covers", 2022): [H("Shaedon Sharpe", 7.5, under=100, book="bet365"), H("Jalen Duren", 10.5, under=-108, book="FanDuel")],
    ("covers", 2023): [H("Taylor Hendricks", 8.5, under=-150, book="bet365"), H("Gregory Jackson", 26.5, over=-160, book="bet365"),
                       H("Bilal Coulibaly", 11.5, under=-132, book="FanDuel"), H("Bilal Coulibaly", 11.5, under=-110, book="bet365"),
                       H("Bilal Coulibaly", k=10, yes=220, book="DraftKings"), H("Anthony Black", 8.5, under=-270, book="bet365")],
    ("dknetwork_popular", 2025): [H("Derik Queen", 11.5, over=-140), H("Derik Queen", 10.5, stage="open", date="2025-06-17"),
                                  H("Kon Knueppel", 5.5), H("Kon Knueppel", 6.5, stage="open", date="2025-06-17"), H("Kon Knueppel", k=5, yes=-1000),
                                  H("Kon Knueppel", k=5, yes=-150, stage="open", date="2025-06-23")],
    ("actionnetwork", 2026): [H("LeBaron Philon", 17.5, under=-110, book="Caesars"), H("Darius Acuff", 6.5), H("Cameron Carr", 16.5),
                              H("Yaxel Lendeborg", 13.5, under=-105, book="bet365"), H("Jayden Quaintance", 23.5),
                              H("Nate Ament", k=10, prob=0.39, book="Kalshi"), H("Izan Almansa", k=10, prob=0.60, book="Kalshi"),
                              H("LeBaron Philon", k=10, prob=0.09, book="Kalshi"), H("Yaxel Lendeborg", k=10, prob=0.18, book="Kalshi"),
                              H("Mikel Brown", k=10, prob=0.05, book="Kalshi"), H("Joshua Jefferson", k=30, prob=0.60, book="Kalshi"),
                              H("Zuby Ejiofor", k=30, prob=0.84, book="Kalshi"), H("Jayden Quaintance", k=30, prob=0.39, book="Kalshi")],
    ("rawchili", 2026): [H("Mikel Brown", 6.5, under=-165), H("Darius Acuff", 6.5, under=-190), H("Keaton Wagler", 7.5, under=-245),
                         H("Kingston Flemings", 9.5, under=-275), H("Kingston Flemings", k=5, yes=750)],
    ("savannahherald", 2026): [H("Keaton Wagler", 7.5, under=-290), H("Keaton Wagler", 7.5, under=-450, book="Caesars"), H("Aday Mara", 9.5),
                               H("Yaxel Lendeborg", 13.5, book="Caesars"), H("Darius Acuff", 6.5, over=-115, book="Caesars")],
    ("rg", 2026): [H("Nate Ament", 10.5, 125, -161, book="20Bet")],
}


def _verify_hand(rows, text, source, year):
    """Every hand row: the surname must occur in the page and each quoted price (and the line, when the page states it as a
    number rather than 'third or better') within ~1000 characters of some occurrence of the surname."""
    flat = text.replace("\n", " ")
    for r in rows:
        if source == "si" and year == 2020 and r["player"] != "Cassius Stanley":
            continue  # transcribed from the cached screenshot, not from the text
        last = _tokens(r["player"])[-1]
        nums = [str(v) for v in (r["over"], r["under"], r["yes"], r["no"]) if v is not None]
        if r["prob"] is not None:
            nums.append(str(int(round(r["prob"] * 100))))
        if r["line"] is not None and str(r["line"]) in flat:
            nums.append(str(r["line"]))
        nums = [n[:-2] if n.endswith(".0") else n for n in nums]
        ok = False
        for m in re.finditer(re.escape(last), flat, re.I):
            window = flat[max(0, m.start() - 300): m.end() + 1000]
            if all(re.search(rf"(?<![\d.]){re.escape(n)}(?![\d])", window) for n in nums):
                ok = True
                break
        if not ok:
            raise ValueError(f"hand row not found in {source} {year}: {r['player']} {nums}")


# --------------------------------------------------------------------------- source registry

class Src:
    def __init__(self, year, site, slug, book, url, parser, wayback=None, published=None, note=""):
        self.year, self.site, self.slug, self.book, self.url = year, site, slug, book, url
        self.parser, self.wayback, self.published, self.note = parser, wayback, published, note

    @property
    def paths(self):
        d = ODDS_DIR / self.site / str(self.year)
        return d / f"{self.slug}.html", d / f"{self.slug}.json"


SI_IMG = "https://www.si.com/.image/t_share/MTc2ODg0NzY5ODMxNzkwNTM0/img-3736.jpg"

SOURCES = [
    # 2017: the first year with player props (offshore books; Nevada only allowed school totals)
    Src(2017, "sportsinsights", "position_overunders", "BetOnline", "https://www.sportsinsights.com/blog/2017-nba-draft-position-overunders/", "sportsinsights"),
    Src(2017, "forbes", "5dimes_preview", "5Dimes", "https://www.forbes.com/sites/alexkay/2017/06/21/2017-nba-draft-latest-odds-preview-predictions-and-prop-betting-picks-for-top-prospects/",
        "forbes", wayback="20170621134915"),
    # 2018
    Src(2018, "sportsbettingdime", "villanova_props", "SBD (book unnamed)", "https://www.sportsbettingdime.com/news/nba/2018-draft-props-villanova/", "sbd2018"),
    Src(2018, "sportingnews", "betonline_props", "BetOnline", "https://www.sportingnews.com/ca/nba/news/nba-draft-2018-odds-picks-props-deandre-ayton-luka-doncic-marvin-bagley-no-1-overall-lottery/1d57fv0cixmfm1u2gi30tmar19", "sn_betonline"),
    Src(2018, "bustedcoverage", "bovada_props", "Bovada", "https://bustedcoverage.com/2018/06/21/for-the-degenerates-nba-draft-prop-bets/", "bovada2018"),
    Src(2018, "getmoresports", "betdsi_props", "BetDSI", "https://www.getmoresports.com/2018-nba-draft-prop-bets/", "betdsi"),
    Src(2018, "sbnation", "oddsshark_preview", "OddsShark consensus", "https://www.sbnation.com/odds/2018/6/20/17481856/nba-draft-2018-odds-betting-preview-deandre-ayton-first-overall", "hand"),
    Src(2018, "actionnetwork", "jaren_jackson", "Action Network (book unnamed)", "https://www.actionnetwork.com/nba/nba-draft-jaren-jackson-pick-betting-prop-selection", "hand"),
    # 2019
    Src(2019, "crossingbroad", "pointsbet_board", "PointsBet", "https://www.crossingbroad.com/2019/06/pointsbet-releases-nba-draft-betting-odds.html", "pointsbet2019"),
    Src(2019, "audacy", "fanduel_guide", "FanDuel", "https://www.audacy.com/articles/2019-nba-draft-betting-guide-props-odds-best-bet", "fanduel2019"),
    Src(2019, "lakersnation", "oddsshark_preview", "OddsShark consensus", "https://lakersnation.com/2019-nba-draft-odds-pelicans-control-board-anthony-davis-trade-lakers/", "hand"),
    Src(2019, "actionnetwork_garland", "garland_prop", "Action Network (book unnamed)", "https://www.actionnetwork.com/nba/2019-nba-draft-props-darius-garland-knicks-pelicans-cavaliers", "hand"),
    Src(2019, "actionnetwork_cheat", "cheat_sheet", "Action Network (book unnamed)", "https://www.actionnetwork.com/nba/nba-draft-odds-prop-betting-darius-garland-rj-barret-deandre-hunter-cheat-sheet", "hand",
        note="published 18:41 ET, first round tipped 19:00 ET"),
    # 2020 (November draft)
    Src(2020, "si", "draftkings_preview", "DraftKings", "https://www.si.com/betting/2020/11/17/nba-draft-betting-preview", "hand", note="lines transcribed from the cached screenshot img-3736.jpg"),
    Src(2020, "actionnetwork", "pointsbet_consensus", "PointsBet", "https://www.actionnetwork.com/nba/nba-draft-odds-lamelo-ball-anthony-edwards-james-wiseman-mock-draft-2020", "an_consensus"),
    Src(2020, "actionnetwork", "pointsbet_consensus_nov13", "PointsBet", "https://www.actionnetwork.com/nba/nba-draft-odds-lamelo-ball-anthony-edwards-james-wiseman-mock-draft-2020", "an_consensus", wayback="20201113124923"),
    Src(2020, "thescore", "draft_spot_props", "theScore Bet", "https://thescore.com/nba/news/2054045", "thescore2020"),
    Src(2020, "rotowire", "draftkings_props", "DraftKings", "https://www.rotowire.com/basketball/article/2020-nba-draft-prop-betting-guide-53840", "rotowire"),
    Src(2020, "gambling911", "betus_props", "BetUS", "https://www.gambling911.com/2020-NBA-Draft-Prop-Bets.html", "betus", published="2020-11-17T12:00:00Z"),
    # 2021 (July draft)
    Src(2021, "sportsbettingdime", "draftkings_overunders", "DraftKings", "https://www.sportsbettingdime.com/news/nba/2021-draft-props-odds-over-unders-when-players-will-be-selected/", "sbd_table", wayback="20210727154351"),
    Src(2021, "williamhill", "position_props", "William Hill (NV)", "https://www.williamhill.us/2021-nba-draft-25-player-position-props-released/", "hand"),
    Src(2021, "cbssports", "williamhill_props", "William Hill (NV)", "https://www.cbssports.com/nba/news/2021-nba-draft-odds-and-prop-bets-cade-cunningham-heavy-favorite-to-go-no-1-to-detroit-pistons/", "cbs_wh2021", wayback="20210729015859"),
    Src(2021, "nba", "betmgm_cheat_sheet", "BetMGM", "https://www.nba.com/nbabet/2021-nba-draft-cheat-sheet-betting-odds-consensus-mock-player-analysis-more", "an_consensus"),
    Src(2021, "actionnetwork", "early_props", None, "https://www.actionnetwork.com/nba/2021-nba-draft-betting-picks-preview-prop-bets", "hand"),
    # 2022
    Src(2022, "sportsbettingdime", "draftkings_overunders", "DraftKings", "https://www.sportsbettingdime.com/news/nba/draft-odds-1st-overall-top-5-player-over-unders-head-to-head-picks/", "sbd_table", wayback="20220623162102"),
    Src(2022, "sportingnews", "draft_day_guide", "DraftKings/FanDuel", "https://www.sportingnews.com/us/nba/news/updated-2022-nba-draft-betting-guide-best-bets-props-top-10-odds/ekclxxgij4g2wypkshqgvhim", "sn_ou", wayback="20220623163158"),
    Src(2022, "sportingnews", "guide_jun16", "DraftKings/FanDuel", "https://www.sportingnews.com/us/nba/news/2022-nba-draft-updated-betting-guide-who-will-go-no-1-best-bets-top-picks-player-draft-position-props-and-top-10-odds/kqrxvxzu5oujujh34kjj1ihm", "side_paren"),
    Src(2022, "joeduffy", "sportsbetting_ag_lines", "SportsBetting.ag", "https://www.joeduffy.net/blog/2022/06/nba-draft-props-2022-major-line-movements/", "joeduffy", published="2022-06-21T12:00:00Z"),
    Src(2022, "covers", "mock_draft_v2", None, "https://www.covers.com/nba/2022-mock-draft-version-2", "hand"),
    # 2023
    Src(2023, "covers", "draft_prop_picks", "DraftKings", "https://www.covers.com/nba/draft-prop-picks-2023", "covers2023", wayback="20230622045420"),
    Src(2023, "dknetwork", "best_bets_jun22", "DraftKings", "https://dknetwork.draftkings.com/2023/06/22/2023-nba-draft-best-bets-on-draftkings-sportsbook-june-22/", "dkn"),
    Src(2023, "dknetwork", "best_bets_jun6", "DraftKings", "https://dknetwork.draftkings.com/2023/06/06/2023-nba-draft-best-bets-on-draftkings-sportsbook/", "dkn"),
    Src(2023, "sportingnews", "draftkings_props_jun16", "DraftKings", "https://www.sportingnews.com/us/nba/news/nba-draft-2023-odds-best-prop-bets-over-under-advice/ybwajjpunnfhjjowtlhfwgum", "side_paren", wayback="20230616184059"),
    Src(2023, "sportingnews", "draftkings_props_jun20", "DraftKings", "https://www.sportingnews.com/us/nba/news/nba-draft-2023-odds-best-prop-bets-over-under-advice/ybwajjpunnfhjjowtlhfwgum", "side_paren"),
    Src(2023, "actionnetwork", "betting_picks", None, "https://www.actionnetwork.com/nba/2023-nba-draft-betting-picks-predictions-scoot-henderson-brandon-miller-jarace-walker", "an2023"),
    Src(2023, "stltoday", "fanduel_props", "FanDuel", "https://www.stltoday.com/sports/betting/article_fe8bd1c9-b594-5c3d-aa52-476aaa75b754.html", "stltoday"),
    Src(2023, "bestonlinesportsbooks", "betanysports_props", "BetAnySports", "https://www.bestonlinesportsbooks.info/news/nba-draft-props-where-will-top-prospects-be-picked-draft-night/", "betanysports"),
    # 2024
    Src(2024, "sportsbettingdime", "draftkings_overunders", "DraftKings", "https://www.sportsbettingdime.com/news/nba/2024-draft-odds-predictions-picks-best-bets-for-bronny-james-more/", "sbd_table",
        note="'Odds as of June 25'; page re-saved 22:39 ET June 26 (during round 1), no results in the text"),
    Src(2024, "covers", "draft_odds_betting", "Covers (book unnamed)", "https://www.covers.com/nba/draft-odds-betting-2024", "covers2024", wayback="20240625205633"),
    Src(2024, "saturdaydownsouth", "kentucky_props", None, "https://www.saturdaydownsouth.com/news/college-football/kentucky-nba-draft-props-where-will-reed-sheppard-rob-dillingham-land-in-2024-nba-draft/", "sds2024"),
    Src(2024, "dknetwork", "best_bets_jun20", "DraftKings", "https://dknetwork.draftkings.com/2024/06/20/2024-nba-draft-best-bets-odds-predictions-to-consider-on-draftkings-sportsbook-2/", "dkn"),
    # 2025
    Src(2025, "lasvegassportsbetting", "bovada_board", "Bovada", "https://lasvegassportsbetting.com/nba-basketball/2025-nba-draft-position-odds/", "lvsb2025",
        published="2025-06-25T12:00:00Z", note="page stamped 'Updated: June 25, 2025' (meta says May 30); every market incl. Flagg/Harper still open, so captured before the first pick"),
    Src(2025, "cbssports", "draftkings_board", "DraftKings", "https://www.cbssports.com/nba/news/2025-nba-draft-odds-for-top-20-and-more-see-where-kon-knueppel-jeremiah-fears-and-others-are-expected-to-go/", "cbs2025"),
    Src(2025, "thescore", "espnbet_props", "ESPN Bet", "https://www.thescore.com/nba/news/3303420/nba-draft-props-how-far-will-ace-bailey-fall", "thescore2025"),
    Src(2025, "gambling911", "betonline_odds", "BetOnline", "https://www.gambling911.com/nba/2025-draft-odds-061825.html", "betonline2025", published="2025-06-18T12:00:00Z"),
    Src(2025, "dknetwork_popular", "most_popular_props", "DraftKings", "https://dknetwork.draftkings.com/2025/06/24/2025-nba-draft-most-popular-props-on-draftkings-sportsbook/", "hand"),
    # 2026 (first round Tuesday June 23)
    Src(2026, "actionnetwork", "draft_preview_jun23", None, "https://www.actionnetwork.com/nba/nba-draft-predictions-picks-preview-betting-odds-for-tuesday-june-23", "hand"),
    Src(2026, "sportingnews", "fanduel_how_to_bet", "FanDuel", "https://www.sportingnews.com/us/betting/news/how-bet-nba-draft-odds-no-1-pick-prediction-best-bets/8f9ecd6794b2b2c5886a33f6", "sn2026"),
    Src(2026, "rawchili", "clippers_pick5", "DraftKings", "https://www.rawchili.com/nba/764247/", "hand"),
    Src(2026, "rockytopinsider", "ament_draftkings", "DraftKings", "https://www.rockytopinsider.com/2026/06/18/nate-ament-nba-draft-odds-the-payouts-on-a-top-10-coin-flip-and-300-to-1-longshot/", "rti2026"),
    Src(2026, "savannahherald", "draft_night_guide", "Savannah Herald (book unnamed)", "https://savannahherald.com/nba-draft-betting-guide-highlights-value-tonights-first-round-including-keaton-wagler/", "hand"),
    Src(2026, "rg", "betting_preview", "20Bet", "https://rg.org/en-ca/news/basketball/2026-nba-draft-betting-preview-dybansta-leads-class", "hand"),
]

# Leads and finds that were NOT used, and why (kept for the record)
REJECTED = [
    (2019, "sportingnews.com/ca .../nba-draft-2019-prop-bets-zion-williamson-is-a-sure-thing-but", "published 2019-06-20T23:48Z = 19:48 ET, after the first pick (19:00 ET start); only Reddish 1-7/8-30 and Edwards R1 yes/no, both covered by FanDuel"),
    (2022, "sportingnews.com draft-day guide LIVE page", "live version re-published 23:40Z / modified 00:07Z June 24 (after tip-off); the 16:31Z Wayback capture is used instead"),
    (2024, "covers.com/nba/draft-odds-betting-2024 LIVE page", "live version stamped 2024-06-26T19:05 CDT (after tip-off); the June 25 Wayback capture is used instead"),
    (2024, "sportsbookadvisor.com 2024-nba-draft-round-1-what-we-learned", "post-draft recap"),
    (2025, "dknetwork 2025-nba-draft-props-best-bets-to-be-a-first-round-pick", "modified 2025-06-26T02:11Z (during round 1); DraftKings first-round market taken from CBS (16:41 ET June 25) instead"),
    (2026, "vegasinsider.com/nba/odds/draft/", "live page is the 2025 board (BetMGM, 'UPDATE JUNE 24 2025'); no 2026 capture"),
    (2026, "basketball.realgm.com/betting/news/nba-draft-betting-odds/", "403 live, no Wayback capture"),
    (2021, "ftnfantasy.com bet-the-mock-2021", "modified 2024-06-14, three lines already covered by BetMGM/DraftKings"),
]


# --------------------------------------------------------------------------- fetch + cache

def _get(url, params=None, tries=4, timeout=120):
    for i in range(tries):
        try:
            r = _S.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (403, 404):
                return None
        except requests.RequestException:
            pass
        time.sleep(4 * (i + 1))
    return None


def _wayback_raw(ts: str, url: str) -> bytes | None:
    r = _get(f"{WAYBACK}/web/{ts}id_/{url}")
    if r is None:
        return None
    if r.headers.get("content-encoding", "").lower() in ("zstd", "br") or r.content[:4] == b"\x28\xb5\x2f\xfd":
        r = _get(f"{WAYBACK}/web/{ts}/{url}")
    return r.content if r is not None else None


def fetch(src: Src, verbose=True) -> dict | None:
    """Download and cache one page (no-op when cached). Returns the json sidecar or None when the page cannot be fetched."""
    hp, jp = src.paths
    if jp.exists():
        return json.loads(jp.read_text())
    if src.wayback:
        raw = _wayback_raw(src.wayback, src.url)
    else:
        r = _get(src.url)
        raw = r.content if r is not None else None
    if raw is None or len(raw) < 2000:
        if verbose:
            print(f"{src.site:22s} {src.year} {src.slug}: fetch failed")
        return None
    pub, mod = _meta_dates(raw)
    meta = {"year": src.year, "site": src.site, "slug": src.slug, "url": src.url, "wayback": src.wayback, "published": pub or src.published,
            "modified": mod, "fetched": datetime.now(timezone.utc).isoformat(timespec="seconds"), "book": src.book}
    hp.parent.mkdir(parents=True, exist_ok=True)
    hp.write_bytes(raw)
    if src.site == "si" and src.year == 2020:
        img = _get(SI_IMG)
        if img is not None:
            (hp.parent / "img-3736.jpg").write_bytes(img.content)
    jp.write_text(json.dumps(meta, indent=1))
    if verbose:
        print(f"{src.site:22s} {src.year} {src.slug}: cached ({'wayback ' + src.wayback if src.wayback else 'published ' + str(meta['published'])})")
    return meta


def download(verbose=True) -> None:
    for s in SOURCES:
        fetch(s, verbose=verbose)


def _capture_time(src: Src, meta: dict) -> pd.Timestamp:
    """UTC time of the evidence: the Wayback capture, else the registry's date (pages whose meta is missing or predates the
    board they show), else the article's published time."""
    if src.wayback:
        return pd.Timestamp(datetime.strptime(src.wayback, "%Y%m%d%H%M%S")).tz_localize("UTC")
    ts = pd.Timestamp(src.published or meta.get("published"))
    return ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")


# --------------------------------------------------------------------------- name matching

ALIASES = {  # page spelling (norm_name) -> draft_table key, where neither normalisation nor a unique surname resolves it
    "edriceadebayo": "bamadebayo", "obadiahtoppin": "obitoppin", "nickellwalkeralexander": "nickeilalexanderwalker",
    "walkeralexander": "nickeilalexanderwalker", "zairewilliams": "ziairewilliams", "alexandresarr": "alexsarr", "ronaldholland": "ronholland",
    "carltoncarrington": "bubcarrington", "walterkessler": "walkerkessler", "nilokajovic": "nikolajovic", "benedictmathurin": "bennedictmathurin",
    "colinmurrayboyles": "collinmurrayboyles", "kharmanmaluach": "khamanmaluach", "khamankaluach": "khamanmaluach", "vjedgecomb": "vjedgecombe",
    "trejohsnon": "trejohnson", "liammcneely": "liammcneeley", "hansenyang": "yanghansen", "gregoryjackson": "ggjackson", "cameronthomas": "camthomas",
    "bjboston": "brandonboston", "lebaronphilon": "labaronphilon", "bradenflemings": "kingstonflemings", "mohamedbamba": "mobamba",
}
SURNAME_ALIASES = {(2019, "johnson"): "keldonjohnson", (2019, "porter"): "kevinporter"}  # surname-only names that are ambiguous in the class

_TRANSLIT = str.maketrans({"ı": "i", "ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "ß": "ss", "ø": "o", "Ø": "O"})


def _alt(s: str) -> str:
    return norm_name(unicodedata.normalize("NFKD", str(s)).translate(_TRANSLIT).replace(".", ""))


def _tokens(s: str) -> list[str]:
    toks = [t for t in re.split(r"[\s\-]+", str(s).replace(".", "").replace("'", "")) if t]
    return [t for t in toks if t.lower() not in ("jr", "sr", "ii", "iii", "iv")] or toks


def match_names(names: list[str], pool: pd.DataFrame, year: int) -> list[str | None]:
    """Page names -> draft_table keys of the class: exact normalised name, transliteration, ALIASES, then a unique surname
    (first name / first initial breaking ties). None = not a draftee (or an unresolved spelling)."""
    keys = set(pool.key)
    alt = {_alt(p): k for k, p in zip(pool.key, pool.player)}
    by_last = {}
    for k, p in zip(pool.key, pool.player):
        toks = _tokens(p)
        by_last.setdefault(toks[-1].lower(), []).append((k, toks[0].lower()))
    out = []
    for n in names:
        n = re.sub(r"\s*['\"][^'\"]+['\"]\s*", " ", n).strip()  # nicknames: Edrice 'Bam' Adebayo
        k = norm_name(n)
        cand = None
        if k in keys:
            cand = k
        elif _alt(n) in alt:
            cand = alt[_alt(n)]
        elif ALIASES.get(k) in keys:
            cand = ALIASES[k]
        else:
            toks = [t.lower() for t in _tokens(n)]
            if (year, toks[-1]) in SURNAME_ALIASES:
                cand = SURNAME_ALIASES[(year, toks[-1])]
            else:
                c = by_last.get(toks[-1], [])
                if len(c) > 1 and len(toks) > 1:
                    same = [x for x in c if x[1] == toks[0]] or [x for x in c if x[1][:1] == toks[0][:1]]
                    c = same
                if len(c) == 1:
                    cand = c[0][0]
        out.append(cand if cand in keys else None)
    return out


# --------------------------------------------------------------------------- build

def _draftees() -> pd.DataFrame:
    d = pd.read_parquet(C.PROC / "draft_table.parquet", columns=["key", "draft_year", "player", "pick"])
    return d.drop_duplicates(KEYS)


def rows_for(src: Src, meta: dict, pool: pd.DataFrame) -> pd.DataFrame:
    """Parsed + hand rows of one page, matched to draftees, with book and evidence date filled in."""
    hp, _ = src.paths
    text = page_text(hp.read_bytes())
    rows = PARSERS[src.parser](text) if src.parser != "hand" else []
    hand = HAND.get((src.site, src.year), [])
    if hand:
        _verify_hand(hand, text, src.site, src.year)
        rows = rows + [dict(r) for r in hand]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for c in ("line", "over", "under", "k", "yes", "no", "prob"):
        df[c] = pd.to_numeric(df[c])
    df["book"] = df.book.fillna(src.book or f"{src.site} (book unnamed)")
    cap = _capture_time(src, meta)
    df["date"] = [pd.Timestamp(d).date() if isinstance(d, str) else cap.date() for d in df.date]
    df["capture"] = cap
    df["source"] = f"{src.site}/{src.slug}"
    df["key"] = match_names(list(df.player), pool, src.year)
    return df


def _agg_player(g: pd.DataFrame, year: int) -> dict:
    out = {}
    ou = g[g.line.notna() & (g.stage == "close")]
    if len(ou):
        # final line per book = the latest-dated observation of that book
        fin = ou.sort_values("date").groupby("book").tail(1)
        p_under = [fair_prob(u, o) for u, o in zip(fin.under, fin.over)]
        out["odds_line"] = float(fin.line.median())
        out["odds_over_price"] = median_price(fin.over)
        out["odds_under_price"] = median_price(fin.under)
        out["odds_expected_pick"] = float(np.median([expected_pick(l, p) for l, p in zip(fin.line, p_under)]))
        out["odds_n_books"] = int(fin.book.nunique())
        out["odds_line_std"] = float(fin.line.std(ddof=1)) if len(fin) > 1 else np.nan
        moves = []
        for b, gb in g[g.line.notna()].groupby("book"):  # opening lines: explicit 'open' rows or earlier-dated captures
            close = gb[gb.stage == "close"].sort_values("date")
            if not len(close):
                continue
            earlier = gb[(gb.stage == "open") | (gb.date < close.date.iloc[-1])]
            if len(earlier):
                earlier = earlier.assign(_o=(earlier.stage != "open")).sort_values(["date", "_o"])  # same day: the stated opener first
                moves.append(close.line.iloc[-1] - earlier.line.iloc[0])
        out["odds_line_move"] = float(np.median(moves)) if moves else np.nan
        dates = list(fin.date)
    else:
        dates = []
    for k, name in TOPK.items():
        exp = g[(g.k == k) & (g.stage == "close")]
        probs = [r.prob if pd.notna(r.prob) else fair_prob(r.yes, r.no) for r in exp.itertuples()]
        # an O/U line at k + 0.5 is the same market: P(under) = P(top k)
        eq = ou[ou.line == k + 0.5] if len(ou) else ou
        if len(eq):
            eq = eq.sort_values("date").groupby("book").tail(1)
            probs += [fair_prob(u, o) for u, o in zip(eq.under, eq.over) if pd.notna(u) or pd.notna(o)]
        probs = [p for p in probs if pd.notna(p)]
        out[f"odds_prob_{name}"] = float(np.median(probs)) if probs else np.nan
        if len(exp):
            dates += list(exp.date)
    if dates:
        out["odds_days_before_draft"] = int((DRAFT_DATE[year] - max(dates)).days)
    return out


COLS = ["odds_line", "odds_over_price", "odds_under_price", "odds_expected_pick", "odds_prob_top5", "odds_prob_top10", "odds_prob_lottery",
        "odds_prob_first_round", "odds_n_books", "odds_line_std", "odds_line_move", "odds_days_before_draft"]


def build(download_missing: bool = False, verbose: bool = True) -> pd.DataFrame:
    d = _draftees()
    frames, prov = [], []
    for src in SOURCES:
        meta = fetch(src, verbose=verbose) if download_missing else (json.loads(src.paths[1].read_text()) if src.paths[1].exists() else None)
        if meta is None:
            continue
        cap = _capture_time(src, meta)
        cutoff = pd.Timestamp(FIRST_ROUND_UTC[src.year]).tz_localize("UTC")
        assert cap < cutoff, (src.site, src.year, src.slug, cap, "captured/published after the first round tipped off")  # THE RULE
        pool = d[d.draft_year == src.year]
        df = rows_for(src, meta, pool)
        n_players = int(df.player.nunique()) if len(df) else 0
        n_matched = int(df.key.dropna().nunique()) if len(df) else 0
        prov.append({"year": src.year, "source": f"{src.site}/{src.slug}", "url": src.url, "snapshot_or_published_date": cap.date(),
                     "n_players_parsed": n_players, "n_draftees_matched": n_matched, "book": src.book or "per row",
                     "evidence": "wayback " + src.wayback if src.wayback else "published " + str(meta.get("published")),
                     "modified": meta.get("modified"), "note": src.note})
        if verbose:
            unmatched = sorted(set(df.player[df.key.isna()])) if len(df) else []
            print(f"{src.site:22s} {src.year} {src.slug:28s} {cap.date()}  {n_players:3d} players / {n_matched:3d} draftees   unmatched: {unmatched}")
        if len(df):
            frames.append(df.assign(draft_year=src.year))
    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prov).to_csv(ODDS_DIR / "provenance.csv", index=False)
    if not frames:
        return pd.DataFrame(columns=KEYS + COLS)
    rows = pd.concat(frames, ignore_index=True).dropna(subset=["key"])
    rows.to_csv(ODDS_DIR / "rows.csv", index=False)  # every market row used, for review
    recs = []
    for (k, y), g in rows.groupby(["key", "draft_year"]):
        a = _agg_player(g, y)
        if a:
            recs.append({"key": k, "draft_year": int(y), **a})
    out = pd.DataFrame(recs)
    for c in COLS:
        if c not in out:
            out[c] = np.nan
    return out[KEYS + COLS]


def load_odds() -> pd.DataFrame:
    return build(download_missing=False, verbose=False)


# --------------------------------------------------------------------------- report

def report(out: pd.DataFrame) -> None:
    d = _draftees()
    m = d.merge(out, on=KEYS, how="left")
    prov = pd.read_csv(ODDS_DIR / "provenance.csv")
    print("\nyear  draftees  with_line  spearman(E[pick], pick)  spearman(line, pick)  sources")
    for y in YEARS:
        g = m[m.draft_year == y]
        has = g.odds_line.notna()
        rho = g.loc[has, ["odds_expected_pick", "pick"]].corr(method="spearman").iloc[0, 1] if has.sum() > 2 else np.nan
        rho_l = g.loc[has, ["odds_line", "pick"]].corr(method="spearman").iloc[0, 1] if has.sum() > 2 else np.nan
        p = prov[prov.year == y]
        srcs = ", ".join(f"{r.source.split('/')[0]}@{r.snapshot_or_published_date}({r.n_draftees_matched})" for r in p.itertuples())
        print(f"{y}  {len(g):8d}  {int(has.sum()):9d}  {rho:23.3f}  {rho_l:20.3f}  {srcs}")
    print(f"\n{len(prov)} pages; {len(out)} player-years; books/player median {out.odds_n_books.median():.0f}; "
          f"line movement recorded for {out.odds_line_move.notna().sum()}; top-5 prob for {out.odds_prob_top5.notna().sum()}, "
          f"top-10 {out.odds_prob_top10.notna().sum()}, lottery {out.odds_prob_lottery.notna().sum()}, first round {out.odds_prob_first_round.notna().sum()}")


if __name__ == "__main__":
    t0 = time.time()
    res = build(download_missing=True)
    print(f"built {res.shape} in {time.time() - t0:.0f}s")
    report(res)
