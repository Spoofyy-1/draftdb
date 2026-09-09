#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DraftDB — build the 30 NBA team pages from data/team_cards.json.

Rewrites, for every team present in the dataset:

    site/teams/<slug>.html      one page per franchise, its own accent colour
    site/teams/index.html       the league index

Standard library only, no build step: the files written here are the files
Vercel serves. Each page keeps the site chrome (announcement bar, nav, footer),
the fonts, css/base.css + css/team.css, and the accent colour already used by
that team's page.

Usage
-----
    # real data, in place
    python3 tools/build_team_pages.py

    # develop against the mock fixture, into a throwaway directory
    python3 tools/build_team_pages.py \
        --data data/team_cards.mock.json \
        --out  tools/_preview \
        --asset-prefix ../../
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from datetime import date

YEARS = (2019, 2020, 2021, 2022, 2023, 2024, 2025)

SITE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# slug, city, nickname, fallback accent (the colour each placeholder page shipped
# with). The accent actually used is re-read from the page being replaced, so a
# hand-tweaked colour survives a rebuild; this table is only the fallback.
TEAMS = [
    ("atlanta-hawks",          "Atlanta",       "Hawks",        "#E03A3E"),
    ("boston-celtics",         "Boston",        "Celtics",      "#007A33"),
    ("brooklyn-nets",          "Brooklyn",      "Nets",         "#FFFFFF"),
    ("charlotte-hornets",      "Charlotte",     "Hornets",      "#1D1160"),
    ("chicago-bulls",          "Chicago",       "Bulls",        "#CE1141"),
    ("cleveland-cavaliers",    "Cleveland",     "Cavaliers",    "#860038"),
    ("dallas-mavericks",       "Dallas",        "Mavericks",    "#0053BC"),
    ("denver-nuggets",         "Denver",        "Nuggets",      "#0E2240"),
    ("detroit-pistons",        "Detroit",       "Pistons",      "#C8102E"),
    ("golden-state-warriors",  "Golden State",  "Warriors",     "#1D428A"),
    ("houston-rockets",        "Houston",       "Rockets",      "#CE1141"),
    ("indiana-pacers",         "Indiana",       "Pacers",       "#002D62"),
    ("la-clippers",            "LA",            "Clippers",     "#C8102E"),
    ("los-angeles-lakers",     "Los Angeles",   "Lakers",       "#552583"),
    ("memphis-grizzlies",      "Memphis",       "Grizzlies",    "#5D76A9"),
    ("miami-heat",             "Miami",         "Heat",         "#98002E"),
    ("milwaukee-bucks",        "Milwaukee",     "Bucks",        "#00471B"),
    ("minnesota-timberwolves", "Minnesota",     "Timberwolves", "#0C2340"),
    ("new-orleans-pelicans",   "New Orleans",   "Pelicans",     "#0C2340"),
    ("new-york-knicks",        "New York",      "Knicks",       "#006BB6"),
    ("oklahoma-city-thunder",  "Oklahoma City", "Thunder",      "#007AC1"),
    ("orlando-magic",          "Orlando",       "Magic",        "#0077C0"),
    ("philadelphia-76ers",     "Philadelphia",  "76ers",        "#006BB6"),
    ("phoenix-suns",           "Phoenix",       "Suns",         "#1D1160"),
    ("portland-trail-blazers", "Portland",      "Trail Blazers","#E03A3E"),
    ("sacramento-kings",       "Sacramento",    "Kings",        "#5A2D81"),
    ("san-antonio-spurs",      "San Antonio",   "Spurs",        "#C4CED4"),
    ("toronto-raptors",        "Toronto",       "Raptors",      "#CE1141"),
    ("utah-jazz",              "Utah",          "Jazz",         "#002B5C"),
    ("washington-wizards",     "Washington",    "Wizards",      "#002B5C"),
]
TEAM_BY_SLUG = {t[0]: t for t in TEAMS}

INDEX_ACCENT = "#a855f7"          # the teams index keeps the DraftDB purple

REPO_SITE = "https://github.com/Spoofyy-1/draftdb"
REPO_DATA = "https://github.com/Spoofyy-1/DraftDB-Data"

DASH = "&mdash;"


# --------------------------------------------------------------------------
# formatting helpers — nothing here may ever emit "None" or "nan"
# --------------------------------------------------------------------------

BAD_STRINGS = {"", "none", "nan", "null", "undefined", "n/a", "na", "-", "--"}


def esc(value, dash=DASH):
    """Escape a value for HTML text. Missing / junk values become an em dash."""
    if value is None:
        return dash
    if isinstance(value, float) and value != value:          # NaN
        return dash
    s = str(value).strip()
    if s.lower() in BAD_STRINGS:
        return dash
    return html.escape(s, quote=True)


def attr(value):
    """Escape a value for an HTML attribute (missing -> empty string)."""
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def to_num(value):
    """Return a finite float, or None for anything that is not a real number."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
    elif isinstance(value, str):
        s = value.strip().replace(",", "")
        if s.lower() in BAD_STRINGS:
            return None
        try:
            v = float(s)
        except ValueError:
            return None
    else:
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return v


def war(value, dash=DASH):
    """WAR, one decimal."""
    v = to_num(value)
    if v is None:
        return dash
    if abs(v) < 0.05:
        v = 0.0
    return "{:.1f}".format(v)


def delta(value, dash=DASH):
    """A WAR difference, one decimal, always signed."""
    v = to_num(value)
    if v is None:
        return dash
    if abs(v) < 0.05:
        return "0.0"
    return "{:+.1f}".format(v)


def whole(value, dash=DASH):
    """An integer count, thousands-separated."""
    v = to_num(value)
    return dash if v is None else "{:,.0f}".format(v)


def pct(value, dash=DASH):
    v = to_num(value)
    return dash if v is None else "{:.1f}%".format(v)


def ordinal(value, dash=DASH):
    v = to_num(value)
    if v is None:
        return dash
    n = int(round(v))
    if 10 <= abs(n) % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(abs(n) % 10, "th")
    return "{}{}".format(n, suffix)


def rank_text(value):
    """A within-class model rank, or the honest 'not scored'."""
    v = to_num(value)
    if v is None:
        return "not scored"
    return "No.&nbsp;{:.0f}".format(v)


def plural(n, one, many=None):
    many = many if many is not None else one + "s"
    return one if n == 1 else many


def iso_date(value):
    """Normalise the dataset's `generated` field to YYYY-MM-DD."""
    s = "" if value is None else str(value).strip()
    m = re.match(r"(\d{4}-\d{2}-\d{2})", s)
    if m:
        return m.group(1)
    return date.today().isoformat()


HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
ACCENT_IN_FILE_RE = re.compile(r"--accent:\s*(#[0-9A-Fa-f]{6})")


def read_accent(path, fallback):
    """Re-use the accent already on disk so a hand-edited colour survives."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            head = fh.read(4000)
    except (OSError, UnicodeDecodeError):
        return fallback
    m = ACCENT_IN_FILE_RE.search(head)
    if m and HEX_RE.match(m.group(1)):
        return m.group(1)
    return fallback


# --------------------------------------------------------------------------
# shared chrome
# --------------------------------------------------------------------------

FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
    '<link href="https://fonts.googleapis.com/css2?family=Geist+Pixel&family=Inter:wght@300;400;500;600'
    '&family=JetBrains+Mono:wght@400;500&family=Fragment+Mono&display=swap" rel="stylesheet">'
)


def logo_svg(accent, size):
    return (
        '<svg viewBox="0 0 40 40" width="{s}" height="{s}" aria-hidden="true" focusable="false">'
        '<rect x="2" y="2" width="16" height="16" fill="#fff"/>'
        '<path d="M22 2h10v4h4v4h2v8H22z" fill="#fff"/>'
        '<path d="M2 22h16v16h-8v-2H6v-4H4v-4H2z" fill="#fff"/>'
        '<rect x="22" y="22" width="16" height="16" fill="#fff"/>'
        '<rect x="18" y="18" width="4" height="4" fill="{a}"/></svg>'
    ).format(s=size, a=accent)


def page_head(title, description, accent, prefix):
    return "\n".join([
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>{} — DraftDB</title>".format(title),
        '<meta name="description" content="{}">'.format(description),
        FONTS,
        '<link rel="stylesheet" href="{}css/base.css">'.format(prefix),
        '<link rel="stylesheet" href="{}css/team.css">'.format(prefix),
        "<style>",
        "  :root{{--accent:{a};--accent-soft:{a}1f;--accent-line:{a}8c}}".format(a=accent),
        "  .btn--accent{background:var(--accent);border-color:var(--accent);color:#fff}",
        "  .btn--accent:hover{filter:brightness(1.12)}",
        "  .chip{color:#fff}",
        "  ::selection{background:var(--accent);color:#fff}",
        "</style>",
        "</head>",
        "<body>",
    ])


def chrome_header(accent, prefix):
    p = prefix
    return """
<div class="announce" role="region" aria-label="Announcement">
  <span class="chip">Announcement</span>
  <span class="announce__text">DraftDB 2026 class projections are live.</span>
  <a class="announce__link" href="{p}index.html#model">View methodology &#8599;</a>
</div>
<header class="nav" id="top">
  <div class="nav__inner">
    <a class="nav__logo" href="{p}index.html" aria-label="DraftDB home">{logo}<span class="wordmark">DraftDB</span></a>
    <nav class="nav__links" aria-label="Primary">
      <a href="{p}index.html#model">Model</a>
      <a href="{p}index.html#backtests">Backtests</a>
      <a href="{p}runs.html">Runs</a>
      <a href="{p}compare.html">Compare</a>
      <a href="{p}index.html#use-cases">Use Cases</a>
      <a href="index.html" class="nav__link is-active">Teams</a>
      <a href="{p}index.html#docs">Documentation</a>
    </nav>
    <button class="nav__menu-btn" type="button" aria-controls="primary-links" aria-expanded="false"><span class="m">Menu</span><span class="x">Close</span></button>
    <div class="nav__actions btn-group">
      <a class="btn btn--dark btn--sm" href="#">Log In</a>
      <a class="btn btn--primary btn--sm" href="#">Sign Up</a>
    </div>
  </div>
</header>
""".format(p=p, logo=logo_svg(accent, 30)).strip("\n")


def chrome_footer(accent, prefix, generated, data_name):
    return """
<footer class="footer" id="contact">
  <div class="frame">
    <div class="container" style="padding:44px 0 40px">
      <a class="nav__logo" href="{p}index.html" aria-label="DraftDB home">{logo}<span class="wordmark">DraftDB</span></a>
      <p class="footer__tag" style="margin-top:18px">Draft smarter with AI.</p>
      <div class="divider-list" style="justify-content:flex-start;margin-top:22px">
        <a class="link" href="{p}index.html#model">Model</a>
        <a class="link" href="{p}index.html#backtests">Backtests</a>
        <a class="link" href="{p}runs.html">Runs</a>
        <a class="link" href="{p}compare.html">Compare</a>
        <a class="link" href="index.html">Teams</a>
      </div>
      <div style="margin-top:26px;font-size:14px;color:var(--text-2)">&copy; 2026 DraftDB Labs, Inc</div>
      <div class="footer__gen">Generated from {data} on {gen}</div>
    </div>
  </div>
  <div class="footer__end"></div>
</footer>

<script src="{p}js/base.js"></script>
</body>
</html>
""".format(p=prefix, logo=logo_svg(accent, 32), gen=esc(generated), data=esc(data_name)).strip("\n")


# --------------------------------------------------------------------------
# page content
# --------------------------------------------------------------------------

VERDICTS = {
    "model": ("pickcard--model", "Model&rsquo;s pick outperformed"),
    "team":  ("pickcard--team",  "Team&rsquo;s pick outperformed"),
    "tie":   ("pickcard--tie",   "Even"),
}


def team_labels(team):
    """(city, nickname, full name) for a team record from the dataset."""
    slug = str(team.get("slug") or "").strip()
    name = str(team.get("name") or "").strip()
    row = TEAM_BY_SLUG.get(slug)
    if row:
        city, nick = row[1], row[2]
        return city, nick, (name or "{} {}".format(city, nick))
    if not name:
        name = slug.replace("-", " ").title() or "Team"
    parts = name.split(" ")
    return " ".join(parts[:-1]) or name, parts[-1], name


def agreed_pick(actual, model):
    """The team's pick outranked every alternative on the model's board at this slot: the model agreed with the team."""
    ar, mr = to_num((actual or {}).get("model_rank")), to_num((model or {}).get("model_rank"))
    return bool(model) and ar is not None and mr is not None and ar < mr


def recount(team):
    """Card verdicts recounted with the agreed rule (a card where the model agreed is neither a win nor a loss)."""
    out = {"model_better": 0, "team_better": 0, "ties": 0, "agreed": 0, "n_cards": 0}
    fr = {"model_better": 0, "team_better": 0, "ties": 0, "agreed": 0}
    for d in team.get("drafts") or []:
        for pick in d.get("picks") or []:
            actual, model = pick.get("actual") or {}, pick.get("model_choice") or {}
            if not model:
                continue
            out["n_cards"] += 1
            if agreed_pick(actual, model) or same_player(actual, model):
                key = "agreed"
            else:
                v = str(pick.get("verdict") or "tie").lower()
                key = {"model": "model_better", "team": "team_better"}.get(v, "ties")
            out[key] += 1
            if to_num(pick.get("round")) == 1:
                fr[key] += 1
    out["first_round_counts"] = fr
    return out


def same_player(a, b):
    pid_a, pid_b = a.get("pid"), b.get("pid")
    if pid_a is not None and pid_b is not None and str(pid_a) == str(pid_b):
        return True
    na, nb = a.get("name"), b.get("name")
    return bool(na) and bool(nb) and str(na).strip().lower() == str(nb).strip().lower()


def rows_html(rows):
    items = ['      <div class="pk-row"><dt>{}</dt><dd>{}</dd></div>'.format(k, v) for k, v in rows]
    return '    <dl class="pk-rows">\n' + "\n".join(items) + "\n    </dl>"


def went_text(player):
    """Where a player actually went, and to whom."""
    pick_no = to_num(player.get("actual_pick"))
    by = esc(player.get("drafted_by"), dash="")
    if pick_no is not None and by:
        return "{} overall <span class=\"q\">&middot;</span> {}".format(ordinal(pick_no), by)
    if pick_no is not None:
        return "{} overall".format(ordinal(pick_no))
    return by or DASH


def pick_card(team_name, year, pick, idx):
    actual = pick.get("actual") or {}
    model = pick.get("model_choice") or {}
    best = pick.get("best_hindsight") or {}

    verdict = str(pick.get("verdict") or "").strip().lower()
    if verdict not in VERDICTS:
        verdict = "tie"
    css_class, verdict_label = VERDICTS[verdict]

    diff = to_num(pick.get("delta_war"))
    if diff is None and model:
        mw, aw = to_num(model.get("war")), to_num(actual.get("war"))
        if mw is not None and aw is not None:
            diff = mw - aw

    is_same = bool(model) and same_player(actual, model)
    if is_same:
        verdict_label = "Same pick"
    agreed = agreed_pick(actual, model) and not is_same
    if agreed:
        css_class, verdict_label, diff = "pickcard--agree", "Model agreed", None

    pick_no = to_num(pick.get("pick"))
    rnd = to_num(pick.get("round"))
    anchor = "p{}-{}".format(year, int(pick_no) if pick_no is not None else "x{}".format(idx))

    # ---- card header ----
    crumbs = ['<span class="label">Pick {} overall</span>'.format(
        "{:.0f}".format(pick_no) if pick_no is not None else DASH)]
    if rnd is not None:
        crumbs.append('<span class="label">Round {:.0f}</span>'.format(rnd))
    crumbs.append('<span class="label">{} draft</span>'.format(year))
    head = ('\n      <span class="pickcard__sep" aria-hidden="true">/</span>\n      ').join(crumbs)

    # ---- left: the team's pick ----
    left_rows = [
        ("Overall pick", ordinal(pick_no)),
        ("Round", "{:.0f}".format(rnd) if rnd is not None else DASH),
        ("College or club", esc(actual.get("org"))),
        ("WAR to date", war(actual.get("war"))),
        ("Seasons scored", whole(actual.get("seasons_scored"))),
        ("Model rank, {} class".format(year), rank_text(actual.get("model_rank"))),
    ]
    games, minutes = to_num(actual.get("games")), to_num(actual.get("minutes"))
    left_note = ""
    if games is not None or minutes is not None:
        bits = []
        if games is not None:
            bits.append("{} {}".format(whole(games), plural(int(games), "game")))
        if minutes is not None:
            bits.append("{} minutes".format(whole(minutes)))
        left_note = '\n    <p class="pk-note">{}</p>'.format(" &middot; ".join(bits))

    left = """  <section class="pickcard__side" aria-label="The {team} picked">
    <div class="pickcard__who">The <b>{team}</b> picked</div>
    <h3 class="pickcard__name">{name}</h3>
{rows}{note}
  </section>""".format(team=esc(team_name), name=esc(actual.get("name"), dash="Unknown player"),
                       rows=rows_html(left_rows), note=left_note)

    # ---- right: the model's choice ----
    sub = ('\n    <div class="pickcard__sub">Its top-ranked player still on the board '
           'at this slot</div>')
    if not model:
        right_body = ('    <p class="pickcard__name">No board entry</p>\n'
                      '    <p class="pk-note">The model has no scored board entry for this slot, '
                      'so this pick is shown without a comparison.</p>')
        sub = ""
        css_class = "pickcard--tie"
        verdict_label = "Not scored"
        diff = None
    elif agreed:
        right_rows = [
            ("Model rank, {} class".format(year), rank_text(actual.get("model_rank"))),
            ("WAR to date", war(actual.get("war"))),
            ("Seasons scored", whole(actual.get("seasons_scored"))),
        ]
        right_body = ('    <h3 class="pickcard__name">{}</h3>\n{}\n'
                      '    <p class="pk-note">The team&rsquo;s pick was also the model&rsquo;s top-ranked player still on the board at this slot. '
                      'Next on its board: {} (rank {}, {} WAR to date, {}).</p>').format(
            esc(actual.get("name")), rows_html(right_rows), esc(model.get("name")),
            rank_text(model.get("model_rank")), war(model.get("war")), went_text(model))
    elif is_same:
        right_rows = [
            ("Model rank, {} class".format(year), rank_text(model.get("model_rank"))),
            ("College or club", esc(model.get("org"))),
            ("WAR to date", war(model.get("war"))),
            ("Seasons scored", whole(model.get("seasons_scored"))),
        ]
        right_body = ('    <h3 class="pickcard__name">{}</h3>\n{}\n'
                      '    <p class="pk-note">Same player: the model&rsquo;s board had him at the top '
                      'of what was still available at this slot.</p>').format(
            esc(model.get("name")), rows_html(right_rows))
    else:
        right_rows = [
            ("Model rank, {} class".format(year), rank_text(model.get("model_rank"))),
            ("Actually drafted", went_text(model)),
            ("College or club", esc(model.get("org"))),
            ("WAR to date", war(model.get("war"))),
            ("Seasons scored", whole(model.get("seasons_scored"))),
        ]
        right_body = '    <h3 class="pickcard__name">{}</h3>\n{}'.format(
            esc(model.get("name")), rows_html(right_rows))

    right = """  <section class="pickcard__side" aria-label="Our model chose">
    <div class="pickcard__who">Our model chose</div>{sub}
{body}
  </section>""".format(sub=sub, body=right_body)

    # ---- centre badge ----
    unit = "same player" if agreed else "WAR, model &minus; team"
    mid = """  <div class="pickcard__mid">
    <div class="vbadge__delta">{d}</div>
    <div class="vbadge__unit">{unit}</div>
    <div class="vbadge__label">{label}</div>
  </div>""".format(d=("&#10003;" if agreed else delta(diff)), unit=unit, label=verdict_label)

    # ---- hindsight foot ----
    foot = ""
    if best and esc(best.get("name"), dash="") :
        tail = ""
        bp, bby = to_num(best.get("actual_pick")), esc(best.get("drafted_by"), dash="")
        if bp is not None and bby:
            tail = " &middot; went {} overall to {}".format(ordinal(bp), bby)
        elif bp is not None:
            tail = " &middot; went {} overall".format(ordinal(bp))
        foot = ('\n  <div class="pickcard__foot">Best available in hindsight: '
                '<b>{name}</b> ({w} WAR){tail}</div>').format(
            name=esc(best.get("name")), w=war(best.get("war")), tail=tail)

    return """<article class="pickcard {cls}" id="{anchor}">
  <header class="pickcard__head">
      <span class="sq sq--sm" aria-hidden="true"></span>
      {head}
  </header>
  <div class="pickcard__body">
{left}
{mid}
{right}
  </div>{foot}
</article>""".format(cls=css_class, anchor=attr(anchor), head=head, left=left, mid=mid,
                     right=right, foot=foot)


def year_section(team_name, year, draft):
    """One <section class="draft-year"> — always emitted, even for a year with no pick."""
    picks = list((draft or {}).get("picks") or [])
    no_pick = bool((draft or {}).get("no_pick")) or not picks

    if no_pick:
        meta = "No selection"
        body = ('    <div class="nopick"><span class="sq" aria-hidden="true"></span>'
                'No selection in {}.</div>'.format(year))
    else:
        team_war = sum(to_num(p.get("actual", {}).get("war")) or 0.0 for p in picks)
        model_war = sum(to_num((p.get("model_choice") or {}).get("war")) or 0.0 for p in picks)
        meta = "{} {} &middot; team {} WAR &middot; model {} WAR".format(
            len(picks), plural(len(picks), "pick"), war(team_war), war(model_war))
        body = "\n".join(pick_card(team_name, year, p, i) for i, p in enumerate(picks))

    return """<section class="draft-year frame" id="y{y}" aria-labelledby="y{y}-h">
    <div class="container">
      <div class="draft-year__head">
        <h2 class="draft-year__title" id="y{y}-h">{y} draft</h2>
        <span class="draft-year__meta">{meta}</span>
      </div>
{body}
    </div>
  </section>""".format(y=year, meta=meta, body=body)


def hero_section(team, meta, years_with_picks):
    city, nick, name = team_labels(team)
    s = dict(team.get("summary") or {})
    rc = recount(team); s.update({k: rc[k] for k in ("model_better", "team_better", "ties", "agreed")})
    fr = dict(s.get("first_round_only") or {}); fr.update(rc["first_round_counts"])

    n_picks = to_num(s.get("n_picks"))
    n_cards = to_num(s.get("n_cards"))
    mb, tb, ti, ag = to_num(s.get("model_better")), to_num(s.get("team_better")), to_num(s.get("ties")), to_num(s.get("agreed"))

    tiles = [
        ('WAR to date &middot; team picks', war(s.get("war_actual_total")),
         "{} {} in the 2019&ndash;2025 drafts".format(
             whole(n_picks), plural(int(n_picks or 0), "pick")), ""),
        ('WAR to date &middot; model picks', war(s.get("war_model_total")),
         "the same slots, taken in the model&rsquo;s board order", " tstat--model"),
        ('Comparison cards', '{}<span class="sl">/</span>{}<span class="sl">/</span>{}'.format(
            whole(mb, "0"), whole(tb, "0"), whole(ti, "0")),
         "won by the model / by the team / even{}".format(
             " &middot; {} where the model agreed with the team".format(whole(ag)) if (ag or 0) > 0 else ""), ""),
    ]
    tiles_html = "\n".join(
        '        <div class="tstat{cls}">\n'
        '          <div class="tstat__k">{k}</div>\n'
        '          <div class="tstat__v">{v}</div>\n'
        '          <div class="tstat__sub">{sub}</div>\n'
        '        </div>'.format(cls=cls, k=k, v=v, sub=sub) for k, v, sub, cls in tiles)

    first_round = ""
    if fr:
        first_round = (" First round only: the team&rsquo;s picks {ta} WAR against the model&rsquo;s "
                       "{tm} WAR summed over cards, {mb} {cards} to the model and {tb} to the team.").format(
            ta=war(fr.get("war_actual_total")), tm=war(fr.get("war_model_total")),
            mb=whole(fr.get("model_better"), "0"),
            cards=plural(int(to_num(fr.get("model_better")) or 0), "card"),
            tb=whole(fr.get("team_better"), "0"))

    ynav = "\n".join(
        '        <a href="#y{y}"{cls}>{y}</a>'.format(
            y=y, cls="" if y in years_with_picks else ' class="is-empty"') for y in YEARS)

    return """<section class="team-hero frame">
    <div class="container">
      <div class="team-hero__eyebrow"><span class="team-swatch"></span>{name} &middot; 2019&ndash;2025 drafts</div>
      <h1 class="team-hero__title">{name}</h1>
      <p class="team-verdict">Across the 2019&ndash;2025 drafts, pick by pick, the model&rsquo;s top available player out-produced the team&rsquo;s selection on <b>{mb} of {ncards} cards</b>; the team&rsquo;s pick did better on <b>{tb}</b>{agtxt}. Summed over cards, the model&rsquo;s choices have <b>{model} WAR</b> to date against <b>{actual}</b> for the team&rsquo;s picks &mdash; the same alternative can appear on several cards, so the sums compare choices, not rosters.</p>
      <div class="tstat-grid">
{tiles}
      </div>
      <div class="team-hero__meta"><span><b>Team</b> {name}</span><span><b>Picks</b> {npicks}</span><span><b>Cards</b> {ncards}</span><span><b>Boards</b> {boards}</span><span><b>Model</b> {label}</span></div>
      <p class="team-foot-note">{wardef}{first}</p>
      <nav class="ynav" aria-label="Jump to a draft class">
{ynav}
      </nav>
    </div>
  </section>""".format(
        name=esc(name), model=war(s.get("war_model_total")), actual=war(s.get("war_actual_total")),
        mb=whole(mb, "0"), tb=whole(tb, "0"), agtxt=(", and on {} the model had the same player at the top of its board".format(whole(ag)) if (ag or 0) > 0 else ""),
        tiles=tiles_html, npicks=whole(n_picks, "0"), ncards=whole(n_cards, "0"),
        boards=esc(meta.get("boards_used")), label=esc(meta.get("model", {}).get("label")),
        wardef=WAR_SHORT, first=first_round, ynav=ynav)


WAR_SHORT = ("WAR to date is counted over at most a player&rsquo;s first five NBA seasons, so the "
             "2022&ndash;2025 classes are still accruing and are only comparable within their own year.")

HONEST_ITEMS = [
    "WAR to date is counted over <b>at most a player&rsquo;s first five NBA seasons</b>. It is a running "
    "total, not a career, and it says nothing about what either player does next.",
    "The recent classes are scored on <b>one or two seasons</b>. Those cards will move, and some of them "
    "will flip, as more seasons are played.",
    "A model rank is a <b>within-class ranking of drafted players only</b>. It is an ordering of that "
    "year&rsquo;s board, not a grade against every player in basketball, and a player with no rank was "
    "simply not scored.",
    "The model&rsquo;s choice is the top-ranked player <b>still on the board</b> at that slot, from a "
    "board built only from what was knowable before the draft. It is not a claim that anyone knew how "
    "the careers would turn out.",
    "<b>A single card is noisy.</b> One pick can go either way for reasons no model or front office "
    "could see coming. The seven-class mean is the real test; the individual cards are how you read it.",
]


def honest_section(meta):
    items = list(HONEST_ITEMS)
    definition = esc(meta.get("war_definition"), dash="")
    if definition:
        items.append("<b>How WAR is measured here:</b> {}".format(definition))
    items = "\n".join(
        '          <li><span class="sq" aria-hidden="true"></span><span>{}</span></li>'.format(t)
        for t in items)
    return """<section class="section--tight frame" id="honest" aria-labelledby="honest-h">
    <div class="container">
      <div class="honest">
        <div class="honest__k">// Caveats //</div>
        <h2 class="honest__h" id="honest-h">Read this honestly</h2>
        <ul class="honest__list">
{items}
        </ul>
      </div>
    </div>
  </section>""".format(items=items)


def how_section(meta):
    model = meta.get("model") or {}
    steps = [
        ("01", "Only what was knowable before draft night",
         "Every input is dated before the pick: college and international box scores, physical "
         "measurements, the dated mock-draft consensus and published scouting grades, and biography. "
         "Nothing from a player&rsquo;s professional career reaches the model."),
        ("02", "Trained on the 2000&ndash;2018 classes",
         "The model learns from the drafts of 2000 through 2018. The classes on this page were never "
         "part of that training, and no draft&rsquo;s own year or later is ever visible to it."),
        ("03", "A gate every change has to pass",
         "Any change to the model must clear a seven-fold walk-forward gate on 2012&ndash;2018 and then "
         "repeat the result under shifted random seeds before its blind years are allowed to be scored."),
        ("04", "Blind classes, sealed vault",
         "The 2019&ndash;2025 classes are blind. They are scored through a sealed vault using three seed "
         "sets, so the years on this page cannot be tuned against."),
        ("05", "What the accuracy actually is",
         "Verified order accuracy is <b>{ea}</b> on expanding boards against <b>{da}</b> for the real "
         "draft order; on strict boards it is <b>{es}</b> against <b>{ds}</b>. This page was scored on "
         "<b>{boards}</b> boards.".format(
             ea=pct(model.get("expanding_accuracy")), da=pct(model.get("draft_accuracy_expanding")),
             es=pct(model.get("strict_accuracy")), ds=pct(model.get("draft_accuracy_strict")),
             boards=esc(meta.get("boards_used")))),
        ("06", "Open to check",
         'Everything is versioned on GitHub: the site and model documentation at '
         '<a href="{site}" rel="noopener">Spoofyy-1/draftdb</a> '
         '(<code>docs/EXPERIMENTS.md</code>, <code>model/CURRENT_MODEL.json</code>, and the board itself '
         'in <code>model/board_current_2019_2025.csv</code>), and the collected pre-draft data and its '
         'collectors at <a href="{data}" rel="noopener">Spoofyy-1/DraftDB-Data</a>.'.format(
             site=REPO_SITE, data=REPO_DATA)),
    ]
    steps_html = "\n".join(
        '        <div class="step">\n'
        '          <div class="step__n">{n}</div>\n'
        '          <div class="step__t">{t}</div>\n'
        '          <p class="step__b">{b}</p>\n'
        '        </div>'.format(n=n, t=t, b=b) for n, t, b in steps)
    return """<section class="section--tight frame" id="how" aria-labelledby="how-h">
    <div class="container">
      <h2 class="draft-year__title" id="how-h">How DraftDB works</h2>
      <div class="steps">
{steps}
      </div>
      <p class="team-foot-note">Model on this page: <b>{label}</b>. Boards used: <b>{boards}</b>.</p>
    </div>
  </section>""".format(steps=steps_html, label=esc((meta.get("model") or {}).get("label")),
                       boards=esc(meta.get("boards_used")))


# --------------------------------------------------------------------------
# whole pages
# --------------------------------------------------------------------------

def team_page(team, meta, accent, prefix):
    city, nick, name = team_labels(team)

    by_year = {}
    for draft in team.get("drafts") or []:
        y = to_num(draft.get("year"))
        if y is not None:
            by_year[int(y)] = draft
    years_with_picks = set(
        y for y, d in by_year.items() if (d.get("picks") and not d.get("no_pick")))

    description = ("Every {name} pick from 2019 to 2025 beside the player DraftDB&#39;s model would have "
                   "taken at the same slot, with WAR to date for both.").format(name=attr(name))

    parts = [
        page_head(name, description, accent, prefix),
        "",
        chrome_header(accent, prefix),
        "",
        '<main id="main">',
        "  " + hero_section(team, meta, years_with_picks),
    ]
    for y in YEARS:
        parts.append("  " + year_section(name, y, by_year.get(y)))
    parts.append("  " + honest_section(meta))
    parts.append("  " + how_section(meta))
    parts.append("</main>")
    parts.append("")
    parts.append(chrome_footer(accent, prefix, meta["generated"], meta["data_name"]))
    return "\n".join(parts) + "\n"


def index_card(slug, city, nick, accent, team):
    if team is None:
        stats = ('      <div class="team-card__stats">\n'
                 '        <div class="tmini"><div class="tmini__k">Team picks</div><div class="tmini__v">{d}</div></div>\n'
                 '        <div class="tmini tmini--model"><div class="tmini__k">Model picks</div><div class="tmini__v">{d}</div></div>\n'
                 '      </div>\n'
                 '      <div class="team-card__cards">Not in this dataset</div>').format(d=DASH)
    else:
        s = dict(team.get("summary") or {}); s.update({k: v for k, v in recount(team).items() if k in ("model_better", "team_better", "ties")})
        stats = ('      <div class="team-card__stats">\n'
                 '        <div class="tmini"><div class="tmini__k">Team picks</div><div class="tmini__v">{ta}</div></div>\n'
                 '        <div class="tmini tmini--model"><div class="tmini__k">Model picks</div><div class="tmini__v">{tm}</div></div>\n'
                 '      </div>\n'
                 '      <div class="team-card__cards">Cards: {mb} model &middot; {tb} team &middot; {ti} {tie}</div>').format(
            ta=war(s.get("war_actual_total")), tm=war(s.get("war_model_total")),
            mb=whole(s.get("model_better"), "0"), tb=whole(s.get("team_better"), "0"),
            ti=whole(s.get("ties"), "0"),
            tie=plural(int(to_num(s.get("ties")) or 0), "tie"))

    return """      <a class="team-card" href="{slug}.html" style="--tc:{tc}">
      <span class="team-card__bar"></span>
      <div class="team-card__city">{city}</div>
      <div class="team-card__name">{nick}</div>
{stats}
      <div class="team-card__go">Open team page <span class="arr">&#8594;</span></div>
      </a>""".format(slug=attr(slug), tc=attr(accent), city=esc(city), nick=esc(nick), stats=stats)


def index_page(doc, meta, by_slug, accents, prefix):
    accent = accents.get("__index__", INDEX_ACCENT)

    war_a = war_m = 0.0
    mb = tb = ti = n_cards = n_picks = 0
    have = 0
    for team in doc.get("teams") or []:
        s = dict(team.get("summary") or {}); s.update({k: v for k, v in recount(team).items() if k in ("model_better", "team_better", "ties")})
        war_a += to_num(s.get("war_actual_total")) or 0.0
        war_m += to_num(s.get("war_model_total")) or 0.0
        mb += int(to_num(s.get("model_better")) or 0)
        tb += int(to_num(s.get("team_better")) or 0)
        ti += int(to_num(s.get("ties")) or 0)
        n_cards += int(to_num(s.get("n_cards")) or 0)
        n_picks += int(to_num(s.get("n_picks")) or 0)
        have += 1

    tiles = [
        ("WAR to date &middot; team picks", war(war_a),
         "{} {} across {} {}".format(whole(n_picks), plural(n_picks, "pick"),
                                     whole(have), plural(have, "team")), ""),
        ("WAR to date &middot; model picks", war(war_m),
         "the same slots, taken in the model&rsquo;s board order", " tstat--model"),
        ("Comparison cards", '{}<span class="sl">/</span>{}<span class="sl">/</span>{}'.format(mb, tb, ti),
         "won by the model / by the team / ties", ""),
    ]
    tiles_html = "\n".join(
        '        <div class="tstat{cls}">\n'
        '          <div class="tstat__k">{k}</div>\n'
        '          <div class="tstat__v">{v}</div>\n'
        '          <div class="tstat__sub">{sub}</div>\n'
        '        </div>'.format(cls=cls, k=k, v=v, sub=sub) for k, v, sub, cls in tiles)

    seen = set()
    cards = []
    for slug, city, nick, fallback in TEAMS:
        seen.add(slug)
        cards.append(index_card(slug, city, nick, accents.get(slug, fallback), by_slug.get(slug)))
    for team in doc.get("teams") or []:
        slug = str(team.get("slug") or "").strip()
        if slug and slug not in seen:
            city, nick, _ = team_labels(team)
            cards.append(index_card(slug, city, nick, accents.get(slug, INDEX_ACCENT), team))

    body = """<main id="main">
  <section class="team-hero frame">
    <div class="container">
      <div class="team-hero__eyebrow"><span class="team-swatch"></span>30 teams &middot; 2019&ndash;2025 drafts</div>
      <h1 class="team-hero__title">Every team&rsquo;s draft, re-run.</h1>
      <p class="team-verdict">Across the 2019&ndash;2025 drafts, pick by pick, the model&rsquo;s top available player out-produced the team&rsquo;s selection on <b>{mb} of {nc} cards</b> and the team&rsquo;s pick did better on <b>{tb}</b>. Summed over cards the model&rsquo;s choices have <b>{wm} WAR</b> to date against <b>{wa}</b> for the teams&rsquo; picks; the same alternative can appear on several cards, so the sums compare choices, not rosters. Open a team to see the pick-by-pick cards.</p>
      <div class="tstat-grid">
{tiles}
      </div>
      <p class="team-foot-note">{short} Boards used: <b>{boards}</b>. Model: <b>{label}</b><br>{wardef}</p>
    </div>
  </section>
  <section class="section--tight frame">
    <div class="container">
      <div class="teams-grid">
{cards}
      </div>
    </div>
  </section>
</main>""".format(wm=war(war_m), wa=war(war_a), mb=mb, tb=tb, nc=n_cards, tiles=tiles_html, cards="\n".join(cards),
                  short=WAR_SHORT, wardef=esc(meta.get("war_definition"), dash=""),
                  boards=esc(meta.get("boards_used")),
                  label=esc((meta.get("model") or {}).get("label")))

    description = ("Every NBA team&#39;s 2019-2025 draft picks beside the players DraftDB&#39;s model "
                   "would have taken at the same slots, with WAR to date.")
    return "\n".join([
        page_head("Teams", description, accent, prefix),
        "",
        chrome_header(accent, prefix),
        "",
        body,
        "",
        chrome_footer(accent, prefix, meta["generated"], meta["data_name"]),
    ]) + "\n"


# --------------------------------------------------------------------------
# audit + validation
# --------------------------------------------------------------------------

def audit_data(doc, by_slug):
    """Data-level warnings — reported, never silently patched."""
    notes = []
    missing = [slug for slug, _, _, _ in TEAMS if slug not in by_slug]
    if missing:
        notes.append("{} of 30 teams are absent from the dataset: {}".format(
            len(missing), ", ".join(missing)))
    unknown = [s for s in by_slug if s not in TEAM_BY_SLUG]
    if unknown:
        notes.append("slugs not in the built-in team table: {}".format(", ".join(sorted(unknown))))

    sign_bad = 0
    no_board = 0
    off_years = set()
    for team in doc.get("teams") or []:
        for draft in team.get("drafts") or []:
            y = to_num(draft.get("year"))
            if y is not None and int(y) not in YEARS:
                off_years.add(int(y))
            for pick in draft.get("picks") or []:
                if not (pick.get("model_choice") or {}):
                    no_board += 1
                    continue
                d = to_num(pick.get("delta_war"))
                v = str(pick.get("verdict") or "").lower()
                if d is None:
                    continue
                if (v == "model" and d < -0.05) or (v == "team" and d > 0.05):
                    sign_bad += 1
    if off_years:
        notes.append("draft years outside 2019-2025 were ignored: {}".format(
            ", ".join(str(y) for y in sorted(off_years))))
    return notes, sign_bad, no_board


FORBIDDEN = [
    (re.compile(r"\bNone\b"), "None"),
    (re.compile(r"\bnan\b"), "nan"),
    (re.compile(r"\bNaN\b"), "NaN"),
    (re.compile(r"\bnull\b"), "null"),
    (re.compile(r"\bundefined\b"), "undefined"),
    (re.compile(r"\{\{|\}\}"), "{{ }}"),
    (re.compile(r"%\(|%s\b"), "%-placeholder"),
]

AMP_RE = re.compile(r"&(?!(?:[a-zA-Z][a-zA-Z0-9]{1,9}|#[0-9]{1,7}|#[xX][0-9a-fA-F]{1,6});)")
PAIRED_TAGS = ("html", "head", "body", "main", "section", "article", "div", "span", "nav",
               "header", "footer", "dl", "dt", "dd", "ul", "li", "p", "h1", "h2", "h3", "a", "style")


def validate(files):
    """Structural check of the written pages. Returns (problems, stats)."""
    problems = []
    stats = {"pages": 0, "cards": 0, "nopick": 0, "years_ok": 0}

    for path in files:
        text = open(path, "r", encoding="utf-8").read()
        name = os.path.basename(path)
        stats["pages"] += 1
        is_team_page = name != "index.html"

        if not text.startswith("<!DOCTYPE html>") or not text.rstrip().endswith("</html>"):
            problems.append("{}: document does not open with <!DOCTYPE html> and close with </html>".format(name))

        if is_team_page:
            sections = len(re.findall(r'<section class="draft-year', text))
            if sections != len(YEARS):
                problems.append("{}: {} draft sections, expected {}".format(name, sections, len(YEARS)))
            else:
                stats["years_ok"] += 1
            for y in YEARS:
                if 'id="y{}"'.format(y) not in text:
                    problems.append("{}: missing anchor for {}".format(name, y))
            stats["cards"] += len(re.findall(r'<article class="pickcard', text))
            stats["nopick"] += len(re.findall(r'<div class="nopick"', text))

        for pattern, label in FORBIDDEN:
            hits = pattern.findall(text)
            if hits:
                problems.append("{}: {} occurrence(s) of the placeholder/absent-value token {!r}".format(
                    name, len(hits), label))

        body = text.split("<main", 1)[-1].split("</main>", 1)[0]
        bad_amp = AMP_RE.findall(body)
        if bad_amp:
            problems.append("{}: {} unescaped '&' inside <main>".format(name, len(bad_amp)))

        for tag in PAIRED_TAGS:
            opened = len(re.findall(r"<{}\b".format(tag), text))
            closed = len(re.findall(r"</{}>".format(tag), text))
            if opened != closed:
                problems.append("{}: <{}> opened {} times, closed {} times".format(
                    name, tag, opened, closed))

    return problems, stats


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def build(data_path, out_dir, prefix):
    with open(data_path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    if not isinstance(doc, dict) or not isinstance(doc.get("teams"), list):
        raise SystemExit("error: {} is not a team-cards document "
                         "(expected an object with a \"teams\" list)".format(data_path))

    data_name = os.path.relpath(os.path.abspath(data_path), SITE).replace(os.sep, "/")
    if data_name.startswith(".."):
        data_name = os.path.basename(data_path)

    meta = {
        "generated": iso_date(doc.get("generated")),
        "boards_used": doc.get("boards_used"),
        "war_definition": doc.get("war_definition"),
        "model": doc.get("model") or {},
        "data_name": data_name,
    }

    by_slug = {}
    for team in doc["teams"]:
        slug = str(team.get("slug") or "").strip()
        if slug:
            by_slug[slug] = team

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # accents come from the pages being replaced, so a hand-tuned colour survives
    accents = {}
    for slug, _city, _nick, fallback in TEAMS:
        accents[slug] = read_accent(os.path.join(out_dir, slug + ".html"),
                                    read_accent(os.path.join(SITE, "teams", slug + ".html"), fallback))
    for slug in by_slug:
        if slug not in accents:
            accents[slug] = read_accent(os.path.join(out_dir, slug + ".html"), INDEX_ACCENT)
    accents["__index__"] = read_accent(os.path.join(out_dir, "index.html"),
                                       read_accent(os.path.join(SITE, "teams", "index.html"), INDEX_ACCENT))

    written = []
    for slug in sorted(by_slug):
        team = by_slug[slug]
        path = os.path.join(out_dir, slug + ".html")
        html_text = team_page(team, meta, accents.get(slug, INDEX_ACCENT), prefix)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html_text)
        written.append(path)

    index_path = os.path.join(out_dir, "index.html")
    with open(index_path, "w", encoding="utf-8") as fh:
        fh.write(index_page(doc, meta, by_slug, accents, prefix))
    written.append(index_path)

    notes, sign_bad, no_board = audit_data(doc, by_slug)
    return doc, meta, written, notes, sign_bad, no_board


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="build_team_pages.py",
        description="Build the DraftDB team pages from data/team_cards.json.")
    parser.add_argument("--data", default=os.path.join(SITE, "data", "team_cards.json"),
                        help="team-cards JSON (default: site/data/team_cards.json)")
    parser.add_argument("--out", default=os.path.join(SITE, "teams"),
                        help="directory to write the pages into (default: site/teams)")
    parser.add_argument("--asset-prefix", default="../",
                        help="path from a written page back to the site root "
                             "(default: ../ ; use ../../ for tools/_preview)")
    args = parser.parse_args(argv)

    data_path = os.path.abspath(args.data)
    out_dir = os.path.abspath(args.out)
    prefix = args.asset_prefix
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    if not os.path.exists(data_path):
        sys.stderr.write(
            "error: no dataset at {}\n"
            "       build it first, or point --data at data/team_cards.mock.json\n".format(data_path))
        return 2

    doc, meta, written, notes, sign_bad, no_board = build(data_path, out_dir, prefix)
    problems, stats = validate(written)

    print("DraftDB team pages")
    print("  data    {}".format(data_path))
    print("  out     {}".format(out_dir))
    print("  assets  {}".format(prefix or "(site root)"))
    print("  dataset generated {} · boards {} · model {}".format(
        meta["generated"], meta.get("boards_used") or "?",
        (meta.get("model") or {}).get("label") or "?"))
    print("")
    print("FILES ({})".format(len(written)))
    for path in written:
        print("  {:<42} {:>8.1f} KB".format(
            os.path.relpath(path, SITE).replace(os.sep, "/"), os.path.getsize(path) / 1024.0))
    print("")
    print("VALIDATION")
    team_pages = stats["pages"] - 1
    print("  team pages written .......... {}".format(team_pages))
    print("  draft sections per page ..... {} of {} pages have all {} ({}-{})".format(
        stats["years_ok"], team_pages, len(YEARS), YEARS[0], YEARS[-1]))
    print("  comparison cards ............ {}".format(stats["cards"]))
    print("  'no selection' rows ......... {}".format(stats["nopick"]))
    print("  picks with no model board ... {}".format(no_board))
    print("  verdict vs delta sign ....... {} disagreement(s)".format(sign_bad))
    print("  placeholder / &-escape / tag-balance scan: {}".format(
        "clean" if not problems else "{} problem(s)".format(len(problems))))
    for problem in problems:
        print("    ! {}".format(problem))
    for note in notes:
        print("  note: {}".format(note))
    print("")
    print("OK" if not problems else "FAILED")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
