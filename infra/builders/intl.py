"""Download and parse pre-draft international / youth basketball sources into data/external/intl_prospects.parquet.

Stages (each idempotent, raw files cached under data/external/intl_raw/):
  bbref     basketball-reference.com international league-season player totals (12 leagues) + NBA player index (birth dates)
  gleague   stats.gleague.nba.com season totals (Regular Season + Showcase Cup; includes G League Ignite)
  timur     timurkulenovic/basketball-dataset EuroLeague / EuroCup / ABA / Slovenian league box scores (PIR, fouls drawn)
  ngt       EuroLeague U18 Next Generation Tournament rosters + per-year player statistics (api-live.euroleague.net)
  eurocamp  adidas Eurocamp archive rosters (grassroots.adidas.com; archived years only)
  fiba      FIBA youth national-team competitions U15-U20 (digital-api.fiba.basketball GDAP gateway behind fiba.basketball)
  build     match everything to the draft table and write data/external/intl_prospects.parquet

Usage: python -m infra.builders.intl [stage ...]   (default: all stages, downloads skipped if cached)
Loader: load_intl() -> DataFrame keyed by (key, draft_year) with the i_* columns.

Leakage rules: club seasons count only if the season ends in the draft year or earlier (season label <= draft_year); FIBA
events and NGT seasons count only if they ended before the draft night (DRAFT_DATES). No NBA statistic, mock rank or
model output is read anywhere.

i_league_level: EuroLeague 5; EuroCup, ACB, NBL Australia 4; ABA, LBA, LNB, BSL, VTB, GBL, ISR, G League 3; CBA, SLO 2; other 1.
i_fiba_max_level / i_fiba_last_level: U17/U19 World Cup 4; European Div A finals 3; other continental finals 2; Div B/C,
sub-zonal qualifiers, Challengers 1.
"""

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from lxml import html

from infra.config import PROC, ROOT
from infra.dataset import norm_name

EXT = ROOT / "data" / "external"
RAW = EXT / "intl_raw"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}
BBREF = "https://www.basketball-reference.com"
BBREF_SLEEP = 3.5  # ~17 requests / minute, under the 20/min limit

BBREF_LEAGUES = {
    "euroleague": "EuroLeague", "eurocup": "EuroCup", "spain-liga-acb": "ACB", "italy-basket-serie-a": "LBA",
    "france-lnb-pro-a": "LNB", "greek-basket-league": "GBL", "israel-super-league": "ISR", "aba-adriatic": "ABA",
    "nbl-australia": "NBL", "cba-china": "CBA", "turkey-super-league": "BSL", "vtb-united": "VTB",
}
SEASONS = range(2006, 2027)  # season label = calendar year the season ends


def _get(url, path: Path, sleep=0.0, ok404=True) -> bool:
    """Download url to path unless cached. Returns True if the file exists afterwards."""
    if path.exists():
        return path.stat().st_size > 0
    for attempt in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=60)
            break
        except requests.exceptions.ConnectionError as e:  # transient TLS resets
            if attempt == 2:
                raise
            print(f"retry {url}: {e}", flush=True)
            time.sleep(5)
    time.sleep(sleep)
    path.parent.mkdir(parents=True, exist_ok=True)
    if r.status_code == 404 and ok404:
        path.write_bytes(b"")  # remember the miss
        return False
    if r.status_code == 429:
        print("rate limited, sleeping 90s", flush=True)
        time.sleep(90)
        return _get(url, path, sleep, ok404)
    r.raise_for_status()
    path.write_bytes(r.content)
    return True


# --------------------------------------------------------------------------- bbref international

def bbref_download():
    d = RAW / "bbref_intl"
    for lg in BBREF_LEAGUES:
        for y in SEASONS:
            _get(f"{BBREF}/international/{lg}/{y}_totals.html", d / f"{lg}_{y}_totals.html", sleep=BBREF_SLEEP)
    # NBA player index: birth dates for everyone who played an NBA game (one page per letter)
    for ch in "abcdefghijklmnopqrstuvwyz":
        _get(f"{BBREF}/players/{ch}/", RAW / "bbref_players" / f"{ch}.html", sleep=BBREF_SLEEP)


def _text(td):
    return td.text_content().strip()


def _doc(path: Path):
    """bbref ships some tables inside HTML comments (lazy loading); un-comment before parsing."""
    return html.fromstring(path.read_bytes().replace(b"<!--", b"").replace(b"-->", b""))


BOX = ("g", "mp", "fg", "fga", "fg3", "fg3a", "fg2", "fg2a", "ft", "fta", "orb", "drb", "trb", "ast", "stl", "blk", "tov", "pf", "pts")


def _bbref_rows(doc, table_id, league, season, team_stat):
    for tr in doc.xpath(f'//table[@id="{table_id}"]/tbody/tr[not(contains(@class,"thead"))]'):
        cells = {td.get("data-stat"): td for td in tr}
        if "player" not in cells or not _text(cells["player"]):
            continue
        rec = {"league": league, "season": season, "player": _text(cells["player"]), "team": _text(cells[team_stat]) if team_stat in cells else None}
        m = re.search(r"/players/(?:\w/)?([^/]+)\.html", (cells["player"].xpath(".//a/@href") or [""])[0])
        rec["bbref_intl_id"] = m.group(1) if m else None
        rec["age"] = pd.to_numeric(_text(cells["age"]), errors="coerce") if "age" in cells else np.nan
        for k in BOX:
            rec[k] = pd.to_numeric(_text(cells[k]), errors="coerce") if k in cells else np.nan
        yield rec


def parse_bbref_totals() -> pd.DataFrame:
    recs = []
    for p in sorted((RAW / "bbref_intl").glob("*_totals.html")):
        if p.stat().st_size == 0:
            continue
        lg, y = re.match(r"(.+)_(\d{4})_totals", p.stem).groups()
        recs += list(_bbref_rows(_doc(p), f"totals-stats-{y}", BBREF_LEAGUES[lg], int(y), "team_name"))
    df = pd.DataFrame(recs)
    print(f"bbref intl: {len(df)} player-season-team rows, {df.groupby('league').season.nunique().to_dict()}")
    return df


# --------------------------------------------------------------------------- NBA G League (stats.gleague.nba.com; includes G League Ignite)
GLG_API = "https://stats.gleague.nba.com/stats/leaguedashplayerstats"
GLG_HEADERS = {**UA, "Referer": "https://gleague.nba.com/", "Origin": "https://gleague.nba.com", "Accept": "application/json"}


def gleague_download():
    for y in range(2008, 2027):
        for st in ("Regular Season", "Showcase"):  # Ignite's 2021-22 games only count in the Showcase Cup
            p = RAW / "gleague" / f"{y}_{st.replace(' ', '_')}.json"
            if p.exists():
                continue
            params = {"College": "", "Conference": "", "Country": "", "DateFrom": "", "DateTo": "", "Division": "", "DraftPick": "", "DraftYear": "",
                      "GameScope": "", "GameSegment": "", "Height": "", "LastNGames": 0, "LeagueID": "20", "Location": "", "MeasureType": "Base",
                      "Month": 0, "OpponentTeamID": 0, "Outcome": "", "PORound": 0, "PaceAdjust": "N", "PerMode": "Totals", "Period": 0,
                      "PlayerExperience": "", "PlayerPosition": "", "PlusMinus": "N", "Rank": "N", "Season": f"{y - 1}-{str(y)[2:]}",
                      "SeasonSegment": "", "SeasonType": st, "ShotClockRange": "", "StarterBench": "", "TeamID": 0, "VsConference": "",
                      "VsDivision": "", "Weight": ""}
            r = requests.get(GLG_API, params=params, headers=GLG_HEADERS, timeout=60)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(r.content if r.ok else b"")
            time.sleep(1)


def parse_gleague() -> pd.DataFrame:
    import json
    frames = []
    for p in sorted((RAW / "gleague").glob("*.json")):
        if p.stat().st_size == 0:
            continue
        rs = json.loads(p.read_bytes())["resultSets"][0]
        df = pd.DataFrame(rs["rowSet"], columns=rs["headers"])
        df["season"] = int(p.stem[:4])
        frames.append(df)
    g = pd.concat(frames, ignore_index=True)
    g = g.rename(columns={"PLAYER_NAME": "player", "TEAM_ABBREVIATION": "team", "AGE": "age", "GP": "g", "MIN": "mp", "FGM": "fg", "FGA": "fga",
                          "FG3M": "fg3", "FG3A": "fg3a", "FTM": "ft", "FTA": "fta", "OREB": "orb", "REB": "trb", "AST": "ast", "STL": "stl",
                          "BLK": "blk", "TOV": "tov", "PF": "pf", "PFD": "pfd", "PTS": "pts"})
    g["league"] = "GLG"
    g["bbref_intl_id"] = "glg" + g.PLAYER_ID.astype(str)
    cols = ["league", "season", "player", "team", "bbref_intl_id", "age", "g", "mp", "fg", "fga", "fg3", "fg3a", "ft", "fta", "orb", "trb", "ast", "stl", "blk", "tov", "pf", "pfd", "pts"]
    g = g[cols].groupby(["league", "season", "player", "bbref_intl_id"], as_index=False).agg({**{c: "sum" for c in cols[6:]}, "team": lambda s: "/".join(dict.fromkeys(s)), "age": "max"})
    print(f"gleague: {len(g)} player-season rows, seasons {g.season.min()}-{g.season.max()}")
    return g


def parse_bbref_birthdates() -> pd.DataFrame:
    recs = []
    for p in sorted((RAW / "bbref_players").glob("*.html")):
        if p.stat().st_size == 0:
            continue
        doc = html.fromstring(p.read_bytes())
        for tr in doc.xpath('//table[@id="players"]/tbody/tr'):
            th = tr.xpath('.//th[@data-stat="player"]')
            bd = tr.xpath('.//td[@data-stat="birth_date"]')
            if not th or not th[0].get("data-append-csv"):
                continue
            recs.append({"bbref_id": th[0].get("data-append-csv"), "birthdate": pd.to_datetime(_text(bd[0]), errors="coerce") if bd else pd.NaT})
    return pd.DataFrame(recs).drop_duplicates("bbref_id")


# --------------------------------------------------------------------------- timurkulenovic/basketball-dataset
TIMUR_RAW = "https://raw.githubusercontent.com/timurkulenovic/basketball-dataset/HEAD/data"
TIMUR_LEAGUES = {"euroleague": "EuroLeague", "eurocup": "EuroCup", "aba": "ABA", "slo": "SLO"}


def timur_download():
    for lg in TIMUR_LEAGUES:
        for f in ("box_score", "games"):
            _get(f"{TIMUR_RAW}/{lg}/games/parquet/{f}.parquet", RAW / "timur" / f"{lg}_{f}.parquet", ok404=False)


def _season_end(s: str) -> int:
    """'2013/14' or '2013/2014' -> 2014."""
    a, b = str(s).split("/")
    return int(a[:2] + b) if len(b) == 2 else int(b)


def parse_timur() -> pd.DataFrame:
    """Per player-season box-score aggregates (incl. PIR = VAL and fouls drawn) with the team name from the games table."""
    out = []
    for lg, name in TIMUR_LEAGUES.items():
        b = pd.read_parquet(RAW / "timur" / f"{lg}_box_score.parquet")
        g = pd.read_parquet(RAW / "timur" / f"{lg}_games.parquet")
        g.columns = [c.upper() for c in g.columns]
        gid = "GAME_ID" if "GAME_ID" in g.columns else "ID"
        g = g.rename(columns={gid: "GAME_ID"})[["GAME_ID", "H_TEAM", "A_TEAM"]].drop_duplicates("GAME_ID")
        b = b[b.TYPE == "Player"].merge(g, on="GAME_ID", how="left")
        b["team"] = np.where(b.TEAM == "H", b.H_TEAM, b.A_TEAM)
        b["player"] = b.PLAYER_NAME.map(_first_last)
        b["season"] = b.SEASON.map(_season_end)
        b["mins"] = b.MINUTES.fillna(0).astype(float) + b.SECONDS.fillna(0).astype(float) / 60
        b["played"] = (b.mins > 0) | (b.POINTS.fillna(0) > 0)
        reb_o, reb_d, reb_t = ("REB_OFF", "REB_DEF", "REB_TOT") if "REB_TOT" in b else ("REB_O", "REB_D", "REB_T")
        if lg == "aba":  # source has offensive/defensive rebounds swapped (71% of all rebounds tagged offensive, every season)
            reb_o, reb_d = reb_d, reb_o
        cols = {"POINTS": "pts", reb_t: "trb", reb_o: "orb", "ASS": "ast", "ST": "stl", "TO": "tov", "BLC_FV": "blk", "FG2_M": "fg2", "FG2_A": "fg2a",
                "FG3_M": "fg3", "FG3_A": "fg3a", "FT_M": "ft", "FT_A": "fta", "FLS_CM": "pf", "FLS_RV": "pfd", "VAL": "pir"}
        for c in cols:
            b[c] = pd.to_numeric(b[c], errors="coerce").astype(float)
        agg = b[b.played].groupby(["season", "player", "team"]).agg(g=("GAME_ID", "nunique"), mp=("mins", "sum"), **{v: (k, "sum") for k, v in cols.items()})
        agg = agg.reset_index()
        agg["league"] = name
        out.append(agg)
    df = pd.concat(out, ignore_index=True)
    df["fg"], df["fga"] = df.fg2 + df.fg3, df.fg2a + df.fg3a
    print(f"timur: {len(df)} player-season-team rows, {df.groupby('league').season.agg(['min', 'max', 'nunique']).to_dict('index')}")
    return df


def _first_last(s: str) -> str:
    s = str(s).strip()
    if "," in s:
        last, first = s.split(",", 1)
        return f"{first.strip()} {last.strip()}"
    return s


# --------------------------------------------------------------------------- FIBA (digital-api.fiba.basketball GDAP gateway)
# Public read-only gateway behind fiba.basketball; the subscription key is embedded in every page of the site.
FIBA_API = "https://digital-api.fiba.basketball/hapi"
FIBA_KEY = "898cd5e7389140028ecb42943c47eb74"
FIBA_YEARS = range(2001, 2027)
_fiba_session = requests.Session()


def fiba_get(endpoint: str, path: Path, retries=4) -> bool:
    """GET {FIBA_API}/{endpoint} to path (json). Cached. Empty file marks a 204/404. Returns False if all attempts failed."""
    if path.exists():
        return True
    for attempt in range(retries):
        try:
            r = _fiba_session.get(f"{FIBA_API}/{endpoint}", headers={**UA, "Ocp-Apim-Subscription-Key": FIBA_KEY}, timeout=90)
            if r.status_code in (204, 404):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"")
                return True
            r.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(r.content)
            return True
        except requests.RequestException as e:
            print(f"fiba retry {attempt} {endpoint}: {e}", flush=True)
            time.sleep(3 * (attempt + 1))
    return False


def fiba_games_files(y: int) -> list[Path]:
    """Games calendar for one year: whole year if the server manages it, else four quarters."""
    d = RAW / "fiba"
    whole = d / f"games_{y}.json"
    if whole.exists() or fiba_get(f"getgdapgamesbetweentwodates?dateFrom={y}-01-01&dateTo={y}-12-31", whole, retries=1):
        return [whole]
    out = []
    for q, (a, b) in enumerate([("01-01", "03-31"), ("04-01", "06-30"), ("07-01", "09-30"), ("10-01", "12-31")], 1):
        p = d / f"games_{y}_q{q}.json"
        if not fiba_get(f"getgdapgamesbetweentwodates?dateFrom={y}-{a}&dateTo={y}-{b}", p):
            raise RuntimeError(f"fiba games {y} q{q} failed")
        out.append(p)
    return out


def _load_json(path: Path):
    import json
    if not path.exists() or path.stat().st_size == 0:
        return None
    return json.loads(path.read_bytes())


def fiba_competitions() -> pd.DataFrame:
    """All men's national-team competitions with at least one game, from the games calendar."""
    files = [p for y in FIBA_YEARS for p in fiba_games_files(y)]
    comps = {}
    for p in files:
        for g in _load_json(p) or []:
            c = g.get("competition") or {}
            if not c.get("competitionId"):
                continue
            cid = c["competitionId"]
            rec = comps.setdefault(cid, {
                "competition_id": cid, "name": c.get("officialName"), "season": c.get("season"), "age": c.get("ageCategory"),
                "gender": c.get("gender"), "zone": c.get("fibaZone"), "type": c.get("competitionType"),
                "cat_code": (c.get("competitionCategory") or {}).get("code"), "cat_name": (c.get("competitionCategory") or {}).get("name"),
                "start": c.get("start"), "end": c.get("end"), "n_games": 0, "last_game": None})
            rec["n_games"] += 1
            gd = g.get("gameDateTime") or ""
            rec["last_game"] = max(rec["last_game"] or "", gd)
    return pd.DataFrame(list(comps.values()))


def fiba_download_youth(comps: pd.DataFrame, workers=4):
    """Rosters (birth dates) and per-player competition statistics for every team of every selected competition."""
    import concurrent.futures as cf
    d = RAW / "fiba"
    for cid in comps.competition_id:
        fiba_get(f"getgdapcompetitionteamsbycompetitionid?gdapCompetitionId={cid}&profile=true", d / "teams" / f"{cid}.json")
    jobs = []
    for cid in comps.competition_id:
        for t in _load_json(d / "teams" / f"{cid}.json") or []:
            tid = t["teamId"]
            jobs.append((f"getgdapcompetitionteamrosterbyteamid?gdapTeamId={tid}", d / "roster" / f"{tid}.json"))
            jobs.append((f"getgdapcompetitionteamstatisticsbyteamid?gdapTeamId={tid}", d / "stats" / f"{tid}.json"))
    todo = [j for j in jobs if not j[1].exists()]
    print(f"fiba: {len(jobs)} team files, {len(todo)} to download", flush=True)
    with cf.ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda j: fiba_get(*j), todo))


def parse_fiba(comps: pd.DataFrame) -> pd.DataFrame:
    """One row per player x competition: bio from the roster, box-score averages from the team statistics call."""
    d = RAW / "fiba"
    recs = []
    for c in comps.itertuples():
        for t in _load_json(d / "teams" / f"{c.competition_id}.json") or []:
            tid = t["teamId"]
            roster = _load_json(d / "roster" / f"{tid}.json") or {}
            bio = {p["personId"]: p for p in roster.get("players") or []}
            stats = (_load_json(d / "stats" / f"{tid}.json") or {}).get("playerInCompetitionTeamStatistics") or []
            seen = set()
            for s in stats:
                pid = s["playerId"]
                seen.add(pid)
                b = bio.get(pid, {})
                recs.append(_fiba_row(c, t, pid, b, s))
            for pid, b in bio.items():  # rostered, no stats row (did not play / stats missing)
                if pid not in seen:
                    recs.append(_fiba_row(c, t, pid, b, {}))
    df = pd.DataFrame(recs)
    print(f"fiba: {len(df)} player-competition rows over {df.competition_id.nunique()} competitions")
    return df


def _fiba_row(c, t, pid, b, s):
    g = s.get("totalGamesPlayed") or b.get("gamesPlayed")
    return {
        "competition_id": c.competition_id, "competition": c.name, "season": c.season, "age_cat": c.age, "zone": c.zone,
        "cat_code": c.cat_code, "comp_start": c.start, "comp_end": c.end,
        "team_id": t["teamId"], "team_code": (t.get("profile") or {}).get("code") or t.get("code"),
        "player_id": pid, "first_name": b.get("firstName", s.get("firstName")), "last_name": b.get("lastName", s.get("lastName")),
        "dob": b.get("dateOfBirth"), "height_cm": b.get("heightInCm"), "position": b.get("position"), "nationality": b.get("nationality"),
        "club": b.get("clubName"), "final_rank": (t.get("profile") or {}).get("finalRanking"), "on_final": b.get("isOnFinalRoster"),
        "gp": g, "min": (s.get("totalPlayTimeInSeconds") or np.nan) / 60 if s else np.nan,
        "pts": s.get("totalPoints"), "reb": s.get("totalRebounds"), "oreb": s.get("totalReboundsOffensive"), "ast": s.get("totalAssists"),
        "stl": s.get("totalSteals"), "blk": s.get("totalBlocks"), "tov": s.get("totalTurnovers"), "pf": s.get("totalFouls"),
        "eff": s.get("totalEfficiency"), "fgm": s.get("totalFieldGoalsMade"), "fga": s.get("totalFieldGoalsAttempted"),
        "fg3m": s.get("totalThreePointsMade"), "fg3a": s.get("totalThreePointsAttempted"), "ftm": s.get("totalFreeThrowsMade"),
        "fta": s.get("totalFreeThrowsAttempted"), "plus_minus": s.get("totalPlusMinus"),
    }


# --------------------------------------------------------------------------- EuroLeague Next Generation Tournament (U18), api-live.euroleague.net
EL_API = "https://api-live.euroleague.net"


def ngt_download():
    d = RAW / "ngt"
    _get(f"{EL_API}/v2/competitions/J/seasons", d / "seasons.json", ok404=False)
    seasons = _load_json(d / "seasons.json")["data"]
    for s in seasons:
        _get(f"{EL_API}/v2/competitions/J/seasons/{s['code']}/people", d / "people" / f"{s['code']}.json", sleep=0.3)
    # player statistics are aggregated per season-year (every tournament of that year), whichever season code is passed
    for y, code in {s["year"]: s["code"] for s in seasons}.items():
        _get(f"{EL_API}/v3/competitions/J/statistics/players/traditional?SeasonMode=Single&SeasonCode={code}&statisticMode=Accumulated&limit=2000",
             d / "stats" / f"{y}.json", sleep=0.3)


def parse_ngt() -> pd.DataFrame:
    d = RAW / "ngt"
    seasons = pd.DataFrame(_load_json(d / "seasons.json")["data"])
    seasons["end"] = pd.to_datetime(seasons.endDate.str.slice(0, 19))
    year_end = seasons.groupby("year").end.max()
    bio = {}
    for p in sorted((d / "people").glob("*.json")):
        for r in (_load_json(p) or {}).get("data", []):
            if r.get("type") != "J":  # J = player
                continue
            per = r["person"]
            bio.setdefault(per["code"], {"dob": per.get("birthDate"), "height": per.get("height"), "country": (per.get("country") or {}).get("code"),
                                         "first": per.get("passportName"), "last": per.get("passportSurname")})
    recs = []
    for p in sorted((d / "stats").glob("*.json")):
        y = int(p.stem)
        for r in (_load_json(p) or {}).get("players", []):
            pl = r["player"]
            b = bio.get(pl["code"], {})
            recs.append({"year": y, "end": year_end.get(y), "code": pl["code"], "player": _first_last(pl["name"]), "age": pl.get("age"),
                         "team": (pl.get("team") or {}).get("name"), "dob": b.get("dob"), "height_cm": b.get("height"), "country": b.get("country"),
                         "g": r["gamesPlayed"], "mp": r["minutesPlayed"], "pts": r["pointsScored"], "fg2": r["twoPointersMade"], "fg2a": r["twoPointersAttempted"],
                         "fg3": r["threePointersMade"], "fg3a": r["threePointersAttempted"], "ft": r["freeThrowsMade"], "fta": r["freeThrowsAttempted"],
                         "orb": r["offensiveRebounds"], "trb": r["totalRebounds"], "ast": r["assists"], "stl": r["steals"], "tov": r["turnovers"],
                         "blk": r["blocks"], "pf": r["foulsCommited"], "pfd": r["foulsDrawn"], "pir": r["pir"]})
    df = pd.DataFrame(recs)
    df["fg"], df["fga"] = df.fg2 + df.fg3, df.fg2a + df.fg3a
    print(f"ngt: {len(df)} player-year rows, years {df.year.min()}-{df.year.max()}, {df.dob.notna().mean():.0%} with birth date")
    return df


# --------------------------------------------------------------------------- adidas Eurocamp archive (full camp rosters, archived years only)
EUROCAMP_YEARS = [2012, 2013, 2014, 2016, 2017, 2023, 2024, 2025]  # years with a roster page on grassroots.adidas.com/eurocamp/archive


def eurocamp_download():
    for y in EUROCAMP_YEARS:
        _get(f"https://grassroots.adidas.com/eurocamp/archive/{y}", RAW / "eurocamp" / f"{y}.html", sleep=1.0)


def parse_eurocamp() -> pd.DataFrame:
    """Roster = player profile images on the archive page (name in the img alt, else in the file name). Opponent squads brought in
    to play the campers (USA Select, 3SSB Select, national U20 teams) are not Eurocamp invitees and are dropped."""
    recs = []
    for y in EUROCAMP_YEARS:
        p = RAW / "eurocamp" / f"{y}.html"
        if not p.exists() or p.stat().st_size == 0:
            continue
        raw = p.read_bytes().decode("utf-8", "ignore")
        for team, fname, alt in re.findall(rf'assets/{y}/team/profiles/([^/"]+)/([^/"]+?)\.(?:png|jpg|jpeg|webp)"(?: alt="([^"]*)")?', raw):
            if re.search(r"usa|select|national|3ssb", team, re.I):
                continue
            name = (alt or fname).replace("%20", " ").strip()
            if name and not name.isdigit():
                recs.append({"year": y, "team": team, "player": name})
    df = pd.DataFrame(recs).drop_duplicates(["year", "player"])
    print(f"eurocamp: {len(df)} camper rows, {df.groupby('year').size().to_dict()}")
    return df


# --------------------------------------------------------------------------- build: match to draftees, derive features

# Draft nights (the leakage cut for events; club seasons are cut on season-end year <= draft_year).
DRAFT_DATES = {2003: "06-26", 2004: "06-24", 2005: "06-28", 2006: "06-28",
               2007: "06-28", 2008: "06-26", 2009: "06-25", 2010: "06-24", 2011: "06-23", 2012: "06-28", 2013: "06-27", 2014: "06-26",
               2015: "06-25", 2016: "06-23", 2017: "06-22", 2018: "06-21", 2019: "06-20", 2020: "11-18", 2021: "07-29", 2022: "06-23",
               2023: "06-22", 2024: "06-26", 2025: "06-25", 2026: "06-24"}

# Ordinal strength of the competition the pre-draft season was played in.
LEAGUE_LEVEL = {"EuroLeague": 5, "EuroCup": 4, "ACB": 4, "NBL": 4, "ABA": 3, "LBA": 3, "LNB": 3, "BSL": 3, "VTB": 3, "GBL": 3, "ISR": 3,
                "GLG": 3, "CBA": 2, "SLO": 2}

# Name fixes: draft-table name -> name used by the club/FIBA sources (nicknames, name changes, transliterations).
ALIASES = {
    "Enes Freedom": ["Enes Kanter"], "Edy Tavares": ["Walter Tavares", "Walter Samuel Tavares"], "Raul Neto": ["Raulzinho Neto", "Raul Togni Neto"],
    "Paulão Prestes": ["Paulo Prestes", "Paulao Prestes"], "Bruno Caboclo": ["Bruno Cabloco"], "Didi Louzada": ["Marcos Louzada", "Marcos Louzada Silva"],
    "Alpha Kaba": ["Alpha Kaba"], "Ricky Rubio": ["Ricard Rubio"], "Rudy Fernández": ["Rodolfo Fernandez"], "Sergio Llull": ["Sergi Llull"],
    "KJ Martin": ["Kenyon Martin Jr.", "Kenyon Martin"], "Nando De Colo": ["Nando de Colo"], "Sun Yue": ["Yue Sun"], "Yi Jianlian": ["Jianlian Yi"],
    "Zhou Qi": ["Qi Zhou"], "Wang Zhelin": ["Zhelin Wang"], "Yang Hansen": ["Hansen Yang"], "Gui Santos": ["Guilherme Santos", "Gui Deodato"],
    "Vít Krejčí": ["Vit Krejci"], "Dāvis Bertāns": ["Davis Bertans"], "Willy Hernangómez": ["Guillermo Hernangomez", "Willy Hernangomez"],
    "Juancho Hernangómez": ["Juan Hernangomez", "Juancho Hernangomez"], "Álex Abrines": ["Alejandro Abrines", "Alex Abrines"],
    "Dani Díez": ["Daniel Diez", "Dani Diez"], "Timothé Luwawu-Cabarrot": ["Timothe Luwawu", "Timothe Luwawu-Cabarrot"],
    "Sasha Vezenkov": ["Aleksandar Vezenkov"], "Georgios Papagiannis": ["Giorgos Papagiannis"], "Georgios Kalaitzakis": ["Giorgos Kalaitzakis"],
    "Georgios Printezis": ["Giorgos Printezis"], "Thanasis Antetokounmpo": ["Athanasios Antetokounmpo"], "Kyrylo Fesenko": ["Kirill Fesenko", "Kyryl Fesenko"],
    "Sergiy Gladyr": ["Sergii Gladyr", "Serhiy Hladyr"], "Vsevolod Ishchenko": ["Vsevolod Ishchenko"], "Ömer Aşık": ["Omer Asik"],
    "Semih Erden": ["Semih Erden"], "Tomáš Satoranský": ["Tomas Satoransky"], "Jonas Valančiūnas": ["Jonas Valanciunas"], "Danté Exum": ["Dante Exum"],
    "Isaïa Cordinier": ["Isaia Cordinier"], "Petr Cornelie": ["Petr Cornelie"], "Théo Maledon": ["Theo Maledon"], "Nolan Traoré": ["Nolan Traore"],
    "Bogoljub Markovic": ["Bogoljub Marković"], "Nikola Djurisic": ["Nikola Đurišić", "Nikola Durisic"], "Marko Simonovic": ["Marko Simonović"],
    "Hugo González": ["Hugo Gonzalez"], "Karim López": ["Karim Lopez"], "Sergio de Larrea": ["Sergio De Larrea"], "Pacôme Dadiet": ["Pacome Dadiet"],
    "Tidjane Salaün": ["Tidjane Salaun"], "Mohamed Diawara": ["Mohamed Diawara"], "Ismael Kamagate": ["Ismaël Kamagate"], "Ariel Hukporti": ["Ariel Hukporti"],
    "Nikola Topić": ["Nikola Topic"], "Alperen Şengün": ["Alperen Sengun"], "Tarik Biberovic": ["Tarik Biberović"], "Anžejs Pasečņiks": ["Anzejs Pasecniks"],
    "Rokas Jokubaitis": ["Rokas Jokubaitis"], "Chukwudiebere Maduabum": ["Chu Maduabum"], "Latavious Williams": ["Latavious Williams"],
    "Nikola Jović": ["Nikola Jovic"], "R.J. Hampton": ["RJ Hampton"], "Cedi Osman": ["Cedi Osman"], "Alex Sarr": ["Alexandre Sarr"],
    "Tristan Vukcevic": ["Tristan Vukcevic-Tsalikis"], "Hugo González": ["Hugo Gonzalez Pena"], "Serge Ibaka": ["Serge Ibaka Ngobila"],
    "Emmanuel Mudiay": ["Emmanuel Mudiay"], "Jalen Green": ["Jalen Green"], "Bilal Coulibaly": ["Bilal Coulibaly"],
}


def _draft_date(y):
    return pd.Timestamp(f"{y}-{DRAFT_DATES[y]}")


def _keys(name):
    """Name variants for matching: normalised full name plus the aliases."""
    return [norm_name(name)] + [norm_name(a) for a in ALIASES.get(name, [])]


def load_pro_seasons() -> pd.DataFrame:
    """Unified club-season table: bbref international + G League, timur box scores for competitions bbref lacks (ABA before 2021,
    Slovenian league) and PIR / fouls-drawn from timur for EuroLeague / EuroCup seasons that both cover."""
    bb = pd.read_parquet(RAW / "bbref_intl_totals.parquet")
    if (RAW / "gleague_seasons.parquet").exists():
        bb = pd.concat([bb, pd.read_parquet(RAW / "gleague_seasons.parquet")], ignore_index=True)
    tm = pd.read_parquet(RAW / "timur_seasons.parquet")
    for d in (bb, tm):
        d["key"] = d.player.map(norm_name)
    bb["source"] = "bbref"
    tm["source"] = "timur"
    # timur ABA/SLO fill the bbref gap; timur EuroLeague/EuroCup only used for PIR & fouls drawn
    have = set(zip(bb.league, bb.season))
    fill = tm[[(l, s) not in have for l, s in zip(tm.league, tm.season)]]
    extra = tm[[(l, s) in have for l, s in zip(tm.league, tm.season)]].groupby(["league", "season", "key"], as_index=False)[["pir", "pfd", "mp"]].sum()
    pro = pd.concat([bb, fill], ignore_index=True)
    num = ["g", "mp", "fg", "fga", "fg3", "fg3a", "ft", "fta", "orb", "trb", "ast", "stl", "blk", "tov", "pf", "pts", "pir", "pfd"]
    for c in num:
        if c not in pro:
            pro[c] = np.nan
        pro[c] = pd.to_numeric(pro[c], errors="coerce")
    # one row per league-season-player (mid-season transfers summed; team names joined)
    agg = {c: "sum" for c in num}
    agg.update({"team": lambda s: " / ".join(dict.fromkeys(s.dropna().astype(str))), "player": "first", "age": "max", "source": "first",
                "bbref_intl_id": lambda s: s.dropna().nunique()})
    pro["age"] = pd.to_numeric(pro.get("age"), errors="coerce")
    pro = pro.groupby(["league", "season", "key"], as_index=False).agg(agg).rename(columns={"bbref_intl_id": "n_ids"})
    pro.loc[pro.source == "bbref", ["pir", "pfd"]] = np.nan
    pro = pro.merge(extra, on=["league", "season", "key"], how="left", suffixes=("", "_t"))
    for c in ("pir", "pfd"):
        pro[c] = pro[c].where(pro.source == "timur", pro[f"{c}_t"])
    # only trust timur PIR when its minutes agree with bbref's (same player, complete box scores)
    pro.loc[(pro.source == "bbref") & ((pro.mp_t - pro.mp).abs() > 0.15 * pro.mp.clip(lower=1)), ["pir", "pfd"]] = np.nan
    pro = pro.drop(columns=["pir_t", "pfd_t", "mp_t"])
    pro["level"] = pro.league.map(LEAGUE_LEVEL).fillna(1)
    return pro


def _rates(d: pd.DataFrame, prefix: str, mp="mp", g="g") -> pd.DataFrame:
    """Per-game, per-36 and shooting rates from totals. Returns a frame of i_{prefix}* columns."""
    out = pd.DataFrame(index=d.index)
    G, M = d[g].replace(0, np.nan), d[mp].replace(0, np.nan)
    out[f"{prefix}gp"], out[f"{prefix}mpg"] = d[g], d[mp] / G
    for c in ("pts", "trb", "ast", "stl", "blk", "tov", "pir"):
        if c in d:
            out[f"{prefix}{c}_pg"] = d[c] / G
            out[f"{prefix}{c}_36"] = 36 * d[c] / M
    if "orb" in d:
        out[f"{prefix}orb_36"] = 36 * d.orb / M
    out[f"{prefix}fga_36"], out[f"{prefix}fg3a_36"], out[f"{prefix}fta_36"] = 36 * d.fga / M, 36 * d.fg3a / M, 36 * d.fta / M
    out[f"{prefix}fg_pct"] = d.fg / d.fga.replace(0, np.nan)
    out[f"{prefix}fg3_pct"] = d.fg3 / d.fg3a.replace(0, np.nan)
    out[f"{prefix}ft_pct"] = d.ft / d.fta.replace(0, np.nan)
    out[f"{prefix}fg3a_rate"] = d.fg3a / d.fga.replace(0, np.nan)
    out[f"{prefix}fta_rate"] = d.fta / d.fga.replace(0, np.nan)
    out[f"{prefix}ts"] = d.pts / (2 * (d.fga + 0.44 * d.fta)).replace(0, np.nan)
    out[f"{prefix}efg"] = (d.fg + 0.5 * d.fg3) / d.fga.replace(0, np.nan)
    out[f"{prefix}ast_tov"] = d.ast / d.tov.replace(0, np.nan)
    # NBA-style efficiency (no fouls drawn / blocks against), computable from every source unlike PIR
    eff = d.pts + d.trb + d.ast + d.stl + d.blk - (d.fga - d.fg) - (d.fta - d.ft) - d.tov
    out[f"{prefix}eff_pg"], out[f"{prefix}eff_36"] = eff / G, 36 * eff / M
    if "pfd" in d:
        out[f"{prefix}pfd_36"] = 36 * d.pfd / M
    return out


def _college_span(draft: pd.DataFrame) -> pd.DataFrame:
    """First/last Torvik season of each modelled draftee (his college years), to reject same-name Europeans playing while he was in college."""
    torvik = pd.read_parquet(PROC / "torvik.parquet")
    tbl = pd.read_parquet(PROC / "draft_table.parquet").reset_index(drop=True)
    pid = torvik.pid.reindex(tbl.torvik_idx.values).values
    span = torvik.groupby("pid").year.agg(["min", "max"])
    out = span.reindex(pid).reset_index(drop=True)
    out.index = draft.index
    return out.rename(columns={"min": "col_first", "max": "col_last"})


def match_pro(draft: pd.DataFrame, pro: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Last pre-draft club season per draftee. Returns (features frame indexed like draft, per-draftee candidate rows for QA)."""
    span = _college_span(draft)
    pro_keys = pro.key.unique()
    rows = []
    for i, r in draft.iterrows():
        window = (pro.season <= r.draft_year) & (pro.season >= r.draft_year - 2)
        keys = _keys(r.player)
        cand = pro[pro.key.isin(keys) & window]
        if cand.empty and len(keys[0]) >= 9:  # "Hugo Gonzalez" -> "Hugo Gonzalez Pena", "Tristan Vukcevic" -> "Tristan Vukcevic-Tsalikis"
            pref = [k for k in pro_keys if k.startswith(keys[0])]
            cand = pro[pro.key.isin(pref) & window]
        if r.player in PRO_OVERRIDES:
            cand = cand[cand.team.str.contains(PRO_OVERRIDES[r.player], case=False, na=False)]
        if pd.notna(span.loc[i, "col_last"]):  # college players: only a club season after leaving college counts (pre-college stints are
            cand = cand[cand.season > span.loc[i, "col_last"]]  # rare and the name-collision risk with journeymen abroad is high)
        if cand.empty:
            continue
        season = cand.season.max()
        c = cand[cand.season == season].copy()
        c["idx"] = i
        rows.append(c)
    cands = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    feats = []
    for i, c in cands.groupby("idx"):
        main = c.sort_values(["mp", "level"], ascending=False).iloc[0]
        top = c.sort_values(["level", "mp"], ascending=False).iloc[0]
        f = {"idx": i, "i_league": main.league, "i_team": main.team, "i_season": main.season, "i_league_level": main.level,
             "i_season_gap": draft.loc[i, "draft_year"] - main.season, "i_n_comps": len(c), "i_top_level": top.level,
             "i_all_gp": c.g.sum(), "i_all_mpg": c.mp.sum() / max(c.g.sum(), 1), "i_name_collision": int(c.n_ids.max() > 1),
             "i_bbref_age": main.age}
        f.update(_rates(pd.DataFrame([main]), "i_").iloc[0].to_dict())
        if top.league != main.league:
            f.update({"i_top_gp": top.g, "i_top_mpg": top.mp / max(top.g, 1), "i_top_pts_36": 36 * top.pts / max(top.mp, 1),
                      "i_top_pir_36": 36 * top.pir / max(top.mp, 1) if pd.notna(top.pir) else np.nan})
        else:
            f.update({"i_top_gp": main.g, "i_top_mpg": main.mp / max(main.g, 1), "i_top_pts_36": 36 * main.pts / max(main.mp, 1),
                      "i_top_pir_36": 36 * main.pir / max(main.mp, 1) if pd.notna(main.pir) else np.nan})
        feats.append(f)
    feats = pd.DataFrame(feats).set_index("idx") if feats else pd.DataFrame()
    return feats, cands


# Same-name collisions among draftees, resolved by team substring of the true pre-draft club.
PRO_OVERRIDES = {"Marko Simonovic": "Mega", "Nikola Jović": "Mega", "Marcus Eriksson": "Barcelona|Manresa", "Dario Šarić": "Cibona|Zagreb"}


def build():
    draft = pd.read_parquet(PROC / "draft_table.parquet")[["player", "draft_year", "bbref_id", "pick", "college", "torvik_idx"]].reset_index(drop=True)
    draft["key"] = draft.player.map(norm_name)
    draft["modelled"] = draft.torvik_idx.notna()  # has a Torvik college season (the table's own `modelled` already folds this file in)
    bd = pd.read_parquet(RAW / "bbref_birthdates.parquet")
    draft = draft.merge(bd, on="bbref_id", how="left")

    pro = load_pro_seasons()
    pro_feats, cands = match_pro(draft, pro)
    out = draft[["key", "draft_year", "player", "bbref_id", "modelled"]].join(pro_feats)
    print(f"pro season matched: {out.i_league.notna().sum()} of {len(out)} draftees; unmodelled {out[~out.modelled].i_league.notna().sum()} / {(~out.modelled).sum()}")

    fy = pd.read_parquet(RAW / "fiba_youth.parquet") if (RAW / "fiba_youth.parquet").exists() else None
    if fy is not None:
        fiba_feats, dob_fiba = match_fiba(draft, fy)
        out = out.join(fiba_feats)
        draft["birthdate"] = draft.birthdate.fillna(dob_fiba)
        print(f"fiba youth matched: {out.i_fiba_youth_events.notna().sum()} of {len(out)} draftees")
    if (RAW / "ngt_seasons.parquet").exists():
        ngt_feats, dob_ngt = match_ngt(draft, pd.read_parquet(RAW / "ngt_seasons.parquet"))
        out = out.join(ngt_feats)
        draft["birthdate"] = draft.birthdate.fillna(dob_ngt)
        print(f"ngt matched: {out.i_ngt.notna().sum()} of {len(out)} draftees")

    if (RAW / "eurocamp_rosters.parquet").exists():
        ec = pd.read_parquet(RAW / "eurocamp_rosters.parquet")
        ec["key"] = ec.player.map(norm_name)
        camps = ec.groupby("key").year.apply(set).to_dict()
        years = set(ec.year)
        flag, last = [], []
        for r in draft.itertuples():
            window = set(range(r.draft_year - 3, r.draft_year + 1))  # camps he could have attended: the June before the draft and three years back
            att = set().union(*[camps.get(k, set()) for k in _keys(r.player)]) & window
            if att:
                flag.append(1), last.append(max(att))
            elif window & years:
                flag.append(0), last.append(np.nan)
            else:  # no archived camp in his window: unknown
                flag.append(np.nan), last.append(np.nan)
        out["i_eurocamp"], out["i_eurocamp_year"] = flag, last
        print(f"eurocamp matched: {int(np.nansum(flag))} draftees ({sum(f == 0 for f in flag)} known non-attendees)")

    # ages from the best birth date we have (bbref NBA index, else FIBA roster)
    draft["birthdate"] = pd.to_datetime(draft.birthdate)
    out["i_age_at_draft"] = [(_draft_date(y) - b).days / 365.25 if pd.notna(b) else np.nan for y, b in zip(draft.draft_year, draft.birthdate)]
    out["i_age_season"] = [((pd.Timestamp(f"{int(s)}-02-01") - b).days / 365.25 if pd.notna(b) and pd.notna(s) else np.nan) for s, b in zip(out.i_season, draft.birthdate)]
    out["i_age_season"] = out.i_age_season.fillna(out.pop("i_bbref_age") if "i_bbref_age" in out else np.nan)
    out["i_has_pro"] = out.i_league.notna().astype(int)
    out = out.drop(columns=["player", "bbref_id", "modelled"]).drop_duplicates(["key", "draft_year"])
    EXT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(EXT / "intl_prospects.parquet", index=False)
    cands.to_parquet(RAW / "qa_pro_candidates.parquet", index=False)
    print(f"wrote {EXT / 'intl_prospects.parquet'}: {out.shape}")
    return out


def load_intl(numeric_only: bool = False) -> pd.DataFrame:
    """The built features keyed by (key, draft_year); `numeric_only` drops the string columns i_league / i_team."""
    df = pd.read_parquet(EXT / "intl_prospects.parquet")
    return df.drop(columns=[c for c in ("i_league", "i_team") if c in df.columns]) if numeric_only else df


# FIBA youth competition level: world > European Div A > other continental finals > Div B/C, sub-zonal qualifiers.
def fiba_level(name: str, zone: str) -> int:
    n = name.lower()
    if zone == "World" or "world" in n:
        return 4
    if "division b" in n or "division c" in n or "qualif" in n or "centrobasket" in n or "south american" in n or "cocaba" in n or "zone" in n \
            or "promotion" in n or "challenge" in n or "seaba" in n or "waba" in n or "cbc " in n or "preliminar" in n:
        return 1
    if zone == "Europe":
        return 3
    return 2


def match_fiba(draft: pd.DataFrame, fy: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Match draftees to FIBA persons by name (both orders, aliases), disambiguated by birth date when we know it; else by
    plausible age at the event. Aggregates every youth event that ended before the player's draft night."""
    fy = fy.copy()
    fy["name"] = fy.first_name.fillna("") + " " + fy.last_name.fillna("")
    fy["key"] = fy.name.map(norm_name)
    fy["key_rev"] = (fy.last_name.fillna("") + " " + fy.first_name.fillna("")).map(norm_name)
    fy["lastkey"] = fy.last_name.fillna("").map(norm_name)
    fy["dob"] = pd.to_datetime(fy.dob.astype(str).str.slice(0, 10), errors="coerce")
    fy["end"] = pd.to_datetime(fy.comp_end.astype(str).str.slice(0, 19), errors="coerce")  # 7-digit fractional seconds break the parser
    fy["level"] = [fiba_level(n, z) for n, z in zip(fy.competition.fillna(""), fy.zone.fillna(""))]
    fy["u"] = pd.to_numeric(fy.age_cat.str.extract(r"(\d+)")[0], errors="coerce")
    # rosters include preliminary long lists; a player took part if he has a box score, or (competitions without any
    # statistics) if he is flagged on the final roster
    comp_has_stats = fy.groupby("competition_id").gp.transform(lambda s: (s.fillna(0) > 0).any())
    fy["took_part"] = (fy.gp.fillna(0) > 0) | (~comp_has_stats & fy.on_final.fillna(False).astype(bool))
    persons = fy.groupby("player_id").agg(k1=("key", lambda s: set(s)), k2=("key_rev", lambda s: set(s)), kl=("lastkey", lambda s: set(s)),
                                          dob=("dob", "first"), first_season=("season", "min"))
    by_key: dict[str, set] = {}
    by_last: dict[str, set] = {}
    for pid, r in persons.iterrows():
        for k in r.k1 | r.k2:
            by_key.setdefault(k, set()).add(pid)
        for k in r.kl:
            by_last.setdefault(k, set()).add(pid)

    feats, dobs = [], {}
    for i, r in draft.iterrows():
        dd = _draft_date(r.draft_year)
        pids = set().union(*[by_key.get(k, set()) for k in _keys(r.player)])
        bd = r.birthdate if pd.notna(r.birthdate) else None
        if bd is not None:
            pids = {p for p in pids if pd.isna(persons.loc[p, "dob"]) or abs((persons.loc[p, "dob"] - bd).days) <= 366}
            if not pids:  # first-name mismatch (nickname): last name + exact birth date
                last = norm_name(r.player.split()[-1])
                pids = {p for p in by_last.get(last, set()) if pd.notna(persons.loc[p, "dob"]) and persons.loc[p, "dob"] == bd}
        else:  # unknown birth date: youth events must sit 1-8 years before the draft
            pids = {p for p in pids if 0 <= r.draft_year - persons.loc[p, "first_season"] <= 8}
        if not pids:
            continue
        allrows = fy[fy.player_id.isin(pids) & (fy.end < dd) & (fy.season <= r.draft_year)]
        if bd is None and allrows.dob.notna().any():
            dobs[i] = allrows.dob.dropna().iloc[0]
        ev = allrows[allrows.took_part]
        if ev.empty:
            continue
        b = bd if bd is not None else dobs.get(i)
        f = {"idx": i, "i_fiba_youth_events": ev.competition_id.nunique(), "i_fiba_ids": len(pids),
             "i_fiba_max_level": ev.level.max(), "i_fiba_last_year": ev.season.max(), "i_fiba_first_year": ev.season.min(),
             "i_fiba_u16": int(ev.u.isin([15, 16]).any() or ((ev.u == 17) & (ev.level < 4)).any()), "i_fiba_u17wc": int(((ev.u == 17) & (ev.level == 4)).any()),
             "i_fiba_u18": int((ev.u == 18).any()), "i_fiba_u19wc": int(((ev.u == 19) & (ev.level == 4)).any()), "i_fiba_u20": int((ev.u == 20).any()),
             "i_fiba_div_b": int(ev.competition.str.contains("DIVISION B|Division B", na=False).any()),
             "i_fiba_best_rank": ev.final_rank.min(), "i_fiba_height_cm": ev.sort_values("season").height_cm.dropna().iloc[-1] if ev.height_cm.notna().any() else np.nan,
             "i_fiba_last_age": (ev.end.max() - b).days / 365.25 if b is not None else np.nan}
        played = ev[ev.gp.fillna(0) > 0]
        if not played.empty:
            tot = played[["gp", "min", "pts", "reb", "oreb", "ast", "stl", "blk", "tov", "eff", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta"]].sum()
            d = pd.DataFrame([{"g": tot.gp, "mp": tot["min"], "pts": tot.pts, "trb": tot.reb, "orb": tot.oreb, "ast": tot.ast, "stl": tot.stl, "blk": tot.blk,
                               "tov": tot.tov, "pir": tot.eff, "fg": tot.fgm, "fga": tot.fga, "fg3": tot.fg3m, "fg3a": tot.fg3a, "ft": tot.ftm, "fta": tot.fta}])
            f.update(_rates(d, "i_fiba_").iloc[0].to_dict())
            pg = played.assign(ppg=played.pts / played.gp, epg=played.eff / played.gp)
            f["i_fiba_best_ppg"], f["i_fiba_best_effpg"] = pg.ppg.max(), pg.epg.max()
            last = pg.sort_values(["season", "level"]).iloc[-1]
            f.update({"i_fiba_last_level": last.level, "i_fiba_last_u": last.u, "i_fiba_last_gp": last.gp, "i_fiba_last_mpg": last["min"] / last.gp,
                      "i_fiba_last_ppg": last.ppg, "i_fiba_last_effpg": last.epg, "i_fiba_last_rpg": last.reb / last.gp, "i_fiba_last_apg": last.ast / last.gp,
                      "i_fiba_last_pts_36": 36 * last.pts / last["min"] if last["min"] else np.nan, "i_fiba_last_eff_36": 36 * last.eff / last["min"] if last["min"] else np.nan})
        feats.append(f)
    feats = pd.DataFrame(feats).set_index("idx") if feats else pd.DataFrame()
    return feats, pd.Series(dobs)


def match_ngt(draft: pd.DataFrame, ngt: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """EuroLeague U18 Next Generation Tournament appearances before the draft (season-year y counts if y + 1 <= draft_year)."""
    ngt = ngt.copy()
    ngt["key"] = ngt.player.map(norm_name)
    ngt["dob"] = pd.to_datetime(ngt.dob.astype(str).str.slice(0, 10), errors="coerce")
    persons = ngt.groupby("code").agg(key=("key", "first"), dob=("dob", "first"), first_year=("year", "min"))
    by_key: dict[str, set] = {}
    for code, r in persons.iterrows():
        by_key.setdefault(r.key, set()).add(code)
    feats, dobs = [], {}
    for i, r in draft.iterrows():
        codes = set().union(*[by_key.get(k, set()) for k in _keys(r.player)])
        bd = r.birthdate if pd.notna(r.birthdate) else None
        if bd is not None:
            codes = {c for c in codes if pd.isna(persons.loc[c, "dob"]) or abs((persons.loc[c, "dob"] - bd).days) <= 366}
        else:
            codes = {c for c in codes if 1 <= r.draft_year - persons.loc[c, "first_year"] <= 7}
        ev = ngt[ngt.code.isin(codes) & (ngt.year + 1 <= r.draft_year)]
        if ev.empty:
            continue
        if bd is None and ev.dob.notna().any():
            dobs[i] = ev.dob.dropna().iloc[0]
        b = bd if bd is not None else dobs.get(i)
        last = ev.year.max()
        f = {"idx": i, "i_ngt": 1, "i_ngt_years": ev.year.nunique(), "i_ngt_last_year": last + 1,
             "i_ngt_last_age": (pd.Timestamp(f"{last + 1}-05-15") - b).days / 365.25 if b is not None else np.nan,
             "i_ngt_height_cm": ev.height_cm.replace(0, np.nan).dropna().iloc[-1] if ev.height_cm.replace(0, np.nan).notna().any() else np.nan}
        tot = ev[["g", "mp", "pts", "trb", "orb", "ast", "stl", "blk", "tov", "fg", "fga", "fg3", "fg3a", "ft", "fta", "pir", "pfd"]].sum().to_frame().T
        f.update(_rates(tot, "i_ngt_").iloc[0].to_dict())
        feats.append(f)
    feats = pd.DataFrame(feats).set_index("idx") if feats else pd.DataFrame()
    return feats, pd.Series(dobs)


if __name__ == "__main__":
    stages = sys.argv[1:] or ["bbref", "gleague", "timur", "ngt", "eurocamp", "fiba", "build"]
    if "bbref" in stages:
        bbref_download()
        parse_bbref_totals().to_parquet(RAW / "bbref_intl_totals.parquet", index=False)
        parse_bbref_birthdates().to_parquet(RAW / "bbref_birthdates.parquet", index=False)
    if "gleague" in stages:
        gleague_download()
        parse_gleague().to_parquet(RAW / "gleague_seasons.parquet", index=False)
    if "timur" in stages:
        timur_download()
        parse_timur().to_parquet(RAW / "timur_seasons.parquet", index=False)
    if "ngt" in stages:
        ngt_download()
        parse_ngt().to_parquet(RAW / "ngt_seasons.parquet", index=False)
    if "eurocamp" in stages:
        eurocamp_download()
        parse_eurocamp().to_parquet(RAW / "eurocamp_rosters.parquet", index=False)
    if "fiba" in stages:
        comps = fiba_competitions()
        comps.to_parquet(RAW / "fiba_competitions.parquet", index=False)
        youth = comps[(comps.gender == "Men") & (comps.type == "National Teams") & comps.age.fillna("").str.startswith("Under")]
        print(f"fiba: {len(comps)} competitions, {len(youth)} men's youth national-team competitions", flush=True)
        fiba_download_youth(youth)
        parse_fiba(youth).to_parquet(RAW / "fiba_youth.parquet", index=False)
    if "build" in stages:
        build()
