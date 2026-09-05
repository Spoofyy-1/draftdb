"""Pre-draft mock-draft momentum (columns mo_*) for the 2008-2026 classes, keyed by (key, draft_year): how each prospect's
consensus mock rank moved over the ~90 days before the draft.

Every board is a Wayback Machine capture taken BEFORE draft night (the cutoff of infra.builders.mocks: 22:00 UTC on the
first night, before the first pick), so nothing here can be derived from the actual selections. Per (source, year) the CDX
API lists the captures of the source's candidate URLs (mocks.CANDIDATES) in the LOOKBACK_DAYS before the cutoff; for each
horizon (90, 60, 30, 7 days before the draft) the closest capture AT OR BEFORE the target date is used, provided it is
within TOLERANCE_DAYS of the target and parses (mocks.PARSERS) as a board of that class under mocks.py's acceptance rules
(>= 25 picks, <= 2 gaps, half of the top 30 are draftees of the class); the final board is the latest pre-draft capture.
Captures are cached as data/external/momentum/<source>/<year>/<timestamp>.html with an index.json per (source, year)
(horizon -> capture and parsed players; misses are recorded too, so re-runs are offline); data/external/momentum/provenance.csv
lists every board used.

Ranks: a draftee absent from a board sits just outside it: 61 on a two-round mock, 31 on a first-round mock (as in mocks.py),
N+1 on an N-name big board. Per horizon the consensus is the mean over the sources that have a board at that horizon, the
disagreement the sample std across them. Boards list many players who later withdraw, so a name matched only by surname
must also agree on the first name (see _match_strict).

  mo_rank_{90d,60d,30d,7d,final}  consensus rank at each horizon (NaN: no source has a board there)
  mo_change_{90d,60d,30d,7d}      rank at the horizon minus the final rank (positive = rose)
  mo_late_momentum                rank_30d - rank_final (identical to mo_change_30d, kept under the requested name)
  mo_velocity                     least-squares slope of consensus rank vs horizon (90/60/30/7/0 days before the draft) over
                                  the available horizons, in ranks per 30 days (positive = rising towards the draft)
  mo_acceleration                 late slope (30d, 7d, final) minus early slope (90d, 60d, 30d), same units
  mo_rank_std_final               cross-source std of the final rank; mo_rank_std_change = final std minus 60d std
  mo_n_snapshots                  distinct boards (source x capture) that list the player
  mo_n_sources_final              sources whose final board lists the player
  mo_first_seen_days              days before the draft of the earliest board that lists him (NaN: never listed)

    from infra.builders.momentum import load_momentum   # cache only, no network
    python -m infra.builders.momentum                    # fetch what is missing, rebuild, sanity report
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from infra import config as C
from infra.builders.mocks import (ALIASES, BOARD_LEN, CANDIDATES, CUTOFF_HHMM, DRAFT_NIGHT, MIN_PARSED, MISSING_RANK,
                                  MISSING_RANK_R1, PARSERS, WAYBACK, _S, _alt, _contiguous, _download, _draftees, _get,
                                  _match_names, _same_class, _tokens)
from infra.dataset import norm_name

MO_DIR = C.ROOT / "data" / "external" / "momentum"
KEYS = ["key", "draft_year"]
YEARS = range(2008, 2027)
SOURCES = list(CANDIDATES)
HORIZONS = {"90d": 90, "60d": 60, "30d": 30, "7d": 7, "final": 0}  # target: this many days before draft night
TOLERANCE_DAYS = 20  # a horizon is skipped when the nearest capture at or before its target is older than this
LOOKBACK_DAYS = max(HORIZONS.values()) + TOLERANCE_DAYS
MAX_TRIES = 4  # captures tried per horizon (closest first) before the horizon is given up
WORKERS = 4  # concurrent (source, year) fetches against the Wayback Machine
CDX_TRIES = 6  # CDX attempts per URL (15 s, 30 s, ... between them) before the (source, year) is abandoned for this run
_CDX_SLOTS = threading.Semaphore(2)  # the CDX API is the fragile part: at most two listings in flight
EARLY, LATE = ["90d", "60d", "30d"], ["30d", "7d", "final"]  # the two halves of the window for mo_acceleration
FEATURES = ([f"mo_rank_{h}" for h in HORIZONS] + [f"mo_change_{h}" for h in HORIZONS if h != "final"]
            + ["mo_late_momentum", "mo_velocity", "mo_acceleration", "mo_rank_std_final", "mo_rank_std_change",
               "mo_n_snapshots", "mo_n_sources_final", "mo_first_seen_days"])
REJECT_LOG = []  # (mock name, key) surname matches refused because the first names disagree, printed by the report
NICKNAMES = {  # mock spelling (norm_name) -> key where the first name is a nickname / typo the first-name rule would refuse
    "bjmullens": "byronmullens", "billwalker": "henrywalker", "edriceadebayo": "bamadebayo", "moeharkless": "mauriceharkless",
    "nahshonhyland": "boneshyland", "nahshonboneshyland": "boneshyland", "roydevynmarble": "devynmarble",
    "waltertavares": "edytavares", "keziekzokpala": "kzokpala", "kezieokpala": "kzokpala", "mstissethybulle": "matissethybulle",
    "deronholmes": "daronholmes", "lrmbahamoute": "lucmbahamoute", "cdouglasroberts": "chrisdouglasroberts",
    "carltoncarrington": "bubcarrington", "aleksandarvezenkov": "sashavezenkov",
}


# --------------------------------------------------------------------------- wayback

def _cutoff(year: int) -> str:
    return DRAFT_NIGHT[year] + CUTOFF_HHMM


def _days_before(ts: str, year: int) -> int:
    """Calendar days from the capture date to draft night."""
    return (pd.Timestamp(DRAFT_NIGHT[year]) - pd.Timestamp(ts[:8])).days


def _captures(url: str, year: int) -> list[tuple[str, str]]:
    """(timestamp, original) of the 200-OK captures of `url` in the LOOKBACK_DAYS before the draft-night cutoff. Raises
    when the CDX API is down (it answers 503 'Temporarily Offline' under load) so the failure is never cached as a miss."""
    frm = (pd.Timestamp(DRAFT_NIGHT[year]) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    p = {"url": url, "output": "json", "filter": "statuscode:200", "from": frm, "to": _cutoff(year), "limit": 5000}
    for i in range(CDX_TRIES):
        with _CDX_SLOTS:
            r = _get(f"{WAYBACK}/cdx/search/cdx", p, tries=1)
        if r is not None and r.text.strip().startswith("["):
            return sorted((row[1], row[2]) for row in json.loads(r.text)[1:] if row[1] <= _cutoff(year))
        if r is not None and not r.text.strip():  # 200 with an empty body: no captures
            return []
        time.sleep(15 * (i + 1))
    raise RuntimeError(f"CDX unavailable for {url}")


def _candidates(caps: list[tuple[str, str]], year: int, days: int) -> list[tuple[str, str]]:
    """Captures at or before the horizon's target date and within TOLERANCE_DAYS of it, closest first. The final board is
    simply the latest pre-draft capture (as in mocks.py), however old."""
    hi = days + TOLERANCE_DAYS if days else LOOKBACK_DAYS
    out = [(ts, u) for ts, u in caps if days <= _days_before(ts, year) <= hi]
    return sorted(out, reverse=True)[:MAX_TRIES]


def _try(source: str, d: Path, ts: str, original: str, pool: pd.DataFrame) -> tuple[str, list] | None:
    """(text, players) when the capture parses as a board of the class, None when it does not. A capture that cannot be
    downloaded raises when the Wayback Machine is throttling or offline (429 / 503): a network failure must not be cached
    as 'no board'. A capture Wayback itself answers with another error for (a 500 on the 2018-06-18 DraftExpress page) is
    broken for good and counts as unusable."""
    text = (d / f"{ts}.html").read_text() if (d / f"{ts}.html").exists() else _download(ts, original)
    if text is None:
        status = _S.get(f"{WAYBACK}/web/{ts}id_/{original}", timeout=60, allow_redirects=False).status_code
        if status in (429, 503) or status < 400:
            raise RuntimeError(f"download failed {ts} {original} ({status})")
        return None
    try:
        players = PARSERS[source](text)
    except Exception:  # empty or truncated capture
        return None
    if len(players) >= MIN_PARSED and _contiguous(players) and _same_class(players, pool):
        return text, players
    return None


# --------------------------------------------------------------------------- fetch + cache

def _dir(source: str, year: int) -> Path:
    return MO_DIR / source / str(year)


def _reparse(source: str, d: Path, meta: dict) -> dict:
    """Re-parse the cached captures so parser fixes in mocks.py apply without a download."""
    for m in meta["horizons"].values():
        if m and (d / f"{m['ts']}.html").exists():
            m["players"] = PARSERS[source]((d / f"{m['ts']}.html").read_text())
            m["n_players"] = len(m["players"])
    return meta


def read_series(source: str, year: int) -> dict | None:
    """Cached {'source','year','n_captures','horizons': {horizon: {'ts','url','n_players','players'} | None}}, or None."""
    idx = _dir(source, year) / "index.json"
    return json.loads(idx.read_text()) if idx.exists() else None


def fetch_series(source: str, year: int, pool: pd.DataFrame, verbose: bool = True) -> dict | None:
    """The source's boards at every horizon for the class (see read_series), downloading what the cache lacks. A cached
    (source, year) never touches the network again; its captures are re-parsed."""
    d = _dir(source, year)
    if (d / "index.json").exists():
        return _reparse(source, d, read_series(source, year))
    urls = CANDIDATES[source](year)
    if not urls:  # the source had no board that year
        return None
    caps = sorted({c for u in urls for c in _captures(u, year)})  # raises when the CDX API is down: nothing is cached
    tried, horizons, failed = {}, {}, 0
    for h, days in HORIZONS.items():
        horizons[h] = None
        for ts, original in _candidates(caps, year, days):
            if ts not in tried:
                try:
                    tried[ts] = _try(source, d, ts, original, pool)
                except RuntimeError:  # the capture is skipped this run and the (source, year) is not cached, so it is retried
                    tried[ts], failed = None, failed + 1
            if tried[ts] is not None:
                text, players = tried[ts]
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{ts}.html").write_text(text)
                horizons[h] = {"ts": ts, "url": original, "n_players": len(players), "players": players}
                break
    meta = {"source": source, "year": year, "n_captures": len(caps), "horizons": horizons}
    d.mkdir(parents=True, exist_ok=True)
    if not failed:
        (d / "index.json").write_text(json.dumps(meta))
    if verbose:
        got = " ".join(f"{h}@{m['ts'][:8]}({m['n_players']})" for h, m in horizons.items() if m)
        note = f"  ({failed} downloads failed, not cached)" if failed else ""
        print(f"{source:12s} {year} {len(caps):3d} captures  {got or 'no board'}{note}\n", end="", flush=True)  # one write: threads
    return meta


# --------------------------------------------------------------------------- name matching

def _match_strict(names: list[str], pool: pd.DataFrame) -> list[tuple[str | None, int]]:
    """mocks._match_names with one extra rule: a match found only through the surname (or fuzzily) must agree on the first
    two letters of the first name. Pre-draft boards list many players who never enter the draft, and 'Justin Robinson' must
    not become the class's Jerome Robinson. Returns (key, tier): tier 0 = exact / transliterated / alias, 1 = surname."""
    alt = {_alt(p): k for k, p in zip(pool.key, pool.player)}
    first = {k: (_tokens(p)[0].lower() if _tokens(p) else "") for k, p in zip(pool.key, pool.player)}
    out = []
    for n, k in zip(names, _match_names(names, pool)):
        if NICKNAMES.get(norm_name(n)) in first:
            out.append((NICKNAMES[norm_name(n)], 0))
        elif k is None:
            out.append((None, 0))
        elif k in (norm_name(n), ALIASES.get(norm_name(n)), alt.get(_alt(n))):
            out.append((k, 0))
        elif _tokens(n) and _tokens(n)[0].lower()[:2] == first[k][:2]:
            out.append((k, 1))
        else:
            REJECT_LOG.append((n, k))
            out.append((None, 1))
    return out


def _board_frame(meta: dict, pool: pd.DataFrame) -> pd.DataFrame:
    """key, rank, listed for every draftee of the class: his rank on this board, or 'just outside it' when absent -- 61 on a
    two-round mock, 31 on a first-round mock (mocks.py's MISSING_RANK / MISSING_RANK_R1), N+1 on an N-name big board (the
    Ringer's grows from 30 to 60+ names over the spring). When two listed names resolve to one draftee the exact match wins
    over the surname match."""
    players = sorted(meta["players"])[:BOARD_LEN]
    matched = _match_strict([n for _, n in players], pool)
    df = pd.DataFrame({"rank": [r for r, _ in players], "key": [k for k, _ in matched], "tier": [t for _, t in matched]})
    df = df.dropna(subset=["key"]).sort_values(["tier", "rank"]).drop_duplicates("key")
    fill = MISSING_RANK if meta["n_players"] >= BOARD_LEN - 3 else MISSING_RANK_R1 if meta["n_players"] <= 30 else len(players) + 1
    out = pool[["key"]].merge(df[["key", "rank"]], on="key", how="left")
    out["listed"] = out["rank"].notna()
    out["rank"] = out["rank"].fillna(fill).astype(float)
    return out


# --------------------------------------------------------------------------- features

def _slope(x: pd.DataFrame, y: pd.DataFrame) -> pd.Series:
    """Row-wise least-squares slope of y on x over the columns where both are present (NaN with < 2 distinct x)."""
    m = x.notna() & y.notna()
    x, y = x.where(m), y.where(m)
    dx, dy = x.sub(x.mean(axis=1), axis=0), y.sub(y.mean(axis=1), axis=0)
    sxx = (dx ** 2).sum(axis=1)
    return ((dx * dy).sum(axis=1) / sxx).where((m.sum(axis=1) >= 2) & (sxx > 0))


def _year_frame(year: int, metas: dict[str, dict], pool: pd.DataFrame) -> tuple[pd.DataFrame | None, list[dict]]:
    rows, prov = [], []
    for source, meta in metas.items():
        for h, m in meta["horizons"].items():
            if not m:
                continue
            days = _days_before(m["ts"], year)
            assert m["ts"] <= _cutoff(year) and 0 <= days <= LOOKBACK_DAYS, (source, year, m["ts"])  # pre-draft guard
            bf = _board_frame(m, pool)
            rows.append(bf.assign(source=source, horizon=h, ts=m["ts"], days_before=days))
            prov.append({"year": year, "source": source, "horizon": h,
                         "target_date": (pd.Timestamp(DRAFT_NIGHT[year]) - pd.Timedelta(days=HORIZONS[h])).date(),
                         "snapshot": m["ts"], "snapshot_date": pd.Timestamp(m["ts"][:8]).date(), "days_before_draft": days,
                         "days_off_target": days - HORIZONS[h], "url": m["url"], "n_players_parsed": m["n_players"],
                         "n_draftees_matched": int(bf.listed.sum())})
    if not rows:
        return None, prov
    long = pd.concat(rows, ignore_index=True)
    g = long.groupby(["key", "horizon"])
    rank = g["rank"].mean().unstack("horizon").reindex(columns=list(HORIZONS))
    std = g["rank"].std(ddof=1).unstack("horizon").reindex(columns=list(HORIZONS))
    n_listed = g["listed"].sum().unstack("horizon").reindex(columns=list(HORIZONS))
    # slopes use the nominal horizon distances: with the actual capture dates a 7d and a final board three days apart
    # turn a one-place move into a slope of 10 ranks / 30 days
    days = pd.DataFrame({h: float(v) for h, v in HORIZONS.items()}, index=rank.index)
    out = pd.DataFrame(index=rank.index)
    for h in HORIZONS:
        out[f"mo_rank_{h}"] = rank[h]
    for h in HORIZONS:
        if h != "final":
            out[f"mo_change_{h}"] = rank[h] - rank["final"]
    out["mo_late_momentum"] = rank["30d"] - rank["final"]
    out["mo_velocity"] = 30 * _slope(days, rank)
    out["mo_acceleration"] = 30 * (_slope(days[LATE], rank[LATE]) - _slope(days[EARLY], rank[EARLY]))
    out["mo_rank_std_final"] = std["final"]
    out["mo_rank_std_change"] = std["final"] - std["60d"]
    listed = long[long.listed]
    out["mo_n_snapshots"] = listed.drop_duplicates(["key", "source", "ts"]).groupby("key").size().reindex(out.index).fillna(0).astype(int)
    out["mo_n_sources_final"] = n_listed["final"]
    out["mo_first_seen_days"] = listed.groupby("key").days_before.max().reindex(out.index)
    return out.reset_index().assign(draft_year=year), prov


def build(years=None, fetch: bool = True, verbose: bool = True) -> pd.DataFrame:
    years = list(years or YEARS)
    d = _draftees()
    pools = {y: d[d.draft_year == y] for y in years}
    jobs = [(s, y) for y in years for s in SOURCES if CANDIDATES[s](y)]
    metas = {}
    if fetch:
        with ThreadPoolExecutor(WORKERS) as ex:
            futs = {ex.submit(fetch_series, s, y, pools[y], verbose): (s, y) for s, y in jobs}
            for f in as_completed(futs):
                try:
                    metas[futs[f]] = f.result()
                except Exception as e:  # one broken (source, year) must not lose the others' work
                    print(f"{futs[f]} failed: {type(e).__name__}: {str(e)[:100]}", flush=True)
    else:
        metas = {(s, y): read_series(s, y) for s, y in jobs}
    frames, prov = [], []
    for y in years:
        frame, p = _year_frame(y, {s: metas[(s, y)] for s in SOURCES if metas.get((s, y))}, pools[y])
        prov += p
        if frame is not None:
            frames.append(frame)
    out = pd.concat(frames, ignore_index=True)[KEYS + FEATURES]
    if fetch:
        MO_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(prov).to_csv(MO_DIR / "provenance.csv", index=False)
    return out


def load_momentum() -> pd.DataFrame:
    return build(fetch=False, verbose=False)


# --------------------------------------------------------------------------- report

def report(out: pd.DataFrame) -> None:
    d = pd.read_parquet(C.PROC / "draft_table.parquet", columns=KEYS + ["pick", "player"]).drop_duplicates(KEYS)
    m = d[d.draft_year.isin(YEARS)].merge(out, on=KEYS, how="left")
    prov = pd.read_csv(MO_DIR / "provenance.csv")
    print("\nyear  draftees  listed>=2h  horizons  boards  rho(late_mom, pick-final)  boards (source: horizon@date ...)")
    for y, g in m.groupby("draft_year"):
        p = prov[prov.year == y]
        listed_h = sum((g[f"mo_rank_{h}"] < MISSING_RANK).astype(int) for h in HORIZONS)
        horizons = [h for h in HORIZONS if g[f"mo_rank_{h}"].notna().any()]
        ok = g.mo_late_momentum.notna() & g.pick.notna()
        rho = g.mo_late_momentum[ok].corr(g.pick[ok] - g.mo_rank_final[ok], method="spearman") if ok.sum() > 5 else np.nan
        boards = "; ".join(f"{s}: " + " ".join(f"{r.horizon}@{r.snapshot_date}" for r in ps.itertuples()) for s, ps in p.groupby("source"))
        print(f"{y}  {len(g):8d}  {int((listed_h >= 2).sum()):10d}  {len(horizons):8d}  {len(p):6d}  {rho:25.3f}  {boards}")
    print(f"\n{len(prov)} boards; {out.shape[0]} rows; velocity available for {m.mo_velocity.notna().mean():.1%} of draftees")
    print("sign: rho > 0 means late risers were drafted LATER than their final mock rank (mocks over-corrected), rho < 0 that the rise continued on draft night")
    for name, y in [("Michael Porter", 2018), ("Shaedon Sharpe", 2022), ("Donovan Mitchell", 2017)]:
        r = m[(m.draft_year == y) & m.player.str.contains(name, regex=False)]
        if len(r):
            r = r.iloc[0]
            print(f"\n{r.player} {y} pick {int(r.pick)}: " + ", ".join(f"{f[3:]}={r[f]:.2f}" for f in FEATURES if pd.notna(r[f])))
    print("\nsurname matches refused (mock name -> key):", sorted(set(REJECT_LOG)))


if __name__ == "__main__":
    t0 = time.time()
    res = build()
    print(f"built {res.shape} in {time.time() - t0:.0f}s")
    report(res)
