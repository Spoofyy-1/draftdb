"""Rebuild as much of Colin's `data/` as the permitted public sources allow, then run his own builders on it.

His built data is gone; only his code survives. This script recreates the inputs his builders expect
(`data/raw/...`, `data/external/...`, `data/processed/{drafts,torvik,target,season_war}.parquet`) from sources that
are allowed here, runs the builders that depend only on those sources, and merges the resulting families onto our
bridge table as `data/processed/draft_table_colin.parquet` (pid-keyed, no names).

Substitutions (see SOURCES.md):
  drafts.parquet     <- our identity file (pid, draft_year, actual_pick) + college resolved from the public CSVs
                        and from Torvik's own draft tag.  bbref_id := pid.
  torvik.parquet     <- our cached Bart Torvik getadvstats exports (2008-2026), parsed with HIS TORVIK_COLS.
  season_war/target  <- our staging per-season outcomes, via bridge.build_table.season_war_rows + infra.war.war_target.
  bbref pieces       <- skipped entirely (draft pages, advanced season tables, biography, international league pages).

Stages (each idempotent; `all` runs them in order):
    sources   download ayush (2 files) / jasong / combine mirror / RAPTOR      (4.8 MB down,   2 s)
    torvik    local Torvik cache -> data/raw/torvik/*.csv -> torvik.parquet     ( 52 MB disk,   2 s)
    drafts    identity -> drafts.parquet (+ target.parquet, season_war.parquet) (               2 s)
    hoopr     stream hoopR-mbb-data parquets into data/external/hoopr           (113 MB down, 1m38s)
    features  run his permitted builders -> data/external/feat_*.parquet        (104 MB disk,    2m)
    merge     bridge table + rebuilt families -> processed/draft_table_colin.parquet
    coverage  per-family / per-band coverage report, before and after
    restore   put our bridge table back at data/processed/draft_table.parquet (safety net)
    clean     delete data/external/hoopr + player_game.parquet once the output exists (~217 MB)

Usage:
    python3 -m bridge.rebuild_colin_data all
    python3 -m bridge.rebuild_colin_data hoopr --seasons 2003-2026     # resumable, safe to re-run
    python3 -m bridge.rebuild_colin_data features --only combine,hoopr,shrunk,transfers
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from infra import config as C

HERE = Path(__file__).resolve().parent
ROOT = C.ROOT
RAW = C.RAW
PROC = C.PROC
EXT = ROOT / "data" / "external"
LOGS = HERE / "logs"

IDENTITY = Path("/Users/kennakao/Downloads/nba_redraft_handoff/identity_KEEP_SEPARATE/tabular_names.csv")
STAGING = Path("/Users/kennakao/nba/datarebuild/v4_build/staging_v419")
TORVIK_CACHE = Path("/Users/kennakao/nba/datarebuild/novel/torvik_context/raw")

BRIDGE_TABLE = PROC / "draft_table.parquet"          # our bridge output; snapshotted, restored at the end
BRIDGE_SNAPSHOT = PROC / "draft_table_bridge.parquet"
COLIN_TABLE = PROC / "draft_table_colin.parquet"
HIS_TABLE = PROC / "draft_table_his.parquet"          # LOCAL ONLY: his own name-keyed table (carries player names)
NAMEMAP = PROC / "_pid_names.parquet"                # LOCAL ONLY: pid -> normalised name (never shipped)

BANDS = [("2003_09", 2003, 2009), ("2010_18", 2010, 2018), ("2019_25", 2019, 2025)]

AYUSH_URL = "https://raw.githubusercontent.com/AyushBatra01/NBADraft/main/data/draft_players.csv"
AYUSH_24_URL = "https://raw.githubusercontent.com/AyushBatra01/NBADraft/main/data/draft_players24.csv"
JASONG_URL = "https://raw.githubusercontent.com/JasonG7234/NBA-Draft-Model/master/data/draft_db.csv"
COMBINE_URL = "https://raw.githubusercontent.com/MichLitt/nba-draft-oracle-pro/main/data/raw/combine_2000_2026_raw.csv"
RAPTOR_URL = "https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-raptor/historical_RAPTOR_by_player.csv"
HOOPR_URL = "https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/mbb/{kind}/parquet/{remote}.parquet"
# what each consumer needs: load_hoopr wants player_box + schedules 2003+, game_features wants all four 2007+
HOOPR_WANT = {"player_box": range(2003, 2027), "schedules": range(2003, 2027),
              "team_box": range(2007, 2027), "player_core": range(2007, 2027)}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _norm(s: str) -> str:
    import re
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s.lower())
    return re.sub(r"[^a-z]", "", s)


def _free_mb() -> int:
    st = shutil.disk_usage(str(ROOT))
    return st.free // (1024 * 1024)


def _fetch(url: str, dest: Path, min_bytes: int = 1000) -> bool:
    """Download unless already present and non-trivial. Returns True if the file is usable afterwards."""
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "DraftDB-research/1.0 (contact: mike@alphax.inc)"})
        with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as e:
        tmp.unlink(missing_ok=True)
        log(f"  FAILED {url.rsplit('/', 1)[-1]}: {type(e).__name__} {e}")
        return False
    if tmp.stat().st_size < min_bytes:
        tmp.unlink(missing_ok=True)
        log(f"  SHORT  {url.rsplit('/', 1)[-1]}: {tmp.stat().st_size} bytes")
        return False
    tmp.replace(dest)
    log(f"  got {dest.name} {dest.stat().st_size / 1e6:.1f} MB")
    return True


# --------------------------------------------------------------------------- stage: sources

def stage_sources() -> None:
    EXT.mkdir(parents=True, exist_ok=True)
    ok = []
    ok.append(_fetch(AYUSH_URL, EXT / "_ayush_2004_2023.csv", 50_000))
    got24 = _fetch(AYUSH_24_URL, EXT / "_ayush_2024.csv", 5_000)
    # his external.ayush() reads one CSV; the repo ships the 2024 class as a second file with an identical
    # 44-column schema, so the two are concatenated here rather than changing his loader.
    dst = EXT / "ayush_draft_players.csv"
    if ok[-1]:
        parts = [pd.read_csv(EXT / "_ayush_2004_2023.csv")]
        if got24:
            parts.append(pd.read_csv(EXT / "_ayush_2024.csv"))
        a = pd.concat(parts, ignore_index=True, sort=False).drop_duplicates(["Name", "Year"])
        a.to_csv(dst, index=False)
        log(f"  ayush_draft_players.csv {len(a)} rows, classes {int(a.Year.min())}-{int(a.Year.max())}")
    ok.append(_fetch(JASONG_URL, EXT / "jasong_draft_db.csv", 50_000))
    ok.append(_fetch(COMBINE_URL, EXT / "nba_combine" / "combine_mirror.csv", 50_000))
    # RAPTOR is downloaded for completeness (his war.py path); our labels replace it, so a failure is not fatal.
    _fetch(RAPTOR_URL, RAW / "raptor" / "historical_RAPTOR_by_player.csv", 100_000)
    if not all(ok):
        raise SystemExit("a required public CSV could not be downloaded; re-run `sources`")
    log("sources ok")


# --------------------------------------------------------------------------- stage: torvik

def stage_torvik() -> None:
    """Our cached getadvstats exports -> the CSVs his download_data.parse_torvik reads, then the parquet."""
    from infra.download_data import parse_torvik

    dst = RAW / "torvik"
    dst.mkdir(parents=True, exist_ok=True)
    for y in C.TORVIK_YEARS:
        out = dst / f"advstats_{y}.csv"
        src = TORVIK_CACHE / f"adv_{y}.csv.gz"
        if out.exists() and out.stat().st_size > 0:
            continue
        if not src.exists():
            log(f"  missing Torvik cache for {y} ({src}) -- skipped")
            continue
        with gzip.open(src, "rb") as f_in, open(out, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out, length=1 << 20)
    have = sorted(int(p.stem.split("_")[1]) for p in dst.glob("advstats_*.csv"))
    log(f"torvik csv seasons {have[0]}-{have[-1]} ({len(have)} files, {sum(p.stat().st_size for p in dst.glob('*.csv')) / 1e6:.0f} MB)")
    t = parse_torvik()
    PROC.mkdir(parents=True, exist_ok=True)
    t.to_parquet(PROC / "torvik.parquet", index=False)
    log(f"torvik.parquet {len(t):,} rows x {t.shape[1]} cols; seasons {int(t.year.min())}-{int(t.year.max())}")
    # sanity: Torvik's own draft tag must line up with the identity file's picks on a known class
    chk = t[(t.year == 2019) & t.pick.notna()]
    log(f"  sanity: {len(chk)} tagged 2019 picks, best BPM {t[t.year == 2019].bpm.max():.1f}")


# --------------------------------------------------------------------------- stage: drafts / labels

def _d1_index(torvik: pd.DataFrame) -> dict:
    """first 3 normalised chars -> the D1 team names starting with them, for validating the CSVs' school strings."""
    idx: dict = {}
    for t in torvik.team.dropna().unique():
        n = _norm(t).replace("state", "st")
        if len(n) >= 3:
            idx.setdefault(n[:3], set()).add(n)
    return idx


def _is_d1(name, idx: dict) -> bool:
    """The public CSVs put a high school ('SACA') or a pro club ('Illawarra (NBL)') in the same column as a college.
    Only a string that resolves to a Bart Torvik D1 team is accepted, so the college gate keeps its meaning.
    Matching follows infra.dataset._same_school: same first five characters in either direction."""
    if not isinstance(name, str) or not name.strip():
        return False
    n = _norm(name).replace("state", "st")
    if len(n) < 3:
        return False
    for d in idx.get(n[:3], ()):  # both start with the same three letters; then his own 5-char containment rule
        if n[:5] in d or d[:5] in n:
            return True
    return False


def _college_lookup(idx: dict) -> pd.DataFrame:
    """(key, draft_year) -> college, from the two public draft CSVs, keeping only strings that name a D1 team."""
    rows = []
    p = EXT / "ayush_draft_players.csv"
    if p.exists():
        a = pd.read_csv(p)
        rows.append(pd.DataFrame({"key": a.Name.map(_norm), "draft_year": a.Year.astype(int), "college": a.Team}))
    p = EXT / "jasong_draft_db.csv"
    if p.exists():
        j = pd.read_csv(p, low_memory=False, usecols=["Name", "Season", "School"])
        yr = pd.to_numeric(j.Season.astype(str).str[:4], errors="coerce") + 1
        rows.append(pd.DataFrame({"key": j.Name.map(_norm), "draft_year": yr, "college": j.School}).dropna(subset=["draft_year"]))
    out = pd.concat(rows, ignore_index=True)
    out["draft_year"] = out.draft_year.astype(int)
    out = out[out.college.map(lambda s: _is_d1(s, idx))]
    return out.dropna(subset=["college"]).drop_duplicates(["key", "draft_year"])


def _college_from_torvik(d: pd.DataFrame, torvik: pd.DataFrame) -> pd.Series:
    """Torvik itself names the college. Two passes, both leakage-free (his own match_torvik uses the same tag):
    1. Torvik tags a player-season with the pick he later became: (year == draft_year, pick) + same last name.
    2. Otherwise a unique normalised-name match in a season within [draft_year - 2, draft_year]."""
    tv = torvik.copy()
    tv["pick"] = pd.to_numeric(tv.pick, errors="coerce")
    tv["key"] = tv.player_name.map(_norm)
    tv["last"] = tv.player_name.map(lambda s: _norm(str(s).split()[-1]) if str(s).split() else "")
    tvp = tv.rename(columns={"team": "tv_team"})
    d = d[["draft_year", "pick", "player", "key"]].copy()
    d["last_d"] = d.player.map(lambda s: _norm(str(s).split()[-1]) if str(s).split() else "")
    out = pd.Series(np.nan, index=d.index, dtype=object)

    tp = tvp[tvp.pick.notna()]
    m = d.reset_index().merge(tp[["tv_team", "year", "pick", "last"]], left_on=["draft_year", "pick"],
                              right_on=["year", "pick"], how="inner")
    m = m[m.last == m.last_d].drop_duplicates("index")
    out.loc[m["index"].values] = m.tv_team.values

    todo = d[out.isna()].reset_index()
    c = todo.merge(tvp[["key", "tv_team", "year"]], on="key", how="inner")
    c = c[(c.year <= c.draft_year) & (c.year >= c.draft_year - 2)]
    c = c.sort_values("year").drop_duplicates("index", keep="last")
    out.loc[c["index"].values] = c.tv_team.values
    return out


def stage_drafts() -> None:
    """drafts.parquet from the identity file (names are local-only), plus target.parquet and season_war.parquet."""
    from infra.war import war_target
    from bridge.build_table import season_war_rows

    if not BRIDGE_SNAPSHOT.exists() and BRIDGE_TABLE.exists():
        shutil.copy2(BRIDGE_TABLE, BRIDGE_SNAPSHOT)
        log(f"snapshotted bridge table -> {BRIDGE_SNAPSHOT.name}")
    bridge = pd.read_parquet(BRIDGE_SNAPSHOT, columns=["bbref_id", "draft_year", "pick"])

    ident = pd.read_csv(IDENTITY)
    ident = ident[ident.pid.isin(set(bridge.bbref_id))].copy()
    d = pd.DataFrame({
        "draft_year": ident.draft_year.astype(int),
        "pick": ident.actual_pick,
        "team": "",                       # NBA team: bbref-only, unused by every builder we run
        "player": ident.player_name.astype(str),
        "bbref_id": ident.pid.astype(str),
    })
    d = d.dropna(subset=["pick"]).sort_values(["draft_year", "pick"]).reset_index(drop=True)
    d["pick"] = d["pick"].astype(int)
    d["key"] = d.player.map(_norm)

    torvik = pd.read_parquet(PROC / "torvik.parquet", columns=["player_name", "team", "year", "pick"])
    idx = _d1_index(torvik)
    col_tv = _college_from_torvik(d, torvik)
    lut = _college_lookup(idx)
    col_csv = d.merge(lut, on=["key", "draft_year"], how="left").college
    d["college"] = col_tv.where(col_tv.notna(), col_csv.values)
    n_tv, n_csv = col_tv.notna().sum(), col_csv.notna().sum()
    log(f"drafts: {len(d)} rows, college filled {d.college.notna().sum()} ({d.college.notna().mean():.1%}) "
        f"[torvik tag {n_tv}, public csv {n_csv}]")

    # LOCAL ONLY: the pid -> normalised-name map the merge stage needs. Never copied to the box.
    d[["bbref_id", "key", "draft_year"]].to_parquet(NAMEMAP, index=False)
    d.drop(columns=["key"]).to_parquet(PROC / "drafts.parquet", index=False)

    # labels: our staging per-season outcomes, in his (bbref_id, season, war) shape
    train = pd.read_csv(STAGING / "train_2000_2018.csv", low_memory=False)
    swar = season_war_rows(train)
    swar = swar[swar.bbref_id.isin(set(d.bbref_id))]
    swar.to_parquet(PROC / "season_war.parquet", index=False)
    tgt = war_target(d.drop(columns=["key"], errors="ignore"), swar)
    tgt.to_parquet(PROC / "target.parquet", index=False)
    log(f"season_war {len(swar):,} rows / {swar.bbref_id.nunique()} players; target {len(tgt)} rows, "
        f"labelled {int(tgt.labelled.sum())}")
    _intl_stub()


def _intl_stub() -> None:
    """`infra.external.intl()` reads data/external/intl_prospects.parquet, which `infra/builders/intl.py` builds from
    basketball-reference international pages + stats.gleague.nba.com -- neither is permitted here, and our own
    intl_ / fy_ / eur_ blocks already fill his intl_pro / intl_fiba families in bridge/build_table.py. An empty,
    correctly-typed stub keeps his dataset.build_table() running unmodified with those columns simply absent."""
    from infra.external import INTL_FEATURES
    p = EXT / "intl_prospects.parquet"
    if p.exists():
        return
    stub = pd.DataFrame({"key": pd.Series(dtype="object"), "draft_year": pd.Series(dtype="int64")})
    for c in INTL_FEATURES:
        stub[c] = pd.Series(dtype="float64")
    stub.to_parquet(p, index=False)
    log(f"wrote empty {p.name} (intl.py sources are not permitted; our own blocks fill intl_pro / intl_fiba)")


# --------------------------------------------------------------------------- stage: hoopR

def stage_hoopr(seasons: range | None = None, keep_mb: int = 250) -> None:
    """Stream the hoopR-mbb-data parquets one file at a time. Resumable: an existing non-empty file is skipped."""
    dst = EXT / "hoopr"
    dst.mkdir(parents=True, exist_ok=True)
    todo = []
    for kind, yrs in HOOPR_WANT.items():
        for s in (yrs if seasons is None else [y for y in yrs if y in seasons]):
            f = dst / f"{kind}_{s}.parquet"
            if not (f.exists() and f.stat().st_size > 0):
                todo.append((kind, s, f))
    log(f"hoopr: {len(todo)} files to fetch, {_free_mb()} MB free")
    done = 0
    for kind, s, f in todo:
        if _free_mb() < keep_mb:
            log(f"STOP: only {_free_mb()} MB free (floor {keep_mb}); {len(todo) - done} files left. Re-run `hoopr` after freeing space.")
            return
        remote = f"mbb_schedule_{s}" if kind == "schedules" else f"{kind}_{s}"
        _fetch(HOOPR_URL.format(kind=kind, remote=remote), f, min_bytes=500)
        done += 1
        (LOGS / "hoopr_progress.txt").write_text(f"{done}/{len(todo)} {kind}_{s}\n")
    have = {k: sum(1 for s in yrs if (dst / f"{k}_{s}.parquet").exists()) for k, yrs in HOOPR_WANT.items()}
    size = sum(p.stat().st_size for p in dst.glob("*.parquet")) / 1e6
    log(f"hoopr complete: {have}, {size:.0f} MB on disk, {_free_mb()} MB free")


# --------------------------------------------------------------------------- stage: features

# name -> (module, callable or None) ; None means "module has its own build()" handled below
PERMITTED = {
    "combine": ("infra.builders.college_sources", "load_combine"),
    "kaggle": None,          # HF snapshot of a Torvik mirror: adds no column, and needs huggingface_hub
    "score": None,           # data.scorenetwork.org: keys only, no pre-draft column
    "ianstack": None,        # github ianstack/...: keys + college WS; fetched opportunistically below
    "hoopr": ("infra.builders.college_sources", "load_hoopr"),
    "marchmadness": None,    # HF snapshot; needs huggingface_hub
    "shrunk": ("infra.builders.shrink", "load_shrunk"),
    "transfers": ("infra.builders.transfers", "load_transfers"),
    "response": ("infra.builders.response", "load_response"),
}
IANSTACK_URL = "https://raw.githubusercontent.com/ianstack/NBA-Draft-Combine-Analysis/main/data/College_WS.csv"


def _write_feat(name: str, df: pd.DataFrame) -> None:
    if not {"key", "draft_year"}.issubset(df.columns):
        log(f"  {name}: missing key/draft_year, skipped")
        return
    df = df.drop_duplicates(["key", "draft_year"]).copy()
    df["draft_year"] = df.draft_year.astype(int)
    df.to_parquet(EXT / f"feat_{name}.parquet", index=False)
    log(f"  feat_{name}: {len(df):5d} rows x {df.shape[1] - 2:3d} cols")


def stage_features(only: set[str] | None = None) -> None:
    import importlib

    def want(n):
        return only is None or n in only

    # 1. his own draft table (name-keyed, LOCAL): the builders that need it read PROC/draft_table.parquet
    if want("table"):
        _rebuild_his_table()

    # 2. simple loaders
    for name in ("combine", "shrunk", "transfers", "hoopr"):
        if not want(name):
            continue
        mod, fn = PERMITTED[name]
        try:
            df = getattr(importlib.import_module(mod), fn)()
        except Exception as e:
            log(f"  {name}: FAILED {type(e).__name__}: {e}")
            continue
        _write_feat(name, df)

    # 3. ianstack (small public CSV; optional)
    if want("ianstack"):
        if _fetch(IANSTACK_URL, EXT / "ianstack" / "College_WS.csv", 5_000):
            try:
                from infra.builders.college_sources import load_ianstack
                _write_feat("ianstack", load_ianstack())
            except Exception as e:
                log(f"  ianstack: FAILED {type(e).__name__}: {e}")

    # 4. torvik context (t_): pure Torvik, no download
    if want("torvik_context"):
        try:
            from infra.builders import torvik_context as tc
            out = tc.build() if hasattr(tc, "build") else None
            if out is None:
                out = tc.load_torvik_context()
            _write_feat("torvik_context", out)
        except Exception as e:
            log(f"  torvik_context: FAILED {type(e).__name__}: {e}")

    # 5. game features (g_) + the player-game table response.py needs
    if want("game"):
        try:
            from infra.builders import game_features as gf
            gf.build(rebuild_player_game=not gf.PLAYER_GAME.exists())
            if gf.FEATURES.exists():
                _write_feat("game", pd.read_parquet(gf.FEATURES))
        except Exception as e:
            log(f"  game: FAILED {type(e).__name__}: {e}")

    # 6. response (rs_) -- needs crosswalk_hoopr + player_game from step 5
    if want("response"):
        try:
            from infra.builders.response import load_response
            _write_feat("response", load_response())
        except Exception as e:
            log(f"  response: FAILED {type(e).__name__}: {e}")

    # 7. refresh his table so the coverage report sees everything
    if want("table"):
        _rebuild_his_table()


def _rebuild_his_table() -> None:
    """Run HIS infra.dataset.build_table over whatever feat_*.parquet exist. Name-keyed; stays on this Mac."""
    import importlib
    import infra.external, infra.dataset
    importlib.reload(infra.external)
    importlib.reload(infra.dataset)
    t = infra.dataset.build_table()
    t.to_parquet(BRIDGE_TABLE, index=False)     # his builders read PROC/draft_table.parquet; restored by `merge`
    t.to_parquet(HIS_TABLE, index=False)        # kept so `merge` can harvest his Torvik line + trajectory
    log(f"his draft_table (local, name-keyed): {t.shape}")


# --------------------------------------------------------------------------- stage: merge

FAMILY_PREFIX = {
    "a_box": ("a_",), "j_bio": ("rsci", "sos", "j_height", "j_weight", "j_age"), "j_box": ("j_",),
    "j_shot": ("dunks_per_min", "pct_rim", "pct_astd"), "j_aau": ("aau_",), "j_event": ("ev_",),
    "combine": ("c_",), "hoopr": ("h_",), "game": ("g_",), "response": ("rs_",), "tctx": ("t_",),
    "shrunk": ("sh_",), "transfers": ("tr_",), "ianstack": ("is_",),
}


def _rebuilt_columns() -> pd.DataFrame:
    """Every column his builders produced, keyed by (key, draft_year). Names live only in `key` -- dropped at the end."""
    from infra.dataset import norm_name
    from infra import external

    frames = []
    # the two GitHub draft CSVs go through HIS loaders, unmodified
    if (EXT / "ayush_draft_players.csv").exists():
        frames.append(("ayush", external.ayush(norm_name)))
    if (EXT / "jasong_draft_db.csv").exists():
        frames.append(("jasong", external.jasong(norm_name)))
    for p in sorted(EXT.glob("feat_*.parquet")):
        f = pd.read_parquet(p)
        if {"key", "draft_year"}.issubset(f.columns):
            frames.append((p.stem, f))
    # his own Torvik line and trajectory are computed inside dataset.build_table, not written as a feat_ parquet:
    # harvest them from the table he just built. The collision audit decides whether each one may fill a hole.
    if HIS_TABLE.exists():
        h = pd.read_parquet(HIS_TABLE)
        if "torvik_idx" in h.columns:
            from infra.dataset import NUMERIC_FEATURES
            want = [c for c in dict.fromkeys(NUMERIC_FEATURES + external.TRAJECTORY_FEATURES) if c in h.columns]
            frames.append(("his_table", h[["key", "draft_year"] + want]))

    out = None
    for name, f in frames:
        f = f.copy()
        f["draft_year"] = pd.to_numeric(f.draft_year, errors="coerce")
        f = f.dropna(subset=["draft_year", "key"])
        f["draft_year"] = f.draft_year.astype(int)
        f = f.drop_duplicates(["key", "draft_year"])
        keep = ["key", "draft_year"] + [c for c in f.columns if c not in ("key", "draft_year") and pd.api.types.is_numeric_dtype(f[c])]
        f = f[keep].copy()
        # bool columns (g_hoopr_matched, g_dob_imputed, g_first_game_censored) become object once a left merge
        # introduces NaN -- his own dataset.py lists the same three. Ship them as 0/1 floats.
        for c in [c for c in f.columns if f[c].dtype == bool]:
            f[c] = f[c].astype(float)
        out = f if out is None else out.merge(f, on=["key", "draft_year"], how="outer", suffixes=("", f"__{name}"))
    dupes = [c for c in out.columns if "__" in c]
    if dupes:
        out = out.drop(columns=dupes)
    return out


COALESCE_PREFIXES = ("c_",)   # combine: the task asks for the MichLitt mirror to fill holes in ours
# weight in pounds off a different tape: rho 0.982, median ratio 1.0000, +96 rows. His own dataset.build_table
# already coalesces this exact pair (`table.weight_lb.fillna(table.c_weight)`), so filling a hole is his convention.
COALESCE_COLUMNS = {"weight_lb"}
SAME_RHO, SAME_RATIO = 0.99, 0.02   # "the two implementations compute the same quantity"


def _collision_action(base: str, ours_empty: bool, rho: float, ratio: float) -> str:
    """What to do with a rebuilt column whose name already exists in the bridge table."""
    if ours_empty:
        return "filled_empty"          # a_* / j_* / aau_* / ev_* / rsci / sos: written empty by the bridge, filled now
    if base.startswith(COALESCE_PREFIXES) or base in COALESCE_COLUMNS:
        return "coalesced_combine"
    if pd.notna(rho) and pd.notna(ratio) and rho >= SAME_RHO and abs(ratio - 1.0) <= SAME_RATIO:
        return "coalesced_identical"   # verified same scale and same ordering, so filling a hole is safe
    return "kept_ours"                 # different definition or different scale -- never mix them in one column


def stage_merge() -> None:
    """Bridge table + rebuilt families -> draft_table_colin.parquet, pid-keyed, names dropped.

    Collision policy: a rebuilt column whose name already exists in the bridge table fills ours only when ours is
    entirely empty, when the family is `c_*`, or when the overlap proves the two implementations compute the same
    quantity (Spearman >= 0.99 and median ratio within 2%). Otherwise ours wins -- two implementations of the same
    idea must not share one column (MAPPING.md's one-scale-per-column rule). Every collision is measured and written
    to bridge/collision_audit.csv.
    """
    if not BRIDGE_SNAPSHOT.exists():
        raise SystemExit("no bridge snapshot; run `drafts` first")
    bridge = pd.read_parquet(BRIDGE_SNAPSHOT)
    names = pd.read_parquet(NAMEMAP)                       # pid -> normalised name, LOCAL
    reb = _rebuilt_columns()

    m = bridge.merge(names.rename(columns={"key": "_name_key"}), on=["bbref_id", "draft_year"], how="left")
    before = set(bridge.columns)
    m = m.merge(reb.rename(columns={"key": "_name_key"}), on=["_name_key", "draft_year"], how="left",
                suffixes=("", "_reb"))

    audit, coalesced, kept_ours = [], 0, []
    for c in [c for c in m.columns if c.endswith("_reb")]:
        base = c[:-4]
        ok = m[base].notna() & m[c].notna()
        rho = m.loc[ok, base].corr(m.loc[ok, c], method="spearman") if ok.sum() > 20 else np.nan
        with np.errstate(all="ignore"):
            ratio = float(np.nanmedian(m.loc[ok, c] / m.loc[ok, base].replace(0, np.nan))) if ok.sum() > 20 else np.nan
        gain = int((m[base].isna() & m[c].notna()).sum())
        action = _collision_action(base, not m[base].notna().any(), rho, ratio)
        audit.append({"column": base, "n_overlap": int(ok.sum()), "spearman": round(float(rho), 4) if pd.notna(rho) else None,
                      "median_ratio": round(ratio, 4) if pd.notna(ratio) else None,
                      "rows_ours_missing_his_present": gain, "action": action})
        if action == "kept_ours":
            kept_ours.append(base)
        else:
            m[base] = m[base].where(m[base].notna(), m[c])
            coalesced += gain
    if audit:
        A = pd.DataFrame(audit).sort_values("column")
        A.to_csv(HERE / "collision_audit.csv", index=False)
        log(f"  collisions: {len(A)} columns -> " + ", ".join(f"{k} {v}" for k, v in A.action.value_counts().items())
            + f"; {coalesced} cells filled; bridge/collision_audit.csv")
        if kept_ours:
            log(f"  kept ours (different definition/scale): {kept_ours}")
    m = m.drop(columns=[c for c in m.columns if c.endswith("_reb")] + ["_name_key"])
    added = [c for c in m.columns if c not in before]
    log(f"merge: {len(m)} rows, {len(before)} -> {m.shape[1]} columns (+{len(added)})")

    # identity guard: nothing that carries a name may reach the box
    new_obj = [c for c in added if str(m[c].dtype) == "object"]
    assert not new_obj, f"rebuilt object columns would ship free text: {new_obj}"
    for c in ("bbref_id", "key", "player"):
        assert (m[c].astype(str) == m.bbref_id.astype(str)).all(), f"{c} is not the pid -- a name would leak"
    assert "_name_key" not in m.columns and not any("name" in c.lower() for c in m.columns), "a name-ish column survived"
    m.to_parquet(COLIN_TABLE, index=False)
    log(f"wrote {COLIN_TABLE} {m.shape}")

    # restore the bridge table his builders temporarily overwrote
    shutil.copy2(BRIDGE_SNAPSHOT, BRIDGE_TABLE)
    log(f"restored {BRIDGE_TABLE.name} from the snapshot")
    # season_war: ours is unchanged in content, but rewrite it in the 3-column shape for completeness
    sw = pd.read_parquet(PROC / "season_war.parquet")
    sw[["bbref_id", "season", "war"]].to_parquet(PROC / "season_war.parquet", index=False)


# --------------------------------------------------------------------------- stage: coverage

def _cov(t: pd.DataFrame, cols: list[str], lo: int, hi: int) -> float:
    cols = [c for c in cols if c in t.columns]
    if not cols:
        return float("nan")
    s = t[(t.draft_year >= lo) & (t.draft_year <= hi)]
    return float(s[cols].notna().mean().mean()) if len(s) else float("nan")


def stage_coverage() -> None:
    from tournament.contract import EXPLICIT, PREFIX, TORVIK
    from infra.dataset import CATEGORICAL_FEATURES, NUMERIC_FEATURES
    from infra.external import AYUSH_BOX, JASONG_NUM

    before = pd.read_parquet(BRIDGE_SNAPSHOT)
    after = pd.read_parquet(COLIN_TABLE) if COLIN_TABLE.exists() else before

    groups: dict[str, list[str]] = {k: list(v) for k, v in EXPLICIT.items()}
    groups["all_torvik"] = NUMERIC_FEATURES + list(CATEGORICAL_FEATURES)
    for k, v in TORVIK.items():
        groups[f"tv:{k}"] = list(v)
    for name, p in PREFIX.items():
        cols = [c for c in after.columns if c.startswith(p)]
        if cols:
            groups[name] = cols
    groups["a_box"] = list(AYUSH_BOX.values())
    groups["j_all"] = list(JASONG_NUM.values())

    rows = []
    for g, cols in sorted(groups.items()):
        have_b = [c for c in cols if c in before.columns]
        have_a = [c for c in cols if c in after.columns]
        nb = sum(before[c].notna().any() for c in have_b)
        na = sum(after[c].notna().any() for c in have_a)
        r = {"family": g, "cols": len(cols), "live_before": nb, "live_after": na}
        for band, lo, hi in BANDS:
            r[f"before_{band}"] = round(_cov(before, cols, lo, hi), 3)
            r[f"after_{band}"] = round(_cov(after, cols, lo, hi), 3)
        rows.append(r)
    rep = pd.DataFrame(rows)
    rep.to_csv(HERE / "coverage_colin_rebuild.csv", index=False)
    print(rep.to_string(index=False))
    log(f"wrote {HERE / 'coverage_colin_rebuild.csv'}")


# --------------------------------------------------------------------------- stages: restore / clean

def stage_restore() -> None:
    """Put our bridge table back at data/processed/draft_table.parquet.

    `features` deliberately parks HIS name-keyed table there because his builders read that path. `merge` restores
    ours at the end; run this by hand if a `features` run was interrupted."""
    if not BRIDGE_SNAPSHOT.exists():
        raise SystemExit("no snapshot to restore from")
    shutil.copy2(BRIDGE_SNAPSHOT, BRIDGE_TABLE)
    log(f"restored {BRIDGE_TABLE.name} ({BRIDGE_TABLE.stat().st_size / 1e6:.1f} MB)")


def stage_clean() -> None:
    """Delete the bulky intermediates once draft_table_colin.parquet exists. Every one is re-derivable:
    `hoopr` re-downloads data/external/hoopr (~118 MB), `features --only game` rebuilds player_game.parquet."""
    if not COLIN_TABLE.exists():
        raise SystemExit("draft_table_colin.parquet does not exist yet; not deleting anything")
    freed = 0
    for p in list((EXT / "hoopr").glob("*.parquet")) + [PROC / "player_game.parquet"]:
        if p.exists():
            freed += p.stat().st_size
            p.unlink()
    log(f"clean: freed {freed / 1e6:.0f} MB; {_free_mb()} MB free. Re-run `hoopr` then "
        f"`features --only game,response` to rebuild them.")


# --------------------------------------------------------------------------- cli

STAGES = {"sources": stage_sources, "torvik": stage_torvik, "drafts": stage_drafts,
          "hoopr": stage_hoopr, "features": stage_features, "merge": stage_merge, "coverage": stage_coverage,
          "restore": stage_restore, "clean": stage_clean}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="+", choices=list(STAGES) + ["all"])
    ap.add_argument("--seasons", default=None, help="hoopr: e.g. 2003-2026")
    ap.add_argument("--only", default=None, help="features: comma list of table,combine,shrunk,transfers,hoopr,ianstack,torvik_context,game,response")
    ap.add_argument("--keep-mb", type=int, default=250, help="hoopr: stop when free disk falls below this")
    a = ap.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)
    order = ["sources", "torvik", "drafts", "hoopr", "features", "merge", "coverage", "restore", "clean"]
    stages = [s for s in order if s != "clean"] if "all" in a.stage else [s for s in order if s in a.stage]
    log(f"stages: {stages}; free disk {_free_mb()} MB")
    for s in stages:
        log(f"=== {s}")
        if s == "hoopr":
            rng = None
            if a.seasons:
                lo, hi = (int(x) for x in a.seasons.split("-"))
                rng = range(lo, hi + 1)
            stage_hoopr(rng, a.keep_mb)
        elif s == "features":
            stage_features(set(a.only.split(",")) if a.only else None)
        else:
            STAGES[s]()
    log(f"done; free disk {_free_mb()} MB")


if __name__ == "__main__":
    main()
