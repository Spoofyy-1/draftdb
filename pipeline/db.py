"""Push every outputs/runs/*/run.json into Supabase Postgres (what the web app reads).

All network I/O is non-blocking: the async supabase-py client, with each run's four tables and every insert batch
issued concurrently under asyncio.gather.

Env (repo-root .env is read automatically):
  SUPABASE_URL         https://<ref>.supabase.co
  SUPABASE_SECRET_KEY  service-role key -- writes bypass RLS; never expose this to the browser
  SUPABASE_DB_URL      optional, only for --schema: postgres connection string for the DDL

  python -m pipeline.db            push every run
  python -m pipeline.db --schema   apply supabase_schema.sql first (needs SUPABASE_DB_URL)
"""

import asyncio
import itertools
import json
import math
import os
import sys
from pathlib import Path

from infra.config import OUT

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_SQL = Path(__file__).with_name("supabase_schema.sql")
BATCH = 500  # rows per insert request; redraft_picks runs to tens of thousands


def _load_env() -> None:
    """Read repo-root .env into os.environ without overriding what is already set."""
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _clean(v):
    """Postgres rejects the NaN/Infinity that json.dump happily writes; send null instead."""
    return None if isinstance(v, float) and not math.isfinite(v) else v


def _rows(dicts: list[dict]) -> list[dict]:
    return [{k: _clean(v) for k, v in d.items()} for d in dicts]


# --------------------------------------------------------------------------- run.json -> table rows

def _run_row(r: dict) -> dict:
    return {
        "run_id": r["run_id"], "created": r["created"], "tag": r["tag"], "models": ",".join(r["models"]),
        "redraft_model": r["redraft_model"], "north_star": r["north_star"], "target": r["target"],
        "target_desc": r.get("target_desc", ""), "feature_hash": r["feature_hash"], "n_features": len(r["features"]),
        "gpus": r["gpus"],
    }


def _split_rows(r: dict) -> list[dict]:
    return [{"run_id": r["run_id"], "year": s["year"], "role": s["role"], "labelled": int(s["labelled"]),
             "causal_context": s["causal_context"]} for s in r["split"]]


def _metric_rows(r: dict) -> list[dict]:
    return [{"run_id": r["run_id"], "split": m["split"], "protocol": m["protocol"], "model": m["model"], "year": m["year"],
             "n_context": m["n_context"], "n": m["n"], "seconds": m["seconds"], "spearman": m["spearman"],
             "spearman_nba": m.get("spearman_nba"),
             "war_captured_pct_14": m["war_captured_pct@14"], "war_captured_pct_30": m["war_captured_pct@30"],
             "ndcg_14": m["ndcg@14"], "ndcg_30": m["ndcg@30"],
             "war_ours_14": m["war_ours@14"], "war_actual_14": m["war_actual@14"], "war_oracle_14": m["war_oracle@14"],
             "war_ours_30": m["war_ours@30"], "war_actual_30": m["war_actual@30"], "war_oracle_30": m["war_oracle@30"]}
            for m in r["metrics"]]


def _pick_rows(r: dict) -> list[dict]:
    return [{"run_id": r["run_id"], "year": d["year"], "model": d["model"], "protocol": d["protocol"],
             "player": p["player"], "bbref_id": p["bbref_id"], "actual_pick": p["actual_pick"], "new_pick": p["new_pick"],
             "team": p["team"], "college": p["college"], "modelled": int(p["modelled"]), "pred": p["pred"],
             "war": p["war"], "seasons_played": p["seasons_played"], "labelled": int(p["labelled"])}
            for d in r["redrafts"] for p in d["picks"]]


# --------------------------------------------------------------------------- push

async def _insert(sb, table: str, rows: list[dict]) -> None:
    """Every batch of one table in flight at once."""
    if not rows:
        return
    chunks = [_rows(rows[i:i + BATCH]) for i in range(0, len(rows), BATCH)]
    await asyncio.gather(*(sb.table(table).insert(c).execute() for c in chunks))


async def _push_run(sb, path: Path) -> tuple[str, int]:
    r = json.loads(path.read_text())
    # Deleting the run cascades to splits / metrics / redraft_picks, so a re-push replaces rather than duplicates.
    await sb.table("runs").delete().eq("run_id", r["run_id"]).execute()
    await sb.table("runs").insert(_rows([_run_row(r)])).execute()
    picks = _pick_rows(r)
    await asyncio.gather(
        _insert(sb, "splits", _split_rows(r)),
        _insert(sb, "metrics", _metric_rows(r)),
        _insert(sb, "redraft_picks", picks),
    )
    return r["run_id"], len(picks)


async def _client():
    from supabase import acreate_client

    _load_env()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SECRET_KEY") or os.environ.get("SUPABASE_PUBLISHABLE_KEY")
    if not url or not key:
        raise SystemExit("set SUPABASE_URL and SUPABASE_SECRET_KEY in the repo-root .env")
    return await acreate_client(url, key)


async def _select(sb, table: str, run_id: str, columns: str = "*") -> list[dict]:
    """Follow PostgREST's 1000-row cap until a short page comes back."""
    rows: list[dict] = []
    for start in itertools.count(0, BATCH):
        page = (await sb.table(table).select(columns).eq("run_id", run_id).range(start, start + BATCH - 1).execute()).data
        rows += page
        if len(page) < BATCH:
            return rows
    return rows


async def _latest_run_picks() -> tuple[str, str, list[dict]]:
    sb = await _client()
    runs = (await sb.table("runs").select("run_id, redraft_model").order("created", desc=True).limit(1).execute()).data
    if not runs:
        raise SystemExit("no runs in Supabase yet -- run `make run` first")
    run_id, model = runs[0]["run_id"], runs[0]["redraft_model"]
    return run_id, model, await _select(sb, "redraft_picks", run_id, "year, actual_pick, pred, war, modelled, labelled")


def latest_run_picks() -> tuple[str, str, list[dict]]:
    """Sync entrypoint for tournament/audit.py: (run_id, redraft_model, every pick of the newest run)."""
    return asyncio.run(_latest_run_picks())


async def push() -> None:
    paths = sorted((OUT / "runs").glob("*/run.json"))
    if not paths:
        print("no outputs/runs/*/run.json to push")
        return
    _load_env()
    if not os.environ.get("SUPABASE_SECRET_KEY"):
        raise SystemExit("pushing needs SUPABASE_SECRET_KEY (the service-role key); the publishable key is read-only")
    sb = await _client()
    for run_id, n in await asyncio.gather(*(_push_run(sb, p) for p in paths)):
        print(f"pushed {run_id}  {n} picks")


async def apply_schema() -> None:
    """Run supabase_schema.sql over a direct Postgres connection (the REST API cannot do DDL)."""
    import asyncpg

    _load_env()
    dsn = os.environ.get("SUPABASE_DB_URL")
    if not dsn:
        raise SystemExit("set SUPABASE_DB_URL (Supabase -> Project Settings -> Database -> Connection string)")
    con = await asyncpg.connect(dsn)
    try:
        await con.execute(SCHEMA_SQL.read_text())
    finally:
        await con.close()
    print("schema applied")


def rebuild() -> None:
    """Sync entrypoint kept for pipeline/run.py, which calls this at the end of a run."""
    asyncio.run(push())


if __name__ == "__main__":
    asyncio.run(apply_schema() if "--schema" in sys.argv else push())
