"""Rebuild outputs/results.sqlite from every outputs/runs/*/run.json (what the web app reads)."""

import json
import sqlite3

from infra.config import OUT

DB = OUT / "results.sqlite"

SCHEMA = """
CREATE TABLE runs (run_id TEXT PRIMARY KEY, created TEXT, tag TEXT, models TEXT, redraft_model TEXT, north_star TEXT, target TEXT, target_desc TEXT,
                   feature_hash TEXT, n_features INTEGER, holdout_evaluated INTEGER, gpus INTEGER);
CREATE TABLE splits (run_id TEXT, year INTEGER, role TEXT, labelled INTEGER, causal_context TEXT);
CREATE TABLE metrics (run_id TEXT, split TEXT, protocol TEXT, model TEXT, year INTEGER, n_context INTEGER, n INTEGER, seconds REAL,
                      spearman REAL, spearman_nba REAL, war_captured_pct_14 REAL, war_captured_pct_30 REAL, ndcg_14 REAL, ndcg_30 REAL,
                      war_ours_14 REAL, war_actual_14 REAL, war_oracle_14 REAL, war_ours_30 REAL, war_actual_30 REAL, war_oracle_30 REAL);
CREATE TABLE redraft_picks (run_id TEXT, year INTEGER, model TEXT, protocol TEXT, player TEXT, bbref_id TEXT, actual_pick INTEGER, new_pick INTEGER,
                            team TEXT, college TEXT, modelled INTEGER, pred REAL, war REAL, seasons_played INTEGER, labelled INTEGER);
CREATE INDEX ix_metrics ON metrics(run_id, split, protocol, model);
CREATE INDEX ix_picks ON redraft_picks(run_id, year);
"""


def rebuild():
    tmp = DB.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    con.executescript(SCHEMA)
    for path in sorted((OUT / "runs").glob("*/run.json")):
        r = json.loads(path.read_text())
        con.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (r["run_id"], r["created"], r["tag"], ",".join(r["models"]), r["redraft_model"], r["north_star"], r["target"], r.get("target_desc", ""),
                     r["feature_hash"], len(r["features"]), int(r["holdout_evaluated"]), r["gpus"]))
        con.executemany("INSERT INTO splits VALUES (?,?,?,?,?)",
                        [(r["run_id"], s["year"], s["role"], int(s["labelled"]), json.dumps(s["causal_context"])) for s in r["split"]])
        con.executemany("INSERT INTO metrics VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        [(r["run_id"], m["split"], m["protocol"], m["model"], m["year"], m["n_context"], m["n"], m["seconds"],
                          m["spearman"], m.get("spearman_nba"), m["war_captured_pct@14"], m["war_captured_pct@30"], m["ndcg@14"], m["ndcg@30"],
                          m["war_ours@14"], m["war_actual@14"], m["war_oracle@14"], m["war_ours@30"], m["war_actual@30"], m["war_oracle@30"]) for m in r["metrics"]])
        con.executemany("INSERT INTO redraft_picks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        [(r["run_id"], d["year"], d["model"], d["protocol"], p["player"], p["bbref_id"], p["actual_pick"], p["new_pick"], p["team"],
                          p["college"], int(p["modelled"]), p["pred"], p["war"], p["seasons_played"], int(p["labelled"]))
                         for d in r["redrafts"] for p in d["picks"]])
    con.commit()
    con.close()
    tmp.replace(DB)
    print("wrote", DB)


if __name__ == "__main__":
    rebuild()
