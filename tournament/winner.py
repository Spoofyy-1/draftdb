"""The winners archive: freeze a scored system into winners/<name>/ so it survives sweeps and `make clean`.

`outputs/` is regenerated (and wiped by `make clean`), and outputs/sweeps/<tag>/best.json is overwritten by the next
sweep that reuses the tag. Anything worth keeping goes here instead: the exact layer-1 config, the layer-2 rule, the
per-player predictions it was scored from, the metrics, and a fingerprint of the draft table it was built on.

  python -m tournament.winner export --name <name> --layer1 <parquet ...> --config "<layer-1 config name>"
                                     [--rule "<layer-2 rule>"] [--window holdout|context] [--note ...]
  python -m tournament.winner list
  python -m tournament.winner show --name <name>

Export re-scores from the prediction files rather than trusting a number typed on the command line, so a winner in
this folder always carries metrics that match its own predictions.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from infra import config as C
from tournament import contract as F

WINNERS = C.ROOT / "winners"


def _table_fingerprint(table: pd.DataFrame) -> str:
    cols = ["bbref_id", "draft_year", "pick", C.TARGET]
    data = pd.util.hash_pandas_object(table[cols], index=False).values
    return hashlib.md5(data.tobytes()).hexdigest()[:12]


def _score(w: pd.DataFrame, col) -> dict:
    """Per-year and mean Spearman / WAR-captured for one ranking column, against the real draft as the baseline."""
    per_year, sp, nba, wc = {}, [], [], []
    for year, g in w.assign(_r=col).groupby("draft_year"):
        g = g[g.labelled.astype(bool) & g[C.TARGET].notna()]
        if len(g) < 3:
            continue
        s = spearmanr(g["_r"], g[C.TARGET]).correlation
        n = spearmanr(-g["pick"], g[C.TARGET]).correlation
        ours = g.sort_values("_r", ascending=False)[C.TARGET].head(14).sum()
        act = g.sort_values("pick")[C.TARGET].head(14).sum()
        best = g[C.TARGET].nlargest(14).sum()
        per_year[int(year)] = {"spearman": float(s), "spearman_nba": float(n), "n": int(len(g))}
        sp.append(s)
        nba.append(n)
        wc.append(100 * (ours - act) / (best - act) if best > act else 0.0)
    return {"spearman": float(np.mean(sp)), "spearman_nba": float(np.mean(nba)), "wc14": float(np.mean(wc)),
            "wins": int(np.sum(np.array(sp) > np.array(nba))), "n_years": len(sp), "per_year": per_year}


def export(a) -> None:
    from tournament.layer2 import _recompute, wide

    l1 = pd.concat([pd.read_parquet(p) for p in a.layer1], ignore_index=True)
    tags = sorted({t for t in l1.years.astype(str) if t.split(":")[0] == a.window})
    if not tags:
        raise SystemExit(f"no rows tagged '{a.window}' in {a.layer1}; found {sorted(l1.years.astype(str).unique())}")
    l1 = l1[l1.years.astype(str).isin(tags)].assign(years=a.window)
    configs = sorted(l1.config.unique())
    if a.config not in configs:
        raise SystemExit(f"config {a.config!r} not in {configs}")

    w = wide(l1, a.window)
    col = _recompute(w, configs, a.rule) if a.rule else w[a.config]
    metrics = _score(w, col)

    table = pd.read_parquet(C.PROC / "draft_table.parquet")
    G = F.validate(table, verbose=False)
    cfg = _find_config(a.config)
    features = F.resolve(cfg.get("features"), G, table.columns) if cfg else None

    dest = WINNERS / a.name
    dest.mkdir(parents=True, exist_ok=True)
    keep = l1[l1.config.isin([a.config])] if not a.rule else l1
    keep.to_parquet(dest / "predictions.parquet", index=False)
    record = {
        "name": a.name, "created": datetime.now(timezone.utc).isoformat(), "note": a.note,
        "window": a.window, "years": sorted(int(y) for y in l1.draft_year.unique()),
        "layer1_config_name": a.config, "layer1_config": cfg, "layer2_rule": a.rule,
        "metrics": metrics,
        "n_features": len(features) if features else None, "features": features,
        "protocol": {"context_years": list(C.CONTEXT_YEARS), "holdout_years": list(C.HOLDOUT_YEARS),
                     "target_kind": C.TARGET_KIND, "target_seasons": C.TARGET_SEASONS,
                     "label_transform": C.LABEL_TRANSFORM, "last_season": C.LAST_SEASON},
        "draft_table_fingerprint": _table_fingerprint(table),
        "source_layer1": [str(p) for p in a.layer1],
        "reproduce": f"python -m tournament.layer1 --years {a.window} --tag <tag> --configs '{json.dumps([cfg])}'" if cfg else None,
    }
    (dest / "winner.json").write_text(json.dumps(record, indent=1))
    print(f"{a.name}: {a.window} spearman {metrics['spearman']:.4f} (scouts {metrics['spearman_nba']:.4f}, "
          f"wc14 {metrics['wc14']:+.1f}, wins {metrics['wins']}/{metrics['n_years']})")
    print("written:", dest / "winner.json")


def _find_config(name: str) -> dict | None:
    """The config dict as the sweep that produced it recorded it."""
    for p in sorted((C.OUT / "sweeps").glob("*/configs_gpu*.json")):
        for c in json.load(open(p)):
            if c.get("name") == name:
                return c
    for p in sorted((C.OUT / "sweeps").glob("*/best.json")):
        b = json.loads(p.read_text())
        if b.get("best_layer1", {}).get("name") == name:
            return b["best_layer1"]
    return None


def _records() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(WINNERS.glob("*/winner.json"))]


def show_list() -> None:
    rs = _records()
    if not rs:
        print("no winners yet")
        return
    df = pd.DataFrame([{"name": r["name"], "window": r["window"], "spearman": r["metrics"]["spearman"],
                        "scouts": r["metrics"]["spearman_nba"], "wc14": r["metrics"]["wc14"],
                        "wins": f"{r['metrics']['wins']}/{r['metrics']['n_years']}",
                        "rule": r["layer2_rule"] or r["layer1_config_name"], "created": r["created"][:19]}
                       for r in rs]).sort_values("spearman", ascending=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)
    print(df.round(4).to_string(index=False))


def show_one(name: str) -> None:
    r = json.loads((WINNERS / name / "winner.json").read_text())
    print(json.dumps({k: v for k, v in r.items() if k != "features"}, indent=1))
    print(f"({r['n_features']} features)")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--name", required=True)
    e.add_argument("--layer1", nargs="+", required=True)
    e.add_argument("--config", required=True, help="layer-1 config name")
    e.add_argument("--rule", help="layer-2 rule name; omit for the pure layer-1 config")
    e.add_argument("--window", default="holdout", choices=["holdout", "context"])
    e.add_argument("--note", default="")
    sub.add_parser("list")
    s = sub.add_parser("show")
    s.add_argument("--name", required=True)
    a = ap.parse_args()
    if a.cmd == "export":
        export(a)
    elif a.cmd == "list":
        show_list()
    else:
        show_one(a.name)


if __name__ == "__main__":
    main()
