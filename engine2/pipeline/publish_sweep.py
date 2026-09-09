"""Publish the holdout predictions for a sweep's context-selected configuration to Supabase.

First score the selected config with `tournament.layer1 --years holdout`, then:
  python -m pipeline.publish_sweep --tag <sweep-tag> --predictions <test.parquet>
"""

import argparse
import asyncio
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd
import torch

from infra import config as C
from tournament import contract as F
from pipeline.db import push
from pipeline.run import _json, _picks, context_years, summarize, year_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--config-name", help="layer-1 config used by the selected layer-2 rule")
    ap.add_argument("--publish-tag", help="clean display tag for the published run")
    ap.add_argument("--rule", help="pre-registered layer-2 rule selected on the context years")
    a = ap.parse_args()

    best = json.loads((C.OUT / "sweeps" / a.tag / "best.json").read_text())
    cfg = best["best_layer1"]
    if a.config_name:
        configs = sum((json.loads(p.read_text()) for p in sorted((C.OUT / "sweeps" / a.tag).glob("configs_gpu*.json"))), [])
        cfg = next(c for c in configs if c["name"] == a.config_name)
    base_name = cfg["name"]
    raw = pd.read_parquet(a.predictions)
    if a.rule:
        from tournament.layer2 import _recompute, wide
        scored = wide(raw, "holdout")
        pred = scored[["draft_year", "bbref_id"]].assign(score=_recompute(scored, [base_name], a.rule))
        name = f"{base_name}+consensus" if a.rule.startswith("consensus") else a.rule
    else:
        name = base_name
        pred = raw[(raw.config == name) & (raw.member == -1)]
    if pred.empty:
        raise SystemExit(f"no predictions found for {name}")
    years = sorted(pred.draft_year.unique())
    if years != list(C.HOLDOUT_YEARS):
        raise SystemExit(f"expected holdout years {list(C.HOLDOUT_YEARS)}, got {years}")

    table = pd.read_parquet(C.PROC / "draft_table.parquet")
    groups = F.validate(table, verbose=False)
    members = cfg.get("stack") or [cfg]
    features = list(dict.fromkeys(sum(
        (F.resolve(m.get("features", cfg.get("features")), groups, table.columns) for m in members), []
    )))
    labelled = sorted(set(table.loc[table.modelled & table.labelled, "draft_year"]))
    metrics, redrafts = [], []
    for year in years:
        scored = pred[pred.draft_year == year][["bbref_id", "score"]]
        cls = table[table.draft_year == year].sort_values("pick")
        modelled = cls[cls.bbref_id.isin(scored.bbref_id)]
        score = scored.set_index("bbref_id").score.reindex(modelled.bbref_id).to_numpy()
        cy = context_years(year, "causal", labelled)
        metrics.append({
            "split": "holdout", "protocol": "causal", "model": name, "year": int(year),
            "n_context": int(table[table.modelled & table.labelled & table.draft_year.isin(cy)].shape[0]),
            "context_years": cy, "seconds": 0.0, **year_metrics(modelled, score),
        })
        redrafts.append({
            "year": int(year), "model": name, "protocol": "causal", "context_years": cy,
            "picks": _picks(cls, scored.set_index("bbref_id").score),
        })

    split = []
    for year in C.DRAFT_YEARS:
        role = "context" if year in C.CONTEXT_YEARS else "holdout" if year in C.HOLDOUT_YEARS else "unlabelled"
        split.append({
            "year": year, "role": role, "labelled": bool(year <= C.LAST_LABELED_DRAFT),
            "causal_context": context_years(year, "causal", labelled),
        })

    now = datetime.now(timezone.utc)
    publish_tag = a.publish_tag or a.tag
    run_id = now.strftime("%Y%m%d_%H%M%S") + f"_{publish_tag}"
    run = {
        "features": features, "run_id": run_id, "created": now.isoformat(), "tag": publish_tag,
        "models": [name], "redraft_model": name, "feature_hash": hashlib.md5(",".join(features).encode()).hexdigest()[:8],
        "model_features": {name: features}, "target": C.TARGET_KIND,
        "target_desc": f"{C.TARGET} = WAR summed over the first {C.TARGET_SEASONS} NBA seasons "
                       f"(FiveThirtyEight RAPTOR through 2022, DARKO/RAPM-equivalent after); "
                       f"never played = {C.NEVER_PLAYED_WAR}",
        "north_star": "spearman", "split": split, "context_years": list(C.CONTEXT_YEARS), "holdout_years": years,
        "gpus": torch.cuda.device_count(), "metrics": metrics,
        "summary": summarize(metrics).to_dict("records"), "redrafts": redrafts,
    }
    out = C.OUT / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(run, indent=1, default=_json))
    asyncio.run(push())
    print(f"published {run_id}: {name}")


if __name__ == "__main__":
    main()
