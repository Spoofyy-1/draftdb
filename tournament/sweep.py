"""The one sweep. Run it, read the report, improve a builder or a rule, run it again.

  1. audit   -- rebuild the table from every materialised source, validate the feature registry, coverage per group
                for college and international draftees, leakage assertions.
  2. layer 1 -- a grid over feature groups x models x label framing, sharded across every GPU as parallel processes,
                on the context years (2013-2017). Every config's per-player predictions land in outputs/layer_1/.
  3. layer 2 -- pre-draft-only rules on top (blends, upside/floor readouts, consensus) on context; leaderboard.
  4. report  -- outputs/sweeps/<tag>/report.md with the leaderboard, the top configs, and what to try next.

The holdout years (2019-2025) are never touched here. Confirm a short pre-registered list with
  python -m tournament.layer2 --layer1 outputs/layer_1/<tag>_gpu*.parquet --holdout --rules ...
which logs every look.

Usage: python -m tournament.sweep --tag s1 [--grid quick|full] [--skip-audit]
       python -m tournament.sweep --tag s1 --status      progress + best so far (also appended live to <repo>/logs.txt)
       python -m tournament.sweep --tag s1 --cut         stop it, finish with what is done, export outputs/sweeps/s1/best.json
"""

import argparse
import itertools
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from infra import config as C
from tournament import contract as F
from pipeline.run import year_metrics

PY = sys.executable


def grid(kind: str, G: dict[str, list[str]]) -> list[dict]:
    """Feature-group x model grid. Groups present in the registry only, so new sources join automatically."""
    # every non-Torvik group the contract knows about, in contract order -- a new source is swept automatically
    ext_groups = [g for g in G if g not in F.TORVIK and g not in ("pipeline", "all_torvik")]
    base_torvik = ["all_torvik", "-cat"]
    cfgs = []
    # anchors: pipeline as published; Torvik-only by group; the widest set; college-only vs everyone; joint vs segmented
    cfgs.append({"name": "pipeline stack", "stack": [{"model": "tabfm_cls", "bins": 5}, {"model": "tabfm", "label": "zscore"}], "features": ["pipeline"]})
    cfgs.append({"name": "torvik rate+adv+bio tabfm_cls", "model": "tabfm_cls", "bins": 5, "features": ["rate", "adv", "bio"]})
    cfgs.append({"name": "torvik all groups tabfm_cls", "model": "tabfm_cls", "bins": 5, "features": ["rate", "adv", "count", "shotloc", "bio", "cat"]})
    cfgs.append({"name": "core+traj+phys exaone_cls", "model": "exaone_cls", "bins": 3, "features": ["core", "traj", "phys"]})
    everything = base_torvik + ext_groups
    lean = ["core", "rate", "traj", "phys", "intl_fiba", "intl_z", "tctx"]
    # Curated broad set: enough signal diversity without feeding every sparse source to every model.
    curated = ["all_torvik", "-cat", "traj", "phys", "intl_pro", "intl_fiba", "combine", "tctx", "intl_z", "mock"]
    fixed_context = {"ctx_start": 2010, "ctx_end": 2018, "label_cutoff": "full", "label": "disc85_gaussrank"}
    xgb_member = {"model": "xgb", "n_estimators": 800, "seeds": 3, "seed_start": 11}
    xgbrank_member = {"model": "xgbrank", "n_estimators": 800, "seeds": 3, "seed_start": 11}
    cat_member = {"model": "catboost", "n_estimators": 800, "seeds": 3, "seed_start": 11}
    catrank_member = {"model": "catrank", "n_estimators": 800, "seeds": 3, "seed_start": 11}
    extra_member = {"model": "extratrees", "n_estimators": 500, "seeds": 3, "seed_start": 11}
    tabicl_member = {"model": "tabicl", "n_estimators": 8}
    tabicl_robust = {"model": "tabicl", "n_estimators": 8, "model_options": {"outlier_threshold": 2.0}}
    cfgs.extend([
        {"name": "XGB fixed 2010-18", "features": curated, **fixed_context, **xgb_member},
        {"name": "XGB pairwise ranker", "features": curated, **fixed_context, **xgbrank_member},
        {"name": "CatBoost fixed 2010-18", "features": curated, **fixed_context, **cat_member},
        {"name": "CatBoost pairwise ranker", "features": curated, **fixed_context, **catrank_member},
        {"name": "ExtraTrees fixed 2010-18", "features": curated, **fixed_context, **extra_member},
        {"name": "LightGBM fixed 2010-18", "features": curated, **fixed_context,
         "model": "lgbm", "n_estimators": 800, "seeds": 3, "seed_start": 11},
        {"name": "Ridge fixed 2010-18", "features": curated, **fixed_context, "model": "ridge"},
        {"name": "TabICL fixed 2010-18", "features": curated, **fixed_context, **tabicl_member},
        {"name": "TabICL robust + XGB ranker", "features": curated, **fixed_context,
         "stack": [tabicl_robust, xgbrank_member]},
        {"name": "TabICL robust + CatBoost", "features": curated, **fixed_context,
         "stack": [tabicl_robust, cat_member]},
        {"name": "TabICL robust + Cat ranker", "features": curated, **fixed_context,
         "stack": [tabicl_robust, catrank_member]},
        {"name": "XGB + TabICL 50-50", "features": curated, **fixed_context,
         "stack": [xgb_member, tabicl_member]},
        {"name": "XGB + TabICL + TabFM", "features": curated, **fixed_context,
         "stack": [xgb_member, tabicl_member, {"model": "tabfm", "n_estimators": 32}]},
    ])
    production = ["all_torvik", "-cat", "traj", "phys", "intl_pro", "intl_fiba", "combine", "intl_z"]
    narrow = ["core", "rate", "traj", "phys", "intl_fiba", "intl_z"]
    tabldm_member = {"model": "tabldm", "n_estimators": 8}
    cfgs.extend([
        {"name": "TabLDM narrow", "features": narrow, **fixed_context, **tabldm_member},
        {"name": "TabLDM lean", "features": lean, **fixed_context, **tabldm_member},
        {"name": "TabLDM production", "features": production, **fixed_context, **tabldm_member},
        {"name": "TabLDM curated", "features": curated, **fixed_context, **tabldm_member},
        {"name": "TabLDM enhanced", "features": narrow, **fixed_context, **tabldm_member,
         "model_options": {"enhance_candidates": True}},
        {"name": "TabLDM direct", "features": narrow, **fixed_context, **tabldm_member,
         "model_options": {"validation": False, "k_fold": False}},
        {"name": "TabLDM 16 estimators", "features": narrow, **fixed_context,
         "model": "tabldm", "n_estimators": 16},
        {"name": "TabLDM + TabICL", "features": narrow, **fixed_context,
         "stack": [tabldm_member, tabicl_member]},
    ])
    if os.environ.get("TABPFN_TOKEN"):
        tabpfn26 = {"model": "tabpfn26", "n_estimators": 8, "internal_only": True}
        tabpfn3 = {"model": "tabpfn3", "n_estimators": 8, "internal_only": True}
        cfgs.extend([
            {"name": "TabPFN 2.6 internal", "features": lean, **fixed_context, **tabpfn26},
            {"name": "TabPFN 2.6 production internal", "features": production, **fixed_context, **tabpfn26},
            {"name": "TabPFN 3 internal", "features": lean, **fixed_context, **tabpfn3},
            {"name": "TabPFN 3 production internal", "features": production, **fixed_context, **tabpfn3},
            {"name": "TabPFN 3 medium-data internal", "features": lean, **fixed_context, **tabpfn3,
             "model_options": {"model_path": "tabpfn-v3-regressor-v3_20260417_mediumdata.ckpt"}},
            {"name": "TabPFN 3 OOD internal", "features": lean, **fixed_context, **tabpfn3,
             "model_options": {"model_path": "tabpfn-v3-regressor-v3_20260506_ood.ckpt"}},
            {"name": "TabPFN 3 + TabICL internal", "features": lean, **fixed_context, "internal_only": True,
             "stack": [tabpfn3, tabicl_member]},
        ])
    tabicl_feature_sets = {
        "lean": lean,
        "production": production,
        "no mocks": curated + ["-mock"],
        "plus shrunk": curated + ["shrunk"],
        "plus game": curated + ["game"],
        "plus hoopR": curated + ["hoopr"],
        "wide": everything,
    }
    for name, features in tabicl_feature_sets.items():
        cfgs.append({"name": f"TabICL {name}", "features": features, **fixed_context, **tabicl_member})
    for start in (2003, 2007):
        cfgs.append({"name": f"TabICL fixed {start}-18", "features": curated, **fixed_context,
                     "ctx_start": start, **tabicl_member})
    cfgs.extend([
        {"name": "TabICL causal all history", "features": curated, "label": "disc85_gaussrank", **tabicl_member},
        {"name": "TabICL causal last 9", "features": curated, "label": "disc85_gaussrank",
         "ctx_last": 9, **tabicl_member},
        {"name": "TabICL full gaussrank", "features": curated, **fixed_context,
         "label": "gaussrank", **tabicl_member},
        {"name": "TabICL full rank", "features": curated, **fixed_context,
         "label": "rank", **tabicl_member},
        {"name": "TabICL norm power", "features": curated, **fixed_context,
         "model_options": {"norm_methods": "power"}, **tabicl_member},
        {"name": "TabICL norm quantile", "features": curated, **fixed_context,
         "model_options": {"norm_methods": "quantile"}, **tabicl_member},
        {"name": "TabICL norm none", "features": curated, **fixed_context,
         "model_options": {"norm_methods": "none"}, **tabicl_member},
        {"name": "TabICL outlier 2", "features": curated, **fixed_context,
         "model_options": {"outlier_threshold": 2.0}, **tabicl_member},
        {"name": "TabICL outlier 8", "features": curated, **fixed_context,
         "model_options": {"outlier_threshold": 8.0}, **tabicl_member},
        {"name": "TabICL random shuffle", "features": curated, **fixed_context,
         "model_options": {"feat_shuffle_method": "random"}, **tabicl_member},
        {"name": "TabICL 16 estimators", "features": curated, **fixed_context,
         "model": "tabicl", "n_estimators": 16},
        {"name": "TabICL 32 estimators", "features": curated, **fixed_context,
         "model": "tabicl", "n_estimators": 32},
    ])
    for topk in (40, 60, 80, 100, 125, 150):
        cfgs.append({"name": f"TabICL top {topk}", "features": curated, **fixed_context,
                     "feature_topk": topk, **tabicl_member})
    for coverage in (0.05, 0.15, 0.30, 0.50):
        cfgs.append({"name": f"TabICL coverage {coverage:.2f}", "features": curated, **fixed_context,
                     "min_coverage": coverage, **tabicl_member})
    cfgs.extend([
        {"name": "TabICL top 80 outlier 2", "features": curated, **fixed_context,
         "feature_topk": 80, "model_options": {"outlier_threshold": 2.0}, **tabicl_member},
        {"name": "TabICL top 100 norm power", "features": curated, **fixed_context,
         "feature_topk": 100, "model_options": {"norm_methods": "power"}, **tabicl_member},
        {"name": "TabLDM top 80", "features": curated, **fixed_context,
         "feature_topk": 80, **tabldm_member},
        {"name": "TabLDM top 100", "features": curated, **fixed_context,
         "feature_topk": 100, **tabldm_member},
    ])
    # Context sensitivity is cheap to map with XGBoost before spending foundation-model compute on a winner.
    for last in (6, 9, 12):
        cfgs.append({"name": f"XGB causal last {last}", "features": curated, "label": "disc85_gaussrank",
                     "ctx_last": last, **xgb_member})
    cfgs.extend([
        {"name": "XGB causal all history", "features": curated, "label": "disc85_gaussrank", **xgb_member},
        {"name": "XGB causal start 2010", "features": curated, "label": "disc85_gaussrank",
         "ctx_start": 2010, **xgb_member},
        {"name": "XGB causal mature 5", "features": curated, "label": "disc85_gaussrank",
         "ctx_min_seasons": 5, **xgb_member},
    ])
    if kind == "full":
        # Balanced, deterministic block masks prevent the hand-authored feature sets from monopolising the search.
        rng = random.Random(20260905)
        seen = {g: 0 for g in ext_groups}
        for i in range(24):
            size = 2 + i % 6
            chosen = sorted(ext_groups, key=lambda g: (seen[g], rng.random()))[:size]
            for g in chosen:
                seen[g] += 1
            model = ("tabicl", "tabldm", "xgb")[i % 3]
            member = tabicl_member if model == "tabicl" else tabldm_member if model == "tabldm" else xgb_member
            cfgs.append({"name": f"feature mask {i + 1:02d} {model}", "features": base_torvik + chosen,
                         **fixed_context, **member})
    # defaults that won rounds 1-2: rank labels for the regressor, 3 bins for the classifiers
    for model, extra in [("tabfm", {"label": "rank"}), ("tabfm_cls", {"bins": 3}), ("exaone_cls", {"bins": 3})]:
        cfgs.append({"name": f"ALL {model}", "model": model, "features": everything, **extra})
    cfgs.append({"name": "ALL tabfm segment", "model": "tabfm", "label": "rank", "features": everything, "segment": True})
    cfgs.append({"name": "STACK ALL rank + lean cls3", "stack": [{"model": "tabfm", "label": "rank"}, {"model": "tabfm_cls", "bins": 3, "features": lean}], "features": everything})
    cfgs.append({"name": "LEAN tabfm rank", "model": "tabfm", "label": "rank", "features": lean})
    cfgs.append({"name": "LEAN tabfm_cls b3", "model": "tabfm_cls", "bins": 3, "features": lean})
    # leave-one-group-out on the widest set (what each source is worth), tabfm rank
    for g in ext_groups:
        cfgs.append({"name": f"ALL -{g} tabfm rank", "model": "tabfm", "label": "rank", "features": everything + [f"-{g}"]})
    # add-one-group on the pipeline set (what each source adds on its own), tabfm rank; narrow sets for exaone
    new_groups = [g for g in ext_groups if g not in ("traj", "phys", "a_box", "j_bio", "j_team", "j_box", "j_shot", "j_aau", "j_event")]
    for g in new_groups:
        cfgs.append({"name": f"pipeline +{g} tabfm rank", "model": "tabfm", "label": "rank", "features": ["pipeline", g]})
        cfgs.append({"name": f"core+traj+phys +{g} exaone_cls", "model": "exaone_cls", "bins": 3, "features": ["core", "traj", "phys", g]})
    if kind == "full":
        for b in (4, 5):
            cfgs.append({"name": f"ALL tabfm_cls b{b}", "model": "tabfm_cls", "bins": b, "features": everything})
        cfgs.append({"name": "ALL tabfm zscore", "model": "tabfm", "label": "zscore", "features": everything})
        cfgs.append({"name": "STACK ALL rank + ALL cls3 + exa lean", "stack": [{"model": "tabfm", "label": "rank"}, {"model": "tabfm_cls", "bins": 3},
                                                                              {"model": "exaone_cls", "bins": 3, "features": lean}], "features": everything})
        for g1, g2 in itertools.combinations([g for g in new_groups if g in ("game", "tctx", "intl_z", "combine", "hoopr", "transfers")], 2):
            cfgs.append({"name": f"LEAN +{g1} +{g2} tabfm rank", "model": "tabfm", "label": "rank", "features": lean + [g1, g2]})
    return cfgs


def shard(cfgs: list[dict], n: int) -> list[list[dict]]:
    """Round-robin by estimated cost so the GPUs finish together (exaone is ~10x cheaper than tabfm)."""
    model_cost = lambda m: (0.1 if m.startswith("exaone") or m in {"ridge", "lgbm"}
                            else 0.2 if m in {"xgb", "xgbrank", "extratrees"}
                            else 0.4 if m in {"catboost", "catrank"}
                            else 1.5 if m.startswith("tabpfn") else 1.2 if m == "tabldm" else 1.0)
    cost = lambda c: sum(model_cost(m.get("model", c.get("model", "tabfm"))) for m in (c.get("stack") or [c]))
    shards, load = [[] for _ in range(n)], [0.0] * n
    for c in sorted(cfgs, key=cost, reverse=True):
        i = load.index(min(load))
        shards[i].append(c)
        load[i] += cost(c)
    return shards


def finish(tag: str, out: Path, cfgs: list[dict], t0: float, l1_paths: list[Path] | None = None, years: str = "context"):
    """Layer 2 on whatever layer-1 predictions exist for this tag, leaderboard, report.md, best.json."""
    from tournament.layer1 import YEARS, config_sig
    table = pd.read_parquet(C.PROC / "draft_table.parquet")
    G = F.validate(table, verbose=False)
    if l1_paths is None:  # gather from the per-config cache: every finished config is there, whether or not its shard completed
        cache = C.OUT / "layer_1" / "cache"
        parts = []
        for c in cfgs:
            p = cache / f"{config_sig(c, years, 0, table, G)}.parquet"
            if p.exists():
                parts.append(pd.read_parquet(p).assign(config=c["name"], years=years))
        if not parts:  # no cached predictions (workers predate the cache): fall back to the scored summary for best.json
            summary = pd.read_csv(C.OUT / "layer_1" / "summary.csv")
            summary = summary[summary.tag.astype(str).str.startswith(tag)].drop_duplicates("config", keep="last").sort_values("spearman", ascending=False)
            if summary.empty:
                print("nothing finished yet for", tag)
                return
            print(summary[["config", "model", "n_feats", "spearman", "spearman_nba", "wc14", "wins"]].round(3).to_string(index=False))
            best_cfg = next(c for c in cfgs if c["name"] == summary.config.iloc[0])
            json.dump({"tag": tag, "finished": int(len(summary)), "of": len(cfgs), "best_layer1": best_cfg, "best_layer1_spearman": float(summary.spearman.iloc[0]),
                       "scouts_spearman": float(summary.spearman_nba.iloc[0]), "layer2_top_rules": [], "note": "no cached predictions; layer 2 skipped"},
                      open(out / "best.json", "w"), indent=1)
            print("best.json:", out / "best.json")
            return
        l1 = pd.concat(parts, ignore_index=True)
        cut_path = C.OUT / "layer_1" / f"{tag}_cut.parquet"
        l1.to_parquet(cut_path, index=False)
        l1_paths = [cut_path]
    pred = pd.concat([pd.read_parquet(p) for p in l1_paths], ignore_index=True)
    done = set(pred.config.unique())
    summary_path = C.OUT / "layer_1" / "summary.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    if not summary.empty:
        summary = summary[summary.tag.astype(str).str.startswith(tag)].drop_duplicates("config", keep="last")
    have = set(summary.config) if "config" in summary else set()
    recovered = []
    for cfg in (c for c in cfgs if c["name"] in done - have):
        rows = pred[(pred.config == cfg["name"]) & (pred.member == -1)]
        per_year = []
        for _, scored in rows.groupby("draft_year"):
            scored = scored[["bbref_id", "draft_year", "score"]].merge(
                table[["bbref_id", "draft_year", C.TARGET, "pick"]], on=["bbref_id", "draft_year"]
            )
            per_year.append(year_metrics(scored, scored.score.values))
        metrics = pd.DataFrame(per_year)
        members = cfg.get("stack") or [cfg]
        model = "+".join(m.get("model", cfg.get("model", "tabfm")) for m in members)
        feats = set(sum((F.resolve(m.get("features", cfg.get("features")), G, table.columns) for m in members), []))
        recovered.append({
            "tag": tag, "years": years, "config": cfg["name"], "model": model, "n_feats": len(feats),
            "spearman": metrics.spearman.mean(), "spearman_nba": metrics.spearman_nba.mean(),
            "wc14": metrics["war_captured_pct@14"].mean(), "wc30": metrics["war_captured_pct@30"].mean(),
            "wins": int((metrics.spearman > metrics.spearman_nba).sum()), "secs": float("nan"),
        })
    summary = pd.concat([summary, pd.DataFrame(recovered)], ignore_index=True)
    summary = summary[summary.config.isin(done)].sort_values("spearman", ascending=False)
    print(f"\n== layer 1: {len(summary)} of {len(cfgs)} configs finished ==")
    print(summary[["config", "model", "n_feats", "spearman", "spearman_nba", "wc14", "wins"]].round(3).to_string(index=False))
    top = summary.config.head(8).tolist()
    print("\n== layer 2 (context) ==", flush=True)
    l2 = subprocess.run([PY, "-m", "tournament.layer2", "--layer1", *map(str, l1_paths), "--configs", *top, "--years", years],
                        capture_output=True, text=True)
    l2_head = "\n".join(l2.stdout.splitlines()[:45])
    print(l2_head or l2.stderr[-2000:])
    # best.json: the best pure layer-1 config (what pipeline.run should publish) and the best rules by name
    best_cfg = next(c for c in cfgs if c["name"] == summary.config.iloc[0])
    rules = []
    for line in l2.stdout.splitlines():
        m = re.match(r"^\s*(.*?)\s+(True|False)\s+(True|False)\s+[-+]?\d+\.\d+", line)
        if m:
            rules.append({"name": m.group(1), "uses_pick": m.group(2) == "True", "robust": m.group(3) == "True"})
    best_rule = next((r["name"] for r in rules if not r["uses_pick"] and r["robust"]), best_cfg["name"])
    best = {"tag": tag, "finished": int(len(summary)), "of": len(cfgs), "best_layer1": best_cfg, "best_layer1_spearman": float(summary.spearman.iloc[0]),
            "scouts_spearman": float(summary.spearman_nba.iloc[0]), "best_layer2_non_market": best_rule,
            "context_years": YEARS[years], "layer2_top_rules": [r["name"] for r in rules[:8]], "layer1_paths": [str(p) for p in l1_paths]}
    json.dump(best, open(out / "best.json", "w"), indent=1)
    with open(out / "report.md", "w") as f:
        f.write(f"# sweep {tag}\n\n{len(summary)} of {len(cfgs)} configs, {time.time() - t0:.0f}s. "
                f"Context years {YEARS[years][0]}-{YEARS[years][-1]}; holdout untouched.\n\n")
        f.write("## layer 1\n\n```\n" + summary[["config", "model", "n_feats", "spearman", "spearman_nba", "wc14", "wins"]].round(3).to_string(index=False) + "\n```\n\n")
        f.write("## layer 2 (top-8 layer-1 configs)\n\n```\n" + l2_head + "\n```\n")
    print(f"\nbest.json: {out / 'best.json'}\nreport:    {out / 'report.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--grid", default="quick", choices=["quick", "full"])
    ap.add_argument("--years", default="context6", choices=["context", "context6", "context7", "context8"])
    ap.add_argument("--skip-audit", action="store_true")
    ap.add_argument("--configs", help="optional JSON file of configs instead of the built-in grid")
    ap.add_argument("--minutes", type=float, help="stop GPU workers after this many minutes and finish from completed config caches")
    ap.add_argument("--cut", action="store_true", help="stop the running sweep for --tag, finish with what is done, export best.json")
    ap.add_argument("--status", action="store_true", help="print progress + best so far for --tag from logs.txt")
    a = ap.parse_args()
    out = C.OUT / "sweeps" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if a.status or a.cut:
        lines = [ln for ln in (C.ROOT / "logs.txt").read_text().splitlines() if ln.startswith(f"[{a.tag} ")] if (C.ROOT / "logs.txt").exists() else []
        print("\n".join(lines[-12:]) if lines else f"no progress logged yet for {a.tag}")
        if a.status:
            return
        import os, signal
        pat = f"tournament.layer1 .*tag {a.tag}_gpu"  # the GPU workers of this sweep only
        pids = [int(p) for p in subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True).stdout.split()]
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        time.sleep(3)
        for pid in subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True).stdout.split():
            os.kill(int(pid), signal.SIGKILL)
        print(f"stopped {len(pids)} GPU worker(s) for {a.tag}")
        cfgs = sum((json.load(open(p)) for p in sorted(out.glob("configs_gpu*.json"))), [])
        finish(a.tag, out, cfgs, t0, years=a.years)
        return

    if not a.skip_audit:
        print("== 1. audit ==", flush=True)
        subprocess.run([PY, "-m", "infra.builders.materialize"], check=True)
        subprocess.run([PY, "-m", "infra.dataset"], check=True, stdout=open(out / "dataset.log", "w"))
        subprocess.run([PY, "-m", "tournament.audit", "--data-only"], check=True)

    table = pd.read_parquet(C.PROC / "draft_table.parquet")
    G = F.validate(table, verbose=False)
    cfgs = json.load(open(a.configs)) if a.configs else grid(a.grid, G)
    F.assert_covered(cfgs, G)
    n_gpu = torch.cuda.device_count()
    assert n_gpu > 0, "no GPU"
    shards = shard(cfgs, n_gpu)
    print(f"\n== 2. layer 1: {len(cfgs)} configs on {n_gpu} GPU(s) ==", flush=True)
    procs = []
    for i, s in enumerate(shards):
        cfg_path = out / f"configs_gpu{i}.json"
        json.dump(s, open(cfg_path, "w"), indent=1)
        log = open(out / f"layer1_gpu{i}.log", "w")
        procs.append(subprocess.Popen([PY, "-m", "tournament.layer1", "--device", f"cuda:{i}", "--years", a.years, "--tag", f"{a.tag}_gpu{i}",
                                       "--configs", str(cfg_path)], stdout=log, stderr=subprocess.STDOUT))
    deadline = time.monotonic() + 60 * a.minutes if a.minutes else None
    while any(p.poll() is None for p in procs):
        if deadline is not None and time.monotonic() >= deadline:
            running = [p for p in procs if p.poll() is None]
            for p in running:
                p.terminate()
            limit = time.monotonic() + 10
            while any(p.poll() is None for p in running) and time.monotonic() < limit:
                time.sleep(0.2)
            for p in running:
                if p.poll() is None:
                    p.kill()
            print(f"\ntime budget reached: stopped {len(running)} worker(s), finishing completed configs", flush=True)
            break
        time.sleep(1)
    for p in procs:
        p.wait()
    l1 = [p for p in (C.OUT / "layer_1" / f"{a.tag}_gpu{i}.parquet" for i in range(n_gpu)) if p.exists()]
    finish(a.tag, out, cfgs, t0, l1_paths=l1 or None, years=a.years)


if __name__ == "__main__":
    main()
