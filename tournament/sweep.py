"""The one sweep. Run it, read the report, improve a builder or a rule, run it again.

  1. audit   -- rebuild the table from every materialised source, validate the feature registry, coverage per group
                for college and international draftees, leakage assertions.
  2. layer 1 -- a grid over feature groups x models x label framing, sharded across every GPU as parallel processes,
                on the validation years (2013-2017). Every config's per-player predictions land in outputs/layer_1/.
  3. layer 2 -- rules on top (blends, upside/floor readouts, market) on validation; leaderboard.
  4. report  -- outputs/sweeps/<tag>/report.md with the leaderboard, the top configs, and what to try next.

The test years (2018-2025) are never touched here. Confirm a short pre-registered list with
  python -m tournament.layer2 --layer1 outputs/layer_1/<tag>_gpu*.parquet --test --rules ...
which logs every look.

Usage: python -m tournament.sweep --tag s1 [--grid quick|full] [--skip-audit]
       python -m tournament.sweep --tag s1 --status      progress + best so far (also appended live to <repo>/logs.txt)
       python -m tournament.sweep --tag s1 --cut         stop it, finish with what is done, export outputs/sweeps/s1/best.json
"""

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from infra import config as C
from tournament import contract as F

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
    # defaults that won rounds 1-2: rank labels for the regressor, 3 bins for the classifiers
    for model, extra in [("tabfm", {"label": "rank"}), ("tabfm_cls", {"bins": 3}), ("exaone_cls", {"bins": 3})]:
        cfgs.append({"name": f"ALL {model}", "model": model, "features": everything, **extra})
        cfgs.append({"name": f"ALL {model} college-only", "model": model, "features": everything, "population": "college", **extra})
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
    cost = lambda c: sum(0.1 if m.get("model", c.get("model", "tabfm")).startswith("exaone") else 1.0 for m in (c.get("stack") or [c]))
    shards, load = [[] for _ in range(n)], [0.0] * n
    for c in sorted(cfgs, key=cost, reverse=True):
        i = load.index(min(load))
        shards[i].append(c)
        load[i] += cost(c)
    return shards


def finish(tag: str, out: Path, cfgs: list[dict], t0: float, l1_paths: list[Path] | None = None):
    """Layer 2 on whatever layer-1 predictions exist for this tag, leaderboard, report.md, best.json."""
    from tournament.layer1 import config_sig
    table = pd.read_parquet(C.PROC / "draft_table.parquet")
    G = F.validate(table, verbose=False)
    if l1_paths is None:  # gather from the per-config cache: every finished config is there, whether or not its shard completed
        cache = C.OUT / "layer_1" / "cache"
        parts = []
        for c in cfgs:
            p = cache / f"{config_sig(c, 'val', 0, table, G)}.parquet"
            if p.exists():
                parts.append(pd.read_parquet(p).assign(config=c["name"], years="val"))
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
    summary = pd.read_csv(C.OUT / "layer_1" / "summary.csv")
    summary = summary[summary.tag.astype(str).str.startswith(tag)].drop_duplicates("config", keep="last").sort_values("spearman", ascending=False)
    done = set(pd.concat([pd.read_parquet(p, columns=["config"]) for p in l1_paths]).config.unique())
    summary = summary[summary.config.isin(done)]
    print(f"\n== layer 1: {len(summary)} of {len(cfgs)} configs finished ==")
    print(summary[["config", "model", "n_feats", "spearman", "spearman_nba", "wc14", "wins"]].round(3).to_string(index=False))
    top = summary.config.head(8).tolist()
    print("\n== layer 2 (validation) ==", flush=True)
    l2 = subprocess.run([PY, "-m", "tournament.layer2", "--layer1", *map(str, l1_paths), "--configs", *top], capture_output=True, text=True)
    l2_head = "\n".join(l2.stdout.splitlines()[:45])
    print(l2_head or l2.stderr[-2000:])
    # best.json: the best pure layer-1 config (what validation.run should publish) and the best rules by name
    best_cfg = next(c for c in cfgs if c["name"] == summary.config.iloc[0])
    rules = [ln.split("  ")[0].strip() for ln in l2.stdout.splitlines() if ln.strip() and ln.lstrip().split("  ")[0].strip() not in ("rule",)]
    best = {"tag": tag, "finished": int(len(summary)), "of": len(cfgs), "best_layer1": best_cfg, "best_layer1_spearman": float(summary.spearman.iloc[0]),
            "scouts_spearman": float(summary.spearman_nba.iloc[0]), "layer2_top_rules": rules[2:8], "layer1_paths": [str(p) for p in l1_paths]}
    json.dump(best, open(out / "best.json", "w"), indent=1)
    with open(out / "report.md", "w") as f:
        f.write(f"# sweep {tag}\n\n{len(summary)} of {len(cfgs)} configs, {time.time() - t0:.0f}s. Validation years 2013-2017; test untouched.\n\n")
        f.write("## layer 1\n\n```\n" + summary[["config", "model", "n_feats", "spearman", "spearman_nba", "wc14", "wins"]].round(3).to_string(index=False) + "\n```\n\n")
        f.write("## layer 2 (top-8 layer-1 configs)\n\n```\n" + l2_head + "\n```\n")
    print(f"\nbest.json: {out / 'best.json'}\nreport:    {out / 'report.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--grid", default="quick", choices=["quick", "full"])
    ap.add_argument("--skip-audit", action="store_true")
    ap.add_argument("--configs", help="optional JSON file of configs instead of the built-in grid")
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
        finish(a.tag, out, cfgs, t0)
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
        procs.append(subprocess.Popen([PY, "-m", "tournament.layer1", "--device", f"cuda:{i}", "--years", "val", "--tag", f"{a.tag}_gpu{i}",
                                       "--configs", str(cfg_path)], stdout=log, stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()
    l1 = [p for p in (C.OUT / "layer_1" / f"{a.tag}_gpu{i}.parquet" for i in range(n_gpu)) if p.exists()]
    finish(a.tag, out, cfgs, t0, l1_paths=l1 or None)


if __name__ == "__main__":
    main()
