"""Optuna refinement for trained tabular models on the walk-forward context years only."""

import argparse
import json
import time

import numpy as np
import optuna
import pandas as pd

from infra import config as C
from infra.models import predict
from infra.war import war_target
from tournament import contract as F
from tournament.layer1 import YEARS, select_features, transform_labels
from pipeline.run import context_years, year_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()

    table = pd.read_parquet(C.PROC / "draft_table.parquet")
    seasons = pd.read_parquet(C.PROC / "season_war.parquet")
    groups = F.groups(table)
    labelled = sorted(table.loc[table.modelled & table.labelled, "draft_year"].unique())
    ext = [g for g in groups if g not in F.TORVIK and g not in {"pipeline", "all_torvik"}]
    specs = {
        "lean": ["core", "rate", "traj", "phys", "intl_fiba", "intl_z", "tctx"],
        "production": ["all_torvik", "-cat", "traj", "phys", "intl_pro", "intl_fiba", "combine", "intl_z"],
        "curated": ["all_torvik", "-cat", "traj", "phys", "intl_pro", "intl_fiba", "combine", "tctx", "intl_z", "mock"],
        "wide": ["all_torvik", "-cat", *ext],
    }

    def objective(trial: optuna.Trial) -> float:
        model = trial.suggest_categorical("model", ["xgb", "xgbrank", "catboost", "catrank", "extratrees"])
        spec = trial.suggest_categorical("features", list(specs))
        label = trial.suggest_categorical("label", ["rank", "gaussrank", "disc85_gaussrank"])
        cutoff = trial.suggest_categorical("label_cutoff", ["causal", "full"])
        start = trial.suggest_categorical("ctx_start", [2003, 2007, 2010])
        last = trial.suggest_categorical("ctx_last", [0, 6, 9, 12])
        topk = trial.suggest_categorical("feature_topk", [0, 60, 100, 150])
        coverage = trial.suggest_categorical("min_coverage", [0.0, 0.05, 0.15, 0.30])
        n_estimators = trial.suggest_int("n_estimators", 300, 1200, step=300)
        if model in {"xgb", "xgbrank"}:
            options = {
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample", 0.35, 1.0),
                "min_child_weight": trial.suggest_int("min_child", 2, 15),
                "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 20.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 5.0, log=True),
            }
        elif model in {"catboost", "catrank"}:
            options = {
                "depth": trial.suggest_int("depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            }
        else:
            options = {
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 15),
                "max_features": trial.suggest_float("max_features", 0.3, 1.0),
            }

        scores = []
        for step, year in enumerate(YEARS["context6"]):
            cy = [y for y in context_years(year, "causal", labelled) if y >= start]
            if last:
                cy = cy[-last:]
            ctx = table[table.modelled & table.labelled & table.draft_year.isin(cy)].copy()
            through = C.LAST_SEASON if cutoff == "full" else year
            ctx[C.TARGET] = war_target(ctx, seasons, through=through)[C.TARGET].values
            ctx = transform_labels(ctx, label, seasons, through)
            feats = F.resolve(specs[spec], groups, table.columns)
            feats = select_features(ctx, feats, coverage, topk or None)
            pool = table[table.draft_year == year]
            score, _ = predict(model, ctx, pool, feats, "cpu", trial.number, n_estimators=n_estimators,
                               model_options=options)
            scores.append(year_metrics(pool, score)["spearman"])
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("mean_spearman", float(np.mean(scores)))
        return float(np.mean(scores) - 0.1 * np.std(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=0),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=12, n_warmup_steps=3),
    )
    study.optimize(objective, timeout=60 * a.minutes, n_jobs=a.jobs, show_progress_bar=False)
    p = study.best_params
    config = {
        "name": f"Optuna {a.tag}", "model": p["model"], "features": specs[p["features"]],
        "label": p["label"], "label_cutoff": p["label_cutoff"], "ctx_start": p["ctx_start"], "ctx_end": 2018,
        "n_estimators": p["n_estimators"], "seeds": 3, "seed_start": 11,
        "model_options": {k: v for k, v in p.items() if k not in {
            "model", "features", "label", "label_cutoff", "ctx_start", "ctx_last", "feature_topk",
            "min_coverage", "n_estimators",
        }},
    }
    if p["ctx_last"]:
        config["ctx_last"] = p["ctx_last"]
    if p["feature_topk"]:
        config["feature_topk"] = p["feature_topk"]
    if p["min_coverage"]:
        config["min_coverage"] = p["min_coverage"]
    out = C.OUT / "optuna"
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "created": time.time(), "trials": len(study.trials), "objective": study.best_value,
        "context_spearman": study.best_trial.user_attrs.get("mean_spearman"), "config": config,
    }
    (out / f"{a.tag}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
