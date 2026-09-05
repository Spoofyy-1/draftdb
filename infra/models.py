"""The model zoo. One place that knows how to fit-and-predict every estimator, used by both the published run
(pipeline/run.py) and the tournament (tournament/layer1.py).

  predict(model, ctx, pool, feats, ...) -> (score, proba)   the zoo: any model, any feature list, bin distribution for classifiers
  fit_predict(name, ctx, pool, ...)     -> score            the published-run API: model name -> pipeline features and stacks

In-context models (TabFM, EXAONE) do no training: `fit` stores the context rows and the forward pass does the learning.
  tabfm       Google TabFM 1.0 regression head (1.65B) on the transformed WAR target
  tabfm_cls   TabFM classification head on CLS_BINS within-class quantile bins, read out as the expected bin
  exaone      LG EXAONE-Tabular regressor (21M); wants narrow feature sets
  exaone_cls  EXAONE-Tabular classifier on within-class bins, expected-bin readout
  xgb         shallow, heavily regularised gradient booster (one seed per call)
  xgbrank     pairwise XGBoost ranker grouped by draft class
  catboost    CatBoost regressor
  catrank     pairwise CatBoost ranker grouped by draft class
  extratrees  randomised-tree regression baseline
  tabicl      TabICL v2 regressor; paired with xgb in a diverse rank stack
  tabicl_cls  TabICL v2 classifier on within-class bins, expected-bin readout
  tabldm      Xiaomi-TabLDM regressor
  tabpfn26    TabPFN 2.6 regressor (internal evaluation only; restricted model license)
  tabpfn3     TabPFN 3 regressor (internal evaluation only; restricted model license)
  ridge/lgbm  trained baselines on the same context rows
  stack       rank-average of tabfm (all features, rank label) and tabfm_cls (3 bins, LEAN_FEATURES) -- the two framings
              err differently; the pair was the best pure configuration of tournament round 2
  +momentum   gives the stack's regression member archived pre-draft mock histories; explicitly not labelled pure AI
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from infra.config import MODEL_DIR, TARGET
from infra.dataset import CATEGORICAL_FEATURES, FEATURES, LEAN_FEATURES, MOMENTUM_FEATURES

MODELS = ["ridge", "lgbm", "xgb", "xgbrank", "catboost", "catrank", "extratrees", "tabicl", "tabldm", "tabpfn26", "tabpfn3", "tabfm", "tabfm_cls", "exaone", "exaone_cls", "stack", "stack+momentum", "stack+consensus", "tabfm_ens"]
ZOO = {"tabfm", "tabfm_cls", "tabfm_ens", "exaone", "exaone_cls", "xgb", "xgbrank", "catboost", "catrank", "extratrees", "tabicl", "tabicl_cls", "tabldm", "tabpfn26", "tabpfn3", "ridge", "lgbm"}
CLS_BINS = 5
# Consensus blends with mock drafts published before draft night, never with the actual pick.
CONSENSUS_WEIGHT = 0.4
CONSENSUS_FEATURE = "mock_rank_consensus"  # players in no pre-draft mock rank last (61)
# TabFM ensemble members scored per batch: identical predictions, a fraction of the peak memory (360 features x ~1200
# context rows in one shot needs ~70 GB).
TABFM_BATCH = 8

_LOADED: dict = {}


def _tabfm(task, device):
    if (task, device) not in _LOADED:
        from tabfm import tabfm_v1_0_0_pytorch as v1
        _LOADED[(task, device)] = v1.load(model_type=task, checkpoint_path=str(MODEL_DIR), device=device)
    return _LOADED[(task, device)]


def _exaone(task, device):
    if ("exaone", task, device) not in _LOADED:
        from exaonetabular import EXAONETabularClassifier, EXAONETabularRegressor
        cls = EXAONETabularClassifier if task == "classification" else EXAONETabularRegressor
        _LOADED[("exaone", task, device)] = cls.from_pretrained(device=device)
    return _LOADED[("exaone", task, device)]


def bins(ctx: pd.DataFrame, k: int = CLS_BINS) -> np.ndarray:
    """Within-class quantile bin of the label: 0 = worst ... k-1 = best. The classifier target."""
    pct = ctx.groupby("draft_year")[TARGET].rank(pct=True)
    return np.minimum(np.floor(pct.values * k), k - 1).astype(int)


def _numeric(ctx, pool, feats):
    """EXAONE wants float arrays: ordinal-code non-numeric columns over the union of both frames (an unsupervised
    relabelling of category names; nothing about the target leaks), keep NaN as NaN."""
    both = pd.concat([ctx[feats], pool[feats]], keys=["c", "t"])
    for c in feats:
        if not pd.api.types.is_numeric_dtype(both[c]):
            codes = pd.Categorical(both[c]).codes.astype(float)
            both[c] = np.where(codes < 0, np.nan, codes)
    both = both.astype(float)
    return both.loc["c"].values, both.loc["t"].values


def _dist(p, classes, k):
    out = np.zeros((p.shape[0], k))
    out[:, np.asarray(classes, dtype=int)] = p
    return out


def predict(model: str, ctx: pd.DataFrame, pool: pd.DataFrame, feats: list[str], device="cpu", seed=0, k=CLS_BINS,
            n_estimators=32, model_options: dict | None = None):
    """Fit on ctx, score pool. Returns (score, proba): proba is the (n_pool, k) bin distribution for classifiers, else None."""
    if "pick" in feats:
        raise ValueError("actual draft position is scoring metadata, never a model input")
    model_options = model_options or {}
    X, y, Xt = ctx[feats], ctx[TARGET].values, pool[feats]
    if model == "tabfm":
        from tabfm import TabFMRegressor
        reg = TabFMRegressor(model=_tabfm("regression", device), n_estimators=n_estimators, batch_size=TABFM_BATCH, random_state=seed)
        return np.asarray(reg.fit(X, y).predict(Xt), dtype=float), None
    if model == "tabfm_ens":  # blog "TabFM-Ensemble": feature crosses + SVD features + NNLS-weighted 32-way ensemble
        from tabfm import TabFMRegressor
        return np.asarray(TabFMRegressor.ensemble(_tabfm("regression", device), batch_size=TABFM_BATCH, random_state=seed).fit(X, y).predict(Xt), dtype=float), None
    if model == "tabfm_cls":
        from tabfm import TabFMClassifier
        clf = TabFMClassifier(model=_tabfm("classification", device), n_estimators=n_estimators, batch_size=TABFM_BATCH, random_state=seed).fit(X, bins(ctx, k))
        p = np.asarray(clf.predict_proba(Xt), dtype=float)
        return p @ np.asarray(clf.classes_, dtype=float), _dist(p, clf.classes_, k)
    if model == "exaone":
        Xc, Xn = _numeric(ctx, pool, feats)
        return np.asarray(_exaone("regression", device).fit(Xc, y.astype(float)).predict(Xn), dtype=float), None
    if model == "exaone_cls":
        Xc, Xn = _numeric(ctx, pool, feats)
        clf = _exaone("classification", device).fit(Xc, bins(ctx, k))
        p = np.asarray(clf.predict_proba(Xn), dtype=float)
        return p @ np.asarray(clf.classes_, dtype=float), _dist(p, clf.classes_, k)
    if model == "xgb":
        import xgboost as xgb
        Xc, Xn = _numeric(ctx, pool, feats)
        # Shallow, heavily regularised trees with a monotone age prior.
        mono = tuple(-1 if c in {"age_at_draft", "draft_age_x"} else 0 for c in feats)
        params = {
            "objective": "reg:squarederror", "max_depth": 3, "learning_rate": 0.02, "n_estimators": n_estimators,
            "subsample": 0.7, "colsample_bytree": 0.5, "min_child_weight": 8, "reg_lambda": 5.0, "reg_alpha": 1.0,
            "monotone_constraints": mono, "random_state": seed, "n_jobs": 8,
        }
        params.update(model_options)
        reg = xgb.XGBRegressor(**params)
        return np.asarray(reg.fit(Xc, y).predict(Xn), dtype=float), None
    if model == "xgbrank":
        import xgboost as xgb
        Xc, Xn = _numeric(ctx, pool, feats)
        order = np.argsort(ctx.draft_year.values, kind="stable")
        params = {
            "objective": "rank:pairwise", "max_depth": 3, "learning_rate": 0.02, "n_estimators": n_estimators,
            "subsample": 0.8, "colsample_bytree": 0.6, "min_child_weight": 8,
            "reg_lambda": 5.0, "reg_alpha": 1.0, "random_state": seed, "n_jobs": 8,
        }
        params.update(model_options)
        ranker = xgb.XGBRanker(**params)
        ranker.fit(Xc[order], y[order], qid=ctx.draft_year.values[order])
        return np.asarray(ranker.predict(Xn), dtype=float), None
    if model in {"catboost", "catrank"}:
        from catboost import CatBoostRanker, CatBoostRegressor
        Xc, Xn = _numeric(ctx, pool, feats)
        params = {
            "iterations": n_estimators, "depth": 5, "learning_rate": 0.03, "l2_leaf_reg": 5.0,
            "random_seed": seed, "verbose": False, "allow_writing_files": False, "thread_count": 8,
        }
        params.update(model_options)
        if "monotone" in params:  # {feature name: +1 / -1}: domain-knowledge monotonicity, by column position
            mono = params.pop("monotone")
            params["monotone_constraints"] = [int(mono.get(f, 0)) for f in feats]
        if model == "catrank":
            order = np.argsort(ctx.draft_year.values, kind="stable")
            ranker = CatBoostRanker(loss_function="YetiRankPairwise", **params)
            ranker.fit(Xc[order], y[order], group_id=ctx.draft_year.values[order])
            return np.asarray(ranker.predict(Xn), dtype=float), None
        reg = CatBoostRegressor(loss_function="RMSE", **params)
        return np.asarray(reg.fit(Xc, y).predict(Xn), dtype=float), None
    if model == "extratrees":
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.impute import SimpleImputer
        Xc, Xn = _numeric(ctx, pool, feats)
        imputer = SimpleImputer(strategy="median")
        params = {"n_estimators": n_estimators, "min_samples_leaf": 5, "max_features": 0.7, "random_state": seed, "n_jobs": 8}
        params.update(model_options)
        reg = ExtraTreesRegressor(**params)
        return np.asarray(reg.fit(imputer.fit_transform(Xc), y).predict(imputer.transform(Xn)), dtype=float), None
    if model == "tabicl":
        from tabicl import TabICLRegressor
        Xc, Xn = _numeric(ctx, pool, feats)
        params = {"device": device, "n_estimators": n_estimators, "batch_size": 8, "random_state": seed}
        params.update(model_options)
        reg = TabICLRegressor(**params)
        return np.asarray(reg.fit(Xc, y.astype(float)).predict(Xn), dtype=float), None
    if model == "tabicl_cls":  # TabICL classification head on within-class quantile bins, expected-bin readout
        from tabicl import TabICLClassifier
        Xc, Xn = _numeric(ctx, pool, feats)
        params = {"device": device, "n_estimators": n_estimators, "batch_size": 8, "random_state": seed}
        params.update(model_options)
        clf = TabICLClassifier(**params).fit(Xc, bins(ctx, k))
        p = np.asarray(clf.predict_proba(Xn), dtype=float)
        return p @ np.asarray(clf.classes_, dtype=float), _dist(p, clf.classes_, k)
    if model == "tabldm":
        from tabldm import TabLDMRegressor
        Xc, Xn = _numeric(ctx, pool, feats)
        params = {"device": device, "n_estimators": n_estimators, "batch_size": 4, "random_state": seed}
        params.update(model_options)
        reg = TabLDMRegressor(**params)
        return np.asarray(reg.fit(Xc, y.astype(float)).predict(Xn), dtype=float), None
    if model in {"tabpfn26", "tabpfn3"}:
        from tabpfn import TabPFNRegressor
        Xc, Xn = _numeric(ctx, pool, feats)
        checkpoint = {
            "tabpfn26": "tabpfn-v2.6-regressor-v2.6_default.ckpt",
            "tabpfn3": "tabpfn-v3-regressor-v3_default.ckpt",
        }[model]
        params = {
            "model_path": checkpoint, "device": device, "n_estimators": n_estimators,
            "random_state": seed, "show_progress_bar": False,
        }
        params.update(model_options)
        reg = TabPFNRegressor(**params)
        return np.asarray(reg.fit(Xc, y.astype(float)).predict(Xn), dtype=float), None
    cats = [c for c in feats if c in CATEGORICAL_FEATURES]
    if model == "ridge":
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        pre = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), [f for f in feats if f not in cats]),
                                 ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), cats)])
        return make_pipeline(pre, RidgeCV(alphas=np.logspace(-1, 3, 20))).fit(X, y).predict(Xt), None
    if model == "lgbm":
        import lightgbm as lgb
        nums = [f for f in feats if f not in cats]
        cat = lambda d: d[nums].apply(pd.to_numeric, errors="coerce").astype(float).assign(**{c: d[c].astype("category") for c in cats})
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.02, num_leaves=7, min_child_samples=15,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, random_state=seed, verbose=-1)
        return m.fit(cat(X), y).predict(cat(Xt)), None
    raise ValueError(model)


def fit_predict(name: str, ctx: pd.DataFrame, pool: pd.DataFrame, device: str = "cpu", seed: int = 0) -> np.ndarray:
    """Published-run API: a MODELS name scored on the pipeline FEATURES."""
    base = name.replace("+consensus", "")
    if base in ("stack", "stack+momentum"):
        reg_features = MOMENTUM_FEATURES if base.endswith("+momentum") else FEATURES
        score = np.mean([rankdata(predict("tabfm", ctx, pool, reg_features, device, seed)[0]),
                         rankdata(predict("tabfm_cls", ctx, pool, LEAN_FEATURES, device, seed, k=3)[0])], axis=0)
        return blend(name, score, pool)
    return np.asarray(predict(base, ctx, pool, FEATURES, device, seed)[0], dtype=float)


def blend(name: str, score: np.ndarray, pool: pd.DataFrame) -> np.ndarray:
    """Apply the pre-draft consensus rank blend named by `name` to base-model scores."""
    if name.endswith("+consensus"):
        return (1 - CONSENSUS_WEIGHT) * rankdata(score) + CONSENSUS_WEIGHT * rankdata(-pool[CONSENSUS_FEATURE].fillna(61).values)
    return np.asarray(score, dtype=float)
