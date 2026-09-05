"""The model zoo. One place that knows how to fit-and-predict every estimator, used by both the published run
(validation/run.py) and the tournament (tournament/layer1.py).

  predict(model, ctx, test, feats, ...) -> (score, proba)   the zoo: any model, any feature list, bin distribution for classifiers
  fit_predict(name, ctx, test, ...)     -> score            the published-run API: model name -> pipeline FEATURES, +market, stacks

In-context models (TabFM, EXAONE) do no training: `fit` stores the context rows and the forward pass does the learning.
  tabfm       Google TabFM 1.0 regression head (1.65B) on the transformed WAR target
  tabfm_cls   TabFM classification head on CLS_BINS within-class quantile bins, read out as the expected bin
  exaone      LG EXAONE-Tabular regressor (21M); wants narrow feature sets
  exaone_cls  EXAONE-Tabular classifier on within-class bins, expected-bin readout
  ridge/lgbm  trained baselines on the same context rows
  stack       rank-average of tabfm (all features, rank label) and tabfm_cls (3 bins, LEAN_FEATURES) -- the two framings
              err differently; the pair was the best pure configuration of tournament round 2
  +momentum   gives the stack's regression member archived pre-draft mock histories; explicitly not labelled pure AI
  +market     (1 - w) * rank(model) + w * rank(-pick): the model adjusts the scouts' consensus. Uses the actual pick,
              so it is never presented as a pure model.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from infra.config import MODEL_DIR, TARGET
from infra.dataset import CATEGORICAL_FEATURES, FEATURES, LEAN_FEATURES, MARKET_FEATURE, MOMENTUM_FEATURES

MODELS = ["ridge", "lgbm", "tabfm", "tabfm_cls", "exaone", "exaone_cls", "stack", "stack+momentum", "stack+consensus", "stack+market", "tabfm_ens", "tabfm+market"]
ZOO = {"tabfm", "tabfm_cls", "tabfm_ens", "exaone", "exaone_cls", "ridge", "lgbm"}
CLS_BINS = 5
# Blend weights: share of the OTHER order in the rank average (1 - w is the AI's share). Chosen on the validation years
# 2011-2018 for the 5-year-WAR target (0.4 beat 0.3/0.5/0.6; under peak WAR the choice was 0.6).
#   +market     blends with the ACTUAL pick: not pre-draft information; reference only.
#   +consensus  blends with the mean rank across mock drafts PUBLISHED BEFORE draft night (infra/builders/mocks.py): fully
#               pre-draft. On validation the consensus alone scores below the scouts; the blend above both.
MARKET_WEIGHT = 0.4
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


def _numeric(ctx, test, feats):
    """EXAONE wants float arrays: ordinal-code non-numeric columns over the union of both frames (an unsupervised
    relabelling of category names; nothing about the target leaks), keep NaN as NaN."""
    both = pd.concat([ctx[feats], test[feats]], keys=["c", "t"])
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


def predict(model: str, ctx: pd.DataFrame, test: pd.DataFrame, feats: list[str], device="cpu", seed=0, k=CLS_BINS, n_estimators=32):
    """Fit on ctx, score test. Returns (score, proba): proba is the (n_test, k) bin distribution for classifiers, else None."""
    X, y, Xt = ctx[feats], ctx[TARGET].values, test[feats]
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
        Xc, Xn = _numeric(ctx, test, feats)
        return np.asarray(_exaone("regression", device).fit(Xc, y.astype(float)).predict(Xn), dtype=float), None
    if model == "exaone_cls":
        Xc, Xn = _numeric(ctx, test, feats)
        clf = _exaone("classification", device).fit(Xc, bins(ctx, k))
        p = np.asarray(clf.predict_proba(Xn), dtype=float)
        return p @ np.asarray(clf.classes_, dtype=float), _dist(p, clf.classes_, k)
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


def fit_predict(name: str, ctx: pd.DataFrame, test: pd.DataFrame, device: str = "cpu", seed: int = 0) -> np.ndarray:
    """Published-run API: a MODELS name scored on the pipeline FEATURES."""
    base = name.replace("+market", "").replace("+consensus", "")
    if base in ("stack", "stack+momentum"):
        reg_features = MOMENTUM_FEATURES if base.endswith("+momentum") else FEATURES
        score = np.mean([rankdata(predict("tabfm", ctx, test, reg_features, device, seed)[0]),
                         rankdata(predict("tabfm_cls", ctx, test, LEAN_FEATURES, device, seed, k=3)[0])], axis=0)
        return blend(name, score, test)
    feats = FEATURES + ([MARKET_FEATURE] if name.endswith("+market") else [])
    return np.asarray(predict(base, ctx, test, feats, device, seed)[0], dtype=float)


def blend(name: str, score: np.ndarray, test: pd.DataFrame) -> np.ndarray:
    """Apply the '+market' / '+consensus' rank blend named by `name` to a base model's scores (no model call)."""
    if name.endswith("+market"):
        return (1 - MARKET_WEIGHT) * rankdata(score) + MARKET_WEIGHT * rankdata(-test[MARKET_FEATURE].values)
    if name.endswith("+consensus"):
        return (1 - CONSENSUS_WEIGHT) * rankdata(score) + CONSENSUS_WEIGHT * rankdata(-test[CONSENSUS_FEATURE].fillna(61).values)
    return np.asarray(score, dtype=float)
