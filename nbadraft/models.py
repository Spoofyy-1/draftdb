"""Scorers: given context (labelled draftees) and a test draft class, return a value score per player.

TabFM is zero-shot: `fit` only stores the context rows; the forward pass does the learning
(in-context). Baselines are ridge and LightGBM trained on the same context rows.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from nbadraft.config import MODEL_DIR
from nbadraft.dataset import CATEGORICAL_FEATURES, FEATURES, MARKET_FEATURE, NUMERIC_FEATURES

TARGET = "peak_war"
MODELS = ["ridge", "lgbm", "tabfm", "tabfm_ens", "tabfm+market", "tabfm_ens+market"]

_TABFM = {}


def _tabfm(device):
    if device not in _TABFM:
        from tabfm import tabfm_v1_0_0_pytorch as v1
        _TABFM[device] = v1.load(model_type="regression", checkpoint_path=str(MODEL_DIR), device=device)
    return _TABFM[device]


def _features(name):
    return FEATURES + [MARKET_FEATURE] if name.endswith("+market") else FEATURES


def fit_predict(name: str, ctx: pd.DataFrame, test: pd.DataFrame, device: str = "cpu", seed: int = 0) -> np.ndarray:
    feats = _features(name)
    X, y, Xt = ctx[feats], ctx[TARGET].values, test[feats]
    base = name.replace("+market", "")

    if base == "ridge":
        nums = [f for f in feats if f not in CATEGORICAL_FEATURES]
        pre = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), nums),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), CATEGORICAL_FEATURES),
        ])
        return make_pipeline(pre, RidgeCV(alphas=np.logspace(-1, 3, 20))).fit(X, y).predict(Xt)

    if base == "lgbm":
        import lightgbm as lgb
        cat = lambda d: d.assign(**{c: d[c].astype("category") for c in CATEGORICAL_FEATURES})
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.02, num_leaves=7, min_child_samples=15,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, random_state=seed, verbose=-1)
        return m.fit(cat(X), y).predict(cat(Xt))

    from tabfm import TabFMRegressor
    model = _tabfm(device)
    if base == "tabfm":
        reg = TabFMRegressor(model=model, n_estimators=32, batch_size=None, random_state=seed)
    elif base == "tabfm_ens":  # blog "TabFM-Ensemble": feature crosses + SVD features + NNLS-weighted 32-way ensemble
        reg = TabFMRegressor.ensemble(model, batch_size=None, random_state=seed)
    else:
        raise ValueError(name)
    return np.asarray(reg.fit(X, y).predict(Xt), dtype=float)
