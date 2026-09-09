"""Shared numpy stand-in learner for the smoke stubs. Ridge regression, median imputation, standardisation.

Not a model anybody should believe: it exists so that `tournament.layer1` can be run end to end on a machine that
has no CatBoost / TabICL, to prove the bridge table and the plumbing work.
"""

import numpy as np


class RidgeStandIn:
    def __init__(self, seed=0, alpha=10.0, **_ignored):
        self.rng = np.random.default_rng(int(seed))
        self.alpha = float(alpha)

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        keep = np.isfinite(y)
        X, y = X[keep], y[keep]
        if len(y) == 0:
            self.med = np.zeros(X.shape[1])
            self.mu = np.zeros(X.shape[1])
            self.sd = np.ones(X.shape[1])
            self.w = np.zeros(X.shape[1] + 1)
            return self
        self.med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
        self.med = np.where(np.isfinite(self.med), self.med, 0.0)
        Z = self._impute(X)
        self.mu, self.sd = Z.mean(0), Z.std(0)
        self.sd = np.where(self.sd > 1e-9, self.sd, 1.0)
        Z = (Z - self.mu) / self.sd
        idx = self.rng.integers(0, len(y), len(y))          # seed-dependent bootstrap, so seed averaging is real
        Z, yb = Z[idx], y[idx]
        A = np.c_[np.ones(len(Z)), Z]
        G = A.T @ A + self.alpha * np.eye(A.shape[1])
        G[0, 0] -= self.alpha
        self.w = np.linalg.solve(G, A.T @ yb)
        return self

    def _impute(self, X):
        X = np.asarray(X, dtype=float)
        return np.where(np.isfinite(X), X, self.med)

    def predict(self, X):
        Z = (self._impute(X) - self.mu) / self.sd
        return np.c_[np.ones(len(Z)), Z] @ self.w
