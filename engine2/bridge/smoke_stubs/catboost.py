"""Stand-in for the `catboost` package. See README.md -- plumbing only, never real numbers."""

import sys

from _ridge import RidgeStandIn

_WARNED = False


def _warn():
    global _WARNED
    if not _WARNED:
        print("!! bridge/smoke_stubs/catboost.py in use -- NOT CatBoost. Plumbing test only.", file=sys.stderr)
        _WARNED = True


class CatBoostRegressor(RidgeStandIn):
    def __init__(self, loss_function=None, iterations=None, depth=None, learning_rate=None, l2_leaf_reg=None,
                 random_seed=0, verbose=False, allow_writing_files=False, thread_count=None,
                 monotone_constraints=None, **kw):
        _warn()
        super().__init__(seed=random_seed)


class CatBoostRanker(CatBoostRegressor):
    def fit(self, X, y, group_id=None, sample_weight=None):
        return super().fit(X, y)
