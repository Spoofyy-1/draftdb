"""Stand-in for the `tabicl` package. See README.md -- plumbing only, never real numbers."""

import sys

from _ridge import RidgeStandIn

_WARNED = False


def _warn():
    global _WARNED
    if not _WARNED:
        print("!! bridge/smoke_stubs/tabicl.py in use -- NOT TabICL. Plumbing test only.", file=sys.stderr)
        _WARNED = True


class TabICLRegressor(RidgeStandIn):
    def __init__(self, device=None, n_estimators=None, batch_size=None, random_state=0, outlier_threshold=None, **kw):
        _warn()
        super().__init__(seed=random_state)


class TabICLClassifier(TabICLRegressor):
    pass
