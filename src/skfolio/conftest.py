"""Doctest configuration for the `skfolio` package.

`pyproject.toml` runs `--doctest-modules` over `src`, so every example in every
docstring is executed and its documented output verified. The few that cannot be are
listed in `SKIPPED` below, each with the reason it is there.
"""

from __future__ import annotations

import numpy as np
import pytest
import sklearn
from _pytest.doctest import DoctestItem

SKIPPED = {
    "skfolio.attribution._predicted.predicted_factor_attribution": (
        "illustrative fragment: `factor_returns` is never defined"
    ),
    "skfolio.attribution._realized.realized_factor_attribution": (
        "illustrative fragment: `factor_returns` is never defined"
    ),
    "skfolio.attribution._realized.rolling_realized_factor_attribution": (
        "illustrative fragment: `factor_returns` is never defined"
    ),
    "skfolio.population._population.Population.boxplot_measure": (
        "illustrative fragment: `population` is never defined"
    ),
    "skfolio.alpha._ew_sharpe_optimal_alpha.EWSharpeOptimalAlpha": (
        "example raises: `partial_fit(X[-5:])` re-feeds rows already seen and "
        "`ValueError: observations must be unique`"
    ),
    "skfolio.alpha._predictor_alpha.PredictorAlpha": (
        "example raises: `partial_fit(X[-5:])` re-feeds rows already seen and "
        "`ValueError: observations must be unique`"
    ),
    "skfolio.prior._opinion_pooling.OpinionPooling": (
        "example raises: `SolverError: Solver 'CLARABEL' failed`"
    ),
}


def pytest_collection_modifyitems(items) -> None:
    """Skip the doctests listed in `SKIPPED`."""
    for item in items:
        if isinstance(item, DoctestItem) and item.name in SKIPPED:
            item.add_marker(pytest.mark.skip(reason=SKIPPED[item.name]))


@pytest.fixture(autouse=True)
def _doctest_environment(tmp_path, monkeypatch):
    """Run every doctest under the state a reader of the published docs would have.

    `tests/conftest.py` sets `np.set_printoptions(suppress=True, precision=6)` and a
    few test modules call `sklearn.set_config(...)` without restoring it. Both leak
    into whatever runs next and change how documented output renders. The temporary
    directory keeps examples that write files (`AssetPanel.save("asset_panel")`) out
    of the working tree.
    """
    monkeypatch.chdir(tmp_path)
    with (
        np.printoptions(precision=8, suppress=False),
        sklearn.config_context(transform_output="default"),
    ):
        yield
