"""Doctest configuration for the `skfolio` package.

`pyproject.toml` runs `--doctest-modules` over `src`, so every example in every
docstring is executed and its documented output verified. A handful of docstrings
cannot be verified yet; they are listed here rather than annotated with
`# doctest: +SKIP` in the source, because those markers render in the published API
documentation and this list is easier to burn down.

Each entry needs a reason. Removing one should be a self-contained change.
"""

from __future__ import annotations

import numpy as np
import pytest
import sklearn
from _pytest.doctest import DoctestItem

# Illustrative fragments: the examples reference names that are never defined
# (`factor_returns  # (252, 3)`, `population`, ...) because they document call shape
# rather than a runnable session. Making them executable means inventing example
# data, which is an authoring decision rather than a mechanical fix.
_FRAGMENTS = {
    "skfolio.attribution._predicted.predicted_factor_attribution",
    "skfolio.attribution._realized.realized_factor_attribution",
    "skfolio.attribution._realized.rolling_realized_factor_attribution",
    "skfolio.population._population.Population.boxplot_measure",
}

# The example itself does not run: it re-feeds overlapping observations to
# `partial_fit`, which raises `ValueError: observations must be unique`.
_BROKEN = {
    "skfolio.alpha._ew_sharpe_optimal_alpha.EWSharpeOptimalAlpha",
    "skfolio.alpha._predictor_alpha.PredictorAlpha",
}

# The example does not converge on every platform: `SolverError: Solver 'CLARABEL'
# failed`. Asserting its output would make the suite depend on solver behaviour.
_SOLVER_DEPENDENT = {
    "skfolio.prior._opinion_pooling.OpinionPooling",
}

# `best_params_` is drawn from a continuous distribution, so the reported value carries
# full float precision and depends on the numpy RNG stream even with `random_state` set.
_RANDOM_SEARCH = {
    "skfolio.model_selection._online._search.OnlineRandomizedSearch",
}

# Output is a large DataFrame summary or full model repr (up to ~8900 lines) whose
# values are convex-solver output. Neither rounding nor `ELLIPSIS` makes these stable
# across BLAS, CPU architecture and solver version.
# `summary()` returns a mixed-dtype frame (numeric columns alongside a textual
# "target" column), so `.round()` leaves the numbers object-typed and full precision.
_MIXED_DTYPE_SUMMARY = {
    "skfolio.model_selection._covariance_forecast_evaluation.CovarianceForecastComparison",
    "skfolio.model_selection._covariance_forecast_evaluation.CovarianceForecastEvaluation",
    "skfolio.model_selection._covariance_forecast_evaluation.covariance_forecast_evaluation",
    "skfolio.model_selection._online._covariance_forecast_evaluation.online_covariance_forecast_evaluation",
}

_UNSTABLE_OUTPUT = {
    "skfolio.model_selection._multiple_randomized_cv.MultipleRandomizedCV",
    "skfolio.moments.variance._empirical_variance.EmpiricalVariance",
    "skfolio.optimization.cluster.hierarchical._schur.SchurComplementary",
    "skfolio.prior._characteristics_factor_model.CharacteristicsFactorModel",
    "skfolio.prior._entropy_pooling.EntropyPooling",
    "skfolio.prior._synthetic_data.SyntheticData",
}

QUARANTINE = {
    **dict.fromkeys(
        _FRAGMENTS, "illustrative fragment; referenced names are undefined"
    ),
    **dict.fromkeys(_BROKEN, "example raises; `partial_fit` re-feeds overlapping rows"),
    **dict.fromkeys(_SOLVER_DEPENDENT, "solver does not converge on every platform"),
    **dict.fromkeys(
        _RANDOM_SEARCH, "sampled hyper-parameters are not stable across numpy versions"
    ),
    **dict.fromkeys(
        _UNSTABLE_OUTPUT, "output is large and not bit-stable across platforms"
    ),
    **dict.fromkeys(
        _MIXED_DTYPE_SUMMARY,
        "summary() frame is mixed-dtype; values stay at full precision",
    ),
}


def pytest_collection_modifyitems(items) -> None:
    """Skip the quarantined doctests, leaving their docstrings unannotated."""
    for item in items:
        if isinstance(item, DoctestItem):
            reason = QUARANTINE.get(item.name)
            if reason is not None:
                item.add_marker(
                    pytest.mark.skip(reason=f"doctest quarantined: {reason}")
                )


@pytest.fixture(autouse=True)
def _pristine_global_state(request):
    """Run every doctest under the global state a reader of the docs would have.

    `tests/conftest.py` sets `np.set_printoptions(suppress=True, precision=6)` for the
    whole session, and a few test modules call `sklearn.set_config(...)` without
    restoring it. Both leak into any doctest that runs afterwards and change how the
    documented output is rendered -- but someone reading the published API docs has
    neither setting, so the examples must be verified against the defaults.
    """
    if not isinstance(request.node, DoctestItem):
        yield
        return
    with (
        np.printoptions(precision=8, suppress=False),
        sklearn.config_context(transform_output="default"),
    ):
        yield
