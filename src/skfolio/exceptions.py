"""
The :mod:`skfolio.exceptions` module includes all custom warnings and error
classes used across skfolio.
"""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

__all__ = [
    "DuplicateGroupsError",
    "EquationToMatrixError",
    "FactorNotFoundError",
    "GroupNotFoundError",
    "NonPositiveVarianceError",
    "OptimizationError",
    "SkfolioError",
    "SolverError",
]


class SkfolioError(Exception):
    """Base class for every error raised by skfolio.

    Catching this catches any error the library raises on purpose, which is what a
    caller wants when a single failure should not stop a wider run -- a backtest loop
    or a scheduled job skipping one period and carrying on -- while still letting
    genuine bugs propagate.
    """


class OptimizationError(SkfolioError):
    """Optimization Did not converge."""


class SolverError(SkfolioError):
    """Solver error."""


class EquationToMatrixError(SkfolioError):
    """Error while processing equations."""


class GroupNotFoundError(SkfolioError):
    """Group name not found in the groups."""


class FactorNotFoundError(SkfolioError):
    """Factor name not found in factor_groups or loading_matrix not provided."""


class DuplicateGroupsError(SkfolioError):
    """Group name appear in multiple group levels."""


class NonPositiveVarianceError(SkfolioError):
    """Variance negative or null."""
