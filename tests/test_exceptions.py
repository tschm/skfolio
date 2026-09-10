"""Keep every skfolio error catchable as one class.

`skfolio.exceptions` used to define seven errors that each inherited `Exception`
directly, so a caller wanting to handle "anything skfolio raises on purpose" had to
enumerate all seven and update that tuple whenever a new one was added. `SkfolioError`
gives them a common base.

These tests exist so the base stays load-bearing: a new exception added without it
would be invisible to `except SkfolioError` and would reintroduce the enumeration
problem for everyone downstream. `test_every_public_error_derives_from_base` fails on
that by name, and discovers the classes rather than listing them, so it covers
exceptions added after it was written.
"""

from __future__ import annotations

import inspect

import pytest

from skfolio import exceptions
from skfolio.exceptions import SkfolioError

PUBLIC_ERRORS = [
    getattr(exceptions, name) for name in exceptions.__all__ if name != "SkfolioError"
]


def test_public_errors_are_discovered() -> None:
    """Guard the discovery itself, so a mistake here cannot silently empty the suite."""
    assert len(PUBLIC_ERRORS) == 7
    assert all(inspect.isclass(error) for error in PUBLIC_ERRORS)


@pytest.mark.parametrize("error", PUBLIC_ERRORS, ids=lambda error: error.__name__)
def test_every_public_error_derives_from_base(error: type[Exception]) -> None:
    """Every exported error must be catchable as `SkfolioError`."""
    assert issubclass(error, SkfolioError)


@pytest.mark.parametrize("error", PUBLIC_ERRORS, ids=lambda error: error.__name__)
def test_raising_is_caught_by_the_base(error: type[Exception]) -> None:
    """The subclass relationship must hold for a raised instance, not just the class."""
    with pytest.raises(SkfolioError):
        raise error("boom")


def test_base_is_exported() -> None:
    """`SkfolioError` is part of the public surface, so callers can import it."""
    assert "SkfolioError" in exceptions.__all__


def test_base_still_derives_from_exception() -> None:
    """Existing `except Exception` handlers must keep working."""
    assert issubclass(SkfolioError, Exception)
