"""Base classes for all estimators and various utility functions."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import suppress

import numpy as np
import sklearn.base as skb

from skfolio.containers import AssetPanel, AssetPanelView
from skfolio.typing import AnyArray, FloatArray, StrArray

__all__ = ["BaseAssetPanelTransformer", "BaseComposition", "validate_asset_panel"]


class BaseAssetPanelTransformer(skb.BaseEstimator, ABC):
    """Base class for estimators that transform asset panel data.

    Descriptors and factor exposure estimators take an
    :class:`~skfolio.containers.AssetPanel` and return transformed values indexed by
    observation and asset. Most transformers return an array with shape
    `(n_observations, n_assets)`. Transformers that produce multiple values per asset,
    such as :class:`~skfolio.factor_exposure.OneHotCategoricalFactors`, return an array
    with shape `(n_observations, n_assets, n_categories)`.

    In scikit-learn, `fit` and `partial_fit` update fitted state, stored in trailing
    underscore attributes, while `transform` returns transformed input data using that
    state. This separation is not suitable for every
    :class:`~skfolio.containers.AssetPanel` transformer. For some estimators, the
    transformed value is produced by the same state transition that updates the
    estimator. A separate `transform` method would either need to mutate state or
    depend on a preceding `partial_fit` call, so the API exposes the combined
    operation directly. For example, the exponentially weighted momentum
    descriptor :class:`~skfolio.descriptor.EWMomentum` needs to update its internal
    EWMA state to compute the transformed value on each observation.

    Other transformers are independent across observations. For example, the
    :class:`~skfolio.descriptor.DividendToPrice` descriptor depends only on the current
    `dividends_ttm` and `market_cap` values and can therefore be declared stateless.

    Accordingly, `fit_transform` is used for full-batch computation and
    `partial_fit_transform` for online computation. Subclasses must implement
    `fit_transform`. Downstream meta-estimators use the presence of
    `partial_fit_transform` to determine whether a transformer supports online
    transformation.

    Supported implementation patterns are:

    - Batch-only transformers implement only `fit_transform`.
    - Stateless transformers declare `stateless=True` and implement only `fit_transform`.
      The base class adds `partial_fit_transform` as a direct delegation to `fit_transform`.
    - Online transformers implement both `fit_transform` and `partial_fit_transform`.

    Attributes
    ----------
    n_assets_ : int
        Number of assets seen during fitting.

    asset_names_ : ndarray of shape (n_assets,)
        Asset names seen during fitting.

    See Also
    --------
    :class:`~skfolio.descriptor.BaseDescriptor` : Computes raw descriptor values.
    :class:`~skfolio.factor_exposure.BaseFactorExposure` : Computes factor exposures.
    """

    n_assets_: int
    asset_names_: StrArray

    stateless: bool = False

    def __init_subclass__(cls, *, stateless: bool | None = None, **kwargs):
        """When `stateless=True`, the subclass declares that `fit_transform` is
        independent across observations. In this case, the base class injects
        `partial_fit_transform` as a delegation to `fit_transform`, so downstream
        meta-estimators can detect online support by checking for that method. When
        `stateless` is omitted, the value is inherited from the parent class.
        """
        super().__init_subclass__(**kwargs)
        if stateless is None:
            stateless = getattr(cls, "stateless", False)
        cls.stateless = stateless

        if stateless and "partial_fit_transform" in cls.__dict__:
            raise TypeError(
                "Classes declared with stateless=True must not define "
                "partial_fit_transform."
            )

        if stateless:

            def partial_fit_transform(
                self, X: AssetPanel, y=None, **fit_params
            ) -> FloatArray:
                """Stateless class delegation to `fit_transform`."""
                return self.fit_transform(X, y, **fit_params)

            cls.partial_fit_transform = partial_fit_transform

    @abstractmethod
    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Fit the transformer if needed and return transformed values.

        Parameters
        ----------
        X : AssetPanel
            Input panel data.

        y : None
            Ignored. Present for API consistency.

        **fit_params : dict
            Additional fit parameters. Metadata routing may pass these parameters to
            sub-estimators when applicable.

        Returns
        -------
        values : ndarray of shape (n_observations, n_assets) or (n_observations, n_assets, n_components)
            Transformed values.
        """


def validate_asset_panel(
    _estimator,
    /,
    asset_panel: AssetPanel | AssetPanelView,
    required_fields: list[str] | None = None,
    reserved_fields: list[str] | None = None,
    finite_or_nan: list[str] | None = None,
    finite_when_active: list[str] | None = None,
    strictly_positive_or_nan: list[str] | None = None,
    strictly_positive_when_active: list[str] | None = None,
    non_negative_or_nan: list[str] | None = None,
    reset: bool = True,
    copy: bool = False,
) -> AssetPanel | AssetPanelView:
    """Validate an AssetPanel and set estimator metadata attributes.

    This function validates that the panel contains required fields, doesn't contain
    reserved ones and sets standard metadata attributes on the estimator.

    Parameters
    ----------
    _estimator : estimator instance
        The estimator on which to set validation attributes.

    asset_panel : AssetPanel or AssetPanelView
        AssetPanel. AssetPanel already validates that all fields have consistent shapes
        and consistent masks, allowing this validation to be lightweight.

    required_fields : list of str, optional
        Fields that must be present in the AssetPanel. If any are missing, a ValueError
        is raised.

    reserved_fields : list of str, optional
        Fields that must NOT be present in the AssetPanel. These are typically names the
        estimator will create internally. If any are found, a ValueError is raised.

    finite_or_nan : list of str, optional
        Fields whose values must be finite or NaN.

    finite_when_active : list of str, optional
        Fields whose values must be finite wherever `active_mask` is True. This is
        typically used for fields like "market_cap" that must be forward-filled for
        holidays before constructing the AssetPanel.

    strictly_positive_or_nan : list of str, optional
        Fields whose values must be strictly positive and finite, or NaN.

    strictly_positive_when_active : list of str, optional
        Fields whose values must be strictly positive and finite wherever
        `active_mask` is True.

    non_negative_or_nan : list of str, optional
        Fields whose values must be non-negative and finite, or NaN.

    reset : bool, default=True
        If True, sets metadata attributes on the estimator:
        - `asset_names_`: array of asset identifiers
        - `n_assets_`: number of assets

        If False, validates that existing attributes match the panel.

    copy : bool, default=False
        If True, returns a shallow copy of the panel (new dict containers with shared
        arrays). Use this when the estimator needs to add fields without mutating the
        user's input.

    Returns
    -------
    AssetPanel or AssetPanelView
        The validated panel, or a shallow copy if `copy=True`.

    Raises
    ------
    TypeError
        If asset_panel is not an AssetPanel or AssetPanelView.

    ValueError
        If required fields are missing, reserved fields are present, a
        field validation rule is violated or (when reset=False) panel doesn't match
        stored metadata.
    """
    # Normalize empty lists to None.
    if required_fields is not None and len(required_fields) == 0:
        required_fields = None
    if reserved_fields is not None and len(reserved_fields) == 0:
        reserved_fields = None
    if finite_or_nan is not None and len(finite_or_nan) == 0:
        finite_or_nan = None
    if finite_when_active is not None and len(finite_when_active) == 0:
        finite_when_active = None
    if strictly_positive_or_nan is not None and len(strictly_positive_or_nan) == 0:
        strictly_positive_or_nan = None
    if (
        strictly_positive_when_active is not None
        and len(strictly_positive_when_active) == 0
    ):
        strictly_positive_when_active = None
    if non_negative_or_nan is not None and len(non_negative_or_nan) == 0:
        non_negative_or_nan = None

    if not isinstance(asset_panel, AssetPanel | AssetPanelView):
        raise TypeError(
            f"Must be an AssetPanel or AssetPanelView, got {type(asset_panel).__name__}"
        )

    field_names = list(asset_panel.keys())
    asset_names = asset_panel.asset_names

    # Check consistency with previous fit (if reset=False).
    if not reset:
        if not np.array_equal(_estimator.asset_names_, asset_names):
            raise ValueError(
                f"asset_names don't match. Expected {list(_estimator.asset_names_)}, "
                f"got {list(asset_names)}."
            )

    # Check reserved fields.
    if reserved_fields is not None:
        reserved = set(reserved_fields) & set(field_names)
        if reserved:
            raise ValueError(
                f"Reserved fields must be removed or renamed: {sorted(reserved)}."
            )

    # Check required fields.
    if required_fields is not None:
        missing = set(required_fields) - set(field_names)
        if missing:
            raise ValueError(
                f"Required fields are missing: {sorted(missing)}. "
                f"Available: {sorted(field_names)}."
            )

    # Check finite-or-NaN constraints.
    if finite_or_nan is not None:
        _check_fields_exist(finite_or_nan, field_names, "finite_or_nan")
        for field in finite_or_nan:
            values = asset_panel[field]
            if values.ndim == 2:
                bad = np.isinf(values)
            else:
                bad = np.isinf(values).any(axis=2)
            if bad.any():
                bad_obs = _bad_observation(bad)
                raise ValueError(
                    f'Field "{field}" contains infinite values '
                    f"(first at observation index {bad_obs}). "
                    f'"{field}" must contain finite values or NaN.'
                )

    # Check finite-when-active constraints.
    if finite_when_active is not None:
        _check_fields_exist(finite_when_active, field_names, "finite_when_active")
        for field in finite_when_active:
            values = asset_panel[field]
            if values.ndim == 2:
                bad = ~np.isfinite(values) & asset_panel.active_mask
            else:
                # 3-D field: check all components
                bad = (~np.isfinite(values)).any(axis=2) & asset_panel.active_mask
            if bad.any():
                bad_obs = _bad_observation(bad)
                raise ValueError(
                    f'Field "{field}" contains NaN/inf for active assets '
                    f"(first at observation index {bad_obs}). "
                    f'"{field}" must be finite wherever `active_mask` is '
                    f'True. Forward-fill "{field}" for holidays '
                    f'or set `active_mask` to False until the first finite "{field}".'
                )

    # Check strictly-positive-or-NaN constraints.
    if strictly_positive_or_nan is not None:
        _check_fields_exist(
            strictly_positive_or_nan, field_names, "strictly_positive_or_nan"
        )
        for field in strictly_positive_or_nan:
            values = asset_panel[field]
            if values.ndim == 2:
                bad = ~np.isnan(values) & (~np.isfinite(values) | (values <= 0))
            else:
                bad = (~np.isnan(values) & (~np.isfinite(values) | (values <= 0))).any(
                    axis=2
                )
            if bad.any():
                bad_obs = _bad_observation(bad)
                raise ValueError(
                    f'Field "{field}" contains non-positive or infinite values '
                    f"(first at observation index {bad_obs}). "
                    f'"{field}" must contain strictly positive finite values or NaN.'
                )

    # Check strictly-positive-when-active constraints.
    if strictly_positive_when_active is not None:
        _check_fields_exist(
            strictly_positive_when_active,
            field_names,
            "strictly_positive_when_active",
        )
        for field in strictly_positive_when_active:
            values = asset_panel[field]
            if values.ndim == 2:
                bad = (~np.isfinite(values) | (values <= 0)) & asset_panel.active_mask
            else:
                bad = (~np.isfinite(values) | (values <= 0)).any(
                    axis=2
                ) & asset_panel.active_mask
            if bad.any():
                bad_obs = _bad_observation(bad)
                raise ValueError(
                    f'Field "{field}" contains non-finite or non-positive values for active '
                    f"assets "
                    f"(first at observation index {bad_obs}). "
                    f'"{field}" must be strictly positive and finite wherever '
                    f"`active_mask` is True."
                )

    # Check non-negative-or-NaN constraints.
    if non_negative_or_nan is not None:
        _check_fields_exist(non_negative_or_nan, field_names, "non_negative_or_nan")
        for field in non_negative_or_nan:
            values = asset_panel[field]
            if values.ndim == 2:
                bad = ~np.isnan(values) & (~np.isfinite(values) | (values < 0))
            else:
                bad = (~np.isnan(values) & (~np.isfinite(values) | (values < 0))).any(
                    axis=2
                )
            if bad.any():
                bad_obs = _bad_observation(bad)
                raise ValueError(
                    f'Field "{field}" contains negative values or infinite values '
                    f"(first at observation index {bad_obs}). "
                    f'"{field}" must contain non-negative finite values or NaN.'
                )

    # Set estimator metadata attributes.
    if reset:
        _estimator.asset_names_ = np.asarray(asset_names)
        _estimator.n_assets_ = len(asset_names)

    return asset_panel.copy() if copy else asset_panel


def _check_fields_exist(
    fields: list[str], field_names: list[str], arg_name: str
) -> None:
    """Check that all validation rule fields exist in the panel."""
    for field in fields:
        if field not in field_names:
            raise ValueError(
                f'{arg_name} lists "{field}", but that field is not in the '
                f"AssetPanel. Available: {sorted(field_names)}."
            )


def _bad_observation(bad: AnyArray) -> int:
    """Return the first observation index containing a validation failure."""
    return int(np.where(bad.any(axis=1))[0][0])


class BaseComposition(skb.BaseEstimator, ABC):
    """Handles parameter management for ensemble estimators."""

    @abstractmethod
    def __init__(self):
        pass

    def _get_params(self, attr, deep=True):
        out = super().get_params(deep=deep)
        if not deep:
            return out

        estimators = getattr(self, attr)
        try:
            out.update(estimators)
        except (TypeError, ValueError):
            # Ignore TypeError for cases where estimators is not a list of
            # (name, estimator) and ignore ValueError when the list is not
            # formatted correctly. This is to prevent errors when calling
            # `set_params`. `BaseEstimator.set_params` calls `get_params` which
            # can error for invalid values for `estimators`.
            return out

        for name, estimator in estimators:
            if hasattr(estimator, "get_params"):
                for key, value in estimator.get_params(deep=True).items():
                    out[f"{name}__{key}"] = value
        return out

    def _set_params(self, attr, **params):
        # Ensure strict ordering of parameter setting:
        # 1. All steps
        if attr in params:
            setattr(self, attr, params.pop(attr))
        # 2. Replace items with estimators in params
        items = getattr(self, attr)
        if isinstance(items, list) and items:
            # Get item names used to identify valid names in params
            # `zip` raises a TypeError when `items` does not contains
            # elements of length 2
            with suppress(TypeError):
                item_names, _ = zip(*items, strict=True)
                for name in params:
                    if "__" not in name and name in item_names:
                        self._replace_estimator(attr, name, params.pop(name))

        # 3. Step parameters and other initialisation arguments
        super().set_params(**params)
        return self

    def _replace_estimator(self, attr, name, new_val):
        # assumes `name` is a valid estimator name
        new_estimators = list(getattr(self, attr))
        for i, (estimator_name, _) in enumerate(new_estimators):
            if estimator_name == name:
                new_estimators[i] = (name, new_val)
                break
        setattr(self, attr, new_estimators)

    def _validate_names(self, names):
        if len(set(names)) != len(names):
            raise ValueError(f"Names provided are not unique: {list(names)!r}")
        invalid_names = set(names).intersection(self.get_params(deep=False))
        if invalid_names:
            raise ValueError(
                f"Estimator names conflict with constructor arguments: {sorted(invalid_names)!r}"
            )
        invalid_names = [name for name in names if "__" in name]
        if invalid_names:
            raise ValueError(
                f"Estimator names must not contain __: got {invalid_names!r}"
            )
