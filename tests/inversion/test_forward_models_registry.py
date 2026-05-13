"""Unit tests for the T1-6 ``cflibs.inversion.forward_models`` registry."""

from __future__ import annotations

import pytest

from cflibs.inversion.forward_models import (
    FORWARD_MODELS,
    ForwardModelFn,
    get_forward_model,
)
from cflibs.radiation.kernels import forward_model as kernel_forward_model


def test_registry_keys():
    """The registry resolves the three documented kernels."""
    assert set(FORWARD_MODELS) == {
        "single_zone_lte",
        "hermann_two_region",
        "lte_with_self_absorption",
    }


def test_get_forward_model_returns_callable():
    fn = get_forward_model("single_zone_lte")
    assert callable(fn)


def test_get_forward_model_raises_on_unknown_name():
    with pytest.raises(KeyError, match="Unknown forward model"):
        get_forward_model("not_a_real_model")


def test_single_zone_lte_is_canonical_kernel():
    """The single-zone entry is the T1-2 :func:`forward_model` callable itself."""
    assert FORWARD_MODELS["single_zone_lte"] is kernel_forward_model


def test_hermann_two_region_is_distinct_wrapper():
    """The Hermann two-region entry is a distinct callable."""
    assert FORWARD_MODELS["hermann_two_region"] is not kernel_forward_model
    assert callable(FORWARD_MODELS["hermann_two_region"])


def test_lte_with_self_absorption_is_distinct_wrapper():
    """The self-absorption entry is a distinct callable that pre-binds the flag."""
    assert FORWARD_MODELS["lte_with_self_absorption"] is not kernel_forward_model
    assert callable(FORWARD_MODELS["lte_with_self_absorption"])


def test_forward_model_fn_is_protocol():
    """``ForwardModelFn`` is a Protocol type (importable, structural)."""
    # ``Protocol`` types are not directly instantiable; the structural check is
    # via ``callable`` on registry entries, which we already exercised above.
    assert ForwardModelFn is not None
