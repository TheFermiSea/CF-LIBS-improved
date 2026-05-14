"""Acceptance test for the T1-1 follow-up host/kernel file split.

ADR-0001 §3 prescribes that JAX-accelerated physics packages expose a
``kernels.py`` sibling next to their host module(s), holding only
``@jit_if_available``-decorated pure functions. This test asserts the
expected modules exist, import cleanly under ``JAX_PLATFORMS=cpu``, and
expose the documented symbols.

Coverage:
    * :mod:`cflibs.radiation.kernels` (landed in T1-2/T1-5)
    * :mod:`cflibs.plasma.kernels` (T1-1 follow-up, this PR)
    * :mod:`cflibs.instrument.kernels` (T1-1 follow-up, this PR)
"""

from __future__ import annotations

import importlib

import pytest

# These kernels.py modules are introduced as part of the T1-1 host/kernel
# split. Each entry: (module dotted path, list of public symbol names that
# must be present after import).
_EXPECTED: list[tuple[str, list[str]]] = [
    (
        "cflibs.plasma.kernels",
        [
            "_partition_sum_jax",
            "_saha_balance_kernel",
            "_boltzmann_populations_kernel",
        ],
    ),
    (
        "cflibs.instrument.kernels",
        [
            "_sigma_at_wavelength_jax",
            "_apply_response_jax",
        ],
    ),
]


@pytest.mark.parametrize("module_name,symbols", _EXPECTED)
def test_kernels_module_imports_and_exposes_symbols(module_name: str, symbols: list[str]) -> None:
    """Each kernels.py module imports cleanly and exposes its kernel symbols."""
    module = importlib.import_module(module_name)
    missing = [name for name in symbols if not hasattr(module, name)]
    assert not missing, f"{module_name} is missing expected kernel symbols: {missing}"


def test_plasma_kernels_back_compat_reexport() -> None:
    """Host module re-exports the kernel symbols for back-compat callers."""
    from cflibs.plasma import kernels
    from cflibs.plasma import saha_boltzmann as host

    for name in (
        "_partition_sum_jax",
        "_saha_balance_kernel",
        "_boltzmann_populations_kernel",
    ):
        assert getattr(host, name) is getattr(kernels, name), (
            f"cflibs.plasma.saha_boltzmann.{name} must re-export the symbol "
            f"from cflibs.plasma.kernels for back-compat with existing callers."
        )


def test_instrument_kernels_back_compat_reexport() -> None:
    """Host module re-exports the kernel symbols for back-compat callers."""
    from cflibs.instrument import kernels
    from cflibs.instrument import model as host

    for name in ("_sigma_at_wavelength_jax", "_apply_response_jax"):
        assert getattr(host, name) is getattr(kernels, name), (
            f"cflibs.instrument.model.{name} must re-export the symbol "
            f"from cflibs.instrument.kernels for back-compat with existing callers."
        )
