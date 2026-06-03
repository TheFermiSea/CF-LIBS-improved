"""Tests for the ``CFLIBS_DISABLE_STARK_T_FACTOR`` env-var toggle.

The toggle exists to support before/after ablation benchmarks of the
Stark temperature-power-law factor (CF-LIBS-improved-vjbh fix landed in
PRs #182 + #183; benchmark plan tracked in CF-LIBS-improved-4rwe).

Off by default — the kernel applies ``factor_T = (T/T_ref)^(-alpha)``.
With ``CFLIBS_DISABLE_STARK_T_FACTOR=1`` set, the kernel collapses
``factor_T`` to 1.0, reproducing the pre-vjbh behaviour for direct
comparison.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.requires_jax]

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from cflibs.core.jax_runtime import AtomicSnapshot  # noqa: E402
from cflibs.radiation.kernels import _per_line_stark_gamma  # noqa: E402


@pytest.fixture
def stark_snapshot():
    """Snapshot with non-trivial alpha so the T-factor branch matters."""
    n_lines = 4
    return AtomicSnapshot(
        species=(("Fe", 1),),
        line_wavelengths_nm=jnp.array([400.0, 410.0, 420.0, 430.0]),
        line_A_ki=jnp.zeros(n_lines),
        line_E_k_ev=jnp.zeros(n_lines),
        line_g_k=jnp.ones(n_lines),
        line_E_i_ev=jnp.zeros(n_lines),
        line_g_i=jnp.ones(n_lines),
        line_species_index=jnp.zeros(n_lines, dtype=jnp.int32),
        line_stark_w=jnp.array([0.02, 0.03, 0.04, 0.05]),
        line_stark_alpha=jnp.array([0.5, 0.6, 0.7, 0.8]),
        line_natural_w=jnp.zeros(n_lines),
        partition_coeffs=jnp.zeros((1, 5)),
        ionization_potential_ev=jnp.array([7.87]),
    )


def test_toggle_off_default_applies_t_factor(stark_snapshot, monkeypatch):
    """With the env var unset, the canonical T-factor is applied."""
    monkeypatch.delenv("CFLIBS_DISABLE_STARK_T_FACTOR", raising=False)

    n_e = 1.0e17
    T_eV = 1.5  # away from REF_T_EV so factor_T != 1
    gamma = np.asarray(_per_line_stark_gamma(stark_snapshot, n_e, T_eV))

    # Reference (A4-CONV-2): stark_w is the stored FWHM at REF_NE=1e17; the
    # kernel returns the Lorentzian HWHM = 0.5 * stark_w * (n_e/1e17) *
    # (T/T_ref)^(-alpha).
    stark_w = np.asarray(stark_snapshot.line_stark_w)
    alpha = np.asarray(stark_snapshot.line_stark_alpha)
    REF_T_EV = 0.86173
    expected = 0.5 * stark_w * (n_e / 1.0e17) * np.power(T_eV / REF_T_EV, -alpha)
    np.testing.assert_allclose(gamma, expected, rtol=1e-6, atol=0.0)
    # Confirm the factor really did do something.
    legacy = 0.5 * stark_w * (n_e / 1.0e17)
    assert not np.allclose(gamma, legacy)


def test_toggle_on_disables_t_factor(stark_snapshot, monkeypatch):
    """With ``CFLIBS_DISABLE_STARK_T_FACTOR=1`` the kernel matches the pre-vjbh formula."""
    monkeypatch.setenv("CFLIBS_DISABLE_STARK_T_FACTOR", "1")

    n_e = 1.0e17
    stark_w = np.asarray(stark_snapshot.line_stark_w)
    # Toggle off the T-factor only: gamma = 0.5 * stark_w * (n_e/1e17).
    expected_legacy = 0.5 * stark_w * (n_e / 1.0e17)

    # Vary T_eV across the LIBS range — the toggled formula has zero T-dependence.
    for T_eV in (0.5, 0.86173, 1.0, 1.5, 2.0):
        gamma = np.asarray(_per_line_stark_gamma(stark_snapshot, n_e, T_eV))
        np.testing.assert_allclose(
            gamma,
            expected_legacy,
            rtol=1e-12,
            atol=0.0,
            err_msg=f"Toggled kernel must be T-independent; got per-T drift at T={T_eV}",
        )


def test_toggle_value_must_be_exactly_1(stark_snapshot, monkeypatch):
    """Any value other than the literal string ``"1"`` keeps the fix enabled.

    Guards against accidental sloppy env-var parsing — only the exact
    documented value disables the factor, mirroring the ``CFLIBS_USE_*``
    pattern used elsewhere in the codebase.
    """
    n_e = 1.0e17
    T_eV = 1.5

    # baseline with T-factor on
    monkeypatch.delenv("CFLIBS_DISABLE_STARK_T_FACTOR", raising=False)
    expected_with_factor = np.asarray(_per_line_stark_gamma(stark_snapshot, n_e, T_eV))

    for bogus in ("", "0", "true", "yes", "True", "ON", "2"):
        monkeypatch.setenv("CFLIBS_DISABLE_STARK_T_FACTOR", bogus)
        gamma = np.asarray(_per_line_stark_gamma(stark_snapshot, n_e, T_eV))
        np.testing.assert_allclose(
            gamma,
            expected_with_factor,
            rtol=1e-12,
            atol=0.0,
            err_msg=f"CFLIBS_DISABLE_STARK_T_FACTOR={bogus!r} should NOT disable the factor",
        )
