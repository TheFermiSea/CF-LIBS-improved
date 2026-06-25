"""Regression guard for the manifold emissivity absolute scale (audit finding C1).

The manifold forward (`_compute_spectrum_snapshot`) computes line emissivity with
SI constants, so the upper-level number density must be in m^-3. ``n_e`` follows
the project cm^-3 convention, so a `* 1.0e6` cm^-3 -> m^-3 conversion is required
(identical to ``kernels.forward_model`` and the two-zone Bayesian ``forward.py``).

Before the fix this conversion was missing, underscaling every stored manifold by
1e6. The existing manifold tests are all shape-normalized and did not catch it.
This test asserts the *absolute* integrated flux matches the m^-3 emissivity, not
the cm^-3 one — a broadening-independent check (a normalized line profile
integrates to 1, so integrated flux equals the emissivity coefficient).
"""

import math
from pathlib import Path

import pytest

jnp = pytest.importorskip("jax.numpy")

from cflibs.core.constants import C_LIGHT, H_PLANCK  # noqa: E402
from cflibs.manifold.config import ManifoldConfig  # noqa: E402
from cflibs.manifold.generator import ManifoldGenerator  # noqa: E402


def _db_path() -> Path:
    here = Path(__file__).resolve().parent.parent.parent
    for cand in (here / "ASD_da" / "libs_production.db", here / "libs_production.db"):
        if cand.exists():
            return cand
    pytest.skip("libs_production.db not available")


@pytest.mark.requires_db
@pytest.mark.requires_jax
@pytest.mark.physics
def test_manifold_emissivity_is_m3_scale(tmp_path):
    cfg = ManifoldConfig(
        db_path=str(_db_path()),
        output_path=str(tmp_path / "unused.h5"),
        elements=["Fe"],
        wavelength_range=(370.0, 390.0),
        pixels=2000,
    )
    gen = ManifoldGenerator(cfg)
    ad = gen.atomic_data
    l_wl, l_aki = ad[0], ad[1]

    T_eV, n_e = 1.0, 1.0e17
    conc = jnp.array([1.0])

    n_upper = ManifoldGenerator._saha_eggert_solver(T_eV, n_e, conc, ad)
    hc4pi_over_lambda = H_PLANCK * C_LIGHT / (4 * jnp.pi * l_wl * 1e-9)
    eps_m3 = float(jnp.sum(hc4pi_over_lambda * l_aki * (n_upper * 1.0e6)))  # fixed
    eps_cm3 = float(jnp.sum(hc4pi_over_lambda * l_aki * n_upper))  # buggy

    wl_grid = jnp.linspace(cfg.wavelength_range[0], cfg.wavelength_range[1], cfg.pixels)
    d_lambda = (cfg.wavelength_range[1] - cfg.wavelength_range[0]) / (cfg.pixels - 1)
    sigma_inst = cfg.instrument_fwhm_nm / 2.3548
    intensity = ManifoldGenerator._compute_spectrum_snapshot(
        wl_grid, T_eV, n_e, conc, ad, sigma_inst
    )
    measured = float(jnp.sum(intensity)) * d_lambda

    # Integrated flux must match the m^-3 emissivity (within window edge effects),
    # and be ~1e6 above the cm^-3 (buggy) scale.
    assert measured == pytest.approx(eps_m3, rel=0.02)
    assert measured / eps_cm3 > 1e5
    assert math.isfinite(measured) and measured > 0
