"""
Tests for the batch forward model composing all GPU kernels.

Tests cover: shape, batch-vs-sequential parity, physical behavior,
graceful degradation, and dimensional consistency.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

# Conditional JAX import
try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None

from cflibs.manifold.batch_forward import (
    BatchAtomicData,
    batch_forward_model,
    pack_atomic_data,
    single_spectrum_forward,
)

pytestmark = pytest.mark.requires_jax


# ---------------------------------------------------------------------------
# Synthetic atomic data fixture
# ---------------------------------------------------------------------------


def _make_synthetic_atomic_data(
    elements: list[str] | None = None,
) -> tuple[BatchAtomicData, list[str]]:
    """Create a small synthetic atomic dataset for testing.

    Returns data for Fe (Z=26), Cu (Z=29), Si (Z=14) by default.
    Each element gets 5 lines per neutral stage, 3 lines per singly-ionized.
    """
    if elements is None:
        elements = ["Fe", "Cu", "Si"]

    # Atomic masses
    masses = {"Fe": 55.845, "Cu": 63.546, "Si": 28.086}
    # Ionization potentials [eV]
    ips = {"Fe": [7.902, 16.19], "Cu": [7.726, 20.29], "Si": [8.152, 16.35]}

    line_data: list[dict] = []

    # Wavelength centers for synthetic lines (nm)
    wl_bases = {"Fe": 260.0, "Cu": 325.0, "Si": 250.0}

    for elem in elements:
        wl_base = wl_bases.get(elem, 300.0)
        mass = masses.get(elem, 56.0)

        # Neutral lines (stage 0 = I)
        for i in range(5):
            line_data.append(
                {
                    "wavelength_nm": wl_base + i * 2.0,
                    "A_ki": 1e8 * (1.0 + 0.5 * i),
                    "g_k": float(2 * (i % 3) + 1),  # 1, 3, 5, 1, 3
                    "E_k_eV": 3.0 + 0.5 * i,  # 3.0 to 5.0 eV
                    "element": elem,
                    "ion_stage": 0,
                    "stark_w_nm": 0.01 + 0.005 * i,
                    "mass_amu": mass,
                }
            )

        # Singly ionized lines (stage 1 = II)
        for i in range(3):
            line_data.append(
                {
                    "wavelength_nm": wl_base + 15.0 + i * 3.0,
                    "A_ki": 2e8 * (1.0 + 0.3 * i),
                    "g_k": float(2 * i + 3),  # 3, 5, 7
                    "E_k_eV": 5.0 + 1.0 * i,  # 5.0 to 7.0 eV (above IP reference)
                    "element": elem,
                    "ion_stage": 1,
                    "stark_w_nm": 0.02 + 0.01 * i,
                    "mass_amu": mass,
                }
            )

    ad = pack_atomic_data(elements, line_data, max_stages=3)

    # Fill in real ionization potentials
    ip_arr = np.zeros((len(elements), 2), dtype=np.float64)
    pf_arr = np.zeros((len(elements), 3, 5), dtype=np.float64)
    for i, elem in enumerate(elements):
        for j, val in enumerate(ips.get(elem, [7.0, 15.0])[:2]):
            ip_arr[i, j] = val
        # Simple partition functions: U ~ g0 * (1 + small T-dependent correction)
        # Using log(U) = a0 + a1*log(T) form
        pf_arr[i, 0, 0] = np.log(9.0)   # Fe I g0 ~ 9, Cu I g0 ~ 2, etc.
        pf_arr[i, 0, 1] = 0.05           # small T dependence
        pf_arr[i, 1, 0] = np.log(10.0)   # ionized ground state
        pf_arr[i, 1, 1] = 0.03
        pf_arr[i, 2, 0] = np.log(1.0)    # doubly ionized (placeholder)

    ad = BatchAtomicData(
        line_wavelengths=ad.line_wavelengths,
        line_A_ki=ad.line_A_ki,
        line_g_k=ad.line_g_k,
        line_E_k=ad.line_E_k,
        line_element_idx=ad.line_element_idx,
        line_ion_stage=ad.line_ion_stage,
        line_stark_w=ad.line_stark_w,
        line_mass_amu=ad.line_mass_amu,
        ionization_potentials=ip_arr,
        partition_coeffs=pf_arr,
        n_elements=len(elements),
        n_stages=3,
    )
    return ad, elements


@pytest.fixture
def synthetic_data():
    """Fixture providing synthetic atomic data and wavelength grid."""
    ad, elements = _make_synthetic_atomic_data()
    wl_grid = np.linspace(245.0, 350.0, 1000)
    return ad, elements, wl_grid


@pytest.fixture
def jax_synthetic_data(synthetic_data):
    """Fixture with JAX arrays."""
    ad, elements, wl_grid = synthetic_data
    if not HAS_JAX:
        pytest.skip("JAX not available")

    # Convert to JAX arrays
    ad_jax = BatchAtomicData(
        line_wavelengths=jnp.asarray(ad.line_wavelengths, dtype=jnp.float64),
        line_A_ki=jnp.asarray(ad.line_A_ki, dtype=jnp.float64),
        line_g_k=jnp.asarray(ad.line_g_k, dtype=jnp.float64),
        line_E_k=jnp.asarray(ad.line_E_k, dtype=jnp.float64),
        line_element_idx=jnp.asarray(ad.line_element_idx, dtype=jnp.int32),
        line_ion_stage=jnp.asarray(ad.line_ion_stage, dtype=jnp.int32),
        line_stark_w=jnp.asarray(ad.line_stark_w, dtype=jnp.float64),
        line_mass_amu=jnp.asarray(ad.line_mass_amu, dtype=jnp.float64),
        ionization_potentials=jnp.asarray(ad.ionization_potentials, dtype=jnp.float64),
        partition_coeffs=jnp.asarray(ad.partition_coeffs, dtype=jnp.float64),
        n_elements=ad.n_elements,
        n_stages=ad.n_stages,
    )
    wl_jax = jnp.asarray(wl_grid, dtype=jnp.float64)
    return ad_jax, elements, wl_jax


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_single_spectrum_shape(jax_synthetic_data):
    """Output is (N_wl,) for single spectrum."""
    ad, _, wl_grid = jax_synthetic_data
    T_eV = jnp.float64(1.0)
    n_e = jnp.float64(1e17)
    C = jnp.array([0.7, 0.2, 0.1], dtype=jnp.float64)

    spectrum = single_spectrum_forward(T_eV, n_e, C, wl_grid, ad)
    assert spectrum.shape == (wl_grid.shape[0],)
    assert spectrum.dtype == jnp.float64


@pytest.mark.unit
def test_batch_spectrum_shape(jax_synthetic_data):
    """Output is (B, N_wl) for batch."""
    ad, _, wl_grid = jax_synthetic_data
    B = 8
    T_eV = jnp.ones(B, dtype=jnp.float64) * 1.0
    n_e = jnp.ones(B, dtype=jnp.float64) * 1e17
    C = jnp.broadcast_to(
        jnp.array([0.7, 0.2, 0.1], dtype=jnp.float64), (B, 3)
    )

    spectra = batch_forward_model(T_eV, n_e, C, wl_grid, ad)
    assert spectra.shape == (B, wl_grid.shape[0])
    assert spectra.dtype == jnp.float64


# ---------------------------------------------------------------------------
# Batch vs sequential parity
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_batch_vs_sequential(jax_synthetic_data):
    """Batch output matches sequential single-spectrum calls to <1e-12."""
    ad, _, wl_grid = jax_synthetic_data
    B = 20

    rng = np.random.RandomState(42)
    T_eV_np = rng.uniform(0.6, 1.5, size=B)
    n_e_np = 10.0 ** rng.uniform(15, 18, size=B)
    # Random compositions on simplex
    raw = rng.dirichlet([1.0, 1.0, 1.0], size=B)
    C_np = raw

    T_eV = jnp.array(T_eV_np, dtype=jnp.float64)
    n_e = jnp.array(n_e_np, dtype=jnp.float64)
    C = jnp.array(C_np, dtype=jnp.float64)

    # Batch computation
    spectra_batch = batch_forward_model(T_eV, n_e, C, wl_grid, ad)

    # Sequential computation
    spectra_seq = []
    for i in range(B):
        s = single_spectrum_forward(T_eV[i], n_e[i], C[i], wl_grid, ad)
        spectra_seq.append(s)
    spectra_seq = jnp.stack(spectra_seq, axis=0)

    # Compare
    max_abs_diff = jnp.max(jnp.abs(spectra_batch - spectra_seq))
    # Relative error where signal is significant
    mask = spectra_seq > jnp.max(spectra_seq) * 1e-10
    if jnp.any(mask):
        rel_err = jnp.max(
            jnp.abs(spectra_batch - spectra_seq)
            / jnp.maximum(jnp.abs(spectra_seq), 1e-300)
            * mask
        )
        assert float(rel_err) < 1e-12, f"Max relative error: {float(rel_err)}"
    assert float(max_abs_diff) < 1e-20 or float(
        max_abs_diff / jnp.max(jnp.abs(spectra_seq))
    ) < 1e-12


# ---------------------------------------------------------------------------
# Single line test
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_single_line():
    """Single line per element produces a single Voigt peak at the correct wavelength."""
    if not HAS_JAX:
        pytest.skip("JAX not available")

    elements = ["Fe"]
    line_data = [
        {
            "wavelength_nm": 300.0,
            "A_ki": 1e8,
            "g_k": 5.0,
            "E_k_eV": 3.5,
            "element": "Fe",
            "ion_stage": 0,
            "stark_w_nm": 0.02,
            "mass_amu": 55.845,
        }
    ]
    ad = pack_atomic_data(elements, line_data, max_stages=3)
    ip_arr = np.array([[7.902, 16.19]])
    pf_arr = np.zeros((1, 3, 5))
    pf_arr[0, 0, 0] = np.log(9.0)
    pf_arr[0, 1, 0] = np.log(10.0)
    pf_arr[0, 2, 0] = 0.0

    ad = BatchAtomicData(
        line_wavelengths=jnp.asarray(ad.line_wavelengths),
        line_A_ki=jnp.asarray(ad.line_A_ki),
        line_g_k=jnp.asarray(ad.line_g_k),
        line_E_k=jnp.asarray(ad.line_E_k),
        line_element_idx=jnp.asarray(ad.line_element_idx),
        line_ion_stage=jnp.asarray(ad.line_ion_stage),
        line_stark_w=jnp.asarray(ad.line_stark_w),
        line_mass_amu=jnp.asarray(ad.line_mass_amu),
        ionization_potentials=jnp.asarray(ip_arr),
        partition_coeffs=jnp.asarray(pf_arr),
        n_elements=1,
        n_stages=3,
    )

    wl_grid = jnp.linspace(295.0, 305.0, 500)
    spectrum = single_spectrum_forward(
        jnp.float64(1.0), jnp.float64(1e17), jnp.array([1.0]), wl_grid, ad
    )

    # Peak should be near 300 nm
    peak_idx = jnp.argmax(spectrum)
    peak_wl = float(wl_grid[peak_idx])
    assert abs(peak_wl - 300.0) < 0.5, f"Peak at {peak_wl} nm, expected ~300 nm"
    assert float(jnp.max(spectrum)) > 0


# ---------------------------------------------------------------------------
# Single element test
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_single_element():
    """Single element (Fe) produces spectrum with correct structure."""
    if not HAS_JAX:
        pytest.skip("JAX not available")

    ad, elements = _make_synthetic_atomic_data(["Fe"])
    ad = BatchAtomicData(
        line_wavelengths=jnp.asarray(ad.line_wavelengths),
        line_A_ki=jnp.asarray(ad.line_A_ki),
        line_g_k=jnp.asarray(ad.line_g_k),
        line_E_k=jnp.asarray(ad.line_E_k),
        line_element_idx=jnp.asarray(ad.line_element_idx),
        line_ion_stage=jnp.asarray(ad.line_ion_stage),
        line_stark_w=jnp.asarray(ad.line_stark_w),
        line_mass_amu=jnp.asarray(ad.line_mass_amu),
        ionization_potentials=jnp.asarray(ad.ionization_potentials),
        partition_coeffs=jnp.asarray(ad.partition_coeffs),
        n_elements=1,
        n_stages=3,
    )

    wl_grid = jnp.linspace(255.0, 285.0, 500)
    spectrum = single_spectrum_forward(
        jnp.float64(1.0), jnp.float64(1e17), jnp.array([1.0]), wl_grid, ad
    )

    assert spectrum.shape == (500,)
    assert float(jnp.max(spectrum)) > 0
    assert jnp.all(jnp.isfinite(spectrum))


# ---------------------------------------------------------------------------
# Multi-element test
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_multi_element(jax_synthetic_data):
    """3 elements produce peaks in distinct wavelength regions."""
    ad, _, wl_grid = jax_synthetic_data
    C = jnp.array([0.5, 0.3, 0.2], dtype=jnp.float64)
    spectrum = single_spectrum_forward(
        jnp.float64(1.0), jnp.float64(1e17), C, wl_grid, ad
    )

    # Fe lines near 260 nm, Cu near 325 nm, Si near 250 nm
    # Check that there is signal in multiple distinct regions
    assert float(jnp.max(spectrum)) > 0
    # Fe region (255-275)
    fe_mask = (wl_grid >= 255) & (wl_grid <= 275)
    # Cu region (320-345)
    cu_mask = (wl_grid >= 320) & (wl_grid <= 345)
    assert float(jnp.max(spectrum[fe_mask])) > 0, "No Fe signal"
    assert float(jnp.max(spectrum[cu_mask])) > 0, "No Cu signal"


# ---------------------------------------------------------------------------
# Zero concentration test
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_zero_concentration(jax_synthetic_data):
    """Setting one element to C=0 removes its lines from spectrum."""
    ad, _, wl_grid = jax_synthetic_data

    # Full spectrum with Cu
    C_full = jnp.array([0.5, 0.3, 0.2], dtype=jnp.float64)
    spec_full = single_spectrum_forward(
        jnp.float64(1.0), jnp.float64(1e17), C_full, wl_grid, ad
    )

    # Zero Cu
    C_nocu = jnp.array([0.6, 0.0, 0.4], dtype=jnp.float64)
    spec_nocu = single_spectrum_forward(
        jnp.float64(1.0), jnp.float64(1e17), C_nocu, wl_grid, ad
    )

    # Cu region should have signal in full but not in nocu
    cu_mask = (wl_grid >= 320) & (wl_grid <= 345)
    cu_full = float(jnp.max(spec_full[cu_mask]))
    cu_nocu = float(jnp.max(spec_nocu[cu_mask]))
    assert cu_full > 0, "Cu signal missing from full spectrum"
    # Cu contribution should be essentially zero (only Fe/Si wing leakage)
    assert cu_nocu < cu_full * 0.01, (
        f"Cu signal not removed: full={cu_full:.2e}, nocu={cu_nocu:.2e}"
    )


# ---------------------------------------------------------------------------
# Temperature dependence test
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_temperature_dependence(jax_synthetic_data):
    """Higher T strengthens high-E_k lines (Boltzmann effect)."""
    ad, _, wl_grid = jax_synthetic_data
    C = jnp.array([0.7, 0.2, 0.1], dtype=jnp.float64)

    spec_low = single_spectrum_forward(
        jnp.float64(0.7), jnp.float64(1e17), C, wl_grid, ad
    )
    spec_high = single_spectrum_forward(
        jnp.float64(1.3), jnp.float64(1e17), C, wl_grid, ad
    )

    # Higher temperature should produce stronger total emission
    # (more upper-level population for same density)
    total_low = float(jnp.sum(spec_low))
    total_high = float(jnp.sum(spec_high))
    assert total_high > total_low, (
        f"Higher T should give more emission: low={total_low:.2e}, high={total_high:.2e}"
    )


# ---------------------------------------------------------------------------
# Density dependence test
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_density_dependence(jax_synthetic_data):
    """Higher n_e produces broader Stark widths."""
    ad, _, wl_grid = jax_synthetic_data
    C = jnp.array([0.7, 0.2, 0.1], dtype=jnp.float64)

    spec_low_ne = single_spectrum_forward(
        jnp.float64(1.0), jnp.float64(1e15), C, wl_grid, ad
    )
    spec_high_ne = single_spectrum_forward(
        jnp.float64(1.0), jnp.float64(1e18), C, wl_grid, ad
    )

    # Higher n_e means broader lines, so peak should be lower relative to
    # total (more spread out)
    # But n_e also scales the population, so normalize by total
    peak_low = float(jnp.max(spec_low_ne))
    total_low = float(jnp.sum(spec_low_ne))
    peak_high = float(jnp.max(spec_high_ne))
    total_high = float(jnp.sum(spec_high_ne))

    ratio_low = peak_low / max(total_low, 1e-300)
    ratio_high = peak_high / max(total_high, 1e-300)

    # Broader lines -> lower peak-to-total ratio
    assert ratio_high < ratio_low, (
        f"Higher n_e should broaden lines: ratio_low={ratio_low:.4e}, ratio_high={ratio_high:.4e}"
    )


# ---------------------------------------------------------------------------
# Batch size invariance test
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_batch_size_invariance(jax_synthetic_data):
    """Same parameters at B=1 and B=64 produce identical results."""
    ad, _, wl_grid = jax_synthetic_data

    T = jnp.float64(1.0)
    ne = jnp.float64(1e17)
    C = jnp.array([0.7, 0.2, 0.1], dtype=jnp.float64)

    # B=1
    spec_1 = batch_forward_model(
        T[None], ne[None], C[None], wl_grid, ad
    )

    # B=64 (all same parameters)
    spec_64 = batch_forward_model(
        jnp.full(64, T),
        jnp.full(64, ne),
        jnp.broadcast_to(C, (64, 3)),
        wl_grid,
        ad,
    )

    # All should be identical
    assert spec_1.shape == (1, wl_grid.shape[0])
    assert spec_64.shape == (64, wl_grid.shape[0])

    for i in range(64):
        max_diff = float(jnp.max(jnp.abs(spec_64[i] - spec_1[0])))
        assert max_diff == 0.0, f"Batch element {i} differs by {max_diff}"


# ---------------------------------------------------------------------------
# CPU fallback test
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cpu_fallback(jax_synthetic_data):
    """Run with JAX_PLATFORMS=cpu produces correct output."""
    # We're already on CPU (set in conftest.py), so just verify it works
    ad, _, wl_grid = jax_synthetic_data
    spectrum = single_spectrum_forward(
        jnp.float64(1.0),
        jnp.float64(1e17),
        jnp.array([0.7, 0.2, 0.1], dtype=jnp.float64),
        wl_grid,
        ad,
    )
    assert jnp.all(jnp.isfinite(spectrum))
    assert float(jnp.max(spectrum)) > 0
    assert spectrum.dtype == jnp.float64


# ---------------------------------------------------------------------------
# Dimensional consistency test
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_dimensional_consistency(jax_synthetic_data):
    """S(lambda) > 0, integral gives total radiance [W m^-3 sr^-1]."""
    ad, _, wl_grid = jax_synthetic_data
    spectrum = single_spectrum_forward(
        jnp.float64(1.0),
        jnp.float64(1e17),
        jnp.array([0.7, 0.2, 0.1], dtype=jnp.float64),
        wl_grid,
        ad,
    )

    # Spectrum should be non-negative
    assert jnp.all(spectrum >= 0), "Spectrum has negative values"

    # Integral over wavelength gives total radiance
    # S(lambda) [W m^-3 sr^-1 nm^-1], integral over nm gives [W m^-3 sr^-1]
    dlambda = float(wl_grid[1] - wl_grid[0])  # nm
    total_radiance = float(jnp.sum(spectrum) * dlambda)
    assert total_radiance > 0, "Total radiance should be positive"
    assert np.isfinite(total_radiance), "Total radiance is not finite"
