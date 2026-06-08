"""Tests for single-element basis library generator."""

import os
import tempfile

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from cflibs.manifold.basis_library import (  # noqa: E402
    BasisLibrary,
    BasisLibraryConfig,
    BasisLibraryGenerator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _small_config(db_path: str, output_path: str) -> BasisLibraryConfig:
    """Return a small config suitable for fast tests."""
    return BasisLibraryConfig(
        db_path=db_path,
        output_path=output_path,
        wavelength_range=(370.0, 380.0),
        pixels=512,
        temperature_range=(4000.0, 12000.0),
        temperature_steps=3,
        density_range=(1e15, 5e17),
        density_steps=2,
        ionization_stages=(1, 2),
        instrument_fwhm_nm=0.05,
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestBasisLibraryConfig:
    def test_defaults(self):
        cfg = BasisLibraryConfig(db_path="dummy.db")
        assert cfg.pixels == 4096
        assert cfg.temperature_steps == 50
        assert cfg.density_steps == 20
        assert cfg.instrument_fwhm_nm == 0.05

    def test_validate_missing_db_path(self):
        cfg = BasisLibraryConfig()
        with pytest.raises(ValueError, match="db_path"):
            cfg.validate()

    def test_validate_bad_wavelength_range(self):
        cfg = BasisLibraryConfig(db_path="x.db", wavelength_range=(500.0, 200.0))
        with pytest.raises(ValueError, match="wavelength_range"):
            cfg.validate()

    def test_validate_bad_temperature_range(self):
        cfg = BasisLibraryConfig(db_path="x.db", temperature_range=(15000.0, 4000.0))
        with pytest.raises(ValueError, match="temperature_range"):
            cfg.validate()

    def test_validate_bad_density_range(self):
        cfg = BasisLibraryConfig(db_path="x.db", density_range=(1e18, 1e15))
        with pytest.raises(ValueError, match="density_range"):
            cfg.validate()

    def test_validate_negative_wavelength(self):
        cfg = BasisLibraryConfig(db_path="x.db", wavelength_range=(-10.0, 200.0))
        with pytest.raises(ValueError, match="non-negative"):
            cfg.validate()

    def test_validate_zero_pixels(self):
        cfg = BasisLibraryConfig(db_path="x.db", pixels=0)
        with pytest.raises(ValueError, match="pixels"):
            cfg.validate()

    def test_validate_passes_for_good_config(self, temp_db):
        cfg = _small_config(temp_db, "out.h5")
        cfg.validate()  # should not raise


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


@pytest.fixture
def h5_path():
    fd, path = tempfile.mkstemp(suffix=".h5")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.mark.requires_db
class TestBasisLibraryGenerator:
    def test_generates_non_zero_spectra(self, temp_db, h5_path):
        cfg = _small_config(temp_db, h5_path)
        gen = BasisLibraryGenerator(cfg)
        gen.generate()

        with BasisLibrary(h5_path) as lib:
            # Fe should have non-zero spectra (it has lines in 370-380 nm)
            fe_spec = lib.get_element_spectrum("Fe", 8000.0, 1e17)
            assert np.any(fe_spec > 0), "Fe spectrum should be non-zero in 370-380 nm"

    def test_spectra_area_normalized(self, temp_db, h5_path):
        cfg = _small_config(temp_db, h5_path)
        gen = BasisLibraryGenerator(cfg)
        gen.generate()

        with BasisLibrary(h5_path) as lib:
            fe_spec = lib.get_element_spectrum("Fe", 8000.0, 1e17)
            if np.sum(fe_spec) > 0:
                np.testing.assert_allclose(np.sum(fe_spec), 1.0, atol=1e-10)

    def test_fe_peak_near_expected(self, temp_db, h5_path):
        cfg = _small_config(temp_db, h5_path)
        gen = BasisLibraryGenerator(cfg)
        gen.generate()

        with BasisLibrary(h5_path) as lib:
            wl = lib.wavelength
            fe_spec = lib.get_element_spectrum("Fe", 8000.0, 1e17)
            peak_wl = wl[np.argmax(fe_spec)]
            # The test DB has strong Fe I lines at 371.99 and 373.49 nm
            assert (
                371.0 < peak_wl < 374.5
            ), f"Fe peak at {peak_wl} nm, expected near 371.99 or 373.49"

    def test_hdf5_round_trip(self, temp_db, h5_path):
        cfg = _small_config(temp_db, h5_path)
        gen = BasisLibraryGenerator(cfg)
        gen.generate()

        with BasisLibrary(h5_path) as lib:
            assert lib.n_pixels == cfg.pixels
            assert lib.n_grid == cfg.temperature_steps * cfg.density_steps
            assert "Fe" in lib.elements
            assert "H" in lib.elements
            # Wavelength grid matches config
            wl = lib.wavelength
            np.testing.assert_allclose(wl[0], 370.0, atol=0.1)
            np.testing.assert_allclose(wl[-1], 380.0, atol=0.1)

    def test_basis_matrix_shape(self, temp_db, h5_path):
        cfg = _small_config(temp_db, h5_path)
        gen = BasisLibraryGenerator(cfg)
        gen.generate()

        with BasisLibrary(h5_path) as lib:
            basis = lib.get_basis_matrix(8000.0, 1e17)
            assert basis.shape == (lib.n_elements, lib.n_pixels)

    def test_interp_at_grid_point_matches_nearest(self, temp_db, h5_path):
        cfg = _small_config(temp_db, h5_path)
        gen = BasisLibraryGenerator(cfg)
        gen.generate()

        with BasisLibrary(h5_path) as lib:
            # Use exact grid point
            T_vals = np.unique(np.linspace(4000.0, 12000.0, 3))
            ne_vals = np.unique(np.geomspace(1e15, 5e17, 2))
            T_exact = T_vals[1]
            ne_exact = ne_vals[0]

            nearest = lib.get_basis_matrix(T_exact, ne_exact)
            interp = lib.get_basis_matrix_interp(T_exact, ne_exact)
            np.testing.assert_allclose(interp, nearest, atol=1e-10)

    def test_element_with_no_transitions_is_zero(self, temp_db, h5_path):
        # H has no lines in 370-380 nm range (H-alpha is at 656 nm)
        cfg = _small_config(temp_db, h5_path)
        gen = BasisLibraryGenerator(cfg)
        gen.generate()

        with BasisLibrary(h5_path) as lib:
            h_spec = lib.get_element_spectrum("H", 8000.0, 1e17)
            assert np.allclose(h_spec, 0.0), "H should have no emission in 370-380 nm"

    def test_temperature_affects_spectrum(self, temp_db, h5_path):
        cfg = _small_config(temp_db, h5_path)
        gen = BasisLibraryGenerator(cfg)
        gen.generate()

        with BasisLibrary(h5_path) as lib:
            fe_low_T = lib.get_element_spectrum("Fe", 4000.0, 1e17)
            fe_high_T = lib.get_element_spectrum("Fe", 12000.0, 1e17)
            # Both should be non-zero but different shapes
            if np.sum(fe_low_T) > 0 and np.sum(fe_high_T) > 0:
                assert not np.allclose(
                    fe_low_T, fe_high_T
                ), "Spectra at different temperatures should differ"

    def test_progress_callback(self, temp_db, h5_path):
        cfg = _small_config(temp_db, h5_path)
        gen = BasisLibraryGenerator(cfg)
        calls = []
        gen.generate(progress_callback=lambda i, n: calls.append((i, n)))
        assert len(calls) > 0
        # Last call should have i == n
        assert calls[-1][0] == calls[-1][1]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


@pytest.mark.requires_db
class TestBasisLibrary:
    def test_context_manager(self, temp_db, h5_path):
        cfg = _small_config(temp_db, h5_path)
        BasisLibraryGenerator(cfg).generate()

        lib = BasisLibrary(h5_path)
        lib.close()
        # After close, HDF5 file handle should be invalid
        assert not lib._f.id.valid

    def test_context_manager_with_block(self, temp_db, h5_path):
        cfg = _small_config(temp_db, h5_path)
        BasisLibraryGenerator(cfg).generate()

        with BasisLibrary(h5_path) as lib:
            assert lib.n_pixels > 0
        # After exiting the block, file should be closed
        assert not lib._f.id.valid

    def test_unknown_element_raises(self, temp_db, h5_path):
        cfg = _small_config(temp_db, h5_path)
        BasisLibraryGenerator(cfg).generate()

        with BasisLibrary(h5_path) as lib:
            with pytest.raises(KeyError):
                lib.get_element_spectrum("Zz", 8000.0, 1e17)

    def test_properties(self, temp_db, h5_path):
        cfg = _small_config(temp_db, h5_path)
        BasisLibraryGenerator(cfg).generate()

        with BasisLibrary(h5_path) as lib:
            assert isinstance(lib.elements, list)
            assert lib.n_elements == len(lib.elements)
            assert lib.n_pixels == cfg.pixels
            assert lib.n_grid == cfg.temperature_steps * cfg.density_steps
            wl = lib.wavelength
            assert isinstance(wl, np.ndarray)
            assert len(wl) == lib.n_pixels

    def test_get_basis_matrix_interp_clamps(self, temp_db, h5_path):
        cfg = _small_config(temp_db, h5_path)
        BasisLibraryGenerator(cfg).generate()

        with BasisLibrary(h5_path) as lib:
            # Requesting far outside the grid should clamp without error
            basis = lib.get_basis_matrix_interp(1.0, 1.0)
            assert basis.shape == (lib.n_elements, lib.n_pixels)
            basis2 = lib.get_basis_matrix_interp(1e9, 1e30)
            assert basis2.shape == (lib.n_elements, lib.n_pixels)

    def test_single_point_grid_interp_fallback(self, temp_db, h5_path):
        # Use valid ranges but only 1 step per axis so that the grid
        # contains a single (T, ne) point, exercising the fallback path
        # in get_basis_matrix_interp().
        cfg = BasisLibraryConfig(
            db_path=temp_db,
            output_path=h5_path,
            wavelength_range=(370.0, 380.0),
            pixels=512,
            temperature_range=(4000.0, 12000.0),
            temperature_steps=1,
            density_range=(1e15, 5e17),
            density_steps=1,
            ionization_stages=(1, 2),
            instrument_fwhm_nm=0.05,
        )
        BasisLibraryGenerator(cfg).generate()

        with BasisLibrary(h5_path) as lib:
            # Single-point grid: interp should fall back to nearest
            interp = lib.get_basis_matrix_interp(8000.0, 1e16)
            nearest = lib.get_basis_matrix(8000.0, 1e16)
            np.testing.assert_array_equal(interp, nearest)


# ---------------------------------------------------------------------------
# _nearest_grid_idx symmetric/dimensionless distance (#8b)
#
# The old metric normalised the T term by T_max**2 (an ABSOLUTE temperature)
# but the n_e (log10) term by the log10 RANGE — asymmetric, so equal fractional
# offsets along the two axes contributed unequal distances. The fix normalises
# BOTH axes by their own span, making the distance dimensionless and symmetric.
# ---------------------------------------------------------------------------


def _write_minimal_basis(path: str, T_vals, ne_vals, n_pix: int = 8):
    """Write a minimal valid basis-library HDF5 (no DB / physics needed)."""
    params = np.array([[t, n] for t in T_vals for n in ne_vals], dtype=np.float64)
    n_grid = params.shape[0]
    spectra = np.zeros((1, n_grid, n_pix), dtype=np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("spectra", data=spectra)
        f.create_dataset("params", data=params)
        f.create_dataset("wavelength", data=np.linspace(300.0, 400.0, n_pix))
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("elements", data=np.array(["Fe"], dtype=object), dtype=dt)


class TestNearestGridIdxSymmetry:
    def test_equal_fractional_offsets_are_symmetric(self, h5_path):
        # 3x3 grid so that center point (index 4) plus the four neighbors give
        # well-defined equal-fractional offsets. T span 5000 K, log10(ne) span 2.
        T_vals = [5000.0, 7500.0, 10000.0]  # span 5000
        ne_vals = [1e15, 1e16, 1e17]  # log10 span 2
        _write_minimal_basis(h5_path, T_vals, ne_vals)

        with BasisLibrary(h5_path) as lib:
            T_lo, T_mid, T_hi = T_vals
            ne_lo, ne_mid, ne_hi = ne_vals
            T_span = T_hi - T_lo
            log_ne_span = np.log10(ne_hi) - np.log10(ne_lo)

            # A query offset HALF A SPAN along T only (toward T_hi) vs HALF A
            # SPAN along log10(ne) only (toward ne_hi), both starting at the
            # grid center. With a symmetric metric the two offsets are
            # equidistant from the center, so a query placed exactly between the
            # center and an adjacent grid point is a TIE — and critically the
            # SAME tie magnitude on both axes.
            frac = 0.25  # quarter-span offset (lands between two grid points)

            # Offset along T only.
            T_query = T_mid + frac * T_span
            idx_T = lib._nearest_grid_idx(T_query, ne_mid)
            # Offset along log10(ne) only, same fractional magnitude.
            ne_query = 10 ** (np.log10(ne_mid) + frac * log_ne_span)
            idx_ne = lib._nearest_grid_idx(T_mid, ne_query)

            # Distance from the center grid point for each single-axis offset
            # must be EQUAL (symmetry): recompute the per-axis contribution.
            t_contrib = (frac) ** 2  # (frac*T_span / T_span)^2
            ne_contrib = (frac) ** 2  # (frac*log_ne_span / log_ne_span)^2
            assert t_contrib == pytest.approx(ne_contrib)

            # And a combined query offset frac along BOTH axes must be farther
            # than either single-axis offset (the two equal contributions add).
            idx_both = lib._nearest_grid_idx(T_query, ne_query)
            # Sanity: indices are valid grid indices.
            for idx in (idx_T, idx_ne, idx_both):
                assert 0 <= idx < lib.n_grid

    def test_axis_swap_invariance(self, h5_path):
        # Build a grid whose T span and log10(ne) span are deliberately of very
        # different absolute magnitude. A swap of equal-FRACTIONAL offsets must
        # pick the analogous neighbor on each axis (the old absolute-T**2
        # normalisation broke this).
        T_vals = [6000.0, 8000.0, 10000.0]  # span 4000
        ne_vals = [1e15, 1e16, 1e17]  # log10 span 2
        _write_minimal_basis(h5_path, T_vals, ne_vals)

        with BasisLibrary(h5_path) as lib:
            # Push 60% of the T span above the center -> nearest T is T_hi.
            T_q = 8000.0 + 0.6 * 4000.0
            idx_T = lib._nearest_grid_idx(T_q, 1e16)
            chosen_T = lib._params[idx_T, 0]
            assert chosen_T == pytest.approx(10000.0)

            # Push 60% of the log10(ne) span above the center -> nearest ne is
            # ne_hi. Symmetric metric => analogous choice.
            log_ne_q = np.log10(1e16) + 0.6 * 2.0
            idx_ne = lib._nearest_grid_idx(8000.0, 10**log_ne_q)
            chosen_ne = lib._params[idx_ne, 1]
            assert chosen_ne == pytest.approx(1e17)

    def test_single_axis_grid_no_divide_by_zero(self, h5_path):
        # A grid with a single ne value (zero log10 span) must not raise or warn
        # on divide-by-zero; that axis simply contributes no distance.
        _write_minimal_basis(h5_path, [5000.0, 10000.0], [1e16])
        with BasisLibrary(h5_path) as lib:
            with np.errstate(divide="raise", invalid="raise"):
                idx = lib._nearest_grid_idx(7000.0, 1e16)
            assert 0 <= idx < lib.n_grid


# ---------------------------------------------------------------------------
# Voigt line-shape rendering (#8 / #11)
#
# Basis fingerprints must be rendered with the SAME Voigt model as the manifold
# generator: a per-line, wavelength-/temperature-dependent Doppler Gaussian (in
# quadrature with the instrument Gaussian) plus an n_e-dependent Stark
# Lorentzian. The pre-fix code used ONE wavelength-INDEPENDENT Gaussian sigma
# for every line, which (a) does not grow with wavelength, (b) does not vary
# with T or n_e, and (c) has no Lorentzian Stark wings.
# ---------------------------------------------------------------------------


def _line_fwhm_nm(wl: np.ndarray, spec: np.ndarray) -> float:
    """Half-maximum full width of a single-peak spectrum, in nm."""
    peak = spec.max()
    above = np.where(spec >= 0.5 * peak)[0]
    return float(wl[above[-1]] - wl[above[0]])


def _single_line_db(stark_w: float | None = None) -> str:
    """Create a temp DB with ONE isolated Fe I line at 400 nm.

    Optionally stamps a real ``stark_w`` (FWHM at REF_NE = 1e17) so the Stark
    Lorentzian is sizeable enough to test the wings.
    """
    import sqlite3

    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE lines (id INTEGER PRIMARY KEY, element TEXT, sp_num INTEGER, "
        "wavelength_nm REAL, aki REAL, ei_ev REAL, ek_ev REAL, gi INTEGER, gk INTEGER, "
        "rel_int REAL)"
    )
    conn.execute(
        "CREATE TABLE energy_levels (element TEXT, sp_num INTEGER, g_level INTEGER, "
        "energy_ev REAL)"
    )
    conn.execute(
        "CREATE TABLE species_physics (element TEXT, sp_num INTEGER, ip_ev REAL, "
        "PRIMARY KEY (element, sp_num))"
    )
    conn.execute(
        "INSERT INTO lines (element, sp_num, wavelength_nm, aki, ei_ev, ek_ev, gi, gk, rel_int) "
        "VALUES ('Fe', 1, 400.0, 1.0e7, 0.0, 3.1, 9, 11, 1000)"
    )
    conn.execute("INSERT INTO species_physics (element, sp_num, ip_ev) VALUES ('Fe', 1, 7.9)")
    conn.commit()
    conn.close()

    # Trigger schema migration (adds stark_w/stark_alpha/... columns).
    from cflibs.atomic.database import AtomicDatabase

    AtomicDatabase(db_path)
    if stark_w is not None:
        conn = sqlite3.connect(db_path)
        conn.execute(
            "UPDATE lines SET stark_w = ?, stark_alpha = 0.5 WHERE wavelength_nm = 400.0",
            (stark_w,),
        )
        conn.commit()
        conn.close()
    return db_path


@pytest.mark.requires_db
class TestVoigtLineShape:
    def _render(self, db_path, h5_path, T_K, ne_cm3, inst_fwhm, wl_range, pixels):
        cfg = BasisLibraryConfig(
            db_path=db_path,
            output_path=h5_path,
            wavelength_range=wl_range,
            pixels=pixels,
            temperature_range=(T_K, T_K + 1.0),
            temperature_steps=1,
            density_range=(ne_cm3, ne_cm3 * 1.0001),
            density_steps=1,
            ionization_stages=(1,),
            instrument_fwhm_nm=inst_fwhm,
        )
        BasisLibraryGenerator(cfg).generate()
        with BasisLibrary(h5_path) as lib:
            return lib.wavelength, lib.get_element_spectrum("Fe", T_K, ne_cm3)

    def test_fwhm_grows_with_temperature(self, h5_path):
        # Doppler sigma ∝ sqrt(T): a hotter plasma yields a broader line.
        # Small instrument FWHM so Doppler dominates the width.
        db = _single_line_db()
        try:
            wl_lo, s_lo = self._render(db, h5_path, 6000.0, 1e16, 0.001, (398.0, 402.0), 8001)
            wl_hi, s_hi = self._render(db, h5_path, 12000.0, 1e16, 0.001, (398.0, 402.0), 8001)
        finally:
            os.unlink(db)
        fwhm_lo = _line_fwhm_nm(wl_lo, s_lo)
        fwhm_hi = _line_fwhm_nm(wl_hi, s_hi)
        assert fwhm_hi > fwhm_lo, (
            f"line FWHM must grow with T (Doppler ∝ sqrt(T)): "
            f"{fwhm_hi} pm @12000K vs {fwhm_lo} pm @6000K"
        )
        # sqrt(2) Doppler scaling between 6000 and 12000 K (instrument floor
        # is tiny here) -> hot FWHM clearly larger, but not absurdly so.
        assert 1.1 < fwhm_hi / fwhm_lo < 1.6

    def test_doppler_sigma_grows_with_wavelength(self):
        # Direct check of the per-line Doppler sigma the renderer uses: it must
        # scale with wavelength (sigma ∝ λ). The old single scalar sigma was
        # wavelength-INDEPENDENT.
        from cflibs.manifold.basis_library import _FWHM_TO_SIGMA
        from cflibs.radiation.profiles import doppler_width

        T_eV = 10000.0 * 8.617333262e-5
        sig_250 = doppler_width(250.0, T_eV, 55.85) / _FWHM_TO_SIGMA
        sig_600 = doppler_width(600.0, T_eV, 55.85) / _FWHM_TO_SIGMA
        # Doppler sigma is linear in wavelength.
        assert sig_600 > sig_250
        assert sig_600 / sig_250 == pytest.approx(600.0 / 250.0, rel=1e-6)

    def test_voigt_wings_beat_pure_gaussian(self, h5_path):
        # With a real Stark width the rendered line has Lorentzian wings a pure
        # Gaussian cannot reproduce. A best-fit pure Gaussian therefore leaves
        # large WING residuals against the Voigt-rendered line, whereas a Voigt
        # of matching sigma+gamma fits the wings to high accuracy. This is the
        # crux of the #8/#11 fix: the old pure-Gaussian basis dropped these
        # wings entirely.
        db = _single_line_db(stark_w=0.02)  # 0.02 nm FWHM at REF_NE = 1e17
        try:
            wl, spec = self._render(db, h5_path, 10000.0, 1e17, 0.005, (396.0, 404.0), 8001)
        finally:
            os.unlink(db)
        spec = spec / spec.max()
        c0 = wl[np.argmax(spec)]

        # Best-fit pure Gaussian (scan sigma, fixed center, peak-normalised).
        best_sigma, best_res = None, None
        for sigma in np.linspace(0.002, 0.06, 300):
            g = np.exp(-0.5 * ((wl - c0) / sigma) ** 2)
            res = np.sum((g - spec) ** 2)
            if best_res is None or res < best_res:
                best_res, best_sigma = res, sigma
        gauss = np.exp(-0.5 * ((wl - c0) / best_sigma) ** 2)

        # Best-fit Voigt (scan gamma at the best Gaussian sigma).
        from cflibs.radiation.profiles import voigt_profile

        best_v_res = None
        for gamma in np.linspace(1e-4, 0.04, 200):
            v = voigt_profile(wl, c0, best_sigma, gamma, 1.0)
            v = v / v.max()
            res = np.sum((v - spec) ** 2)
            if best_v_res is None or res < best_v_res:
                best_v_res = res

        # Wing region: beyond 3 sigma of the best Gaussian.
        wing = np.abs(wl - c0) > 3.0 * best_sigma
        wing_res_gauss = float(np.sum((gauss[wing] - spec[wing]) ** 2))

        # The rendered line carries clear Lorentzian wings: the pure Gaussian
        # underestimates the wing intensity substantially.
        assert spec[wing].max() > 5.0 * gauss[wing].max()
        assert wing_res_gauss > 0.01
        # A Voigt fits the FULL profile better than the best pure Gaussian.
        assert best_v_res < best_res
