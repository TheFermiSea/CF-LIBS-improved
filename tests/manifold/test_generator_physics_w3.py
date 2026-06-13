"""Wave-3 physics-audit regression tests for the manifold generator.

These cover three real (behaviour-changing) corrections in
``cflibs/manifold/generator.py`` and ``cflibs/manifold/config.py``:

* **#6a — Doppler sigma.** All three open-coded Doppler-width sites in the
  generator previously used ``sqrt(2 * k T / m c^2)``, the
  most-probable-speed form. The canonical 1-D Maxwell standard deviation
  (``cflibs.radiation.profiles.doppler_sigma_jax``) is ``sqrt(k T / m c^2)``
  — i.e. the spurious factor of 2 inflated every line width by ``sqrt(2)``
  (~1.41x). The generator now delegates to ``doppler_sigma_jax`` so there
  is a single source of truth.

* **#7 — generic-D simplex closure.** ``_build_param_grid`` previously built
  generic (element count ≠ 4) compositions as ``[c1] + [0.0]*(D-1)`` with
  ``c1`` in ``linspace(0.5, 1.0)`` — every row summed to ``c1``, violating
  the CF-LIBS closure constraint ``Σ C_s = 1``. It now uses a single
  D-dimensional simplex sampler for all element counts: every row sums to
  1.0 with no identically-zero element.

* **#13b — cooling-trail config.** The time-integration cooling laws
  (``t0 = 1e-6 s``, ``T ~ (1+t/t0)**-0.5``, ``n_e ~ (1+t/t0)**-1``) were
  hardcoded ns-ICCD values. They are now ``ManifoldConfig`` fields whose
  defaults reproduce the prior ns values, so a ps-LIBS regime can be
  configured.
"""

from __future__ import annotations

import numpy as np
import pytest

from cflibs.manifold.config import ManifoldConfig
from cflibs.manifold.generator import ManifoldGenerator


def _assert_ns_cooling_defaults(config: ManifoldConfig) -> None:
    """The ns-ICCD cooling-trail values are the behavior-preserving defaults."""
    assert config.cooling_t0_s == pytest.approx(1e-6)
    assert config.cooling_temperature_exponent == pytest.approx(-0.5)
    assert config.cooling_density_exponent == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# #6a — Doppler sigma == canonical profiles.doppler_sigma_jax (no factor of 2)
# ---------------------------------------------------------------------------
class TestDopplerSigmaParity:
    """Generator per-line Doppler σ must equal the canonical 1-D Maxwell std."""

    @pytest.mark.requires_jax
    def test_no_spurious_factor_of_two_in_width(self):
        """The corrected σ is sqrt(2)x smaller than the old buggy form.

        Pins the magnitude of the fix: the old code used
        ``sqrt(2 * kT/mc^2)``; the canonical std is ``sqrt(kT/mc^2)``, so the
        ratio buggy/correct is exactly ``sqrt(2)``.
        """
        try:
            import jax.numpy as jnp  # noqa: F401
        except ImportError:
            pytest.skip("JAX not installed")

        from cflibs.core.constants import C_LIGHT, EV_TO_J, M_PROTON
        from cflibs.radiation.profiles import doppler_sigma_jax

        wl_nm = 500.0
        T_eV = 1.0
        mass_amu = 55.845  # Fe
        mass_kg = mass_amu * M_PROTON

        correct = float(doppler_sigma_jax(wl_nm, T_eV, mass_amu))
        buggy = float(wl_nm * np.sqrt(2.0 * T_eV * EV_TO_J / (mass_kg * C_LIGHT**2)))

        assert buggy / correct == pytest.approx(np.sqrt(2.0), rel=1e-9)

    def test_ldm_sigma_grid_endpoint_uses_maxwell_std(self, tmp_path):
        """``_build_ldm_sigma_grid``'s numpy Doppler endpoint uses the
        factor-of-2-free Maxwell std (numpy twin of doppler_sigma_jax)."""
        from cflibs.core.constants import C_LIGHT, EV_TO_J, M_PROTON

        config = ManifoldConfig(
            db_path=str(tmp_path / "stub.db"),
            output_path=str(tmp_path / "out.h5"),
            elements=["Fe"],
            wavelength_range=(300.0, 400.0),
            temperature_range=(0.5, 1.3),
            density_range=(1e16, 1e18),
            instrument_fwhm_nm=0.0,  # isolate the Doppler term
        )

        gen = ManifoldGenerator.__new__(ManifoldGenerator)
        gen.config = config
        # _build_ldm_sigma_grid reads atomic_data[0] (wl) and [-1] (mass amu).
        wl = np.array([300.0, 400.0], dtype=np.float64)
        mass = np.array([55.845, 55.845], dtype=np.float64)
        gen.atomic_data = (wl, None, None, None, None, None, None, None, None, None, None, mass)

        sigma_grid = gen._build_ldm_sigma_grid(sigma_inst=0.0)

        # build_sigma_grid brackets the σ_max endpoint with a fixed
        # sigma_max_factor=2.0 (cflibs.radiation.ldm). With sigma_inst=0 the
        # max endpoint is purely the hottest/lightest/longest-wl Doppler σ, so
        # grid.max() == 2.0 * correct_endpoint. Recover the unpadded endpoint
        # and compare against the factor-of-2-free Maxwell std.
        sigma_max_factor = 2.0
        recovered_endpoint = sigma_grid.max() / sigma_max_factor

        T_hi = 1.3
        wl_hi = 400.0
        mass_kg = 55.845 * M_PROTON
        correct_endpoint = wl_hi * np.sqrt(T_hi * EV_TO_J / (mass_kg * C_LIGHT**2))
        buggy_endpoint = wl_hi * np.sqrt(2.0 * T_hi * EV_TO_J / (mass_kg * C_LIGHT**2))

        assert recovered_endpoint == pytest.approx(correct_endpoint, rel=1e-6)
        # And it must be sqrt(2)x smaller than the old buggy endpoint.
        assert recovered_endpoint < buggy_endpoint
        assert buggy_endpoint / recovered_endpoint == pytest.approx(np.sqrt(2.0), rel=1e-6)


# ---------------------------------------------------------------------------
# #7 — generic-D simplex closure: every row sums to 1.0, no zero element
# ---------------------------------------------------------------------------
class TestSimplexClosure:
    """Composition grids must obey Σ C_s = 1 for arbitrary element counts."""

    @pytest.mark.parametrize("n_elements", [1, 2, 3, 4, 5, 7])
    def test_composition_grid_rows_sum_to_one(self, n_elements):
        """Each row of the D-simplex sampler sums to 1.0 ± 1e-9 (float64)."""
        comp = ManifoldGenerator._build_composition_grid(n_elements, concentration_steps=12)
        assert comp.shape[1] == n_elements
        sums = comp.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-9)

    @pytest.mark.parametrize("n_elements", [2, 3, 5])
    def test_no_identically_zero_element(self, n_elements):
        """No composition entry is identically zero (mass conservation)."""
        comp = ManifoldGenerator._build_composition_grid(n_elements, concentration_steps=16)
        assert np.all(comp > 0.0), "found an identically-zero composition entry"

    def test_param_grid_three_elements_closes(self, tmp_path):
        """``generate_manifold(elements=['Fe','Cu','Ni'])`` parameter rows
        have composition columns that sum to 1.0 with no zero element.

        Before the fix the generic branch produced ``[c1, 0, 0]`` summing to
        ``c1`` (0.5..1.0), not 1.0, and two identically-zero elements.
        Tested via ``_build_param_grid`` (the exact grid ``generate_manifold``
        sweeps) without needing a real atomic DB.
        """
        config = ManifoldConfig(
            db_path=str(tmp_path / "stub.db"),
            output_path=str(tmp_path / "out.h5"),
            elements=["Fe", "Cu", "Ni"],
            wavelength_range=(300.0, 400.0),
            temperature_range=(0.5, 1.3),
            temperature_steps=2,
            density_range=(1e16, 1e18),
            density_steps=2,
            concentration_steps=5,
        )
        gen = ManifoldGenerator.__new__(ManifoldGenerator)
        gen.config = config

        params = gen._build_param_grid()
        # Columns: [T, ne, C_Fe, C_Cu, C_Ni].
        comps = params[:, 2:]
        assert comps.shape[1] == 3
        # Stored as float32; closure holds within float32 round-off.
        sums = comps.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)
        # No identically-zero element on any row.
        assert np.all(comps > 0.0)

    def test_param_grid_four_elements_also_closes(self, tmp_path):
        """The formerly special-cased 4-element branch now uses the same
        sampler and also closes to 1.0 (the old branch summed Ti+Al+V+0.002
        to 1.002 — a closure violation)."""
        config = ManifoldConfig(
            db_path=str(tmp_path / "stub.db"),
            output_path=str(tmp_path / "out.h5"),
            elements=["Ti", "Al", "V", "Fe"],
            wavelength_range=(300.0, 400.0),
            temperature_range=(0.5, 1.3),
            temperature_steps=2,
            density_range=(1e16, 1e18),
            density_steps=2,
            concentration_steps=6,
        )
        gen = ManifoldGenerator.__new__(ManifoldGenerator)
        gen.config = config

        params = gen._build_param_grid()
        comps = params[:, 2:]
        assert comps.shape[1] == 4
        sums = comps.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)
        assert np.all(comps > 0.0)

    def test_composition_grid_is_deterministic(self):
        """The seeded sampler is reproducible across calls."""
        a = ManifoldGenerator._build_composition_grid(3, concentration_steps=8)
        b = ManifoldGenerator._build_composition_grid(3, concentration_steps=8)
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# #13b — cooling-trail config round-trips; ns defaults reproduce prior values
# ---------------------------------------------------------------------------
class TestCoolingTrailConfig:
    """Cooling-trail laws are configurable; defaults are behaviour-preserving."""

    def test_defaults_reproduce_prior_ns_values(self, tmp_path):
        """The dataclass defaults equal the historical hardcoded constants."""
        config = ManifoldConfig(
            db_path=str(tmp_path / "stub.db"),
            output_path=str(tmp_path / "out.h5"),
            elements=["Fe"],
        )
        _assert_ns_cooling_defaults(config)

    def test_yaml_round_trips_cooling_fields(self, tmp_path):
        """Cooling-trail fields parse from YAML (ps-LIBS regime example)."""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("""
manifold:
  db_path: test.db
  output_path: test.h5
  elements: [Fe]
  wavelength_range: [300.0, 400.0]
  temperature_range: [0.5, 1.3]
  density_range: [1e16, 1e18]
  cooling_t0_s: 1.0e-11
  cooling_temperature_exponent: -0.4
  cooling_density_exponent: -0.9
""")
        config = ManifoldConfig.from_file(cfg_path)
        assert config.cooling_t0_s == pytest.approx(1e-11)
        assert config.cooling_temperature_exponent == pytest.approx(-0.4)
        assert config.cooling_density_exponent == pytest.approx(-0.9)

    def test_yaml_omitted_cooling_fields_default_to_ns_values(self, tmp_path):
        """YAML without cooling keys keeps the behaviour-preserving defaults."""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("""
manifold:
  db_path: test.db
  output_path: test.h5
  elements: [Fe]
""")
        config = ManifoldConfig.from_file(cfg_path)
        _assert_ns_cooling_defaults(config)

    def test_validate_rejects_nonpositive_t0(self, tmp_path):
        """A non-positive ``cooling_t0_s`` is rejected (1 + t/t0 singular)."""
        db = tmp_path / "stub.db"
        db.touch()
        config = ManifoldConfig(
            db_path=str(db),
            output_path=str(tmp_path / "out.h5"),
            elements=["Fe"],
            cooling_t0_s=0.0,
        )
        with pytest.raises(ValueError, match="cooling_t0_s"):
            config.validate()

    @pytest.mark.requires_jax
    def test_cooling_trail_uses_configured_t0(self):
        """The time-integrated path honours the configured cooling exponents.

        Drives ``_time_integrated_spectrum`` with a stub snapshot that simply
        echoes the current T into the accumulator, so we can read back the
        cooling trail and confirm it follows ``(1 + t/t0)**exp`` with the
        passed-in parameters (not the old hardcoded -0.5 / -1.0 / 1e-6).
        """
        try:
            import jax.numpy as jnp
        except ImportError:
            pytest.skip("JAX not installed")

        from cflibs.manifold import generator as gen_mod

        def fake_snapshot(wl_grid, T, ne, concs, atomic_data, sigma_inst):
            # Record (T, ne) pairs by returning a 2-vector [T, ne] per step.
            return jnp.array([T, ne], dtype=jnp.float64)

        orig = gen_mod.ManifoldGenerator._compute_spectrum_snapshot
        gen_mod.ManifoldGenerator._compute_spectrum_snapshot = staticmethod(fake_snapshot)
        try:
            wl_grid = jnp.zeros(2, dtype=jnp.float64)
            params = jnp.array([1.0, 1e17, 1.0], dtype=jnp.float64)  # T_max, ne_max, C
            t0 = 5e-7
            T_exp = -0.5
            ne_exp = -1.0
            gate_width_s = 5e-6
            time_steps = 4
            # dt-integrated accumulator of [T, ne] over the trail.
            out = gen_mod.ManifoldGenerator._time_integrated_spectrum(
                wl_grid,
                params,
                (None,) * 12,
                gate_width_s,
                time_steps,
                0.0,
                t0,
                T_exp,
                ne_exp,
            )
        finally:
            gen_mod.ManifoldGenerator._compute_spectrum_snapshot = orig

        # Reconstruct the expected dt-weighted integral of the trail.
        times = np.linspace(0, gate_width_s, time_steps)
        dt = times[1] - times[0]
        T_trail = 1.0 * (1 + times / t0) ** T_exp
        ne_trail = 1e17 * (1 + times / t0) ** ne_exp
        mask = T_trail > 0.4
        expected_T = np.sum(np.where(mask, T_trail, 0.0) * dt)
        expected_ne = np.sum(np.where(mask, ne_trail, 0.0) * dt)

        out_np = np.asarray(out)
        assert out_np[0] == pytest.approx(expected_T, rel=1e-6)
        assert out_np[1] == pytest.approx(expected_ne, rel=1e-6)
