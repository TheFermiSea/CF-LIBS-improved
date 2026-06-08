"""Tests for ``ManifoldConfig.broadening_mode`` and generator dispatch.

ADR-0001 T1-4 follow-up (bead CF-LIBS-improved-8n4i): ``ManifoldGenerator``
now selects between the per-line Voigt path and the Line Distribution
Method path based on ``ManifoldConfig.broadening_mode``. These tests cover
the config plumbing (default, YAML loader, invalid values) and the
generator dispatch path.
"""

from __future__ import annotations

import pytest

from cflibs.manifold.config import ManifoldConfig
from cflibs.radiation.profiles import BroadeningMode


def _base_config_kwargs(tmp_path):
    """Construct the minimum kwargs needed to instantiate a ``ManifoldConfig``."""
    return dict(
        db_path=str(tmp_path / "missing.db"),
        output_path=str(tmp_path / "out.h5"),
        elements=["Fe"],
        wavelength_range=(300.0, 400.0),
        temperature_range=(1.0, 1.5),
        temperature_steps=2,
        density_range=(1e16, 1e17),
        density_steps=2,
    )


class TestBroadeningModeField:
    """Direct dataclass-construction tests for the new field."""

    def test_default_is_physical_doppler(self, tmp_path):
        """Config built without the field defaults to PHYSICAL_DOPPLER.

        This preserves the pre-bead-8n4i behaviour: existing manifolds keep
        the per-line Voigt path until users explicitly opt in to LDM.
        """
        config = ManifoldConfig(**_base_config_kwargs(tmp_path))
        assert config.broadening_mode is BroadeningMode.PHYSICAL_DOPPLER

    def test_explicit_ldm_gaussian(self, tmp_path):
        """The field accepts a ``BroadeningMode`` enum directly."""
        config = ManifoldConfig(
            broadening_mode=BroadeningMode.LDM_GAUSSIAN,
            **_base_config_kwargs(tmp_path),
        )
        assert config.broadening_mode is BroadeningMode.LDM_GAUSSIAN


class TestBroadeningModeYAMLLoader:
    """YAML round-trip tests for the new ``broadening_mode`` key."""

    def test_yaml_omitted_defaults_to_physical_doppler(self, tmp_path):
        """YAML without ``broadening_mode`` still parses to the default.

        Backward compatibility: every manifold YAML in the wild predates
        this knob and must continue to load unchanged.
        """
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("""
manifold:
  db_path: test.db
  output_path: test.h5
  elements: [Fe]
  wavelength_range: [300.0, 400.0]
  temperature_range: [1.0, 1.5]
  temperature_steps: 2
  density_range: [1e16, 1e17]
  density_steps: 2
""")

        config = ManifoldConfig.from_file(cfg_path)
        assert config.broadening_mode is BroadeningMode.PHYSICAL_DOPPLER

    def test_yaml_parses_ldm_gaussian(self, tmp_path):
        """``broadening_mode: ldm_gaussian`` parses to the LDM enum value."""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("""
manifold:
  db_path: test.db
  output_path: test.h5
  elements: [Fe]
  wavelength_range: [300.0, 400.0]
  temperature_range: [1.0, 1.5]
  temperature_steps: 2
  density_range: [1e16, 1e17]
  density_steps: 2
  broadening_mode: ldm_gaussian
""")

        config = ManifoldConfig.from_file(cfg_path)
        assert config.broadening_mode is BroadeningMode.LDM_GAUSSIAN

    def test_yaml_parses_physical_doppler(self, tmp_path):
        """Explicit ``broadening_mode: physical_doppler`` round-trips."""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("""
manifold:
  db_path: test.db
  output_path: test.h5
  elements: [Fe]
  wavelength_range: [300.0, 400.0]
  temperature_range: [1.0, 1.5]
  temperature_steps: 2
  density_range: [1e16, 1e17]
  density_steps: 2
  broadening_mode: physical_doppler
""")

        config = ManifoldConfig.from_file(cfg_path)
        assert config.broadening_mode is BroadeningMode.PHYSICAL_DOPPLER

    def test_yaml_parses_legacy_and_nist_parity(self, tmp_path):
        """The other two enum values also parse via YAML."""
        for value, expected in (
            ("legacy", BroadeningMode.LEGACY),
            ("nist_parity", BroadeningMode.NIST_PARITY),
        ):
            cfg_path = tmp_path / f"config_{value}.yaml"
            cfg_path.write_text(f"""
manifold:
  db_path: test.db
  output_path: test.h5
  elements: [Fe]
  wavelength_range: [300.0, 400.0]
  temperature_range: [1.0, 1.5]
  temperature_steps: 2
  density_range: [1e16, 1e17]
  density_steps: 2
  broadening_mode: {value}
""")
            config = ManifoldConfig.from_file(cfg_path)
            assert config.broadening_mode is expected, value

    def test_yaml_rejects_unknown_value(self, tmp_path):
        """A typo in ``broadening_mode`` raises ``ValueError`` at load time."""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("""
manifold:
  db_path: test.db
  output_path: test.h5
  elements: [Fe]
  wavelength_range: [300.0, 400.0]
  temperature_range: [1.0, 1.5]
  temperature_steps: 2
  density_range: [1e16, 1e17]
  density_steps: 2
  broadening_mode: not_a_real_mode
""")

        with pytest.raises(ValueError, match="broadening_mode"):
            ManifoldConfig.from_file(cfg_path)


class TestGeneratorDispatch:
    """Generator-side dispatch via the ``broadening_mode`` knob.

    A full manifold sweep is too expensive for a unit test (needs an
    atomic DB). Instead, we verify the dispatch logic by stubbing the
    instance state and checking which branch of ``generate_manifold``'s
    ``batch_spectrum`` closure was selected — done via call observation
    on the two static snapshot helpers.
    """

    @pytest.mark.requires_jax
    def test_generator_dispatches_ldm_for_ldm_mode(self, tmp_path, monkeypatch):
        """When ``broadening_mode == LDM_GAUSSIAN`` the generator routes
        through ``_time_integrated_spectrum_ldm``, not the per-line Voigt
        ``_time_integrated_spectrum``.

        We bypass ``__init__`` (which needs an atomic database) and
        construct a minimal generator with the absolute minimum atomic
        data the dispatch needs (line wavelengths + masses for sigma-grid
        construction) plus the config. The two time-integrated entry
        points are monkey-patched at the class level so we observe which
        branch ``generate_manifold`` selected without needing a real
        physics computation.
        """
        try:
            import jax  # noqa: F401
            import jax.numpy as jnp
            import numpy as np
        except ImportError:
            pytest.skip("JAX not installed")

        from cflibs.manifold import generator as gen_mod

        # Stub atomic_data with the minimum the dispatch consumes:
        # tuple[0] = line wavelengths (nm), tuple[-1] = mass (amu).
        # _build_ldm_sigma_grid uses lines_wl + lines_mass_amu (positions
        # 0 and 11). Other entries are zero-filled stubs since the
        # time-integrated path is monkey-patched.
        n_lines = 4
        stub_wl = np.array([350.0, 360.0, 370.0, 380.0], dtype=np.float32)
        stub_mass = np.array([55.85, 55.85, 55.85, 55.85], dtype=np.float32)
        stub_zeros = np.zeros(n_lines, dtype=np.float32)
        stub_int = np.zeros(n_lines, dtype=np.int32)
        stub_coeffs = np.zeros((1, 3, 5), dtype=np.float32)
        stub_ips = np.zeros((1, 3), dtype=np.float32)
        atomic_data = (
            stub_wl,
            stub_zeros,
            stub_zeros,
            stub_zeros,
            stub_zeros,
            stub_int,
            stub_int,
            stub_coeffs,
            stub_ips,
            stub_zeros,
            stub_zeros,
            stub_mass,
        )

        config = ManifoldConfig(
            db_path=str(tmp_path / "stub.db"),
            output_path=str(tmp_path / "out.h5"),
            elements=["Fe"],
            wavelength_range=(300.0, 400.0),
            temperature_range=(1.0, 1.5),
            temperature_steps=2,
            density_range=(1e16, 1e17),
            density_steps=2,
            concentration_steps=2,
            pixels=16,
            gate_delay_s=300e-9,
            gate_width_s=5e-6,
            time_steps=2,
            batch_size=2,
            broadening_mode=BroadeningMode.LDM_GAUSSIAN,
        )

        gen = gen_mod.ManifoldGenerator.__new__(gen_mod.ManifoldGenerator)
        gen.config = config
        gen.atomic_data = atomic_data

        # Record which time-integrated path is called. Patching the
        # static entry points dodges JIT-cache contamination between
        # tests because ``batch_spectrum`` is a fresh closure each call.
        calls = {"ldm": 0, "voigt": 0}

        # Signatures updated for D3 fix (sigma_inst) and the Wave-3 cooling-trail
        # config fix (#13b): time-integrated paths now also accept the three
        # cooling-trail parameters (t0, T-exponent, ne-exponent) from
        # ``ManifoldConfig`` so ps-LIBS regimes can be configured.
        def fake_ldm_time_int(
            wl_grid, p, ad, sigma_grid, gate_width_s, time_steps, sigma_inst, *cooling
        ):
            calls["ldm"] += 1
            return jnp.zeros_like(wl_grid)

        def fake_voigt_time_int(wl_grid, p, ad, gate_width_s, time_steps, sigma_inst, *cooling):
            calls["voigt"] += 1
            return jnp.zeros_like(wl_grid)

        monkeypatch.setattr(
            gen_mod.ManifoldGenerator,
            "_time_integrated_spectrum_ldm",
            staticmethod(fake_ldm_time_int),
        )
        monkeypatch.setattr(
            gen_mod.ManifoldGenerator,
            "_time_integrated_spectrum",
            staticmethod(fake_voigt_time_int),
        )

        gen.generate_manifold()

        assert calls["ldm"] > 0, "LDM time-integrated path was not invoked"
        assert calls["voigt"] == 0, (
            "Voigt time-integrated path was invoked even though " "broadening_mode=LDM_GAUSSIAN"
        )

    @pytest.mark.requires_jax
    def test_generator_dispatches_voigt_for_default_mode(self, tmp_path, monkeypatch):
        """Default (``PHYSICAL_DOPPLER``) routes through the per-line Voigt
        path, matching pre-bead-8n4i behaviour."""
        try:
            import jax  # noqa: F401
            import jax.numpy as jnp
            import numpy as np
        except ImportError:
            pytest.skip("JAX not installed")

        from cflibs.manifold import generator as gen_mod

        n_lines = 4
        stub_wl = np.array([350.0, 360.0, 370.0, 380.0], dtype=np.float32)
        stub_mass = np.array([55.85, 55.85, 55.85, 55.85], dtype=np.float32)
        stub_zeros = np.zeros(n_lines, dtype=np.float32)
        stub_int = np.zeros(n_lines, dtype=np.int32)
        stub_coeffs = np.zeros((1, 3, 5), dtype=np.float32)
        stub_ips = np.zeros((1, 3), dtype=np.float32)
        atomic_data = (
            stub_wl,
            stub_zeros,
            stub_zeros,
            stub_zeros,
            stub_zeros,
            stub_int,
            stub_int,
            stub_coeffs,
            stub_ips,
            stub_zeros,
            stub_zeros,
            stub_mass,
        )

        config = ManifoldConfig(
            db_path=str(tmp_path / "stub.db"),
            output_path=str(tmp_path / "out.h5"),
            elements=["Fe"],
            wavelength_range=(300.0, 400.0),
            temperature_range=(1.0, 1.5),
            temperature_steps=2,
            density_range=(1e16, 1e17),
            density_steps=2,
            concentration_steps=2,
            pixels=16,
            gate_delay_s=300e-9,
            gate_width_s=5e-6,
            time_steps=2,
            batch_size=2,
            # broadening_mode omitted — defaults to PHYSICAL_DOPPLER.
        )

        gen = gen_mod.ManifoldGenerator.__new__(gen_mod.ManifoldGenerator)
        gen.config = config
        gen.atomic_data = atomic_data

        calls = {"ldm": 0, "voigt": 0}

        # Signatures updated for D3 fix (sigma_inst) and the Wave-3 cooling-trail
        # config fix (#13b): time-integrated paths now also accept the three
        # cooling-trail parameters (t0, T-exponent, ne-exponent) from
        # ``ManifoldConfig`` so ps-LIBS regimes can be configured.
        def fake_ldm_time_int(
            wl_grid, p, ad, sigma_grid, gate_width_s, time_steps, sigma_inst, *cooling
        ):
            calls["ldm"] += 1
            return jnp.zeros_like(wl_grid)

        def fake_voigt_time_int(wl_grid, p, ad, gate_width_s, time_steps, sigma_inst, *cooling):
            calls["voigt"] += 1
            return jnp.zeros_like(wl_grid)

        monkeypatch.setattr(
            gen_mod.ManifoldGenerator,
            "_time_integrated_spectrum_ldm",
            staticmethod(fake_ldm_time_int),
        )
        monkeypatch.setattr(
            gen_mod.ManifoldGenerator,
            "_time_integrated_spectrum",
            staticmethod(fake_voigt_time_int),
        )

        gen.generate_manifold()

        assert calls["voigt"] > 0, "Voigt time-integrated path was not invoked"
        assert calls["ldm"] == 0, (
            "LDM time-integrated path was invoked even though "
            "broadening_mode is the PHYSICAL_DOPPLER default"
        )
