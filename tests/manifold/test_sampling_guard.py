"""Regression tests for the manifold Nyquist sampling guard (Wave 1 D1 fix).

The physics audit (docs/architecture/2026-05-27-physics-audit.md) flagged the
4096-pixel default over 250-550 nm as severely undersampled relative to the
0.05 nm instrument FWHM — only ~0.68 px/FWHM, well below the Robertson 2017 /
Magnier 2025 / Demidov 2022 / van den Bekerom & Pannier 2021 standard of
≥3 px/FWHM. ``ManifoldConfig.validate()`` now rejects sub-Nyquist configs at
load time, and the default ``pixels`` was bumped to 18432 to land just above
the threshold. These tests pin both behaviours.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cflibs.manifold.config import ManifoldConfig


@pytest.fixture
def stub_db(tmp_path: Path) -> Path:
    """Materialize an empty file at the configured ``db_path``.

    ``validate()`` checks ``Path(db_path).exists()`` first, so without this
    fixture every test would trip the database-missing branch before reaching
    the new Nyquist guard.
    """
    db = tmp_path / "stub.db"
    db.touch()
    return db


class TestNyquistGuard:
    """``ManifoldConfig.validate()`` enforces ≥3 px/FWHM."""

    def test_rejects_legacy_4096_pixel_default(self, stub_db: Path) -> None:
        """The pre-fix default (4096 pixels over 250-550 nm at FWHM=0.05 nm)
        is sub-Nyquist (~0.68 px/FWHM) and must be rejected."""
        config = ManifoldConfig(
            db_path=str(stub_db),
            output_path=str(stub_db.parent / "out.h5"),
            elements=["Fe"],
            wavelength_range=(250.0, 550.0),
            temperature_range=(0.5, 2.0),
            temperature_steps=10,
            density_range=(1e16, 1e19),
            density_steps=5,
            pixels=4096,
            instrument_fwhm_nm=0.05,
        )
        with pytest.raises(ValueError) as excinfo:
            config.validate()
        msg = str(excinfo.value)
        assert "px/FWHM" in msg
        # Must surface both the computed under-sampling and the >=3 minimum.
        assert "0.68" in msg or "0.69" in msg, f"expected ~0.68 px/FWHM in message; got: {msg}"
        assert "3" in msg

    def test_accepts_18432_pixel_default(self, stub_db: Path) -> None:
        """The new 18432-pixel default lands at ~3.07 px/FWHM and must pass."""
        config = ManifoldConfig(
            db_path=str(stub_db),
            output_path=str(stub_db.parent / "out.h5"),
            elements=["Fe"],
            wavelength_range=(250.0, 550.0),
            temperature_range=(0.5, 2.0),
            temperature_steps=10,
            density_range=(1e16, 1e19),
            density_steps=5,
            pixels=18432,
            instrument_fwhm_nm=0.05,
        )
        # Should not raise.
        assert config.validate() is True

    def test_default_pixels_is_18432(self) -> None:
        """The dataclass default itself must be the Nyquist-safe value."""
        config = ManifoldConfig(
            db_path="ignored.db",
            output_path="ignored.h5",
            elements=["Fe"],
        )
        assert config.pixels == 18432

    def test_yaml_default_pixels_is_18432(self, tmp_path: Path) -> None:
        """``ManifoldConfig.from_file`` defaults ``pixels`` to 18432 too."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("""
manifold:
  db_path: test.db
  output_path: test.h5
  elements: [Fe]
""".strip())
        config = ManifoldConfig.from_file(cfg)
        assert config.pixels == 18432

    def test_borderline_at_3_px_per_fwhm_passes(self, stub_db: Path) -> None:
        """Exactly 3 px/FWHM (Δλ = FWHM/3) is the boundary and must be allowed."""
        # 300 nm window / 0.05 FWHM × 3 = 18000 pixels gives Δλ = FWHM/3 exactly.
        config = ManifoldConfig(
            db_path=str(stub_db),
            output_path=str(stub_db.parent / "out.h5"),
            elements=["Fe"],
            wavelength_range=(250.0, 550.0),
            temperature_range=(0.5, 2.0),
            temperature_steps=10,
            density_range=(1e16, 1e19),
            density_steps=5,
            pixels=18000,
            instrument_fwhm_nm=0.05,
        )
        assert config.validate() is True
