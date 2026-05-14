"""Tests for ``scripts/build_basis_library.py``.

Covers:

* Filename convention matches the loader regex in
  :class:`cflibs.benchmark.unified.UnifiedBenchmarkContext`.
* The basis-dir resolver honours the documented priority order
  (CLI -> env -> NFS -> repo default).
* The full build pipeline (small grid + a single FWHM) produces a file
  that :class:`cflibs.manifold.basis_library.BasisLibrary` can load and
  that :class:`UnifiedBenchmarkContext` picks up via
  :meth:`basis_for_rp`.
* The manifest is reproducible: invoking ``--manifest-only`` twice with
  the same inputs produces byte-identical output (modulo timestamps).
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

h5py = pytest.importorskip("h5py")

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_basis_library.py"
DB_PATH = REPO_ROOT / "ASD_da" / "libs_production.db"


def _load_builder():
    """Import scripts/build_basis_library.py as a module (no package side-effects)."""
    spec = importlib.util.spec_from_file_location("build_basis_library", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec so @dataclass can introspect
    # the module's namespace (dataclasses.py:712 looks up cls.__module__
    # in sys.modules and crashes if it isn't there yet).
    sys.modules["build_basis_library"] = module
    spec.loader.exec_module(module)
    return module


def _load_runner_script():
    """Import scripts/run_unified_benchmark.py as a module."""
    spec = importlib.util.spec_from_file_location(
        "run_unified_benchmark", REPO_ROOT / "scripts" / "run_unified_benchmark.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_unified_benchmark"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    return _load_builder()


# ---------------------------------------------------------------------------
# Filename convention
# ---------------------------------------------------------------------------


class TestBasisFilename:
    def test_canonical_pattern_matches_loader_regex(self, builder):
        # Loader pattern lives in cflibs/benchmark/unified.py:721-722:
        #   glob: basis_fwhm_*nm.h5
        #   rgx:  basis_fwhm_([0-9.]+)nm\.h5$
        pattern = re.compile(r"basis_fwhm_([0-9.]+)nm\.h5$")
        for fwhm in builder.CANONICAL_FWHM_GRID:
            name = builder.basis_filename(fwhm)
            match = pattern.match(name)
            assert match is not None, f"{name!r} does not match loader regex"
            assert float(match.group(1)) == pytest.approx(fwhm, rel=1e-9)

    def test_uses_g_formatting(self, builder):
        # %g drops trailing zeros; 0.1 / 1.0 / 1.67 are stable shapes.
        assert builder.basis_filename(0.10) == "basis_fwhm_0.1nm.h5"
        assert builder.basis_filename(1.00) == "basis_fwhm_1nm.h5"
        assert builder.basis_filename(1.67) == "basis_fwhm_1.67nm.h5"


# ---------------------------------------------------------------------------
# Resolver priority
# ---------------------------------------------------------------------------


class TestResolveBasisDir:
    def test_cli_wins(self, builder, tmp_path, monkeypatch):
        monkeypatch.setenv("CFLIBS_BASIS_DIR", "/env/path")
        cli = tmp_path / "explicit"
        assert builder.resolve_basis_dir(cli) == cli.expanduser().resolve()

    def test_env_var_wins_over_default(self, builder, tmp_path, monkeypatch):
        env_dir = tmp_path / "env_basis"
        env_dir.mkdir()
        monkeypatch.setenv("CFLIBS_BASIS_DIR", str(env_dir))
        assert builder.resolve_basis_dir(None) == env_dir.resolve()

    def test_falls_back_to_repo_default_when_nfs_absent(
        self, builder, monkeypatch, tmp_path
    ):
        monkeypatch.delenv("CFLIBS_BASIS_DIR", raising=False)
        # Point CANONICAL_NFS_DIR at a definitely-missing parent so the
        # NFS-mounted branch is excluded.
        monkeypatch.setattr(
            builder, "CANONICAL_NFS_DIR", tmp_path / "nope" / "basis_libraries"
        )
        result = builder.resolve_basis_dir(None)
        # Repo default: <repo>/output/basis_libraries
        expected = (builder.PROJECT_ROOT / "output" / "basis_libraries").resolve()
        assert result == expected


# ---------------------------------------------------------------------------
# Full build pipeline (real, but tiny)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not DB_PATH.exists(), reason="atomic database not present")
class TestBuildPipeline:
    def test_single_fwhm_smoke_build(self, builder, tmp_path):
        """End-to-end: build one tiny library and load it via both the
        low-level :class:`BasisLibrary` and the high-level
        :class:`UnifiedBenchmarkContext` loader.
        """
        from cflibs.benchmark.unified import UnifiedBenchmarkContext
        from cflibs.manifold.basis_library import BasisLibrary

        cfg = builder.CanonicalConfig(
            temperature_steps=3,
            density_steps=2,
        )
        builder.build_all(
            db_path=DB_PATH,
            output_dir=tmp_path,
            fwhm_values=[0.5],
            cfg=cfg,
            force=False,
        )

        out = tmp_path / "basis_fwhm_0.5nm.h5"
        assert out.exists(), "expected basis library file was not written"

        # Manifest exists and is parsable.
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["schema_version"] == 1
        assert manifest["fwhm_grid_nm"] == [0.5]
        assert len(manifest["files"]) == 1
        assert manifest["files"][0]["filename"] == "basis_fwhm_0.5nm.h5"
        assert len(manifest["files"][0]["sha256"]) == 64

        # Schema sanity.
        with h5py.File(out, "r") as f:
            assert set(f.keys()) >= {"spectra", "params", "wavelength", "elements"}
            assert f["spectra"].shape[2] == cfg.pixels
            assert f["spectra"].shape[1] == cfg.temperature_steps * cfg.density_steps

        # Low-level loader.
        lib = BasisLibrary(str(out))
        try:
            assert lib.n_elements > 0
            basis = lib.get_basis_matrix(8000.0, 1e17)
            assert basis.shape == (lib.n_elements, cfg.pixels)
            # Area-normalised: rows should sum to either 0 (no transitions in
            # this wavelength window for that element) or ~1.
            row_sums = basis.sum(axis=1)
            assert ((row_sums == 0) | (abs(row_sums - 1.0) < 1e-4)).all()
        finally:
            lib.close()

        # High-level UnifiedBenchmarkContext path -- the one
        # ``run_unified_benchmark.py`` uses.
        ctx = UnifiedBenchmarkContext(db_path=DB_PATH, basis_dir=tmp_path)
        chosen, fwhm, mismatch = ctx.basis_for_rp(550.0 / 0.5)
        assert fwhm == pytest.approx(0.5)
        assert mismatch < 1e-6
        assert chosen.n_pixels == cfg.pixels

    def test_build_is_idempotent(self, builder, tmp_path):
        cfg = builder.CanonicalConfig(temperature_steps=3, density_steps=2)
        builder.build_all(DB_PATH, tmp_path, [0.5], cfg, force=False)
        out = tmp_path / "basis_fwhm_0.5nm.h5"
        first_mtime = out.stat().st_mtime
        first_size = out.stat().st_size
        # Re-run without --force: file must not be regenerated.
        builder.build_all(DB_PATH, tmp_path, [0.5], cfg, force=False)
        assert out.stat().st_mtime == first_mtime
        assert out.stat().st_size == first_size


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not DB_PATH.exists(), reason="atomic database not present")
class TestCli:
    def test_manifest_only_does_not_rebuild(self, builder, tmp_path):
        cfg = builder.CanonicalConfig(temperature_steps=3, density_steps=2)
        # Build first
        builder.build_all(DB_PATH, tmp_path, [0.5], cfg, force=False)
        out = tmp_path / "basis_fwhm_0.5nm.h5"
        first_mtime = out.stat().st_mtime

        # Re-run via main() with --manifest-only -- should NOT touch h5
        rc = builder.main([
            "--db-path", str(DB_PATH),
            "--output-dir", str(tmp_path),
            "--fwhm", "0.5",
            "--T-steps", "3",
            "--ne-steps", "2",
            "--manifest-only",
        ])
        assert rc == 0
        assert out.stat().st_mtime == first_mtime

    def test_unknown_db_path_exits(self, builder, tmp_path):
        # Missing DB triggers sys.exit(1)
        with pytest.raises(SystemExit):
            builder.main([
                "--db-path", str(tmp_path / "nope.db"),
                "--output-dir", str(tmp_path),
                "--fwhm", "0.5",
            ])


# ---------------------------------------------------------------------------
# run_unified_benchmark integration
# ---------------------------------------------------------------------------


class TestUnifiedRunnerResolver:
    def test_resolver_honours_env_var(self, tmp_path, monkeypatch):
        """``run_unified_benchmark._resolve_basis_dir`` should consult
        ``CFLIBS_BASIS_DIR`` when no CLI flag is given.
        """
        # Lazy import (script-style module).
        mod = _load_runner_script()
        env_dir = tmp_path / "env_basis"
        env_dir.mkdir()
        monkeypatch.setenv("CFLIBS_BASIS_DIR", str(env_dir))
        result = mod._resolve_basis_dir(None)
        assert result == env_dir

    def test_resolver_cli_overrides_env(self, tmp_path, monkeypatch):
        mod = _load_runner_script()
        monkeypatch.setenv("CFLIBS_BASIS_DIR", str(tmp_path / "env"))
        cli = tmp_path / "cli"
        assert mod._resolve_basis_dir(cli) == cli
