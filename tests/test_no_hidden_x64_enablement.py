"""Regression tests for arch review #2 candidate 2.

The JAX-x64 enablement contract used to live as a hidden side effect
inside ``cflibs.benchmark.unified._jax_identifier_flags_for``. The
refactor lifted it to an explicit named seam
(``cflibs.core.jax_runtime.configure_for_identifiers``) and added
fail-fast guards in identifier constructors via
:func:`cflibs.core.jax_runtime.check_jax64bit`. These tests pin that
shape so the side effect cannot be silently reintroduced.

The tests run in **subprocesses** because ``tests/conftest.py:25``
enables x64 suite-wide; in-process tests would always see x64 already
on and could not observe the contract.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.requires_jax


def _run_subprocess(script: str, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run *script* in a fresh Python interpreter and return the result.

    The subprocess inherits the parent's interpreter (so ``cflibs`` is
    importable) but starts with a fresh ``jax.config`` — x64 is OFF.
    """
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_jax_identifier_flags_for_does_not_mutate_x64() -> None:
    """``_jax_identifier_flags_for`` must NOT touch ``jax.config``.

    This is the canary for bead ``CF-LIBS-improved-jbfg.1``: regressing
    the hidden side effect would silently re-enable x64 mid-call.
    """
    script = textwrap.dedent("""
        import os
        os.environ["CFLIBS_USE_JAX_IDENTIFIER"] = "1"
        import jax
        assert jax.config.jax_enable_x64 is False, (
            "subprocess unexpectedly started with x64 already enabled"
        )
        from cflibs.benchmark.unified import _jax_identifier_flags_for
        from cflibs.inversion.identify.alias import ALIASIdentifier
        flags = _jax_identifier_flags_for(ALIASIdentifier)
        assert flags, (
            "_jax_identifier_flags_for should return a non-empty dict "
            "when CFLIBS_USE_JAX_IDENTIFIER=1"
        )
        assert jax.config.jax_enable_x64 is False, (
            "_jax_identifier_flags_for mutated jax.config — regression "
            "of bead jbfg.1. Move the side effect back to "
            "cflibs.core.jax_runtime.configure_for_identifiers."
        )
        print("PURE_OK")
        """)
    result = _run_subprocess(script)
    assert result.returncode == 0, (
        f"subprocess failed (rc={result.returncode}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "PURE_OK" in result.stdout


def test_configure_for_identifiers_enables_x64() -> None:
    """Explicit ``configure_for_identifiers()`` actually enables x64."""
    script = textwrap.dedent("""
        import jax
        assert jax.config.jax_enable_x64 is False
        from cflibs.core.jax_runtime import configure_for_identifiers
        configure_for_identifiers()
        assert jax.config.jax_enable_x64 is True, (
            "configure_for_identifiers() did not enable jax_enable_x64"
        )
        print("CONFIGURE_OK")
        """)
    result = _run_subprocess(script)
    assert result.returncode == 0, (
        f"subprocess failed (rc={result.returncode}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "CONFIGURE_OK" in result.stdout


def test_identifier_construction_fails_fast_without_x64() -> None:
    """An identifier constructor with ``use_jax_*=True`` raises when x64 is off.

    Catches the case where someone constructs ``SpectralNNLSIdentifier(
    use_jax_nnls=True)`` from a script that forgot to call
    ``configure_for_identifiers()`` first. Without the guard, FISTA at
    float32 silently produces ~95% coefficient error on column-correlated
    spectra.
    """
    script = textwrap.dedent("""
        import jax
        assert jax.config.jax_enable_x64 is False
        from cflibs.inversion.identify.spectral_nnls import SpectralNNLSIdentifier
        try:
            SpectralNNLSIdentifier(basis_library=None, use_jax_nnls=True)
        except ValueError as exc:
            msg = str(exc).lower()
            assert "x64" in msg or "float64" in msg or "jax_enable_x64" in msg, (
                f"guard fired but message was wrong: {exc}"
            )
            print("GUARD_OK")
        else:
            raise AssertionError(
                "Expected ValueError from missing x64 — guard regressed"
            )
        """)
    result = _run_subprocess(script)
    assert result.returncode == 0, (
        f"subprocess failed (rc={result.returncode}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "GUARD_OK" in result.stdout


def test_identifier_construction_passes_with_explicit_configure() -> None:
    """Round-trip: explicit configure + identifier construction works."""
    script = textwrap.dedent("""
        import jax
        assert jax.config.jax_enable_x64 is False
        from cflibs.core.jax_runtime import configure_for_identifiers
        configure_for_identifiers()
        from cflibs.inversion.identify.spectral_nnls import SpectralNNLSIdentifier
        # Should NOT raise now that x64 is on.
        ident = SpectralNNLSIdentifier(basis_library=None, use_jax_nnls=True)
        assert ident.use_jax_nnls is True
        print("ROUND_TRIP_OK")
        """)
    result = _run_subprocess(script)
    assert result.returncode == 0, (
        f"subprocess failed (rc={result.returncode}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "ROUND_TRIP_OK" in result.stdout
