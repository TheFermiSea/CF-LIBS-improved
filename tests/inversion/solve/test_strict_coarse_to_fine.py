"""Strict / no-fallback tests for the coarse-to-fine hybrid solver.

Covers the four highest-impact silent substitutions catalogued for this area:

1. ``HybridInverter`` silently swaps in the toy ``_default_forward_model`` when no
   physics forward model is wired -> strict refuses at construction.
2. ``invert``/``fit`` wrap the optimizer in ``except Exception`` and return the
   coarse/initial seed as the "answer" -> strict re-raises ``OptimizerFailure``.
3. ``_run_optimizer`` returns ``result.x``/``result.fun`` verbatim (NaN included)
   and never consults ``result.success`` -> strict refuses non-finite /
   ``success=False`` results.
4. ``invert`` fabricates a default seed (T=1, n_e=1e17, uniform C) with zero
   provenance, and accepts an out-of-manifold manifold match -> strict refuses.

All tests monkeypatch ``_run_optimizer`` so they are fast and need no real
optimization, DB, or heavy XLA compile. The default (strict off) path is asserted
to be byte-identical to current production: same numeric result and metadata.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")  # HybridInverter/SpectralFitter require JAX

from cflibs.inversion.common.strict import (  # noqa: E402
    NonIdentifiable,
    NotConverged,
    OptimizerFailure,
)
from cflibs.inversion.solve import coarse_to_fine as c2f  # noqa: E402
from cflibs.inversion.solve.coarse_to_fine import (  # noqa: E402
    HybridInverter,
    HybridInversionResult,
    SpectralFitter,
)


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #
class FakeManifold:
    """Minimal manifold stub: fixed wavelength/elements + a canned NN match."""

    def __init__(self, similarity: float = 0.99):
        self.wavelength = np.linspace(400.0, 410.0, 16)
        self.elements = ["Fe", "Cu"]
        self._similarity = similarity

    def find_nearest_spectrum(self, measured, method="cosine", use_jax=True):
        params = {"T_eV": 1.2, "n_e_cm3": 2e17, "Fe": 0.6, "Cu": 0.4}
        return 0, self._similarity, params


def _toy_forward(T_eV, n_e, conc, wavelength):
    """Tiny smooth, finite, differentiable forward model (2 elements)."""
    import jax.numpy as jnp

    base = jnp.exp(-((wavelength - 405.0) ** 2) / 4.0)
    return (conc[0] + conc[1]) * T_eV * base


def _measured(manifold):
    return np.asarray(_toy_forward(1.2, 2e17, np.array([0.6, 0.4]), manifold.wavelength))


def _patch_optimizer(monkeypatch, *, success=True, finite=True, raises=None):
    """Replace _run_optimizer with a deterministic stub (no real optimization)."""

    def fake(loss_fn, x0, method, max_iterations, bounds=None, *, return_status=False):
        if raises is not None:
            raise raises
        x = np.asarray(x0, dtype=float).copy()
        fun = 0.5
        if not finite:
            x[0] = np.nan
            fun = np.nan
        five = (x, float(fun), bool(success), 7, "scipy")
        if return_status:
            status = {
                "backend": "scipy",
                "success": bool(success),
                "status": 0 if success else 1,
                "message": "ok" if success else "max iters",
                "nit": 7,
                "x_finite": finite,
                "fun_finite": finite,
            }
            return five + (status,)
        return five

    monkeypatch.setattr(c2f, "_run_optimizer", fake)


# --------------------------------------------------------------------------- #
# (i) strict OFF == current production behaviour                               #
# --------------------------------------------------------------------------- #
def test_invert_strict_off_is_production_behaviour(monkeypatch):
    _patch_optimizer(monkeypatch, success=True, finite=True)
    man = FakeManifold()
    inv = HybridInverter(man, forward_model=_toy_forward)
    res = inv.invert(_measured(man), method="L-BFGS-B")

    assert isinstance(res, HybridInversionResult)
    assert res.converged is True
    assert res.iterations == 7
    # Byte-identical metadata: only the production key, no strict diagnostics.
    assert res.metadata == {"optimizer_backend": "scipy"}
    assert "diagnostics" not in res.metadata


def test_invert_strict_off_optimizer_crash_falls_back(monkeypatch):
    """strict off: an optimizer crash is still swallowed -> coarse seed returned."""
    _patch_optimizer(monkeypatch, raises=RuntimeError("LinAlgError"))
    man = FakeManifold()
    inv = HybridInverter(man, forward_model=_toy_forward)
    res = inv.invert(_measured(man))

    assert res.converged is False
    assert res.iterations == 0
    assert res.metadata == {"optimizer_backend": "fallback"}


def test_fit_strict_off_optimizer_crash_falls_back(monkeypatch):
    _patch_optimizer(monkeypatch, raises=RuntimeError("nan gradient"))
    man = FakeManifold()
    fitter = SpectralFitter(_toy_forward, man.elements, man.wavelength)
    res = fitter.fit(_measured(man), initial_T_eV=1.0, initial_n_e=1e17)

    assert res.converged is False
    assert res.temperature_eV == pytest.approx(1.0)
    assert res.electron_density_cm3 == pytest.approx(1e17)
    assert res.metadata == {"optimizer_backend": "fallback"}


# --------------------------------------------------------------------------- #
# (ii) strict ON refuses / flags                                              #
# --------------------------------------------------------------------------- #
def test_strict_refuses_missing_forward_model():
    """The toy _default_forward_model is disabled under strict mode."""
    with pytest.raises(ValueError, match="explicit physics forward_model"):
        HybridInverter(FakeManifold(), forward_model=None, strict=True)


def test_strict_allows_toy_forward_model_opt_in():
    """Tests can still opt back into the toy via allow_toy_forward_model=True."""
    inv = HybridInverter(
        FakeManifold(), forward_model=None, strict=True, allow_toy_forward_model=True
    )
    assert inv.forward_model is inv._default_forward_model


def test_strict_invert_optimizer_crash_raises(monkeypatch):
    _patch_optimizer(monkeypatch, raises=RuntimeError("scipy LinAlgError"))
    man = FakeManifold()
    inv = HybridInverter(man, forward_model=_toy_forward, strict=True)
    with pytest.raises(OptimizerFailure) as ei:
        inv.invert(_measured(man))
    diag = ei.value.diagnostics
    assert diag is not None
    assert diag.extra["exc_type"] == "RuntimeError"
    assert diag.extra["stage"] == "fine"
    assert "loss_at_seed" in diag.extra


def test_strict_invert_nonfinite_result_raises(monkeypatch):
    """A diverged run returning NaN x/fun must not be reported as a fit."""
    _patch_optimizer(monkeypatch, success=True, finite=False)
    man = FakeManifold()
    inv = HybridInverter(man, forward_model=_toy_forward, strict=True)
    with pytest.raises(OptimizerFailure, match="non-finite"):
        inv.invert(_measured(man))


def test_strict_invert_not_converged_raises(monkeypatch):
    """success=False is refused (production would launder it to converged=False)."""
    _patch_optimizer(monkeypatch, success=False, finite=True)
    man = FakeManifold()
    inv = HybridInverter(man, forward_model=_toy_forward, strict=True)
    with pytest.raises(NotConverged):
        inv.invert(_measured(man))


def test_strict_invert_default_seed_refused(monkeypatch):
    """No manifold init + no guess -> fabricated default seed is refused."""
    _patch_optimizer(monkeypatch, success=True, finite=True)
    man = FakeManifold()
    inv = HybridInverter(man, forward_model=_toy_forward, strict=True)
    with pytest.raises(ValueError, match="fabricated default seed"):
        inv.invert(_measured(man), use_manifold_init=False, initial_guess=None)


def test_strict_invert_out_of_manifold_refused(monkeypatch):
    """A near-orthogonal manifold match below the floor is refused as OOD."""
    _patch_optimizer(monkeypatch, success=True, finite=True)
    man = FakeManifold(similarity=0.05)
    inv = HybridInverter(man, forward_model=_toy_forward, strict=True)
    with pytest.raises(NonIdentifiable, match="out-of-manifold"):
        inv.invert(_measured(man), min_coarse_similarity=0.5)


def test_strict_invert_success_records_diagnostics(monkeypatch):
    """Healthy strict run succeeds and surfaces diagnostics + provenance."""
    _patch_optimizer(monkeypatch, success=True, finite=True)
    man = FakeManifold()
    inv = HybridInverter(man, forward_model=_toy_forward, strict=True)
    res = inv.invert(_measured(man), min_coarse_similarity=0.5)

    assert res.converged is True
    assert res.metadata["init_source"] == "manifold"
    assert res.metadata["coarse_similarity"] == pytest.approx(0.99)
    diag = res.metadata["diagnostics"]
    assert diag["strict"] is True
    assert diag["optimizer_success"] is True
    assert diag["optimizer_status"]["backend"] == "scipy"


def test_strict_fit_optimizer_crash_raises(monkeypatch):
    _patch_optimizer(monkeypatch, raises=RuntimeError("OOM"))
    man = FakeManifold()
    fitter = SpectralFitter(_toy_forward, man.elements, man.wavelength, strict=True)
    with pytest.raises(OptimizerFailure) as ei:
        fitter.fit(_measured(man))
    assert ei.value.diagnostics.extra["stage"] == "fit"


def test_strict_fit_nonfinite_result_raises(monkeypatch):
    _patch_optimizer(monkeypatch, success=True, finite=False)
    man = FakeManifold()
    fitter = SpectralFitter(_toy_forward, man.elements, man.wavelength, strict=True)
    with pytest.raises(OptimizerFailure, match="non-finite"):
        fitter.fit(_measured(man))


def test_strict_env_flag_resolves(monkeypatch):
    """CFLIBS_NO_FALLBACK=1 enables strict without an explicit kwarg."""
    monkeypatch.setenv("CFLIBS_NO_FALLBACK", "1")
    with pytest.raises(ValueError, match="explicit physics forward_model"):
        HybridInverter(FakeManifold(), forward_model=None)


def test_per_call_strict_override_beats_instance(monkeypatch):
    """invert(strict=False) on a strict instance restores fallback behaviour."""
    _patch_optimizer(monkeypatch, raises=RuntimeError("boom"))
    man = FakeManifold()
    inv = HybridInverter(man, forward_model=_toy_forward, strict=True)
    res = inv.invert(_measured(man), strict=False)  # override -> fallback, no raise
    assert res.converged is False
    assert res.metadata == {"optimizer_backend": "fallback"}
