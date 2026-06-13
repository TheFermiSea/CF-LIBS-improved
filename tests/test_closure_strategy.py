"""
Tests for the unified ClosureStrategy protocol and its three adapters.

Verifies:

* :class:`ILRClosure` / :class:`PWLRClosure` simplex behavior (sum-to-one,
  zero safety, input validation).
* All three adapters satisfy the :class:`ClosureStrategy` protocol.
* :meth:`gradient_check` returns sensible values in trace-element regimes.
"""

from __future__ import annotations

import numpy as np
import pytest

from cflibs.inversion.physics.closure_strategy import (
    ClosureStrategy,
    ILRClosure,
    PWLRClosure,
    SoftmaxClosure,
)

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


@pytest.mark.requires_jax
def test_softmax_closure_satisfies_protocol():
    pytest.importorskip("jax")
    strat: ClosureStrategy = SoftmaxClosure()
    assert isinstance(strat, ClosureStrategy)
    assert strat.name == "softmax"
    assert strat.backend == "jax"


def test_ilr_closure_satisfies_protocol():
    strat: ClosureStrategy = ILRClosure()
    assert isinstance(strat, ClosureStrategy)
    assert strat.name == "ilr"
    assert strat.backend == "numpy"


def test_pwlr_closure_satisfies_protocol():
    strat: ClosureStrategy = PWLRClosure()
    assert isinstance(strat, ClosureStrategy)
    assert strat.name == "pwlr"
    assert strat.backend == "numpy"


# ---------------------------------------------------------------------------
# ILR closure behavior
# ---------------------------------------------------------------------------


class TestILRClosureParity:
    """ILRClosure.apply simplex behavior."""

    def test_sums_to_one(self):
        raw = np.array([1.0, 2.0, 3.0, 4.0])
        c = ILRClosure().apply(raw)
        assert abs(float(np.sum(c)) - 1.0) < 1e-12

    def test_handles_zero_safely(self):
        # All-zero input should fall back to uniform composition.
        raw = np.zeros(4)
        c = ILRClosure().apply(raw)
        np.testing.assert_allclose(c, np.full(4, 0.25))

    def test_rejects_high_dim_input(self):
        with pytest.raises(ValueError):
            ILRClosure().apply(np.zeros((2, 3)))

    def test_rejects_bad_eps(self):
        with pytest.raises(ValueError):
            ILRClosure(eps=-1.0)


# ---------------------------------------------------------------------------
# PWLR closure behavior
# ---------------------------------------------------------------------------


class TestPWLRClosureParity:
    """PWLRClosure.apply simplex behavior."""

    def test_sums_to_one(self):
        raw = np.array([1.0, 2.0, 3.0, 4.0])
        c = PWLRClosure().apply(raw)
        assert abs(float(np.sum(c)) - 1.0) < 1e-12

    def test_handles_zero_safely(self):
        raw = np.zeros(4)
        c = PWLRClosure().apply(raw)
        np.testing.assert_allclose(c, np.full(4, 0.25))

    def test_rejects_negative_regularization(self):
        with pytest.raises(ValueError):
            PWLRClosure(regularization_strength=-1.0)


# ---------------------------------------------------------------------------
# gradient_check semantics
# ---------------------------------------------------------------------------


@pytest.mark.requires_jax
def test_softmax_gradient_check_flags_trace_elements():
    pytest.importorskip("jax")
    closure = SoftmaxClosure()
    # Well-conditioned composition: smallest component well above floor.
    assert closure.gradient_check(np.array([0.5, 0.3, 0.2])) is True
    # Trace component below the floor — softmax gradients vanish.
    assert closure.gradient_check(np.array([0.9999995, 0.0000005])) is False


def test_ilr_gradient_check_accepts_open_simplex():
    closure = ILRClosure()
    assert closure.gradient_check(np.array([0.5, 0.3, 0.2])) is True
    # ILR remains well-conditioned for tiny components — gradient_check
    # only fails on exact zeros (boundary of the open simplex).
    assert closure.gradient_check(np.array([0.99999, 1e-5])) is True
    assert closure.gradient_check(np.array([1.0, 0.0])) is False


def test_pwlr_gradient_check_accepts_open_simplex():
    closure = PWLRClosure()
    assert closure.gradient_check(np.array([0.5, 0.3, 0.2])) is True
    assert closure.gradient_check(np.array([0.99999, 1e-5])) is True
    assert closure.gradient_check(np.array([1.0, 0.0])) is False


# ---------------------------------------------------------------------------
# Solver backend-guard tests
#
# JAX solvers (HybridInverter, SpectralFitter, JointOptimizer) trace the loss
# through jax.value_and_grad. A NumPy-backend closure (ILR/PWLR) would either
# ConcretizationTypeError on a tracer or silently produce wrong gradients.
# The constructors must reject a non-JAX closure with a clear ValueError
# instead of leaving the failure latent until the first .invert()/.fit() call.
# ---------------------------------------------------------------------------


@pytest.mark.requires_jax
def test_hybrid_inverter_rejects_numpy_closure():
    pytest.importorskip("jax")
    from cflibs.inversion.solve.coarse_to_fine import HybridInverter

    class _StubManifold:
        def __init__(self, n_elements=2, n_wl=8):
            import jax.numpy as jnp

            self.wavelength = jnp.linspace(400.0, 500.0, n_wl)
            self.elements = [f"E{i}" for i in range(n_elements)]

    manifold = _StubManifold()
    with pytest.raises(ValueError, match=r"requires a JAX-backend closure.*ilr"):
        HybridInverter(manifold=manifold, closure=ILRClosure())
    with pytest.raises(ValueError, match=r"requires a JAX-backend closure.*pwlr"):
        HybridInverter(manifold=manifold, closure=PWLRClosure())


@pytest.mark.requires_jax
def test_spectral_fitter_rejects_numpy_closure():
    pytest.importorskip("jax")
    from cflibs.inversion.solve.coarse_to_fine import SpectralFitter

    def _stub_forward(*args, **kwargs):  # pragma: no cover - never called
        raise AssertionError("forward model should not be invoked before guard fires")

    with pytest.raises(ValueError, match=r"requires a JAX-backend closure.*ilr"):
        SpectralFitter(
            forward_model=_stub_forward,
            elements=["Fe", "Ni"],
            wavelength=np.linspace(400.0, 500.0, 8),
            closure=ILRClosure(),
        )


@pytest.mark.requires_jax
def test_joint_optimizer_rejects_numpy_closure():
    pytest.importorskip("jax")
    from cflibs.inversion.solve.joint_optimizer import JointOptimizer

    def _stub_forward(*args, **kwargs):  # pragma: no cover - never called
        raise AssertionError("forward model should not be invoked before guard fires")

    with pytest.raises(ValueError, match=r"requires a JAX-backend closure.*pwlr"):
        JointOptimizer(
            forward_model=_stub_forward,
            elements=["Fe", "Ni"],
            wavelength=np.linspace(400.0, 500.0, 8),
            closure=PWLRClosure(),
        )


@pytest.mark.requires_jax
def test_jax_solvers_accept_default_softmax():
    """Sanity check: default (None) and explicit SoftmaxClosure both pass the guard."""
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from cflibs.inversion.solve.coarse_to_fine import SpectralFitter

    def _stub_forward(*args, **kwargs):  # pragma: no cover - never called
        raise AssertionError("forward model should not be invoked at construction")

    SpectralFitter(
        forward_model=_stub_forward,
        elements=["Fe", "Ni"],
        wavelength=jnp.linspace(400.0, 500.0, 8),
    )
    SpectralFitter(
        forward_model=_stub_forward,
        elements=["Fe", "Ni"],
        wavelength=jnp.linspace(400.0, 500.0, 8),
        closure=SoftmaxClosure(),
    )
