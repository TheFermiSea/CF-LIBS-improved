"""
Tests for the unified ClosureStrategy protocol and its three adapters.

Verifies:

* :class:`SoftmaxClosure` produces *bit-identical* output to the legacy
  :func:`softmax_closure` (the parity test required by architecture-review
  Candidate 3).
* :class:`ILRClosure` produces output matching the existing
  :meth:`ClosureEquation.apply_ilr` round-trip path.
* :class:`PWLRClosure` produces output matching
  :meth:`ClosureEquation.apply_pwlr` with the dominant-component pivot
  selection.
* All three adapters satisfy the :class:`ClosureStrategy` protocol.
* :meth:`gradient_check` returns sensible values in trace-element regimes.
"""

from __future__ import annotations

import numpy as np
import pytest

from cflibs.inversion.physics.closure import ClosureEquation
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
# SoftmaxClosure parity (the marquee test of Candidate 3)
# ---------------------------------------------------------------------------


@pytest.mark.requires_jax
class TestSoftmaxClosureParity:
    """SoftmaxClosure.apply must be bit-identical to softmax_closure."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")
        from cflibs.inversion.physics.softmax_closure import softmax_closure

        self.jax = jax
        self.jnp = jnp
        self.legacy = softmax_closure
        self.adapter = SoftmaxClosure()

    def _random_theta(self, D: int, seed: int):
        key = self.jax.random.PRNGKey(seed)
        return self.jax.random.normal(key, shape=(D,), dtype=self.jnp.float64)

    def test_bit_identical_1d(self):
        for seed in range(50):
            D = (seed % 12) + 2
            theta = self._random_theta(D, seed=seed)
            legacy_out = self.legacy(theta)
            adapter_out = self.adapter.apply(theta)
            err = float(self.jnp.max(self.jnp.abs(legacy_out - adapter_out)))
            assert err == 0.0, f"Non-bit-identical at seed={seed}, D={D}: err={err}"

    def test_bit_identical_batched(self):
        key = self.jax.random.PRNGKey(123)
        theta_batch = self.jax.random.normal(key, shape=(7, 4), dtype=self.jnp.float64)
        legacy_out = self.legacy(theta_batch)
        adapter_out = self.adapter.apply(theta_batch)
        err = float(self.jnp.max(self.jnp.abs(legacy_out - adapter_out)))
        assert err == 0.0, f"Non-bit-identical for batch: err={err}"

    def test_bit_identical_extreme_values(self):
        theta = self.jnp.array([500.0, -500.0, 0.0], dtype=self.jnp.float64)
        legacy_out = self.legacy(theta)
        adapter_out = self.adapter.apply(theta)
        err = float(self.jnp.max(self.jnp.abs(legacy_out - adapter_out)))
        assert err == 0.0, f"Non-bit-identical for extreme theta: err={err}"


# ---------------------------------------------------------------------------
# ILR parity vs ClosureEquation.apply_ilr
# ---------------------------------------------------------------------------


class TestILRClosureParity:
    """ILRClosure.apply must match ClosureEquation.apply_ilr round-trip."""

    def test_matches_apply_ilr(self):
        elements = ["Fe", "Cu", "Zn"]
        intercepts = {"Fe": -1.0, "Cu": -2.5, "Zn": -3.0}
        partition_funcs = {"Fe": 25.0, "Cu": 2.0, "Zn": 1.0}

        legacy = ClosureEquation.apply_ilr(intercepts, partition_funcs)
        legacy_arr = np.array([legacy.concentrations[el] for el in sorted(elements)])

        raw = np.array(
            [
                partition_funcs[el] * np.exp(intercepts[el])
                for el in sorted(elements)
            ]
        )
        adapter = ILRClosure().apply(raw)
        np.testing.assert_allclose(adapter, legacy_arr, rtol=1e-12, atol=1e-15)

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
# PWLR parity vs ClosureEquation.apply_pwlr
# ---------------------------------------------------------------------------


class TestPWLRClosureParity:
    """PWLRClosure.apply must match ClosureEquation.apply_pwlr."""

    def test_matches_apply_pwlr(self):
        intercepts = {"Fe": -1.0, "Cu": -2.5, "Zn": -3.0}
        partition_funcs = {"Fe": 25.0, "Cu": 2.0, "Zn": 1.0}

        legacy = ClosureEquation.apply_pwlr(intercepts, partition_funcs)
        legacy_arr = np.array(
            [legacy.concentrations[el] for el in sorted(intercepts)]
        )

        raw = np.array(
            [
                partition_funcs[el] * np.exp(intercepts[el])
                for el in sorted(intercepts)
            ]
        )
        adapter = PWLRClosure().apply(raw)
        np.testing.assert_allclose(adapter, legacy_arr, rtol=1e-12, atol=1e-15)

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
