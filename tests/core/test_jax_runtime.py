"""Tests for ``cflibs.core.jax_runtime`` — T1-1 acceptance criteria #11
and §7 test plan rows.
"""

from __future__ import annotations

import logging
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from cflibs.core import jax_runtime
from cflibs.core.jax_runtime import (
    AtomicSnapshot,
    JaxMemoryPolicy,
    check_jax64bit,
    jax_policy,
    jit_if_available,
    set_jax_policy,
    vmap_if_available,
)

# ---------------------------------------------------------------------------
# jit_if_available — both argument forms, both HAS_JAX branches.
# ---------------------------------------------------------------------------


def test_jit_if_available_bare_decorator_preserves_function():
    @jit_if_available
    def square(x):
        return x * x

    assert pytest.approx(square(3.0)) == 9.0


def test_jit_if_available_with_jit_kwargs():
    @jit_if_available(static_argnums=(1,))
    def scale(x, n):
        return x * n

    assert pytest.approx(scale(4.0, 3)) == 12.0


def test_jit_if_available_with_and_without_jax(monkeypatch):
    """Decorator must behave the same when HAS_JAX is forcibly False."""
    monkeypatch.setattr(jax_runtime, "HAS_JAX", False)
    monkeypatch.setattr(jax_runtime, "jax", None)

    @jax_runtime.jit_if_available
    def plain(x):
        return x + 1

    @jax_runtime.jit_if_available(static_argnums=(1,))
    def kwarged(x, n):
        return x + n

    assert plain(4.0) == 5.0
    assert kwarged(10.0, 3) == 13.0


# ---------------------------------------------------------------------------
# vmap_if_available — covers the NumPy-stack fallback path.
# ---------------------------------------------------------------------------


def test_vmap_if_available_with_jax_runs_on_array():
    @vmap_if_available(in_axes=0)
    def double(x):
        return x * 2

    out = double(np.arange(5.0))
    np.testing.assert_array_equal(np.asarray(out), np.arange(5.0) * 2)


def test_vmap_if_available_falls_back_to_numpy_stack(monkeypatch):
    monkeypatch.setattr(jax_runtime, "HAS_JAX", False)
    monkeypatch.setattr(jax_runtime, "jax", None)

    @jax_runtime.vmap_if_available(in_axes=0)
    def f(x):
        return x * x

    out = f(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(out, np.array([1.0, 4.0, 9.0]))


def test_vmap_if_available_fallback_handles_tuple_axes(monkeypatch):
    monkeypatch.setattr(jax_runtime, "HAS_JAX", False)
    monkeypatch.setattr(jax_runtime, "jax", None)

    @jax_runtime.vmap_if_available(in_axes=(0, None))
    def add_offset(x, offset):
        return x + offset

    out = add_offset(np.array([1.0, 2.0, 3.0]), 10.0)
    np.testing.assert_array_equal(out, np.array([11.0, 12.0, 13.0]))


# ---------------------------------------------------------------------------
# JaxMemoryPolicy — frozen + hashable, real_dtype reflects allow_32bit.
# ---------------------------------------------------------------------------


def test_jax_memory_policy_is_hashable_and_frozen():
    p1 = JaxMemoryPolicy()
    p2 = JaxMemoryPolicy()
    assert hash(p1) == hash(p2)
    assert {p1, p2} == {p1}

    with pytest.raises(FrozenInstanceError):
        p1.allow_32bit = True  # type: ignore[misc]


def test_jax_memory_policy_real_dtype_reflects_allow_32bit():
    fp64 = JaxMemoryPolicy().real_dtype
    fp32 = JaxMemoryPolicy(allow_32bit=True).real_dtype

    # Cannot compare directly across JAX/NumPy dtype objects, so compare string repr.
    assert "64" in str(fp64)
    assert "32" in str(fp32)


def test_jax_policy_round_trip():
    default = jax_policy()
    try:
        custom = JaxMemoryPolicy(allow_32bit=True, nstitch=4)
        set_jax_policy(custom)
        assert jax_policy() is custom
        assert "32" in str(jax_policy().real_dtype)
    finally:
        set_jax_policy(default)
    assert jax_policy() is default


# ---------------------------------------------------------------------------
# check_jax64bit — raises when x64 disabled, opt-in waiver for Metal.
# ---------------------------------------------------------------------------


@pytest.mark.requires_jax
def test_check_jax64bit_passes_under_test_conftest():
    """conftest.py sets jax_enable_x64=True; running JAX on CPU should pass."""
    check_jax64bit()


class _FakeJaxConfig:
    def __init__(self, x64_enabled: bool) -> None:
        self.jax_enable_x64 = x64_enabled


@pytest.mark.requires_jax
def test_check_jax64bit_raises_when_x64_disabled(monkeypatch):
    import jax

    monkeypatch.setattr(jax, "config", _FakeJaxConfig(False))
    monkeypatch.setattr(jax_runtime, "jax", jax)
    with pytest.raises(ValueError, match="x64 not active"):
        check_jax64bit()


@pytest.mark.requires_jax
def test_check_jax64bit_warns_only_when_raise_disabled(monkeypatch, caplog):
    import jax

    monkeypatch.setattr(jax, "config", _FakeJaxConfig(False))
    monkeypatch.setattr(jax_runtime, "jax", jax)
    with caplog.at_level(logging.WARNING, logger="cflibs.core.jax_runtime"):
        check_jax64bit(raise_on_violation=False)
    assert any("x64 not active" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# AtomicSnapshot — shape consistency + frozen + pytree registration.
# ---------------------------------------------------------------------------


def _make_snapshot(n_lines=4, n_species=2):
    rng = np.random.default_rng(0)
    return AtomicSnapshot(
        species=tuple(("Fe", s) for s in range(1, n_species + 1)),
        line_wavelengths_nm=rng.uniform(300, 700, size=n_lines),
        line_A_ki=rng.uniform(1e6, 1e8, size=n_lines),
        line_E_k_ev=rng.uniform(0.0, 5.0, size=n_lines),
        line_g_k=rng.integers(1, 12, size=n_lines).astype(float),
        line_E_i_ev=rng.uniform(0.0, 2.0, size=n_lines),
        line_g_i=rng.integers(1, 12, size=n_lines).astype(float),
        line_species_index=rng.integers(0, n_species, size=n_lines).astype(np.int32),
        line_stark_w=np.full(n_lines, 0.01),
        line_stark_alpha=np.zeros(n_lines),
        line_natural_w=np.zeros(n_lines),
        partition_coeffs=rng.normal(size=(n_species, 5)),
        ionization_potential_ev=np.array([7.87, 16.18][:n_species]),
    )


def test_atomic_snapshot_arrays_have_consistent_shape():
    snap = _make_snapshot(n_lines=5, n_species=2)
    assert snap.line_wavelengths_nm.shape == (5,)
    assert snap.line_A_ki.shape == (5,)
    assert snap.line_E_k_ev.shape == (5,)
    assert snap.line_g_k.shape == (5,)
    assert snap.line_g_i.shape == (5,)
    assert snap.line_species_index.shape == (5,)
    assert snap.line_stark_w.shape == (5,)
    assert snap.partition_coeffs.shape[0] == 2
    assert snap.ionization_potential_ev.shape == (2,)
    assert len(snap.species) == 2


def test_atomic_snapshot_is_frozen():
    snap = _make_snapshot()
    with pytest.raises(FrozenInstanceError):
        snap.line_A_ki = np.zeros(4)  # type: ignore[misc]


def test_atomic_snapshot_built_from_database_round_trip(atomic_db):
    """AC #9 — ``AtomicDatabase.snapshot()`` returns a snapshot whose per-line
    arrays mirror the underlying SQLite transitions (rtol=1e-12).
    """
    snap = atomic_db.snapshot(
        elements=["Fe"], wavelength_range=(200.0, 800.0), min_relative_intensity=0.0
    )

    transitions = []
    for stage in (1, 2):
        for t in atomic_db.get_transitions(element="Fe", ionization_stage=stage):
            if 200.0 <= t.wavelength_nm <= 800.0:
                transitions.append(t)

    assert snap.line_wavelengths_nm.shape == (len(transitions),)
    snap_wls = np.asarray(snap.line_wavelengths_nm)
    snap_aki = np.asarray(snap.line_A_ki)
    snap_ek = np.asarray(snap.line_E_k_ev)
    snap_gk = np.asarray(snap.line_g_k)

    by_wl = sorted(transitions, key=lambda t: t.wavelength_nm)
    order = np.argsort(snap_wls)
    np.testing.assert_allclose(
        snap_wls[order], [t.wavelength_nm for t in by_wl], rtol=1e-12, atol=0.0
    )
    np.testing.assert_allclose(snap_aki[order], [t.A_ki for t in by_wl], rtol=1e-12)
    np.testing.assert_allclose(snap_ek[order], [t.E_k_ev for t in by_wl], rtol=1e-12)
    np.testing.assert_allclose(snap_gk[order], [t.g_k for t in by_wl], rtol=1e-12)


def test_atomic_snapshot_rejects_inverted_wavelength_range(atomic_db):
    with pytest.raises(ValueError, match="wavelength_range"):
        atomic_db.snapshot(elements=["Fe"], wavelength_range=(800.0, 200.0))


def test_atomic_snapshot_pad_to_n_elements(atomic_db):
    snap = atomic_db.snapshot(elements=["Fe"], wavelength_range=(200.0, 800.0), pad_to_n_elements=4)
    assert len(snap.species) == 4
    assert snap.ionization_potential_ev.shape == (4,)
    assert snap.partition_coeffs.shape == (4, 5)


@pytest.mark.requires_jax
def test_singlezone_lte_plasma_is_a_jax_pytree():
    """AC #10 — ``jax.vmap`` over a 16-element ``SingleZoneLTEPlasma``
    batch traces without ``TypeError: Argument has no leaves``.
    """
    import jax
    import jax.numpy as jnp

    from cflibs.core.jax_runtime import _ensure_pytrees_registered
    from cflibs.plasma.state import SingleZoneLTEPlasma

    _ensure_pytrees_registered()
    plasma = SingleZoneLTEPlasma(T_e=10000.0, n_e=1e17, species={"Fe": 1e15, "H": 1e16})
    batch = jax.tree_util.tree_map(lambda x: jnp.stack([x] * 16), plasma)
    result = jax.vmap(lambda p: p.T_e)(batch)
    assert result.shape == (16,)
    np.testing.assert_allclose(np.asarray(result), 10000.0)


@pytest.mark.requires_jax
def test_instrument_model_is_a_jax_pytree():
    import jax

    from cflibs.core.jax_runtime import _ensure_pytrees_registered
    from cflibs.instrument.model import InstrumentModel

    _ensure_pytrees_registered()
    instr = InstrumentModel(resolution_fwhm_nm=0.05, resolving_power=2000.0)
    leaves, treedef = jax.tree_util.tree_flatten(instr)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert float(rebuilt.resolution_fwhm_nm) == pytest.approx(0.05)
    assert float(rebuilt.resolving_power) == pytest.approx(2000.0)


@pytest.mark.requires_jax
def test_atomic_snapshot_is_a_jax_pytree():
    import jax

    snap = _make_snapshot(n_lines=3, n_species=2)
    leaves, treedef = jax.tree_util.tree_flatten(snap)
    # 12 mandatory array leaves (level_g/level_E_ev/level_mask are None →
    # not exposed as leaves by jax.tree_util).
    assert len(leaves) == 12
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.species == snap.species
    np.testing.assert_array_equal(rebuilt.line_wavelengths_nm, snap.line_wavelengths_nm)
