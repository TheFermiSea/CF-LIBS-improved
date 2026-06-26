"""Tests for ``SingleZoneLTEPlasmaJax``.

The JAX wrapper must preserve the existing ``SingleZoneLTEPlasma`` surface
(so it can be passed transparently into the existing forward model) while
exposing JAX-friendly tensor views of the plasma state.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from cflibs.plasma.state import (  # noqa: E402
    SingleZoneLTEPlasma,
    SingleZoneLTEPlasmaJax,
)


def _realistic_plasma_kwargs(seed: int = 42) -> dict:
    """Multi-element composition matching the task brief."""
    rng = np.random.default_rng(seed)
    return dict(
        T_e=10000.0 + rng.uniform(-1000.0, 1000.0),
        n_e=1.0e16 * (1.0 + 0.1 * rng.standard_normal()),
        species={
            "Si": 4.0e15,
            "Fe": 3.0e15,
            "Mg": 1.5e15,
            "Ca": 1.0e15,
            "Al": 0.5e15,
        },
    )


def test_jax_plasma_preserves_numpy_surface():
    kwargs = _realistic_plasma_kwargs()
    plasma_np = SingleZoneLTEPlasma(**kwargs)
    plasma_jax = SingleZoneLTEPlasmaJax(**kwargs)

    assert plasma_jax.T_e == plasma_np.T_e
    assert plasma_jax.n_e == plasma_np.n_e
    assert plasma_jax.species == plasma_np.species
    assert plasma_jax.T_e_eV == pytest.approx(plasma_np.T_e_eV)
    plasma_jax.validate()


def test_jax_plasma_exposes_jnp_views():
    kwargs = _realistic_plasma_kwargs()
    plasma = SingleZoneLTEPlasmaJax(**kwargs)

    # Tensor views must live in the JAX namespace
    assert "jax" in type(plasma.T_e_jax).__module__
    assert "jax" in type(plasma.species_densities_jax).__module__

    # Numerical equivalence with the Python-float surface
    np.testing.assert_allclose(float(plasma.T_e_jax), plasma.T_e, rtol=1e-12)
    np.testing.assert_allclose(float(plasma.n_e_jax), plasma.n_e, rtol=1e-12)
    np.testing.assert_allclose(float(plasma.T_e_eV_jax), plasma.T_e_eV, rtol=1e-12)

    # species_densities_jax must be 1-D and ordered consistently with species_keys
    assert plasma.species_densities_jax.shape == (len(plasma.species),)
    for k, v in zip(plasma.species_keys, np.asarray(plasma.species_densities_jax)):
        np.testing.assert_allclose(v, plasma.species[k], rtol=1e-12)


def test_from_plasma_round_trip():
    plasma_np = SingleZoneLTEPlasma(**_realistic_plasma_kwargs())
    plasma_jax = SingleZoneLTEPlasmaJax.from_plasma(plasma_np)

    assert plasma_jax.T_e == plasma_np.T_e
    assert plasma_jax.n_e == plasma_np.n_e
    assert plasma_jax.species == plasma_np.species


def test_two_region_plasma_concrete_jnp_array():
    """TwoRegionPlasma must not raise ConcretizationTypeError with concrete jnp arrays (M1-11).

    The old isinstance(T_core, _JAX_TRACER) guard missed concrete jnp.ndarray values
    (they have .ndim but are not Tracer); _is_jax_tracer_or_array gates on both.
    """
    jax = pytest.importorskip("jax")
    jnp = jax.numpy

    from cflibs.plasma.state import TwoRegionPlasma

    T_core = jnp.float32(10000.0)
    T_corona = jnp.float32(6000.0)
    n_e = jnp.float32(1.0e16)
    species = {"Fe": 1.0e15, "Si": 5.0e14}

    plasma = TwoRegionPlasma(T_core=T_core, T_corona=T_corona, n_e=n_e, species=species)
    assert plasma.T_core == pytest.approx(float(T_core))
    assert plasma.T_corona == pytest.approx(float(T_corona))
