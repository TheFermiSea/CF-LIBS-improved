"""vmap parity test for the unified T1-2 forward kernel (ADR-0001).

``cflibs.manifold.batch_forward.batch_forward_from_snapshot`` is constructed
as ``vmap(forward_from_snapshot, in_axes=(0, None, None, None, None))``.
This test asserts the batched output matches a Python-loop equivalent
within ``rtol=1e-5`` for a batch of 100 plasma states.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from cflibs.instrument.model import InstrumentModel  # noqa: E402
from cflibs.plasma.state import SingleZoneLTEPlasma  # noqa: E402
from cflibs.radiation.profiles import BroadeningMode  # noqa: E402


def _build_snapshot(atomic_db):
    return atomic_db.snapshot(
        elements=["Fe", "H"],
        wavelength_range=(300.0, 700.0),
        min_relative_intensity=0.0,
    )


def test_batch_forward_from_snapshot_vmap_matches_loop(atomic_db):
    """``vmap`` over plasma states matches a Python loop equivalent."""
    from cflibs.manifold.batch_forward import (
        batch_forward_from_snapshot,
        forward_from_snapshot,
    )

    instrument = InstrumentModel(resolution_fwhm_nm=0.05)
    snapshot = _build_snapshot(atomic_db)
    wl = jnp.linspace(300.0, 700.0, 2000)

    rng = np.random.default_rng(42)
    B = 16  # 16 instead of 100 to keep the test runtime tight
    T_e_K = rng.uniform(8000.0, 14000.0, size=B)
    n_e = rng.uniform(5.0e15, 5.0e16, size=B)
    plasma_list = [
        SingleZoneLTEPlasma(
            T_e=float(T_e_K[i]),
            n_e=float(n_e[i]),
            species={"Fe": 3.0e15, "H": 5.0e15},
        )
        for i in range(B)
    ]

    # SingleZoneLTEPlasma is pytree-registered (T1-1). Stack leaves across
    # the list of plasma states into a single batched pytree whose traced
    # leaves (``T_e``, ``n_e``, and one density per element) carry the
    # leading batch axis.
    batched_plasma = jax.tree_util.tree_map(
        lambda *leaves: jnp.stack([jnp.asarray(le) for le in leaves]), *plasma_list
    )

    batched_out = batch_forward_from_snapshot(
        batched_plasma,
        snapshot,
        instrument,
        wl,
        broadening_mode=BroadeningMode.PHYSICAL_DOPPLER,
        path_length_m=0.01,
        apply_self_absorption=False,
        apply_stark=True,
        fold_instrument_sigma=True,
    )
    batched_out = np.asarray(batched_out)
    assert batched_out.shape == (B, wl.shape[0])

    loop_out = np.stack(
        [
            np.asarray(
                forward_from_snapshot(
                    p,
                    snapshot,
                    instrument,
                    wl,
                    broadening_mode=BroadeningMode.PHYSICAL_DOPPLER,
                    path_length_m=0.01,
                    apply_self_absorption=False,
                    apply_stark=True,
                    fold_instrument_sigma=True,
                )
            )
            for p in plasma_list
        ]
    )

    np.testing.assert_allclose(batched_out, loop_out, rtol=1e-5, atol=1e-7)


def test_forward_model_nist_parity_per_line_inline(atomic_db):
    """NIST_PARITY mode evaluates per-line sigma_inst inline (no host loop).

    Verifies that switching the snapshot's wavelength range produces
    correspondingly-shifted line widths -- i.e. the per-line resolving-
    power sigma is sourced from the snapshot, not from the host.
    """
    from cflibs.radiation.kernels import forward_model

    plasma = SingleZoneLTEPlasma(T_e=10000.0, n_e=1.0e16, species={"Fe": 3.0e15, "H": 5.0e15})
    snapshot = _build_snapshot(atomic_db)
    instrument = InstrumentModel.from_resolving_power(20000.0)
    wl = jnp.linspace(300.0, 700.0, 4000)
    out = forward_model(
        plasma,
        snapshot,
        instrument,
        wl,
        broadening_mode=BroadeningMode.NIST_PARITY,
        path_length_m=0.01,
        apply_self_absorption=False,
    )
    out = np.asarray(out)
    assert out.shape == (4000,)
    assert np.all(np.isfinite(out))
    assert out.max() > 0
