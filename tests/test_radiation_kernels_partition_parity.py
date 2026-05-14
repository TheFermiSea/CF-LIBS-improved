"""Regression test: cflibs.radiation.kernels._polynomial_partition_function_jax
must agree with the canonical cflibs.plasma.partition implementation
to within machine precision across the LIBS-relevant T range.

The kernels.py helper is inlined to avoid jit import overhead, which
made it easy to drift away from the canonical basis. CF-LIBS-improved-ddwh
caught a log10 / natural-log basis mismatch that produced an 18-orders-
of-magnitude error (Fe I @ 10000K: true U=33.84 vs buggy U=1.95e+20).
This test would have caught that drift at CI time.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from cflibs.plasma.partition import (  # noqa: E402
    polynomial_partition_function_jax as canonical_pf,
)
from cflibs.radiation.kernels import (  # noqa: E402
    _polynomial_partition_function_jax as kernel_pf,
)


# Real Irwin (1981) coefficients for Fe I (natural-log basis), pulled
# from scripts/populate_partition_functions.py. At 10000 K the true
# value is U ~= 33.84; the pre-fix kernel returned U ~= 1.95e+20.
FE_I_COEFFS = jnp.asarray([239.34, -113.17, 20.30, -1.62, 0.049])


def test_kernel_partition_matches_canonical_fe_i_at_10000k():
    """Fe I @ 10000 K must match the canonical (ln-basis) reference.

    Pre-fix the buggy log10 kernel returned ~1.95e+20 here; the
    canonical implementation returns ~33.84.
    """
    T_K = jnp.asarray(10000.0)
    u_kernel = float(kernel_pf(T_K, FE_I_COEFFS))
    u_canonical = float(canonical_pf(T_K, FE_I_COEFFS))
    # Tight tolerance — both functions are mathematically equivalent
    # in exact arithmetic; the only difference should be jax dtype
    # precision (which we don't pin here).
    assert np.isclose(u_kernel, u_canonical, rtol=1e-12, atol=0.0), (
        f"Fe I @ 10000K mismatch: kernel={u_kernel:.6e}, "
        f"canonical={u_canonical:.6e}"
    )
    # Sanity check the absolute magnitude — guards against the
    # canonical also being wrong in some future regression. Loose
    # bound; we don't want to encode the exact coefficient set's
    # output here, just rule out the pre-fix 1e+20 magnitude. The
    # tighter Fe-I physical-bounds check lives in
    # tests/test_partition.py::test_partition_function_physical_bounds
    # which uses the actual database coefficients (not the bd's
    # documentary transcription).
    assert 1.0 < u_kernel < 1e6, (
        f"Fe I @ 10000K U={u_kernel:.6e} is outside the order-of-magnitude "
        "sanity range (1..1e6). Pre-fix this returned ~1.95e+20."
    )


@pytest.mark.parametrize("T_K", [2000.0, 5000.0, 8000.0, 10000.0, 15000.0, 20000.0, 25000.0])
def test_kernel_partition_matches_canonical_across_libs_range(T_K: float):
    """rtol=1e-12 parity across T in [2000, 25000] K — the full LIBS
    operating range. Real Irwin coefficients (Fe I)."""
    T = jnp.asarray(T_K)
    u_kernel = float(kernel_pf(T, FE_I_COEFFS))
    u_canonical = float(canonical_pf(T, FE_I_COEFFS))
    assert np.isclose(u_kernel, u_canonical, rtol=1e-12, atol=0.0), (
        f"T={T_K}K: kernel={u_kernel:.6e}, canonical={u_canonical:.6e}"
    )


def test_kernel_partition_batch_matches_canonical():
    """Both helpers must produce identical results under batched (vmap-style)
    inputs — the kernel runs inside a jit'd forward model where T_K and
    coeffs broadcast over (N_species,) at minimum.
    """
    T_K = jnp.asarray(10000.0)
    # 3 fictitious species with the same coeffs — tests that both
    # helpers broadcast identically over the (N_species, 5) shape.
    coeffs_batch = jnp.tile(FE_I_COEFFS[None, :], (3, 1))
    u_kernel = np.asarray(kernel_pf(T_K, coeffs_batch))
    u_canonical = np.asarray(canonical_pf(T_K, coeffs_batch))
    assert u_kernel.shape == u_canonical.shape == (3,)
    assert np.allclose(u_kernel, u_canonical, rtol=1e-12, atol=0.0)


def test_kernel_partition_clamps_t_to_floor():
    """Both helpers must clamp T_K to >= 1.0 K so log/log10 of zero
    or negative T doesn't propagate NaN. Caller may pass garbage T;
    the helper must not crash."""
    for bad_T in (0.0, -100.0, 0.5):
        T = jnp.asarray(bad_T)
        u_kernel = float(kernel_pf(T, FE_I_COEFFS))
        u_canonical = float(canonical_pf(T, FE_I_COEFFS))
        assert np.isfinite(u_kernel), f"kernel U not finite for T={bad_T}"
        assert np.isfinite(u_canonical), f"canonical U not finite for T={bad_T}"
        # And the two must agree even on the clamp.
        assert np.isclose(u_kernel, u_canonical, rtol=1e-12, atol=0.0)
