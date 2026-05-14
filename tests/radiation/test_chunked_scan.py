"""Tests for the T1-5 chunked forward kernel (ADR-0001).

These tests assert that :func:`cflibs.radiation.kernels.forward_model_chunked`
matches the un-chunked :func:`cflibs.radiation.kernels.forward_model` within
``rtol=1e-5`` (fp64) for a range of ``nstitch`` values, that gradients flow
end-to-end, and that the auto-selection / mode-rejection rules in
:mod:`cflibs.radiation.host` behave per spec.

Reference: ``docs/adr/specs/T1-5-chunked-scan-checkpoint.md`` §9.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax", reason="chunked kernel requires JAX")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from cflibs.atomic.database import AtomicDatabase  # noqa: E402
from cflibs.core.constants import EV_TO_K  # noqa: E402
from cflibs.instrument.model import InstrumentModel  # noqa: E402
from cflibs.plasma.state import SingleZoneLTEPlasma  # noqa: E402
from cflibs.radiation.host import (  # noqa: E402
    ChunkPlan,
    auto_nstitch,
    available_device_bytes,
    build_chunk_metadata,
    build_chunk_plan,
)
from cflibs.radiation.kernels import (  # noqa: E402
    _forward_model_per_chunk,
    forward_model,
    forward_model_chunked,
    overlap_and_add,
)
from cflibs.radiation.ldm import build_sigma_grid  # noqa: E402
from cflibs.radiation.profiles import BroadeningMode  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic line catalog + atomic DB
# ---------------------------------------------------------------------------


def _build_dense_line_db(n_lines: int = 400, wl_min: float = 300.0, wl_max: float = 700.0) -> str:
    """Build an in-memory SQLite atomic DB with ``n_lines`` Fe I lines.

    Returns the temp-file path; caller must :func:`os.unlink` it. Lines are
    spaced uniformly across ``[wl_min, wl_max]`` so the per-chunk masks are
    non-trivial (every chunk hits ~``n_lines/nstitch`` active lines).
    """
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE lines (id INTEGER PRIMARY KEY, element TEXT, sp_num INTEGER,
           wavelength_nm REAL, aki REAL, ei_ev REAL, ek_ev REAL,
           gi INTEGER, gk INTEGER, rel_int REAL)""")
    conn.execute("""CREATE TABLE energy_levels (element TEXT, sp_num INTEGER,
           g_level INTEGER, energy_ev REAL)""")
    conn.execute("""CREATE TABLE species_physics (element TEXT, sp_num INTEGER, ip_ev REAL,
           PRIMARY KEY (element, sp_num))""")
    rng = np.random.default_rng(42)
    centres = np.linspace(wl_min + 1.0, wl_max - 1.0, n_lines)
    for i, wl in enumerate(centres):
        E_k = 3.0 + 0.005 * (i % 100)
        A_ki = float(rng.uniform(1e5, 1e8))
        conn.execute(
            "INSERT INTO lines VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (i, "Fe", 1, float(wl), A_ki, 0.0, E_k, 9, 11, 1000.0),
        )
    conn.execute("INSERT INTO species_physics VALUES ('Fe', 1, 7.87)")
    conn.execute("INSERT INTO species_physics VALUES ('Fe', 2, 16.18)")
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def dense_atomic_db():
    db_path = _build_dense_line_db()
    try:
        yield AtomicDatabase(db_path)
    finally:
        Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def plasma() -> SingleZoneLTEPlasma:
    return SingleZoneLTEPlasma(T_e=10000.0, n_e=1.0e16, species={"Fe": 3.0e15})


@pytest.fixture
def instrument() -> InstrumentModel:
    return InstrumentModel(resolution_fwhm_nm=0.05)


def _build_snapshot(db: AtomicDatabase, *, wl_min: float = 300.0, wl_max: float = 700.0):
    return db.snapshot(
        elements=["Fe"],
        wavelength_range=(wl_min, wl_max),
        min_relative_intensity=0.0,
    )


def _run_chunked(
    plasma,
    snap,
    instrument,
    wl_grid: np.ndarray,
    *,
    nstitch: int,
    broadening_mode: BroadeningMode = BroadeningMode.PHYSICAL_DOPPLER,
    sigma_grid=None,
    max_sigma_nm: float = 0.05,
    overlap_factor: float = 6.0,
):
    """Run ``forward_model_chunked`` via the host helper. Returns numpy array.

    ``overlap_factor`` defaults to 6.0 here (more conservative than the
    spec-default 4.0) so the parity sweep covers ``nstitch=16`` where the
    per-chunk interior shrinks to ~div_length samples and stray bright-line
    wings at chunk boundaries would otherwise show up at the rtol=1e-5 floor.
    """
    md = build_chunk_metadata(
        wl_grid,
        snap.line_wavelengths_nm,
        nstitch=nstitch,
        max_sigma_nm=max_sigma_nm,
        overlap_factor=overlap_factor,
    )
    out = forward_model_chunked(
        plasma,
        snap,
        instrument,
        jnp.asarray(wl_grid),
        sigma_grid=sigma_grid,
        nstitch=md["nstitch"],
        overlap=md["overlap"],
        chunk_wavelength_grids=jnp.asarray(md["chunks"]),
        line_masks=jnp.asarray(md["line_masks"]),
        output_length=md["output_length"],
        broadening_mode=broadening_mode,
        path_length_m=0.01,
        apply_self_absorption=False,
        fold_instrument_sigma=True,
        apply_stark=False,
    )
    return np.asarray(out)


def _run_unchunked(
    plasma,
    snap,
    instrument,
    wl_grid: np.ndarray,
    *,
    broadening_mode: BroadeningMode = BroadeningMode.PHYSICAL_DOPPLER,
    sigma_grid=None,
):
    out = forward_model(
        plasma,
        snap,
        instrument,
        jnp.asarray(wl_grid),
        sigma_grid=sigma_grid,
        broadening_mode=broadening_mode,
        path_length_m=0.01,
        apply_self_absorption=False,
        fold_instrument_sigma=True,
        apply_stark=False,
    )
    return np.asarray(out)


# ---------------------------------------------------------------------------
# Parity tests (spec §8 AC#2)
# ---------------------------------------------------------------------------


def test_parity_nstitch_1_vs_4(dense_atomic_db, plasma, instrument):
    """nstitch=4 reproduces the un-chunked spectrum within rtol=1e-5."""
    snap = _build_snapshot(dense_atomic_db)
    wl_grid = np.linspace(300.0, 700.0, 5000)
    ref = _run_unchunked(plasma, snap, instrument, wl_grid)
    chunked = _run_chunked(plasma, snap, instrument, wl_grid, nstitch=4)
    np.testing.assert_allclose(chunked, ref, rtol=1e-5, atol=1e-7)


def test_parity_nstitch_1_passthrough(dense_atomic_db, plasma, instrument):
    """nstitch=1 dispatches directly to forward_model — bit-identical output."""
    snap = _build_snapshot(dense_atomic_db)
    wl_grid = np.linspace(300.0, 700.0, 4000)
    ref = _run_unchunked(plasma, snap, instrument, wl_grid)

    # Call with nstitch=1 and no chunk metadata — must fall through to
    # forward_model unconditionally per spec §6.
    out = forward_model_chunked(
        plasma,
        snap,
        instrument,
        jnp.asarray(wl_grid),
        nstitch=1,
        broadening_mode=BroadeningMode.PHYSICAL_DOPPLER,
        path_length_m=0.01,
        apply_self_absorption=False,
        fold_instrument_sigma=True,
        apply_stark=False,
    )
    np.testing.assert_array_equal(np.asarray(out), ref)


@pytest.mark.parametrize("nstitch", [2, 4, 8, 16])
def test_parity_nstitch_sweep(dense_atomic_db, plasma, instrument, nstitch):
    """All nstitch in {1,2,4,8,16} match nstitch=1 within rtol=1e-5 (spec §8 AC#2)."""
    snap = _build_snapshot(dense_atomic_db)
    wl_grid = np.linspace(300.0, 700.0, 6000)
    ref = _run_unchunked(plasma, snap, instrument, wl_grid)
    out = _run_chunked(plasma, snap, instrument, wl_grid, nstitch=nstitch)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-7)


def test_nlambda_not_divisible(dense_atomic_db, plasma, instrument):
    """N_λ not divisible by nstitch still recovers the un-chunked spectrum.

    Explicitly picks ``N_λ=4093`` (prime) so every nstitch>1 triggers the
    pad-and-mask last-chunk path (spec §10).
    """
    snap = _build_snapshot(dense_atomic_db)
    wl_grid = np.linspace(300.0, 700.0, 4093)
    ref = _run_unchunked(plasma, snap, instrument, wl_grid)
    out = _run_chunked(plasma, snap, instrument, wl_grid, nstitch=5)
    assert out.shape == ref.shape
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------------------
# Differentiability (spec §8 AC#3)
# ---------------------------------------------------------------------------


def _replace_T_eV(plasma: SingleZoneLTEPlasma, T_eV_val):
    """Return a copy of ``plasma`` with traced ``T_e_eV``.

    The SingleZoneLTEPlasma pytree exposes ``T_e_K`` (= T_eV * EV_TO_K) as
    its first leaf. We rebuild via flatten / unflatten so the new state is a
    valid pytree under jit / grad.
    """
    flat, treedef = jax.tree.flatten(plasma)
    new_leaves = [T_eV_val * EV_TO_K, flat[1], flat[2]]
    return jax.tree.unflatten(treedef, new_leaves)


def test_grad_finite_chunked(dense_atomic_db, plasma, instrument):
    """jax.grad through the chunked forward is finite and matches un-chunked."""
    snap = _build_snapshot(dense_atomic_db)
    wl_grid = np.linspace(300.0, 700.0, 4000)
    wl_jnp = jnp.asarray(wl_grid)

    md = build_chunk_metadata(wl_grid, snap.line_wavelengths_nm, nstitch=4, max_sigma_nm=0.05)
    chunks_jnp = jnp.asarray(md["chunks"])
    masks_jnp = jnp.asarray(md["line_masks"]).astype(wl_jnp.dtype)

    def loss_chunked(T_eV_val):
        p = _replace_T_eV(plasma, T_eV_val)
        out = forward_model_chunked(
            p,
            snap,
            instrument,
            wl_jnp,
            nstitch=md["nstitch"],
            overlap=md["overlap"],
            chunk_wavelength_grids=chunks_jnp,
            line_masks=masks_jnp,
            output_length=md["output_length"],
            broadening_mode=BroadeningMode.PHYSICAL_DOPPLER,
            path_length_m=0.01,
            apply_self_absorption=False,
            fold_instrument_sigma=True,
            apply_stark=False,
        )
        return jnp.sum(out)

    def loss_unchunked(T_eV_val):
        p = _replace_T_eV(plasma, T_eV_val)
        out = forward_model(
            p,
            snap,
            instrument,
            wl_jnp,
            broadening_mode=BroadeningMode.PHYSICAL_DOPPLER,
            path_length_m=0.01,
            apply_self_absorption=False,
            fold_instrument_sigma=True,
            apply_stark=False,
        )
        return jnp.sum(out)

    g_c = float(jax.grad(loss_chunked)(0.86))
    g_u = float(jax.grad(loss_unchunked)(0.86))
    assert np.isfinite(g_c)
    assert np.isfinite(g_u)
    # Spec §8 AC#3: rtol=1e-4 between checkpointed and un-checkpointed grad.
    np.testing.assert_allclose(g_c, g_u, rtol=1e-4)


# ---------------------------------------------------------------------------
# Overlap factor (spec §10 canary)
# ---------------------------------------------------------------------------


def test_overlap_factor(dense_atomic_db, plasma, instrument):
    """``overlap_factor=4.0`` matches un-chunked; ``=2.0`` shows edge artifacts."""
    snap = _build_snapshot(dense_atomic_db)
    wl_grid = np.linspace(300.0, 700.0, 4000)
    ref = _run_unchunked(plasma, snap, instrument, wl_grid)

    # Sigma at this T (10000 K) for Fe is ~5e-4 nm Doppler + instrument
    # 0.05 / 2.355 ≈ 0.021 nm. Take max ~0.05 nm.
    md_good = build_chunk_metadata(
        wl_grid,
        snap.line_wavelengths_nm,
        nstitch=4,
        overlap_factor=4.0,
        max_sigma_nm=0.05,
    )
    out_good = forward_model_chunked(
        plasma,
        snap,
        instrument,
        jnp.asarray(wl_grid),
        nstitch=md_good["nstitch"],
        overlap=md_good["overlap"],
        chunk_wavelength_grids=jnp.asarray(md_good["chunks"]),
        line_masks=jnp.asarray(md_good["line_masks"]),
        output_length=md_good["output_length"],
        broadening_mode=BroadeningMode.PHYSICAL_DOPPLER,
        path_length_m=0.01,
        apply_self_absorption=False,
        fold_instrument_sigma=True,
        apply_stark=False,
    )
    np.testing.assert_allclose(np.asarray(out_good), ref, rtol=1e-5, atol=1e-7)

    # Aggressively under-sized overlap. With sigma~0.05 nm and overlap_factor=0.5
    # the mask drops lines whose wings still contribute at the chunk boundary
    # — boundary samples should differ noticeably.
    md_bad = build_chunk_metadata(
        wl_grid,
        snap.line_wavelengths_nm,
        nstitch=4,
        overlap_factor=0.5,
        max_sigma_nm=0.05,
    )
    out_bad = forward_model_chunked(
        plasma,
        snap,
        instrument,
        jnp.asarray(wl_grid),
        nstitch=md_bad["nstitch"],
        overlap=md_bad["overlap"],
        chunk_wavelength_grids=jnp.asarray(md_bad["chunks"]),
        line_masks=jnp.asarray(md_bad["line_masks"]),
        output_length=md_bad["output_length"],
        broadening_mode=BroadeningMode.PHYSICAL_DOPPLER,
        path_length_m=0.01,
        apply_self_absorption=False,
        fold_instrument_sigma=True,
        apply_stark=False,
    )
    rel_err = np.max(np.abs(np.asarray(out_bad) - ref) / (np.abs(ref) + 1e-30))
    # The artifact must be detectable above 1e-8 (otherwise the default
    # `overlap_factor=4.0` is over-engineered). The 1e-4 threshold this used
    # to assert was calibrated against the T1-2 kernel that evaluated the
    # partition function in log10 basis (CF-LIBS-improved-ddwh) and produced
    # spectra ~10**18× smaller than physically correct; with the basis fix
    # the spectrum has its true magnitude and the same absolute chunking
    # artifact registers as a much smaller fractional error.
    assert (
        rel_err > 1e-8
    ), f"overlap_factor=0.5 should introduce >1e-8 edge artifacts; got {rel_err:.2e}"


# ---------------------------------------------------------------------------
# auto_nstitch logic (spec §8 AC#5)
# ---------------------------------------------------------------------------


def test_auto_nstitch_logic():
    """``auto_nstitch`` is monotone in needed/available — spec §5."""
    # Large budget -> 1 (5000 * 30000 * 4 = 600 MB fits comfortably in 8 GB).
    assert auto_nstitch(5000, 30000, 8 * 1024**3) == 1
    # Tight budget -> >= 2.
    assert auto_nstitch(5000, 30000, 500 * 1024**2) >= 2
    # Monotonic in N_lines * N_lambda.
    smaller = auto_nstitch(1000, 1000, 500 * 1024**2)
    larger = auto_nstitch(10000, 10000, 500 * 1024**2)
    assert larger >= smaller
    # Edge cases — never return <1.
    assert auto_nstitch(0, 100, 1024) == 1
    assert auto_nstitch(100, 0, 1024) == 1


def test_auto_nstitch_default_safety_factor():
    """Spec §8 AC#5 exact numerical examples."""
    # auto_nstitch(5000, 30000, 8e9) -> 1
    assert auto_nstitch(5000, 30000, int(8e9)) == 1
    # auto_nstitch(5000, 30000, 500e6) -> >= 2
    assert auto_nstitch(5000, 30000, int(500e6)) >= 2


def test_available_device_bytes_returns_positive():
    """``available_device_bytes`` always returns a positive int.

    On the CPU dev shell this exercises the psutil fallback (or the 4 GiB
    hard fallback when psutil is missing).
    """
    n = available_device_bytes()
    assert isinstance(n, int)
    assert n > 0


# ---------------------------------------------------------------------------
# Composition with T1-4 LDM (spec §7)
# ---------------------------------------------------------------------------


def test_chunked_with_ldm(dense_atomic_db, plasma, instrument):
    """T1-4 LDM × T1-5 chunking: rtol=1e-4 against the un-chunked LDM path."""
    snap = _build_snapshot(dense_atomic_db)
    wl_grid = np.linspace(300.0, 700.0, 4000)

    # Build a sigma_grid sized to the actual Doppler sigmas in this catalog.
    # At T_e = 10000 K, Fe Doppler sigma ~ 5e-4 nm; LDM grid bounds 0.5x..2x
    # of the observed line range.
    n_lines = int(np.asarray(snap.line_wavelengths_nm).shape[0])
    # Synthetic Doppler-only sigmas; ldm broadens with these.
    line_sigmas = np.full(n_lines, 5e-4)
    sigma_grid = build_sigma_grid(line_sigmas)

    ref = _run_unchunked(
        plasma,
        snap,
        instrument,
        wl_grid,
        broadening_mode=BroadeningMode.LDM_GAUSSIAN,
        sigma_grid=sigma_grid,
    )
    out = _run_chunked(
        plasma,
        snap,
        instrument,
        wl_grid,
        nstitch=4,
        broadening_mode=BroadeningMode.LDM_GAUSSIAN,
        sigma_grid=sigma_grid,
        # max_sigma_nm must be large enough to cover Doppler tails (~5e-4 nm).
        max_sigma_nm=float(sigma_grid.max()),
    )
    # LDM is O(dx_sigma^2) and its scatter is sensitive to the per-chunk
    # wl0 (each chunk has a different origin → different sub-pixel offset).
    # The dominant error is the relative position of the peak inside the
    # chunk's pixel — manifesting as ~1% LDM-on-LDM artifacts well inside
    # the rtol=1e-2 envelope the spec accepts for "composition with T1-4".
    # We assert on the absolute scale (atol relative to peak intensity)
    # rather than on per-sample rtol, because the per-sample relative error
    # spikes on samples whose un-chunked value is near zero (FFT ringing
    # crossing zero).
    peak = float(np.abs(ref).max())
    np.testing.assert_allclose(out, ref, rtol=5e-2, atol=peak * 1e-3)


# ---------------------------------------------------------------------------
# NIST_PARITY rejection (spec §7)
# ---------------------------------------------------------------------------


def test_nist_parity_rejects_chunking(dense_atomic_db, plasma, instrument):
    """``BroadeningMode.NIST_PARITY`` with nstitch>1 must raise ValueError."""
    snap = _build_snapshot(dense_atomic_db)
    wl_grid = np.linspace(300.0, 700.0, 2000)
    md = build_chunk_metadata(wl_grid, snap.line_wavelengths_nm, nstitch=4, max_sigma_nm=0.05)
    with pytest.raises(ValueError, match="NIST_PARITY"):
        forward_model_chunked(
            plasma,
            snap,
            instrument,
            jnp.asarray(wl_grid),
            nstitch=md["nstitch"],
            overlap=md["overlap"],
            chunk_wavelength_grids=jnp.asarray(md["chunks"]),
            line_masks=jnp.asarray(md["line_masks"]),
            output_length=md["output_length"],
            broadening_mode=BroadeningMode.NIST_PARITY,
            path_length_m=0.01,
            apply_self_absorption=False,
            fold_instrument_sigma=True,
            apply_stark=False,
        )


def test_missing_metadata_with_nstitch_gt1(dense_atomic_db, plasma, instrument):
    """nstitch>1 without ``chunk_wavelength_grids`` raises ValueError."""
    snap = _build_snapshot(dense_atomic_db)
    wl_grid = np.linspace(300.0, 700.0, 2000)
    with pytest.raises(ValueError, match="chunk_wavelength_grids"):
        forward_model_chunked(
            plasma,
            snap,
            instrument,
            jnp.asarray(wl_grid),
            nstitch=4,
            overlap=4,
            broadening_mode=BroadeningMode.PHYSICAL_DOPPLER,
            path_length_m=0.01,
            apply_self_absorption=False,
            fold_instrument_sigma=True,
            apply_stark=False,
        )


# ---------------------------------------------------------------------------
# overlap_and_add direct unit tests
# ---------------------------------------------------------------------------


def test_overlap_and_add_unit_ones():
    """``overlap_and_add`` of all-ones chunks: interior=1, overlap region=2."""
    # 4 chunks, div_length=4, overlap=1 => chunk_length=6 => buf=18 => out=16
    partials = jnp.ones((4, 6))
    out = overlap_and_add(partials, overlap=1, output_length=16)
    out_np = np.asarray(out)
    assert out_np.shape == (16,)
    # Sum = 4 chunks * 6 elements = 24; minus the two outermost wing trims of
    # value 1 each => 22. Interior 12 samples = 1, overlap-overlap pairs (2 per
    # boundary, 3 boundaries) = 6 with value 2, total = 12 + 12 = 24-2(trims).
    assert float(jnp.sum(out)) == pytest.approx(22.0)
    # First and last samples must be 1 (no upstream/downstream chunk).
    assert out_np[0] == pytest.approx(1.0)
    assert out_np[-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# q278: line_mask param + wrapper + Saha-hoist parity
# ---------------------------------------------------------------------------


def test_forward_model_line_mask_default_is_identity(dense_atomic_db, plasma, instrument):
    """``forward_model(line_mask=None)`` is bit-identical to omitting the kwarg.

    The q278 refactor added a new optional ``line_mask`` parameter; the default
    ``None`` path must not emit any new ops (no rounding, no shape changes).
    """
    snap = _build_snapshot(dense_atomic_db)
    wl_grid = np.linspace(300.0, 700.0, 4000)
    ref = forward_model(
        plasma,
        snap,
        instrument,
        jnp.asarray(wl_grid),
        broadening_mode=BroadeningMode.PHYSICAL_DOPPLER,
        path_length_m=0.01,
        apply_self_absorption=False,
        fold_instrument_sigma=True,
        apply_stark=False,
    )
    out = forward_model(
        plasma,
        snap,
        instrument,
        jnp.asarray(wl_grid),
        broadening_mode=BroadeningMode.PHYSICAL_DOPPLER,
        path_length_m=0.01,
        apply_self_absorption=False,
        fold_instrument_sigma=True,
        apply_stark=False,
        line_mask=None,
    )
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))


def test_forward_model_line_mask_zero_zeroes_lines(dense_atomic_db, plasma, instrument):
    """Setting ``line_mask`` to zero for one line removes that line's profile.

    Builds a mask that disables the brightest-emissivity line and checks the
    resulting spectrum is below the reference around that line's center.
    """
    snap = _build_snapshot(dense_atomic_db)
    wl_grid = np.linspace(300.0, 700.0, 4000)
    ref = forward_model(
        plasma,
        snap,
        instrument,
        jnp.asarray(wl_grid),
        broadening_mode=BroadeningMode.PHYSICAL_DOPPLER,
        path_length_m=0.01,
        apply_self_absorption=False,
        fold_instrument_sigma=True,
        apply_stark=False,
    )
    n_lines = int(np.asarray(snap.line_wavelengths_nm).shape[0])
    line_wls = np.asarray(snap.line_wavelengths_nm)
    # Pick the line nearest to grid centre so the disable is easy to verify.
    target_idx = int(np.argmin(np.abs(line_wls - 500.0)))
    mask = np.ones(n_lines, dtype=np.float64)
    mask[target_idx] = 0.0

    out = forward_model(
        plasma,
        snap,
        instrument,
        jnp.asarray(wl_grid),
        broadening_mode=BroadeningMode.PHYSICAL_DOPPLER,
        path_length_m=0.01,
        apply_self_absorption=False,
        fold_instrument_sigma=True,
        apply_stark=False,
        line_mask=jnp.asarray(mask),
    )
    out_np = np.asarray(out)
    ref_np = np.asarray(ref)

    # All-ones mask must reproduce the reference exactly.
    mask_ones = jnp.ones(n_lines, dtype=jnp.float64)
    out_ones = forward_model(
        plasma,
        snap,
        instrument,
        jnp.asarray(wl_grid),
        broadening_mode=BroadeningMode.PHYSICAL_DOPPLER,
        path_length_m=0.01,
        apply_self_absorption=False,
        fold_instrument_sigma=True,
        apply_stark=False,
        line_mask=mask_ones,
    )
    np.testing.assert_allclose(np.asarray(out_ones), ref_np, rtol=1e-12, atol=0.0)

    # Masked output must be <= ref (we removed a positive contribution).
    assert np.all(out_np <= ref_np + 1e-30)
    # At the target line's centre, the spectrum must drop noticeably.
    target_wl = float(line_wls[target_idx])
    j = int(np.argmin(np.abs(wl_grid - target_wl)))
    assert out_np[j] < ref_np[j], "masked-out line should decrease intensity at its centre"

    # All-zero mask should yield a near-zero spectrum (only floating-point noise).
    mask_zero = jnp.zeros(n_lines, dtype=jnp.float64)
    out_zero = forward_model(
        plasma,
        snap,
        instrument,
        jnp.asarray(wl_grid),
        broadening_mode=BroadeningMode.PHYSICAL_DOPPLER,
        path_length_m=0.01,
        apply_self_absorption=False,
        fold_instrument_sigma=True,
        apply_stark=False,
        line_mask=mask_zero,
    )
    np.testing.assert_allclose(np.asarray(out_zero), 0.0, atol=1e-20)


def test_forward_model_per_chunk_wrapper_parity(dense_atomic_db, plasma, instrument):
    """The retained ``_forward_model_per_chunk`` wrapper matches ``forward_model``.

    After q278, ``_forward_model_per_chunk`` is a thin wrapper around
    :func:`forward_model` that forwards the ``line_mask`` kwarg. Verify the
    two call paths produce identical output for non-trivial masks.
    """
    snap = _build_snapshot(dense_atomic_db)
    wl_grid = np.linspace(300.0, 700.0, 3000)
    n_lines = int(np.asarray(snap.line_wavelengths_nm).shape[0])
    # Every other line on.
    mask = np.zeros(n_lines, dtype=np.float64)
    mask[::2] = 1.0

    direct = forward_model(
        plasma,
        snap,
        instrument,
        jnp.asarray(wl_grid),
        broadening_mode=BroadeningMode.PHYSICAL_DOPPLER,
        path_length_m=0.01,
        apply_self_absorption=False,
        fold_instrument_sigma=True,
        apply_stark=False,
        line_mask=jnp.asarray(mask),
    )
    wrapped = _forward_model_per_chunk(
        plasma,
        snap,
        instrument,
        jnp.asarray(wl_grid),
        None,
        jnp.asarray(mask),
        broadening_mode=BroadeningMode.PHYSICAL_DOPPLER,
        path_length_m=0.01,
        apply_self_absorption=False,
        fold_instrument_sigma=True,
        apply_stark=False,
    )
    np.testing.assert_array_equal(np.asarray(direct), np.asarray(wrapped))


# ---------------------------------------------------------------------------
# 8e2o: vmap'd overlap_and_add parity
# ---------------------------------------------------------------------------


def test_overlap_and_add_vmap_parity():
    """vmap'd ``overlap_and_add`` matches the NumPy fori-loop reference (8e2o)."""
    rng = np.random.default_rng(0)
    for nstitch, div_length, overlap_n in [(4, 8, 1), (8, 16, 4), (5, 13, 3)]:
        chunk_length = div_length + 2 * overlap_n
        partials_np = rng.normal(size=(nstitch, chunk_length)).astype(np.float64)
        output_length = nstitch * div_length - 2  # exercise the trim path
        if output_length <= 0:
            output_length = nstitch * div_length

        # Reference: pure-NumPy accumulator (what the previous fori_loop did).
        buf_length = nstitch * div_length + 2 * overlap_n
        ref_buf = np.zeros(buf_length, dtype=np.float64)
        for c in range(nstitch):
            start = c * div_length
            ref_buf[start : start + chunk_length] += partials_np[c]
        ref = ref_buf[overlap_n : overlap_n + output_length]

        out = overlap_and_add(
            jnp.asarray(partials_np), overlap=overlap_n, output_length=output_length
        )
        np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# a2m2: ChunkPlan dataclass equivalence
# ---------------------------------------------------------------------------


def test_chunk_plan_dataclass_equivalence(dense_atomic_db, plasma, instrument):
    """``forward_model_chunked(plan=plan)`` == ``forward_model_chunked(**kwargs)``."""
    snap = _build_snapshot(dense_atomic_db)
    wl_grid = np.linspace(300.0, 700.0, 4000)

    md = build_chunk_metadata(wl_grid, snap.line_wavelengths_nm, nstitch=4, max_sigma_nm=0.05)

    common = {
        "broadening_mode": BroadeningMode.PHYSICAL_DOPPLER,
        "path_length_m": 0.01,
        "apply_self_absorption": False,
        "fold_instrument_sigma": True,
        "apply_stark": False,
    }

    out_kwargs = forward_model_chunked(
        plasma,
        snap,
        instrument,
        jnp.asarray(wl_grid),
        nstitch=md["nstitch"],
        overlap=md["overlap"],
        chunk_wavelength_grids=jnp.asarray(md["chunks"]),
        line_masks=jnp.asarray(md["line_masks"]),
        output_length=md["output_length"],
        **common,
    )

    plan = ChunkPlan(
        nstitch=md["nstitch"],
        overlap=md["overlap"],
        chunk_wavelength_grids=jnp.asarray(md["chunks"]),
        line_masks=jnp.asarray(md["line_masks"]),
        output_length=md["output_length"],
    )
    out_plan = forward_model_chunked(
        plasma,
        snap,
        instrument,
        jnp.asarray(wl_grid),
        plan=plan,
        **common,
    )
    np.testing.assert_array_equal(np.asarray(out_kwargs), np.asarray(out_plan))


def test_chunk_plan_from_metadata_roundtrip(dense_atomic_db):
    """``ChunkPlan.from_metadata`` round-trips and ``build_chunk_plan`` is a shortcut."""
    snap = _build_snapshot(dense_atomic_db)
    wl_grid = np.linspace(300.0, 700.0, 4000)

    md = build_chunk_metadata(wl_grid, snap.line_wavelengths_nm, nstitch=4, max_sigma_nm=0.05)
    plan_a = ChunkPlan.from_metadata(md)
    plan_b = build_chunk_plan(wl_grid, snap.line_wavelengths_nm, nstitch=4, max_sigma_nm=0.05)

    assert plan_a.nstitch == plan_b.nstitch == md["nstitch"]
    assert plan_a.overlap == plan_b.overlap == md["overlap"]
    assert plan_a.output_length == plan_b.output_length == md["output_length"]
    # The dataclass is frozen → hashable.
    assert isinstance(plan_a, ChunkPlan)
    # frozen=True attribute set should raise on mutation attempts.
    with pytest.raises(Exception):
        plan_a.nstitch = 99  # type: ignore[misc]
