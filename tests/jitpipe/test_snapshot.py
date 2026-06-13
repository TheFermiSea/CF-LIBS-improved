"""J0 AC2/AC3/AC4 — ``PipelineSnapshot`` build, cache, byte-stability, bridges.

Covers:
* build from the production DB (shapes from spec §2);
* pytree round-trip + jit/vmap smoke (AC2);
* byte-stable ``.npz`` cache; cache hit skips the SQLite scan; invalidation on
  a DB content change (AC3);
* bridge parity vs BOTH legacy builders — ``AtomicDatabase.snapshot`` (forward
  kernel) and ``_AtomicSnapshot.from_solver`` (lax solver) — for a 15-element
  candidate set (AC4: N_species=30, level pad (30, 676));
* no-SQLite-after-build query-count guard (style of
  ``tests/inversion/test_iterative_lax.py:415-429``).

All CPU-x64 (conftest forces it), well under the 600 s watchdog.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.requires_db, pytest.mark.requires_jax]


# --- AC4 reference candidate set (15 elements) ---------------------------
CANDIDATE_ELEMENTS = [
    "Si",
    "Ti",
    "Al",
    "Fe",
    "Mn",
    "Mg",
    "Ca",
    "Na",
    "K",
    "Cu",
    "Zn",
    "Cr",
    "Ni",
    "Co",
    "V",
]
WL_RANGE = (200.0, 900.0)


def _db_path() -> str:
    for p in (
        Path("ASD_da/libs_production.db"),
        Path("libs_production.db"),
        Path(__file__).resolve().parent.parent.parent / "ASD_da" / "libs_production.db",
    ):
        if p.exists():
            return str(p)
    pytest.skip("Production database not found")
    raise AssertionError  # unreachable


@pytest.fixture(scope="module")
def db_path() -> str:
    return _db_path()


# ---------------------------------------------------------------------------
# Build + shapes (spec §2).
# ---------------------------------------------------------------------------


def test_build_shapes(db_path, tmp_path):
    from cflibs.jitpipe import build_snapshot

    snap = build_snapshot(db_path, cache=True, cache_dir=tmp_path)
    # Spec §2 measured values for libs_production.db.
    assert snap.n_lines > 20000, snap.n_lines
    assert snap.n_species == 175, snap.n_species
    # Padded level block: (175, 676) — Fe II has the most levels.
    pad = snap.level_pad
    assert pad[0] == 175
    assert pad[1] == 676, pad
    # Every per-species block has the right first axis.
    for name in (
        "partition_coeffs",
        "partition_coeffs_stored",
        "species_physics",
        "canonical_fallback",
        "oxide_stoichiometry",
        "partition_t_min",
        "partition_g0",
    ):
        assert np.asarray(getattr(snap, name)).shape[0] == 175, name
    # Doublet pairs + oxide present.
    assert snap.doublet_pairs.shape[1] == 2
    assert snap.doublet_pairs.shape[0] == snap.doublet_rho.shape[0]
    assert snap.doublet_pairs.shape[0] == snap.doublet_r_thin.shape[0]


def test_build_time_under_budget(db_path, tmp_path):
    """Spec §6 risk: the eager canonical re-fit is a one-time build cost.

    A cold build pays the ``partition_spec_for`` direct-sum re-fit (process-
    level cached thereafter); we allow a generous bound, and assert the
    cache-hit path is near-instant.
    """
    import time

    from cflibs.jitpipe import build_snapshot

    t0 = time.time()
    build_snapshot(db_path, cache=True, cache_dir=tmp_path)
    cold = time.time() - t0
    assert cold < 30.0, f"cold build too slow: {cold:.1f}s"

    t0 = time.time()
    build_snapshot(db_path, cache=True, cache_dir=tmp_path)
    warm = time.time() - t0
    assert warm < 5.0, f"cache-hit build too slow: {warm:.1f}s"


# ---------------------------------------------------------------------------
# AC2 — pytree round-trip + jit/vmap smoke.
# ---------------------------------------------------------------------------


def test_pytree_roundtrip(db_path, tmp_path):
    import jax

    from cflibs.jitpipe import build_snapshot

    snap = build_snapshot(db_path, cache=True, cache_dir=tmp_path)
    leaves, treedef = jax.tree_util.tree_flatten(snap)
    snap2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert snap2.species == snap.species
    assert snap2.element_symbols == snap.element_symbols
    assert np.array_equal(np.asarray(snap2.line_wavelength_nm), np.asarray(snap.line_wavelength_nm))
    # Static aux is in the treedef, not the leaves.
    assert all(hasattr(leaf, "shape") for leaf in leaves)


def test_jit_vmap_smoke(db_path, tmp_path):
    import jax
    import jax.numpy as jnp

    from cflibs.jitpipe import build_snapshot

    snap = build_snapshot(db_path, cache=True, cache_dir=tmp_path)

    @jax.jit
    def total_aki(s):
        return jnp.sum(s.line_A_ki)

    out = total_aki(snap)
    assert np.isfinite(float(out))

    # vmap a trivial per-line reduction over a batched leaf.
    batched = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (2,) + x.shape) if hasattr(x, "shape") else x, snap
    )
    vout = jax.vmap(lambda s: jnp.sum(s.line_g_k))(batched)
    assert vout.shape == (2,)


# ---------------------------------------------------------------------------
# AC3 — byte-stable cache, cache hit skips scan, invalidation.
# ---------------------------------------------------------------------------


def test_cache_byte_stable(db_path, tmp_path):
    from cflibs.jitpipe import build_snapshot
    from cflibs.jitpipe import host

    chash = host.db_content_hash(db_path)
    cache_path = host.default_cache_path(chash, cache_dir=tmp_path)

    build_snapshot(db_path, cache=True, cache_dir=tmp_path)
    assert cache_path.exists()
    bytes1 = cache_path.read_bytes()

    # Rebuild over a fresh cache dir and compare bytes.
    other = tmp_path / "other"
    other.mkdir()
    build_snapshot(db_path, cache=True, cache_dir=other)
    bytes2 = host.default_cache_path(chash, cache_dir=other).read_bytes()
    assert bytes1 == bytes2, "two builds from the same DB produced differing .npz"


def test_cache_hit_skips_scan(db_path, tmp_path):
    """A cache hit must NOT re-run the SQLite scan (the ``_scan_counter`` hook)."""
    from cflibs.jitpipe.snapshot import build_snapshot

    counter: list[int] = []
    build_snapshot(db_path, cache=True, cache_dir=tmp_path, _scan_counter=counter)
    assert counter == [1], "first build should scan once"

    build_snapshot(db_path, cache=True, cache_dir=tmp_path, _scan_counter=counter)
    assert counter == [1], "second build (cache hit) must skip the SQLite scan"


def test_cache_invalidates_on_db_change(db_path, tmp_path):
    """A different DB content hash must invalidate the cache (force a rescan)."""
    from cflibs.jitpipe.snapshot import build_snapshot

    # Copy the DB so we can mutate it without touching the shared file.
    db_copy = tmp_path / "db.sqlite"
    shutil.copy(db_path, db_copy)

    counter: list[int] = []
    snap1 = build_snapshot(str(db_copy), cache=True, cache_dir=tmp_path, _scan_counter=counter)
    assert counter == [1]

    # Append a byte to change the content hash (SQLite tolerates trailing bytes
    # for the read path we exercise via copy; the hash is over raw file bytes).
    with open(db_copy, "ab") as fh:
        fh.write(b"\x00")

    snap2 = build_snapshot(str(db_copy), cache=True, cache_dir=tmp_path, _scan_counter=counter)
    assert counter == [1, 1], "DB change must invalidate cache and rescan"
    assert snap2.db_content_hash != snap1.db_content_hash


def test_cache_hit_does_not_open_sqlite(db_path, tmp_path, monkeypatch):
    """No-SQLite-after-build guard (style of test_iterative_lax.py:415-429).

    Once the ``.npz`` cache exists, ``build_snapshot`` must satisfy the request
    without opening a single SQLite connection.
    """
    from cflibs.jitpipe import build_snapshot
    from cflibs.jitpipe import host

    # Prime the cache.
    build_snapshot(db_path, cache=True, cache_dir=tmp_path)

    # Now intercept sqlite3.connect; a cache hit must not call it.
    import sqlite3

    calls = {"n": 0}
    real_connect = sqlite3.connect

    def _counting_connect(*a, **k):
        calls["n"] += 1
        return real_connect(*a, **k)

    monkeypatch.setattr(host.sqlite3, "connect", _counting_connect)
    build_snapshot(db_path, cache=True, cache_dir=tmp_path)
    assert calls["n"] == 0, f"cache hit opened {calls['n']} SQLite connection(s)"


# ---------------------------------------------------------------------------
# AC4 — bridge parity vs BOTH existing builders.
# ---------------------------------------------------------------------------


def test_forward_bridge_parity(db_path, tmp_path):
    """``to_atomic_snapshot`` matches ``AtomicDatabase.snapshot`` field-for-field.

    The reference 15-element candidate set yields N_species=30, level pad
    (30, 676) (AC4). We compare per-species IP, the canonical partition poly,
    its validity window, and the full line wavelength set in the window.
    """
    from cflibs.jitpipe import build_snapshot
    from cflibs.jitpipe import parity

    snap = build_snapshot(db_path, cache=True, cache_dir=tmp_path)
    ref = parity.reference_atomic_snapshot(
        CANDIDATE_ELEMENTS, WL_RANGE, db_path=db_path, include_levels=True
    )

    # AC4 measured reference shapes.
    assert len(ref.species) == 30, len(ref.species)
    assert np.asarray(ref.level_g).shape == (30, 676)

    sp_to_row = {sp: i for i, sp in enumerate(snap.species)}
    ip_mine = np.asarray(snap.species_physics)[:, 0]
    ip_ref = np.asarray(ref.ionization_potential_ev)
    cm = np.asarray(snap.partition_coeffs)
    cr = np.asarray(ref.partition_coeffs)
    tmin_m = np.asarray(snap.partition_t_min)
    tmax_m = np.asarray(snap.partition_t_max)
    tmin_r = np.asarray(ref.partition_t_min)
    tmax_r = np.asarray(ref.partition_t_max)

    for j, sp in enumerate(ref.species):
        assert sp in sp_to_row, f"candidate species {sp} missing from superset"
        i = sp_to_row[sp]
        assert np.isclose(ip_mine[i], ip_ref[j], equal_nan=True), sp
        assert np.allclose(cm[i], cr[j], rtol=1e-10, atol=1e-10, equal_nan=True), sp
        assert np.isclose(tmin_m[i], tmin_r[j]) and np.isclose(tmax_m[i], tmax_r[j]), sp

    # Line block: gather superset lines in the window for the candidate species
    # and compare the sorted wavelength set to the reference's.
    cand_rows = {sp_to_row[sp] for sp in ref.species}
    my_wl = np.asarray(snap.line_wavelength_nm)
    my_sp = np.asarray(snap.line_species_index)
    mask = (my_wl >= WL_RANGE[0]) & (my_wl <= WL_RANGE[1]) & np.isin(my_sp, list(cand_rows))
    my_sorted = np.sort(my_wl[mask])
    ref_sorted = np.sort(np.asarray(ref.line_wavelengths_nm))
    assert my_sorted.shape == ref_sorted.shape, (my_sorted.shape, ref_sorted.shape)
    assert np.allclose(my_sorted, ref_sorted)


def test_lax_bridge_parity(db_path, tmp_path):
    """``to_lax_snapshot`` matches ``_AtomicSnapshot.from_solver`` field-for-field.

    Covers the fields consumed inside ``_run_lax_while_loop`` (AC4): IPs,
    ``use_direct``, padded level g/E + masks, stored poly coeffs (NaN-aware),
    and the eager scalar fallbacks.
    """
    from cflibs.jitpipe import build_snapshot
    from cflibs.jitpipe import parity

    snap = build_snapshot(db_path, cache=True, cache_dir=tmp_path)
    mine = snap.to_lax_snapshot(CANDIDATE_ELEMENTS)
    ref = parity.reference_lax_snapshot(CANDIDATE_ELEMENTS, db_path=db_path)

    assert mine.elements == ref.elements
    assert np.allclose(mine.ip0_eV, ref.ip0_eV)
    assert np.allclose(mine.ip_I_for_direct, ref.ip_I_for_direct)
    assert np.allclose(mine.ip_II_for_direct, ref.ip_II_for_direct)
    assert np.array_equal(mine.use_direct, ref.use_direct)
    assert np.array_equal(mine.g_levels_I, ref.g_levels_I)
    assert np.array_equal(mine.E_levels_I, ref.E_levels_I)
    assert np.array_equal(mine.levels_mask_I, ref.levels_mask_I)
    assert np.array_equal(mine.g_levels_II, ref.g_levels_II)
    assert np.array_equal(mine.E_levels_II, ref.E_levels_II)
    assert np.array_equal(mine.levels_mask_II, ref.levels_mask_II)
    assert np.allclose(mine.coeffs_I, ref.coeffs_I, equal_nan=True)
    assert np.allclose(mine.coeffs_II, ref.coeffs_II, equal_nan=True)
    assert np.allclose(mine.fallback_U_I, ref.fallback_U_I)
    assert np.allclose(mine.fallback_U_II, ref.fallback_U_II)


def test_atomic_snapshot_roundtrip(db_path, tmp_path):
    """``from_atomic_snapshot(to_atomic_snapshot(snap))`` preserves shared fields."""
    from cflibs.jitpipe import build_snapshot
    from cflibs.jitpipe.snapshot import PipelineSnapshot

    snap = build_snapshot(db_path, cache=True, cache_dir=tmp_path)
    atomic = snap.to_atomic_snapshot(include_levels=True)
    back = PipelineSnapshot.from_atomic_snapshot(atomic)

    assert back.species == snap.species
    assert np.allclose(np.asarray(back.line_wavelength_nm), np.asarray(snap.line_wavelength_nm))
    assert np.allclose(np.asarray(back.line_A_ki), np.asarray(snap.line_A_ki))
    assert np.array_equal(np.asarray(back.line_species_index), np.asarray(snap.line_species_index))
    assert np.allclose(np.asarray(back.partition_coeffs), np.asarray(snap.partition_coeffs))
