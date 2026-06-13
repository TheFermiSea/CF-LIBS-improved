"""Impure host helpers for the jittable inversion pipeline (ADR-0004 §5.1).

This is one of the *three* modules under :mod:`cflibs.jitpipe` permitted to
touch impure inputs (``sqlite3``, :mod:`cflibs.atomic.database`,
:mod:`cflibs.io`) — the others are :mod:`cflibs.jitpipe.snapshot` and
:mod:`cflibs.jitpipe.parity`. Every other ``jitpipe`` module must stay free of
SQLite so the jit-traced kernels never hold a live database connection inside a
trace (enforced by ``tests/jitpipe/test_import_hygiene.py``; see also the
``kernels.py`` host/kernel split documented at
``cflibs/radiation/kernels.py:72-78``).

Responsibilities
----------------
* **DB -> raw arrays.** One single SQLite scan reads the whole atomic database
  (~6 MB) into the struct-of-arrays blocks of §2 of the J0 spec. The scan is
  isolated in :func:`scan_database` so that the rest of the snapshot builder is
  pure NumPy.
* **Content hashing.** :func:`db_content_hash` computes a sha256 over the DB
  file bytes; it is the cache key that drives ``.npz`` invalidation (AC3).
* **Cache I/O.** :func:`default_cache_path`, :func:`load_npz_cache`,
  :func:`save_npz_cache` implement the byte-stable ``.npz`` cache.
* **Padding / bucketing.** :func:`pad_ragged` and :func:`bucket_for_n_lines`
  implement the per-bucket candidate-set assembly seam (the only per-spectrum
  host<->device hop, §2 of the spec).
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants — the canonical DB location + cache layout.
# ---------------------------------------------------------------------------

#: Default production atomic database, relative to the repo root / CWD.
DEFAULT_DB_PATH = "ASD_da/libs_production.db"

#: Polynomial order of ``log U(T)`` partition fits (a0..a4) — 5 coefficients.
N_POLY = 5

#: Canonical scalar partition fallbacks (eager at build, spec §2). These mirror
#: the per-stage defaults used by ``_AtomicSnapshot`` in
#: ``cflibs/inversion/solve/iterative.py:449-450`` (25.0 for stage I, 15.0 for
#: stage II) before the closed-shell-exact refinement.
FALLBACK_U_I = 25.0
FALLBACK_U_II = 15.0

#: NaN sentinel marking "no partition polynomial available" — matches the
#: convention used by the lax solver (``iterative.py:487``).
POLY_NAN_SENTINEL = float("nan")


# ---------------------------------------------------------------------------
# Content hashing + cache paths.
# ---------------------------------------------------------------------------


def db_content_hash(db_path: str | os.PathLike) -> str:
    """Return a sha256 hex digest of the atomic-DB file *contents*.

    The digest is the cache key for the ``.npz`` snapshot cache: two builds
    from byte-identical DBs collide (cache hit, no SQLite scan), and any change
    to the DB file invalidates the cache (AC3).

    Parameters
    ----------
    db_path : str or os.PathLike
        Path to the SQLite atomic database.

    Returns
    -------
    str
        64-character lowercase hex sha256 digest of the file bytes.
    """
    h = hashlib.sha256()
    with open(db_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def default_cache_dir() -> Path:
    """Return the per-user jitpipe snapshot cache directory.

    Uses ``$CFLIBS_JITPIPE_CACHE`` when set, else ``~/.cache/cflibs/jitpipe``
    (the same per-user convention the J0 spec mandates for the JAX compile
    cache, §3). The directory is created on demand.
    """
    env = os.environ.get("CFLIBS_JITPIPE_CACHE")
    base = Path(env) if env else Path.home() / ".cache" / "cflibs" / "jitpipe"
    base.mkdir(parents=True, exist_ok=True)
    return base


def default_cache_path(content_hash: str, *, cache_dir: Path | None = None) -> Path:
    """Return the ``.npz`` cache path for a given DB content hash."""
    base = cache_dir if cache_dir is not None else default_cache_dir()
    return base / f"snapshot-{content_hash}.npz"


# ---------------------------------------------------------------------------
# Padding / bucketing — the per-spectrum host<->device seam.
# ---------------------------------------------------------------------------


def pad_ragged(rows: list[np.ndarray], *, dtype=np.float64) -> tuple[np.ndarray, np.ndarray]:
    """Right-pad a ragged list of 1-D arrays into a dense ``(R, W)`` block + mask.

    Mirrors the ``_pad_ragged_arrays`` gather pattern at
    ``cflibs/inversion/solve/iterative.py`` so the level blocks have identical
    semantics to the lax solver.

    Parameters
    ----------
    rows : list of ndarray
        Ragged per-row 1-D arrays.
    dtype : numpy dtype, optional
        Output dtype for the padded value block. Default ``float64``.

    Returns
    -------
    values : ndarray, shape (R, W)
        Zero-padded value block; ``W`` is the longest row length (>= 1).
    mask : ndarray of bool, shape (R, W)
        ``True`` where a real value sits, ``False`` in padding.
    """
    n = len(rows)
    width = max((len(r) for r in rows), default=1)
    width = max(width, 1)
    values = np.zeros((n, width), dtype=dtype)
    mask = np.zeros((n, width), dtype=bool)
    for i, row in enumerate(rows):
        k = len(row)
        if k:
            values[i, :k] = np.asarray(row, dtype=dtype)
            mask[i, :k] = True
    return values, mask


#: Shape buckets for padded line counts (ADR-0004 §5.2). A spectrum's
#: candidate line set is padded up to the smallest bucket that fits it so the
#: jit cache key (``StaticConfig.bucket_id``) takes only a handful of values.
LINE_BUCKETS: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048, 4096)


def bucket_for_n_lines(n_lines: int) -> int:
    """Return the padded line count (bucket size) for ``n_lines`` real lines.

    Picks the smallest entry of :data:`LINE_BUCKETS` that is ``>= n_lines``;
    counts larger than the biggest bucket are rounded up to the next power of
    two so the cache key still saturates to a small set.

    Parameters
    ----------
    n_lines : int
        Number of real candidate lines for a spectrum.

    Returns
    -------
    int
        Padded line count to allocate.
    """
    for b in LINE_BUCKETS:
        if n_lines <= b:
            return b
    # Beyond the table: round up to next power of two.
    p = 1
    while p < n_lines:
        p <<= 1
    return p


# ---------------------------------------------------------------------------
# Single SQLite scan -> raw struct-of-arrays blocks.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawScan:
    """Pure-NumPy struct-of-arrays result of a single full DB scan.

    Holds every block of J0 spec §2 in DB order (lines ordered by element,
    sp_num, wavelength; species in a stable ``(element, sp_num)`` order). The
    pure builder in :mod:`cflibs.jitpipe.snapshot` turns this into the
    pytree-registered ``PipelineSnapshot`` without any further SQLite access.
    """

    # Species axis (the canonical ordering for every per-species block).
    species: tuple[tuple[str, int], ...]

    # --- lines block (N_lines,) ---
    line_element_index: np.ndarray  # int16, index into the element symbol table
    line_sp_num: np.ndarray  # int8
    line_wavelength_nm: np.ndarray
    line_A_ki: np.ndarray
    line_E_i_ev: np.ndarray
    line_E_k_ev: np.ndarray
    line_g_i: np.ndarray
    line_g_k: np.ndarray
    line_species_index: np.ndarray  # int32 index into ``species``
    line_stark_w: np.ndarray
    line_stark_alpha: np.ndarray
    line_stark_shift: np.ndarray
    line_aki_uncertainty: np.ndarray
    line_is_resonance: np.ndarray  # bool
    line_stark_source_class: np.ndarray  # uint8
    line_gamma_vdw_log: np.ndarray

    # Distinct element symbols, indexed by ``line_element_index``.
    element_symbols: tuple[str, ...]

    # --- energy_levels block, padded (N_species, L_max) ---
    level_g: np.ndarray
    level_E_ev: np.ndarray
    level_mask: np.ndarray

    # --- partition polys (N_species, N_POLY) + window ---
    # ``partition_coeffs`` is the CANONICAL forward-kernel poly from
    # ``partition_spec_for`` (direct-sum-derived re-fit when levels exist), so
    # ``to_atomic_snapshot`` matches ``AtomicDatabase.snapshot`` exactly.
    # ``partition_coeffs_stored`` is the RAW ``partition_functions`` table row
    # with a NaN sentinel when absent, for the lax-solver bridge.
    partition_coeffs: np.ndarray
    partition_coeffs_stored: np.ndarray
    partition_t_min: np.ndarray
    partition_t_max: np.ndarray
    partition_g0: np.ndarray
    partition_from_direct_sum: np.ndarray

    # --- eager canonical scalar fallbacks (N_species, 2): [U_I, U_II] ---
    canonical_fallback: np.ndarray  # (N_species, 2)

    # --- species_physics (N_species, 2): [ip_ev, atomic_mass] ---
    species_physics: np.ndarray

    # --- doublet pairs (P, 2) int32 + rho (P,) + r_thin (P,) ---
    doublet_pairs: np.ndarray
    doublet_rho: np.ndarray
    doublet_r_thin: np.ndarray

    # --- oxide stoichiometry (N_species,): O atoms per cation per species ---
    oxide_stoichiometry: np.ndarray


#: Mapping from the textual ``stark_w_source`` column to the measured class
#: codes of J0 spec §2 (konjevic-λ²-scaled / interpolated / hydrogenic /
#: stark_b / null). Unknown strings map to 0 (null).
STARK_SOURCE_CLASS = {
    None: 0,
    "": 0,
    "konjevic": 1,
    "konjevic-lambda2": 1,
    "interpolated": 2,
    "hydrogenic": 3,
    "stark_b": 4,
}


def _classify_stark_source(raw: str | None) -> int:
    """Map a ``stark_w_source`` string to its J0 §2 class code (uint8)."""
    if raw is None:
        return 0
    key = str(raw).strip().lower()
    if key in STARK_SOURCE_CLASS:
        return STARK_SOURCE_CLASS[key]
    # Heuristic fallbacks for the dominant measured classes.
    if "konjevic" in key or "lambda" in key:
        return 1
    if "interp" in key:
        return 2
    if "hydrog" in key:
        return 3
    if key in ("stark_b", "starkb", "b"):
        return 4
    return 0


def scan_database(
    db_path: str | os.PathLike,
    *,
    oxide_map: dict[str, float] | None = None,
) -> RawScan:
    """Read the entire atomic DB into a :class:`RawScan` in ONE SQLite scan.

    This is the only place a full-database SQLite read happens for a snapshot
    build. The scan covers every §2 block: lines (with stark metadata),
    energy levels (padded), partition polynomials + windows, eager canonical
    scalar fallbacks, ``species_physics``, candidate doublet pairs, and the
    oxide stoichiometry vector.

    Parameters
    ----------
    db_path : str or os.PathLike
        Path to the SQLite atomic database.
    oxide_map : dict, optional
        ``element -> O atoms per cation`` map. Defaults to
        :data:`cflibs.inversion.physics.closure.OXIDE_OXYGEN_PER_CATION`.

    Returns
    -------
    RawScan
        Pure-NumPy struct-of-arrays of every snapshot block.
    """
    if oxide_map is None:
        from cflibs.inversion.physics.closure import OXIDE_OXYGEN_PER_CATION

        oxide_map = dict(OXIDE_OXYGEN_PER_CATION)

    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        species = _scan_species(conn)
        species_idx = {key: i for i, key in enumerate(species)}

        lines = _scan_lines(conn, species_idx)
        levels = _scan_levels(conn, species, species_idx)
        stored = _scan_partition_stored(conn, species, species_idx)
        species_physics = _scan_species_physics(conn, species, species_idx)
        doublets = _scan_doublets(lines)
        oxide = _scan_oxide(species, oxide_map)
        fallbacks = _scan_canonical_fallbacks(species, stored, levels)
    finally:
        conn.close()

    # Canonical forward-kernel partition specs (direct-sum-preferred re-fit).
    # Lives here, in the host carve-out module, because ``partition_spec_for``
    # is a DB-backed factory; baking it into the snapshot makes
    # ``to_atomic_snapshot`` match ``AtomicDatabase.snapshot`` exactly (AC4).
    canon = _scan_partition_canonical(db_path, species)

    return RawScan(
        species=species,
        oxide_stoichiometry=oxide,
        canonical_fallback=fallbacks,
        species_physics=species_physics,
        partition_coeffs=canon[0],
        partition_coeffs_stored=stored[0],
        partition_t_min=canon[1],
        partition_t_max=canon[2],
        partition_g0=canon[3],
        partition_from_direct_sum=canon[4],
        level_g=levels[0],
        level_E_ev=levels[1],
        level_mask=levels[2],
        doublet_pairs=doublets[0],
        doublet_rho=doublets[1],
        doublet_r_thin=doublets[2],
        **lines,
    )


def _scan_species(conn: sqlite3.Connection) -> tuple[tuple[str, int], ...]:
    """Return the canonical ``(element, sp_num)`` species axis ordering.

    Drawn from ``species_physics`` (the authoritative per-species table) and
    sorted deterministically by ``(element, sp_num)`` so the snapshot is
    byte-stable across builds (AC3).
    """
    rows = conn.execute(
        "SELECT element, sp_num FROM species_physics ORDER BY element, sp_num"
    ).fetchall()
    return tuple((str(r["element"]), int(r["sp_num"])) for r in rows)


def _scan_lines(
    conn: sqlite3.Connection, species_idx: dict[tuple[str, int], int]
) -> dict[str, object]:
    """Read the full ``lines`` table into the §2 line block (sorted, stable)."""
    rows = conn.execute("""
        SELECT element, sp_num, wavelength_nm, aki, ei_ev, ek_ev, gi, gk,
               stark_w, stark_alpha, stark_shift, aki_uncertainty,
               is_resonance, stark_w_source, gamma_vdw_log
        FROM lines
        ORDER BY element, sp_num, wavelength_nm, id
        """).fetchall()

    n = len(rows)
    elements_seen: dict[str, int] = {}
    el_idx = np.zeros(n, dtype=np.int16)
    sp_num = np.zeros(n, dtype=np.int8)
    wl = np.zeros(n, dtype=np.float64)
    aki = np.zeros(n, dtype=np.float64)
    ei = np.zeros(n, dtype=np.float64)
    ek = np.zeros(n, dtype=np.float64)
    gi = np.zeros(n, dtype=np.float64)
    gk = np.zeros(n, dtype=np.float64)
    sp_index = np.zeros(n, dtype=np.int32)
    stark_w = np.zeros(n, dtype=np.float64)
    stark_alpha = np.zeros(n, dtype=np.float64)
    stark_shift = np.zeros(n, dtype=np.float64)
    aki_unc = np.zeros(n, dtype=np.float64)
    is_res = np.zeros(n, dtype=bool)
    stark_cls = np.zeros(n, dtype=np.uint8)
    vdw = np.zeros(n, dtype=np.float64)

    def _f(v: object) -> float:
        return 0.0 if v is None else float(v)

    for i, r in enumerate(rows):
        el = str(r["element"])
        if el not in elements_seen:
            elements_seen[el] = len(elements_seen)
        el_idx[i] = elements_seen[el]
        sp = int(r["sp_num"])
        sp_num[i] = sp
        wl[i] = _f(r["wavelength_nm"])
        aki[i] = _f(r["aki"])
        ei[i] = _f(r["ei_ev"])
        ek[i] = _f(r["ek_ev"])
        gi[i] = _f(r["gi"])
        gk[i] = _f(r["gk"])
        sp_index[i] = species_idx.get((el, sp), -1)
        stark_w[i] = _f(r["stark_w"])
        stark_alpha[i] = _f(r["stark_alpha"])
        stark_shift[i] = _f(r["stark_shift"])
        aki_unc[i] = _f(r["aki_uncertainty"])
        is_res[i] = bool(r["is_resonance"]) if r["is_resonance"] is not None else False
        stark_cls[i] = _classify_stark_source(r["stark_w_source"])
        vdw[i] = _f(r["gamma_vdw_log"])

    element_symbols = tuple(sorted(elements_seen, key=lambda e: elements_seen[e]))

    return {
        "line_element_index": el_idx,
        "line_sp_num": sp_num,
        "line_wavelength_nm": wl,
        "line_A_ki": aki,
        "line_E_i_ev": ei,
        "line_E_k_ev": ek,
        "line_g_i": gi,
        "line_g_k": gk,
        "line_species_index": sp_index,
        "line_stark_w": stark_w,
        "line_stark_alpha": stark_alpha,
        "line_stark_shift": stark_shift,
        "line_aki_uncertainty": aki_unc,
        "line_is_resonance": is_res,
        "line_stark_source_class": stark_cls,
        "line_gamma_vdw_log": vdw,
        "element_symbols": element_symbols,
    }


def _scan_levels(
    conn: sqlite3.Connection,
    species: tuple[tuple[str, int], ...],
    species_idx: dict[tuple[str, int], int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read ``energy_levels`` into a padded ``(N_species, L_max)`` g/E/mask block."""
    g_rows: list[np.ndarray] = [np.empty(0) for _ in species]
    e_rows: list[np.ndarray] = [np.empty(0) for _ in species]
    cur = conn.execute(
        "SELECT element, sp_num, g_level, energy_ev FROM energy_levels "
        "ORDER BY element, sp_num, energy_ev"
    )
    buckets: dict[int, list[tuple[float, float]]] = {}
    for r in cur:
        key = (str(r["element"]), int(r["sp_num"]))
        si = species_idx.get(key)
        if si is None:
            continue
        g = r["g_level"]
        e = r["energy_ev"]
        if g is None or e is None:
            continue
        buckets.setdefault(si, []).append((float(g), float(e)))
    for si, pairs in buckets.items():
        g_rows[si] = np.array([p[0] for p in pairs], dtype=np.float64)
        e_rows[si] = np.array([p[1] for p in pairs], dtype=np.float64)
    g_pad, mask = pad_ragged(g_rows)
    e_pad, _ = pad_ragged(e_rows)
    return g_pad, e_pad, mask


def _scan_partition_stored(
    conn: sqlite3.Connection,
    species: tuple[tuple[str, int], ...],
    species_idx: dict[tuple[str, int], int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the RAW ``partition_functions`` table into ``(N_species, 5)`` coeffs.

    Species rows without a stored polynomial carry the NaN sentinel
    (:data:`POLY_NAN_SENTINEL`), matching the lax solver convention. Returns
    ``(coeffs, t_min, t_max)``; this stored poly drives the lax-solver bridge,
    NOT the forward kernel (which uses the canonical re-fit, see
    :func:`_scan_partition_canonical`).
    """
    n = len(species)
    coeffs = np.full((n, N_POLY), POLY_NAN_SENTINEL, dtype=np.float64)
    t_min = np.full(n, 2000.0, dtype=np.float64)
    t_max = np.full(n, 25000.0, dtype=np.float64)
    cur = conn.execute(
        "SELECT element, sp_num, a0, a1, a2, a3, a4, t_min, t_max FROM partition_functions"
    )
    for r in cur:
        si = species_idx.get((str(r["element"]), int(r["sp_num"])))
        if si is None:
            continue
        coeffs[si] = [
            float(r["a0"] or 0.0),
            float(r["a1"] or 0.0),
            float(r["a2"] or 0.0),
            float(r["a3"] or 0.0),
            float(r["a4"] or 0.0),
        ]
        if r["t_min"] is not None:
            t_min[si] = float(r["t_min"])
        if r["t_max"] is not None:
            t_max[si] = float(r["t_max"])
    return coeffs, t_min, t_max


def _scan_partition_canonical(
    db_path: str | os.PathLike,
    species: tuple[tuple[str, int], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bake the CANONICAL forward-kernel partition specs per species.

    Calls :meth:`cflibs.atomic.database.AtomicDatabase.partition_spec_for` —
    the single factory the reference forward path uses (``database.py:1148``).
    It prefers a direct-sum-derived ln-polynomial re-fit when energy levels
    exist (PF-3/PF-4 fix); otherwise the stored polynomial; otherwise the
    canonical scalar fallback baked as ``[ln(U), 0, 0, 0, 0]``. Baking the same
    spec here is what makes :meth:`PipelineSnapshot.to_atomic_snapshot` match
    ``AtomicDatabase.snapshot`` field-for-field (AC4).

    Returns
    -------
    coeffs : ndarray, shape (N_species, 5)
    t_min, t_max, g0 : ndarray, shape (N_species,)
    from_direct_sum : ndarray, shape (N_species,)
        1.0 where the poly was re-fit from a direct sum, else 0.0.
    """
    from cflibs.atomic.database import AtomicDatabase
    from cflibs.plasma.partition import canonical_partition_fallback as _cpf

    db = AtomicDatabase(str(db_path))
    n = len(species)
    coeffs = np.zeros((n, N_POLY), dtype=np.float64)
    t_min = np.full(n, 2000.0, dtype=np.float64)
    t_max = np.full(n, 25000.0, dtype=np.float64)
    g0 = np.ones(n, dtype=np.float64)
    fds = np.zeros(n, dtype=np.float64)

    for i, (el, sp) in enumerate(species):
        spec = db.partition_spec_for(el, sp)
        if spec is not None:
            row = [float(c) for c in spec.coefficients]
            coeffs[i, : len(row)] = row[:N_POLY]
            t_min[i] = float(spec.t_min)
            t_max[i] = float(spec.t_max)
            g0[i] = float(spec.g0)
            fds[i] = 1.0 if spec.from_direct_sum else 0.0
        else:
            u_fallback = _cpf(el, sp, db)
            coeffs[i, 0] = float(np.log(u_fallback))
    return coeffs, t_min, t_max, g0, fds


def _scan_species_physics(
    conn: sqlite3.Connection,
    species: tuple[tuple[str, int], ...],
    species_idx: dict[tuple[str, int], int],
) -> np.ndarray:
    """Read ``species_physics`` into ``(N_species, 2)`` = [ip_ev, atomic_mass]."""
    out = np.zeros((len(species), 2), dtype=np.float64)
    cur = conn.execute("SELECT element, sp_num, ip_ev, atomic_mass FROM species_physics")
    for r in cur:
        si = species_idx.get((str(r["element"]), int(r["sp_num"])))
        if si is None:
            continue
        out[si, 0] = float(r["ip_ev"]) if r["ip_ev"] is not None else 0.0
        out[si, 1] = float(r["atomic_mass"]) if r["atomic_mass"] is not None else 0.0
    return out


def _scan_doublets(
    lines: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive candidate doublet pairs (same species + shared upper level).

    For two lines (k -> i1), (k -> i2) sharing the upper level ``k`` of the
    same species, the optically-thin emission ratio and density coupling are
    (``physics/self_absorption.py:221-230``)::

        r_thin = (g_k1 A_1 / lambda_1) / (g_k2 A_2 / lambda_2)
        rho    = (g_k1 A_1 lambda_1**3) / (g_k2 A_2 lambda_2**3)

    Because the two lines share the upper level, ``g_k1 == g_k2``, so the
    weights cancel; we keep the general form for robustness. Pairs are emitted
    with the stronger line (larger ``g_k A / lambda``) first so ``rho >= 1``.

    Returns
    -------
    pairs : ndarray, shape (P, 2), int32
        Global line indices of each pair (idx1, idx2).
    rho : ndarray, shape (P,)
    r_thin : ndarray, shape (P,)
    """
    sp_idx = np.asarray(lines["line_species_index"])
    ek = np.asarray(lines["line_E_k_ev"])
    gk = np.asarray(lines["line_g_k"])
    aki = np.asarray(lines["line_A_ki"])
    wl = np.asarray(lines["line_wavelength_nm"])

    # Group line indices by (species, rounded upper-level energy). Rounding to
    # 1e-4 eV tolerates float noise while keeping genuinely distinct levels
    # apart.
    groups: dict[tuple[int, int], list[int]] = {}
    for i in range(sp_idx.shape[0]):
        if sp_idx[i] < 0 or wl[i] <= 0.0 or aki[i] <= 0.0:
            continue
        key = (int(sp_idx[i]), int(round(ek[i] * 1.0e4)))
        groups.setdefault(key, []).append(i)

    pairs: list[tuple[int, int]] = []
    rho_list: list[float] = []
    rthin_list: list[float] = []
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        # Emit all unordered pairs within the shared-upper-level group.
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i1, i2 = idxs[a], idxs[b]
                s1 = gk[i1] * aki[i1] / wl[i1]
                s2 = gk[i2] * aki[i2] / wl[i2]
                if s2 <= 0.0:
                    continue
                # Order stronger line first.
                if s2 > s1:
                    i1, i2 = i2, i1
                    s1, s2 = s2, s1
                r_thin = s1 / s2
                rho = (gk[i1] * aki[i1] * wl[i1] ** 3) / (gk[i2] * aki[i2] * wl[i2] ** 3)
                pairs.append((i1, i2))
                rho_list.append(float(rho))
                rthin_list.append(float(r_thin))

    if not pairs:
        return (
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )
    return (
        np.asarray(pairs, dtype=np.int32),
        np.asarray(rho_list, dtype=np.float64),
        np.asarray(rthin_list, dtype=np.float64),
    )


def _scan_oxide(species: tuple[tuple[str, int], ...], oxide_map: dict[str, float]) -> np.ndarray:
    """Build the per-species oxide stoichiometry vector (O atoms per cation).

    Elements absent from ``oxide_map`` fall back to 1.0 (treated as
    elemental/metal), matching ``apply_oxide_mode`` semantics
    (``physics/closure.py:60-61``).
    """
    out = np.ones(len(species), dtype=np.float64)
    for i, (el, _sp) in enumerate(species):
        out[i] = float(oxide_map.get(el, 1.0))
    return out


def _scan_canonical_fallbacks(
    species: tuple[tuple[str, int], ...],
    polys: tuple[np.ndarray, np.ndarray, np.ndarray],
    levels: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Build eager canonical scalar partition fallbacks (N_species, 2).

    Column 0 is the stage-I default (:data:`FALLBACK_U_I`), column 1 the
    stage-II default (:data:`FALLBACK_U_II`). Where a species has neither a
    polynomial (non-NaN coeffs) nor any energy levels, the closed-shell exact
    refinement is applied via
    :func:`cflibs.inversion.solve.iterative.canonical_partition_fallback`.

    Spec §2: these fallbacks are evaluated **eagerly** at build (one-time
    process cost) rather than lazily per solve, restoring simple semantics.
    """
    from cflibs.inversion.solve.iterative import canonical_partition_fallback

    coeffs = polys[0]
    level_mask = levels[2]
    n = len(species)
    out = np.zeros((n, 2), dtype=np.float64)
    out[:, 0] = FALLBACK_U_I
    out[:, 1] = FALLBACK_U_II

    for i, (el, sp) in enumerate(species):
        has_poly = not np.any(np.isnan(coeffs[i]))
        has_levels = bool(np.any(level_mask[i]))
        if has_poly or has_levels:
            continue
        # No data: refine the scalar fallback (closed-shell ions get exact U).
        try:
            out[i, 0 if sp == 1 else 1] = float(
                canonical_partition_fallback(el, sp, None, warn=False)
            )
        except Exception:  # pragma: no cover - defensive; keep the default
            pass
    return out


# ---------------------------------------------------------------------------
# .npz cache I/O.
# ---------------------------------------------------------------------------


def save_npz_cache(path: str | os.PathLike, payload: dict[str, np.ndarray]) -> None:
    """Write the snapshot's array payload to a byte-stable ``.npz`` archive.

    Uses uncompressed ``np.savez`` with keys written in sorted order so two
    builds from the same DB produce byte-identical files (AC3).
    """
    ordered = {k: payload[k] for k in sorted(payload)}
    np.savez(path, **ordered)


def load_npz_cache(path: str | os.PathLike) -> dict[str, np.ndarray]:
    """Load a snapshot ``.npz`` cache into a plain dict of arrays."""
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}
