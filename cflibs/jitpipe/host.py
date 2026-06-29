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

import functools
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
    """Read the EMITTING ``lines`` into the §2 line block (sorted, stable).

    Mirrors the reference forward path's emitting-line filter in
    ``AtomicDatabase._build_transitions_query`` (``get_transitions``): only
    rows with a usable spontaneous-emission rate and upper level
    (``aki > 0`` and non-NULL ``ek_ev``/``gk``) belong in the CF-LIBS line
    pool. The M5 complete-DB ingest added ~74k observation-only transitions
    with NULL ``aki`` (and sometimes NULL ``ek_ev``/``gk``) that carry
    position+intensity but cannot emit; without this cut the snapshot line
    block would carry tens of thousands of non-emitting lines that the
    reference snapshot/forward/detect path never sees -- a line-set parity
    break (J0 AC4) and a polluted detection/identification catalog.
    """
    from cflibs.atomic.database import EMITTING_LINE_PREDICATE

    rows = conn.execute(f"""
        SELECT element, sp_num, wavelength_nm, aki, ei_ev, ek_ev, gi, gk,
               stark_w, stark_alpha, stark_shift, aki_uncertainty,
               is_resonance, stark_w_source, gamma_vdw_log
        FROM lines
        WHERE {EMITTING_LINE_PREDICATE.format(p="")}
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


# ---------------------------------------------------------------------------
# J8 host glue — per-spectrum gather/scatter (ADR-0004 §5.1; J8 plan §2).
#
# These impure helpers bridge between the reference front-end (response
# correction, detect/identify/calibrate producing a ``LineObservation`` list)
# and the device-pure solve kernels (``cflibs.jitpipe.solve``). They live here
# (the host carve-out) because they touch reference SQLite-backed objects and
# scipy; the jit-traced ``run_one`` core never sees a DB connection.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservationBlock:
    """Padded ``(E, N_max)`` Boltzmann block + element order (J8 plan §2.C).

    The host gather seam between the front-end ``LineObservation`` list and the
    fixed-shape solve kernels (``LaxKernelInputs``). Mirrors the reference
    ``_build_padded_arrays_from_obs`` output layout exactly so the kernel sees
    byte-identical data to ``_run_lax_while_loop``.

    Attributes
    ----------
    elements : list[str]
        Element symbols, length ``E``, in observation-insertion order.
    x, y, w : ndarray, shape (E, N_max)
        Upper-level energies (eV), Boltzmann ``y`` ordinates and
        inverse-variance weights. Zero in padding.
    stage : ndarray, shape (E, N_max), int32
        Ionisation stage per line (1 neutral / 2 ionic); 1 in padding.
    mask : ndarray of bool, shape (E, N_max)
        ``True`` where a real observation sits.
    n_observations : int
        Total number of valid observations (``mask.sum()``); 0 signals the
        failure-policy path (J8 plan §4).
    """

    elements: list[str]
    x: np.ndarray | None
    y: np.ndarray | None
    w: np.ndarray | None
    stage: np.ndarray | None
    mask: np.ndarray | None
    n_observations: int


def build_observation_block(observations, *, weight_cap: float = 0.0) -> ObservationBlock:
    """Gather a front-end ``LineObservation`` list into a padded ``(E, N_max)`` block.

    Reuses the REAL reference ``_build_padded_arrays_from_obs`` (the parity
    oracle for the line-block layout) so the gather is byte-identical to the
    reference solve path. Groups observations by element preserving insertion
    order (``defaultdict(list)`` semantics).

    Parameters
    ----------
    observations : list[LineObservation]
        Selected line observations from the front-end (detect/identify/select).
    weight_cap : float, optional
        Per-element Boltzmann weight dynamic-range cap (``boltzmann_weight_cap``);
        0 disables (default).

    Returns
    -------
    ObservationBlock
        Padded arrays + element order. ``n_observations == 0`` (and ``x is
        None``) when there are no usable observations — the failure-policy path.
    """
    from collections import defaultdict

    from cflibs.inversion.solve.iterative import _build_padded_arrays_from_obs

    obs_by_element: dict[str, list] = defaultdict(list)
    for obs in observations:
        obs_by_element[obs.element].append(obs)

    elements, x, y, w, stage, mask = _build_padded_arrays_from_obs(
        dict(obs_by_element), weight_cap=weight_cap
    )
    if x is None or mask is None:
        return ObservationBlock(
            elements=list(elements),
            x=None,
            y=None,
            w=None,
            stage=None,
            mask=None,
            n_observations=0,
        )
    n_obs = int(np.asarray(mask).sum())
    return ObservationBlock(
        elements=list(elements), x=x, y=y, w=w, stage=stage, mask=mask, n_observations=n_obs
    )


def lax_inputs_from_observation_block(snapshot, block: ObservationBlock):
    """Assemble device ``LaxKernelInputs`` from a snapshot + observation block.

    Bridges the per-bucket candidate-set assembly seam (J8 plan §2.E): gathers
    the per-element atomic block via
    :meth:`cflibs.jitpipe.snapshot.PipelineSnapshot.to_lax_snapshot` (no DB at
    solve time — the snapshot is the baked superset) and feeds it plus the
    padded obs block to :meth:`LaxKernelInputs.from_snapshot`.

    Parameters
    ----------
    snapshot : PipelineSnapshot
        The host-built atomic snapshot.
    block : ObservationBlock
        Padded observation block from :func:`build_observation_block`.

    Returns
    -------
    LaxKernelInputs
        Device-ready padded inputs for the solve kernels.
    """
    from cflibs.jitpipe.solve import LaxKernelInputs

    lax_snap = snapshot.to_lax_snapshot(block.elements)
    return LaxKernelInputs.from_snapshot(
        lax_snap, block.x, block.y, block.w, block.stage, block.mask
    )


def oxide_factors_for_elements(snapshot, elements: list[str]) -> np.ndarray:
    """Per-element oxide stoichiometry vector (O atoms per cation) for ``oxide`` closure.

    Mirrors the reference ``default_oxide_stoichiometry`` mapping: the host
    closure path weights each element's relative concentration by its
    stage-I oxide stoichiometry. Falls back to 1.0 for elements absent from the
    snapshot species axis (treated as elemental/metal), matching
    ``apply_oxide_mode`` semantics.

    Parameters
    ----------
    snapshot : PipelineSnapshot
        Atomic snapshot carrying ``oxide_stoichiometry`` + ``species``.
    elements : list[str]
        Element order (length ``E``) the factors must align to.

    Returns
    -------
    ndarray, shape (E,)
        Oxide stoichiometry per element.
    """
    from cflibs.inversion.physics.closure import default_oxide_stoichiometry

    stoich = default_oxide_stoichiometry(list(elements))
    return np.asarray([float(stoich.get(el, 1.0)) for el in elements], dtype=np.float64)


def cflibs_result_from_loopstate(
    final_state,
    elements: list[str],
    *,
    iterations: int | None = None,
    converged: bool | None = None,
):
    """Reconstitute a reference :class:`CFLIBSResult` from a solve ``LoopState`` (J8 plan §2.F).

    Maps the E-indexed device arrays back to element-keyed dicts via
    ``elements``, reusing the reference ``CFLIBSResult`` type
    (``iterative.py:81``) so downstream consumers and the parity adapters see
    the identical dataclass.

    Parameters
    ----------
    final_state : LoopState
        Final (frozen) solve state from :func:`cflibs.jitpipe.solve.scan_solve`.
    elements : list[str]
        Element order (length ``E``) of the solve.
    iterations, converged : optional
        Overrides; default reads from the loop state.

    Returns
    -------
    CFLIBSResult
    """
    from cflibs.inversion.solve.iterative import CFLIBSResult

    conc_arr = np.asarray(final_state.concentrations)
    concentrations = {el: float(conc_arr[i]) for i, el in enumerate(elements)}
    T_K = float(final_state.T_K)
    n_e = float(final_state.n_e_cm3)
    r2 = float(final_state.r_squared)
    degenerate = bool(final_state.boltzmann_degenerate)
    iters = int(final_state.i) if iterations is None else int(iterations)
    conv = bool(final_state.converged) if converged is None else bool(converged)
    if degenerate:
        conv = False
    return CFLIBSResult(
        temperature_K=T_K,
        temperature_uncertainty_K=0.0,
        electron_density_cm3=n_e,
        concentrations=concentrations,
        concentration_uncertainties={},
        iterations=iters,
        converged=conv,
        temperature_corona_K=None,
        quality_metrics={
            "boltzmann_r_squared": r2,
            "boltzmann_degenerate": float(degenerate),
        },
        electron_density_uncertainty_cm3=0.0,
        boltzmann_covariance=None,
    )


def all_fn_result(elements: list[str]):
    """The all-FN :class:`CFLIBSResult` the reference produces at zero observations.

    Failure-policy parity (J8 plan §4, AC4): the reference raises
    ``ValueError`` at zero observations (``pipeline.py:872``) which the
    scoreboard scores as all false-negative. The jit pipeline must emit the
    SAME all-FN record — NaN-free concentrations (all 0.0), ``converged=False``
    — rather than crash or return NaN. Element keys are preserved (all-zero)
    so the scoreboard's presence rule scores them as missed (FN), not crashed.

    Parameters
    ----------
    elements : list[str]
        The requested element set; concentrations are 0.0 for each.

    Returns
    -------
    CFLIBSResult
    """
    from cflibs.inversion.solve.iterative import CFLIBSResult

    return CFLIBSResult(
        temperature_K=0.0,
        temperature_uncertainty_K=0.0,
        electron_density_cm3=0.0,
        concentrations={el: 0.0 for el in elements},
        concentration_uncertainties={},
        iterations=0,
        converged=False,
        temperature_corona_K=None,
        quality_metrics={"failed": 1.0, "reason": 0.0},
        electron_density_uncertainty_cm3=0.0,
        boltzmann_covariance=None,
    )


@dataclass(frozen=True)
class FrontEndResult:
    """Host front-end output: selected observations + Stark n_e + diagnostics.

    Bundles the reference front-end stages the J8 ``run_one`` composes ahead of
    the jit solve kernels (response -> calibrate -> detect -> identify ->
    select, then the Stark n_e diagnostic). The detect/identify/calibrate jit
    kernels are individually parity-tested (J2/J3/J4); this host wrapper drives
    the reference detection path so the observation set fed to the jit solve
    spine is byte-faithful to ``run_pipeline`` (the parity oracle). The
    ``n_observations == 0`` case is the failure-policy trigger (J8 plan §4).

    Attributes
    ----------
    observations : list[LineObservation]
        Selected line observations (front-end output).
    elements : list[str]
        Requested element set (the run identity).
    ne_stark_cm3 : float or None
        Stark-measured electron density (cm^-3); pins the solve when present.
    diagnostics : dict
        The reference detection/selection diagnostics dict.
    n_observations : int
        Number of selected observations; 0 -> failure path.
    """

    observations: list
    elements: list[str]
    ne_stark_cm3: float | None
    diagnostics: dict
    n_observations: int


def run_front_end(wavelength, intensity, atomic_db, pipeline) -> FrontEndResult:
    """Host front-end: response -> detect/identify/calibrate/select -> Stark n_e.

    Reproduces the front-end half of ``run_pipeline`` (``pipeline.py:806-916``)
    on the host so the jit solve spine downstream consumes byte-identical
    observations. Response correction is applied, then the shared
    ``detect_and_select_lines`` path, then the Stark n_e diagnostic — exactly
    mirroring the reference stage order. Never raises on zero observations: it
    returns ``n_observations == 0`` for the caller to interpret (J8 plan §4),
    where the reference would raise ``ValueError``.

    Parameters
    ----------
    wavelength, intensity : ndarray
        Spectrum axes (nm, intensity).
    atomic_db : AtomicDatabase
        Reference atomic database (host front-end consumes SQLite directly).
    pipeline : AnalysisPipelineConfig
        Resolved pipeline config carrying every front-end knob.

    Returns
    -------
    FrontEndResult
    """
    from cflibs.inversion.pipeline import detect_and_select_lines

    wl = np.asarray(wavelength, dtype=float)
    inten = np.asarray(intensity, dtype=float)
    if pipeline.response_curve:
        inten = inten * response_multiplier(wl, pipeline.response_curve)

    observations, diagnostics = detect_and_select_lines(
        wl,
        inten,
        atomic_db,
        pipeline.elements,
        min_relative_intensity=pipeline.min_relative_intensity,
        top_k_per_element=pipeline.top_k_per_element,
        resolving_power=pipeline.resolving_power,
        wavelength_tolerance_nm=pipeline.wavelength_tolerance_nm,
        min_peak_height=pipeline.min_peak_height,
        peak_width_nm=pipeline.peak_width_nm,
        apply_self_absorption=pipeline.apply_self_absorption,
        exclude_resonance=pipeline.exclude_resonance,
        min_snr=pipeline.min_snr,
        min_energy_spread_ev=pipeline.min_energy_spread_ev,
        min_lines_per_element=pipeline.min_lines_per_element,
        isolation_wavelength_nm=pipeline.isolation_wavelength_nm,
        max_lines_per_element=pipeline.max_lines_per_element,
        wavelength_calibration=pipeline.wavelength_calibration,
        shift_coherence_veto=pipeline.shift_coherence_veto,
        residual_shift_scan_nm=pipeline.residual_shift_scan_nm,
        global_shift_scan_nm=pipeline.global_shift_scan_nm,
        affine_coverage_gate=pipeline.affine_coverage_gate,
        line_residual_gate=pipeline.line_residual_gate,
        # Forward the SAME calibration/selection knobs the reference
        # ``run_pipeline`` passes (pipeline.py): without these the delegated
        # front-end silently fell back to ``detect_and_select_lines`` defaults
        # and was NOT byte-faithful to the reference (M1 parity break).
        # ``ransac_early_exit`` in particular is True in the raw preset, so
        # omitting it ran a different RANSAC wavelength calibration -> a
        # different matched-line set -> diverging concentrations on real
        # ChemCam spectra.
        calib_pool_cache=pipeline.calib_pool_cache,
        hough_calib_seed=pipeline.hough_calib_seed,
        ransac_early_exit=(pipeline.ransac_early_exit or None),
        grade_aware_selection=pipeline.grade_aware_selection,
        target_sigma_t=pipeline.target_sigma_t,
        plasma_temperature_K=pipeline.plasma_temperature_K,
        reliability_ranked_selection=pipeline.reliability_ranked_selection,
        matrix_isolation_element=pipeline.matrix_isolation_element,
        matrix_isolation_n_fwhm=pipeline.matrix_isolation_n_fwhm,
        matrix_isolation_contamination_ratio=pipeline.matrix_isolation_contamination_ratio,
        detection_overrides=pipeline.detection_overrides,
        return_diagnostics=True,
    )

    ne_stark: float | None = None
    if pipeline.stark_ne and observations:
        from cflibs.inversion.physics.stark_ne import measure_stark_ne

        try:
            stark_result = measure_stark_ne(
                wl, inten, observations, atomic_db, resolving_power=pipeline.resolving_power
            )
        except Exception:  # pragma: no cover - defensive
            stark_result = None
        if stark_result is not None and stark_result.usable:
            ne_stark = float(stark_result.ne_median_cm3)

    return FrontEndResult(
        observations=list(observations),
        elements=list(pipeline.elements),
        ne_stark_cm3=ne_stark,
        diagnostics=dict(diagnostics),
        n_observations=len(observations),
    )


def response_multiplier(wavelength: np.ndarray, response_curve_path: str | None) -> np.ndarray:
    """Per-channel spectral-response multiplier array (J8 plan §2.A).

    The reference ``run_pipeline`` divides the measured intensity by the
    relative detection efficiency ``E(lambda)`` (``pipeline.py:816``). The only
    thing that crosses to device is the per-channel multiplier; this host helper
    computes it. Identity (all-ones) when no curve is configured — ChemCam CCS
    spectra arrive response-corrected upstream and must not be corrected twice.

    Parameters
    ----------
    wavelength : ndarray
        Wavelength axis, nm.
    response_curve_path : str or None
        Path to a response curve file, or None for identity.

    Returns
    -------
    ndarray
        Per-channel multiplier; ``intensity_corrected = intensity * mult`` is
        equivalent to the reference ``correction.apply`` (which divides by
        ``E(lambda)``).
    """
    wl = np.asarray(wavelength, dtype=float)
    if not response_curve_path:
        return np.ones_like(wl)
    from cflibs.inversion.preprocess.response_correction import SpectralResponseCorrection

    correction = SpectralResponseCorrection.from_file(response_curve_path)
    ones = np.ones_like(wl)
    corrected = correction.apply(wl, ones)
    return np.asarray(corrected, dtype=float)


# ---------------------------------------------------------------------------
# J9/M2 ON-DEVICE FRONT-END host glue (Wave 3; J8 plan §2.B / §2.C).
#
# These helpers assemble the J3 :class:`~cflibs.jitpipe.identify.FrontEndSnapshot`
# (the padded comb arrays + the padded peak arrays) so that ``run_one`` can run
# the detection / identification **gate stack on-device** (the JIT kernels in
# ``cflibs.jitpipe.{preprocess,detect,calibrate,identify}``) instead of
# delegating byte-for-byte to the reference ``detect_and_select_lines``.
#
# Host vs device split (ADR-0004 §5.1; the host gathers MAY be dynamic-shaped
# because they run BEFORE the trace):
#   * HOST (dynamic): catalog SQL + the gA-Boltzmann comb ranking
#     (``_rank_transitions_by_strength`` / ``_select_comb_transitions``); the
#     padded scipy ``find_peaks`` -> ``(P_max,)`` peak arrays; the obs -> line
#     block gather. These touch the reference DB / scipy and stay impure.
#   * DEVICE (fixed-shape JIT kernels): the comb shift-scan / shift selection /
#     shift-coherence veto / observation build (J3 ``identify.py``), trapezoid
#     intensity extraction, the J1 detect kernel for the detection-path peaks.
#
# Wave-3b additions (this change): the kdet pre-filter (coherence branch) and
# the post-detection ``LineSelector`` ALSO run on-device now:
#   * kdet (coherence branch) -> J3 ``kdet_keep_mask`` over a full-transition-set
#     snapshot (:func:`_ondevice_kdet_filter`); the density-score branch
#     (``shift_coherence_veto=False``: Rust + density) is not ported and falls
#     back to the reference;
#   * ``LineSelector`` SNR / isolation / composite-score -> a fixed-shape device
#     kernel (:func:`_line_selector_scores`) + a host gather applying the gate +
#     per-element top-K cap in the reference order (:func:`_ondevice_line_select`).
#
# Wave-3c / J9 addition (this change): the segmented WAVELENGTH CALIBRATION now
# runs ON-DEVICE in :func:`run_front_end_ondevice` via the kernel-backed driver
# :func:`_ondevice_calibrate_segmented` (the reference segmented-driver structure
# — CCD seams, always-computed global fit, per-segment trust/coverage/disagreement
# gates, global-offset + neighbour fallbacks, seam-monotonicity restore, revert
# gates — with every robust RANSAC core routed through the J2
# :func:`cflibs.jitpipe.calibrate.calibrate_axis_kernel`). Each segment re-detects
# its OWN peaks + line pool (the reference ``_fit_one_segment`` behaviour); feeding
# the kernel a per-segment candidate set (not the global peaks masked to a window)
# plus an UNCONDITIONAL dense-hull coverage tiebreak on the winning slope model
# closes the J2 §7 R8 model-class flip on the real ChemCam BHVO-2 confounder
# (obs Jaccard 1.0 on raw + geological; the ye6t 877 nm Al doublet is preserved).
# It is ~8x faster than the old monolithic ``calibrate_segmented_kernel`` (BHVO-2:
# warm ~4 s vs reference ~1.4 s) via jit-compiling+caching the per-segment kernel
# and k_pair 48->16. The byte-faithful reference (:func:`_ld_calibrate`) is kept
# as the parity-test cross-check.
#
# Stages that DO NOT yet reach byte-parity as on-device kernels and therefore
# stay reference-delegated (reported honestly in ``impl_completeness`` /
# ``remaining_todo``):
#   * the Stark n_e diagnostic (``measure_stark_ne_jit`` is parity-tested in
#     isolation, but the per-candidate DB multiplet-blend gate + the
#     ``break``-after-``max_lines``-SUCCESSES sequencing are delicate, and the
#     ``ne_median`` directly pins the production solve — kept delegated to protect
#     the n_e <= 10 % M1 tolerance).
# The on-device gate stack consumes the SAME reference-computed inputs for the
# remaining delegated stage, so the line-key set it produces matches the reference
# ``detect_and_select_lines`` exactly (obs Jaccard 1.0 on synthetic + real BHVO-2,
# raw + geological presets).
# ---------------------------------------------------------------------------


def build_front_end_snapshot(
    peaks,
    comb_transitions_by_element,
    *,
    e_max: int,
    k_comb: int,
    p_max: int,
):
    """Assemble the J3 :class:`FrontEndSnapshot` from peaks + comb transitions.

    The J8-plan §2.B host gather: pads the per-element comb arrays to
    ``(E_max, K_comb)`` from the *reference catalog order* (the gA-Boltzmann
    ranking the host already applied via ``_select_comb_transitions``) and the
    detected peak arrays to ``(P_max,)``. Mirrors the J3 parity-test
    ``build_snapshot`` exactly (``tests/jitpipe/test_parity_j3.py``) so the
    on-device kernel sees byte-identical inputs to its parity oracle.

    Parameters
    ----------
    peaks : list[tuple[int, float]]
        Reference detected peaks ``(sample_index, wavelength_nm)`` on the
        calibrated axis (insertion order = ascending position).
    comb_transitions_by_element : dict[str, list[Transition]]
        Per-element comb-ranked transitions (the top-K gA-Boltzmann subset),
        dict insertion order == the reference element-loop order.
    e_max, k_comb, p_max : int
        Padded shape bounds (static per compile bucket).

    Returns
    -------
    tuple
        ``(FrontEndSnapshot, elements)`` — the padded pytree + the element-slot
        order (length ``E``, the ``comb_transitions_by_element`` key order).
    """
    import jax.numpy as jnp

    from cflibs.jitpipe.identify import FrontEndSnapshot

    elements = list(comb_transitions_by_element.keys())
    n_e = len(elements)
    n_p = len(peaks)
    if n_e > e_max:
        raise ValueError(f"e_max={e_max} too small for {n_e} elements")
    if n_p > p_max:
        raise ValueError(f"p_max={p_max} too small for {n_p} peaks")

    peak_wl = np.zeros(p_max, dtype=np.float64)
    peak_idx = np.full(p_max, -1, dtype=np.int64)
    peak_mask = np.zeros(p_max, dtype=bool)
    for j, (idx, wl) in enumerate(peaks):
        peak_wl[j] = float(wl)
        peak_idx[j] = int(idx)
        peak_mask[j] = True

    comb_wl = np.zeros((e_max, k_comb), dtype=np.float64)
    comb_ek = np.zeros((e_max, k_comb), dtype=np.float64)
    comb_gk = np.ones((e_max, k_comb), dtype=np.float64)
    comb_aki = np.ones((e_max, k_comb), dtype=np.float64)
    comb_ei = np.zeros((e_max, k_comb), dtype=np.float64)
    comb_stage = np.zeros((e_max, k_comb), dtype=np.int64)
    comb_res = np.full((e_max, k_comb), -1, dtype=np.int64)
    comb_mask = np.zeros((e_max, k_comb), dtype=bool)
    element_mask = np.zeros(e_max, dtype=bool)
    full_n = np.zeros(e_max, dtype=np.int64)

    for e, el in enumerate(elements):
        trans = comb_transitions_by_element[el]
        element_mask[e] = True
        full_n[e] = len(trans)
        for k, t in enumerate(trans[:k_comb]):
            comb_wl[e, k] = float(t.wavelength_nm)
            comb_ek[e, k] = float(t.E_k_ev)
            comb_gk[e, k] = float(t.g_k)
            comb_aki[e, k] = float(t.A_ki)
            comb_ei[e, k] = float(t.E_i_ev)
            comb_stage[e, k] = int(t.ionization_stage)
            comb_res[e, k] = -1 if t.is_resonance is None else int(bool(t.is_resonance))
            comb_mask[e, k] = True

    snap = FrontEndSnapshot(
        peak_wavelength_nm=jnp.asarray(peak_wl),
        peak_index=jnp.asarray(peak_idx),
        peak_mask=jnp.asarray(peak_mask),
        comb_wavelength_nm=jnp.asarray(comb_wl),
        comb_E_k_ev=jnp.asarray(comb_ek),
        comb_g_k=jnp.asarray(comb_gk),
        comb_A_ki=jnp.asarray(comb_aki),
        comb_E_i_ev=jnp.asarray(comb_ei),
        comb_stage=jnp.asarray(comb_stage),
        comb_is_resonance=jnp.asarray(comb_res),
        comb_mask=jnp.asarray(comb_mask),
        element_mask=jnp.asarray(element_mask),
        full_n_lines=jnp.asarray(full_n),
    )
    return snap, elements


def _estimate_wl_step(wavelength: np.ndarray) -> float:
    """Median wavelength step (mirrors reference ``_estimate_wl_step``)."""
    wl = np.asarray(wavelength, dtype=float)
    if wl.size < 2:
        return 0.0
    diffs = np.diff(wl)
    diffs = diffs[np.isfinite(diffs)]
    return float(np.median(diffs)) if diffs.size else 0.0


def _detect_peaks_ondevice(wavelength, intensity, min_peak_height, peak_width_nm):
    """Run the J1 ``detect_peaks_detection`` kernel -> reference-shaped peak list.

    Reproduces the reference ``_find_peaks`` (``line_detection.py:2424``,
    ``intensity/max`` normalise, height=threshold, prom=threshold/2, distance
    from ``peak_width_nm``) on-device via the parity-tested fixed-shape kernel,
    then gathers the accepted peak slots back into the host
    ``list[(sample_index, calibrated_wavelength_nm)]`` the rest of the front-end
    consumes. The wavelength axis passed in is the (calibrated) detection axis,
    so ``wavelength[idx]`` matches the reference gather.

    Returns ``(peaks, count, truncated)``.
    """
    import jax.numpy as jnp

    from cflibs.jitpipe.detect import detect_peaks_detection

    wl = np.asarray(wavelength, dtype=float)
    inten = np.asarray(intensity, dtype=float)
    wl_step = _estimate_wl_step(wl)
    distance_px = max(int(float(peak_width_nm) / max(wl_step, 1e-9)), 1)

    res = detect_peaks_detection(
        jnp.asarray(inten),
        min_peak_height=float(min_peak_height),
        distance_px=int(distance_px),
    )
    indices = np.asarray(res.indices)
    mask = np.asarray(res.mask)
    truncated = bool(res.truncated)
    peaks = [
        (int(indices[j]), float(wl[int(indices[j])])) for j in range(indices.shape[0]) if mask[j]
    ]
    return peaks, len(peaks), truncated


def _next_pow2(n: int) -> int:
    """Smallest power of two >= ``n`` (>= 1) — the front-end snapshot bucketer."""
    n = max(int(n), 1)
    p = 1
    while p < n:
        p <<= 1
    return p


def _ld_calibrate(wl, inten, atomic_db, elements, pipeline):
    """Reference segmented wavelength calibration (delegated; see module note)."""
    from cflibs.inversion.preprocess.wavelength_calibration import (
        calibrate_wavelength_axis_segmented,
    )

    return calibrate_wavelength_axis_segmented(
        wavelength=np.asarray(wl, dtype=float),
        intensity=np.asarray(inten, dtype=float),
        atomic_db=atomic_db,
        elements=elements,
        affine_coverage_gate=pipeline.affine_coverage_gate,
    )


# Reference robust-fit defaults of ``calibrate_wavelength_axis`` /
# ``calibrate_wavelength_axis_segmented`` (the production segmented driver
# signature). The host front-end (peak detection + line-pool SQL/ranking) must
# stay byte-faithful to the reference so the on-device kernel sees the SAME
# (peaks, lines) the reference robust core does (ADR-0001 host/device split).
_CAL_INLIER_TOL_NM = 0.08
_CAL_PAIR_WINDOW_NM = 2.0
_CAL_MAX_LINES_PER_ELEMENT = 60
_CAL_MIN_AKI_GK = 3e3
_CAL_REF_T_K = 10000.0
_CAL_THRESHOLD_FACTOR = 4.0
#: ``h_affine`` for the on-device stratified affine sampler. The reference uses
#: 600 random RANSAC draws; the kernel replaces them with H deterministic
#: stratified samples (J2 §3). 64 already saturates the affine search on the
#: production bands — the corrected axis is bit-identical (max|Δ| 0.00025 nm vs
#: the reference) at 64/128/256, since the dense-hull tiebreak, not the sample
#: count, decides the model class. Fewer hypotheses = a smaller H-chunked
#: residual matrix and a faster trace.
_CAL_H_AFFINE = 64
#: Banded fan-out per peak for the segmented-calib kernel. The J2 default is 48,
#: but the measured ±2.0 nm window fan-out is <=16 on the production ChemCam /
#: BHVO bands, so 16 is bit-identical to 48 (global fit: model/n_inliers/BIC
#: unchanged) while shrinking the (P_max*K_pair) candidate axis ~3x — the
#: dominant cost of the exhaustive shift search + H-chunked residual matrix.
#: This is the speed lever that brings the global fit from ~15 s to ~2 s.
_CAL_K_PAIR = 16
#: Padded segment-slot count for the legacy monolithic ``calibrate_segmented_kernel``
#: (retained importable for the J2 parity tests; the production front-end now uses
#: the per-segment kernel-backed driver below). BHVO-2 has 2 seams / 3 segments.
_CAL_SEG_MAX = 8
#: Reference ``calibrate_wavelength_axis_segmented`` per-segment gate defaults
#: (the kernel-backed driver mirrors them exactly so the gate sequencing matches).
_CAL_MIN_SEGMENT_POINTS = 16
_CAL_SPARSE_SEGMENT_POINTS = 400
_CAL_SEGMENT_MIN_INLIERS = 10
_CAL_SEGMENT_MAX_RMSE_NM = 0.06
_CAL_SEGMENT_MAX_GLOBAL_DISAGREEMENT_NM = 0.5


def _next_pow2_min(n: int, floor: int) -> int:
    """Smallest power of two ``>= max(n, floor)`` — segmented-calib bucketer."""
    return _next_pow2(max(int(n), int(floor)))


@functools.lru_cache(maxsize=64)
def _jitted_calibrate_axis_kernel(
    apply_quality_gate: bool,
    candidate_models: tuple,
    k_pair: int,
    h_affine: int,
    exact_dedupe_score: bool,
    seed: int,
):
    """Return a ``jax.jit``-compiled :func:`calibrate_axis_kernel` closure.

    ``calibrate_axis_kernel`` is not jitted at module scope (the J2 parity tests
    wrap it themselves), so calling it eagerly per segment dispatches op-by-op
    through the ``lax.map``/``lax.scan`` over thousands of candidate slots —
    ~45 s/spectrum. Compiling once per static signature (and caching the compiled
    closure across spectra; the array shapes are the per-call ``jax.jit`` cache
    key) drops the warm per-spectrum cost by ~30x. All non-array arguments are
    baked in as Python statics so the trace specialises on them.
    """
    import jax

    from cflibs.jitpipe import calibrate as _C

    return jax.jit(
        functools.partial(
            _C.calibrate_axis_kernel,
            inlier_tolerance_nm=_CAL_INLIER_TOL_NM,
            max_pair_window_nm=_CAL_PAIR_WINDOW_NM,
            apply_quality_gate=apply_quality_gate,
            candidate_models=candidate_models,
            k_pair=k_pair,
            h_affine=h_affine,
            exact_dedupe_score=exact_dedupe_score,
            seed=seed,
        )
    )


def _calibrate_axis_kernel_backed(
    wavelength,
    intensity,
    atomic_db,
    elements,
    *,
    candidate_models,
    apply_quality_gate,
    random_seed,
    h_affine=_CAL_H_AFFINE,
):
    """Kernel-backed single-axis calibrator — a drop-in for ``calibrate_wavelength_axis``.

    Runs the reference front-end (``detect_peaks_auto`` + ``_build_reference_line_pool``
    with the production defaults; ADR-0001 host split: scipy peak detection +
    catalog SQL + the gA-Boltzmann ranking stay host-side) on EXACTLY the passed
    axis/intensity, pads to a per-call power-of-two bucket, and routes the robust
    RANSAC core through the parity-tested fixed-shape JIT kernel
    :func:`cflibs.jitpipe.calibrate.calibrate_axis_kernel`. Returns a real
    :class:`WavelengthCalibrationResult` so it composes inside the reference
    segmented driver helpers (gates / fallbacks / monotonicity restore /
    assembly).

    Critically, this re-detects peaks and re-builds the line pool on the GIVEN
    slice (the same thing the reference ``calibrate_wavelength_axis`` does when
    called per segment by ``_fit_one_segment``). Feeding the kernel a per-segment
    candidate set — rather than the GLOBAL peaks masked to a segment window — is
    what closes the J2 §7 R8 model-class flip: the affine/shift decision is made
    on the same (peaks, lines) pairs the reference scores, so the BIC-selected
    class matches (the ye6t BHVO-2 segment-0 fit returns ``shift``, not the
    global-peak ``affine`` that shifted the axis ~0.08 nm and dropped the Al
    doublet).
    """
    import jax.numpy as jnp

    from cflibs.inversion.preprocess.preprocessing import detect_peaks_auto
    from cflibs.inversion.preprocess.wavelength_calibration import (
        WavelengthCalibrationResult,
        _build_reference_line_pool,
        _eval_model,
        _is_monotonic_on_grid,
    )

    from cflibs.jitpipe import calibrate as _C

    wl_np = np.asarray(wavelength, dtype=float)
    inten_np = np.asarray(intensity, dtype=float)
    n_w = wl_np.size

    def _fail(reason, n_peaks=0, n_cand=0):
        return WavelengthCalibrationResult(
            success=False,
            model="none",
            coefficients=(),
            corrected_wavelength=wl_np.copy(),
            bic=float("inf"),
            rmse_nm=float("inf"),
            n_inliers=0,
            n_peaks=int(n_peaks),
            n_candidates=int(n_cand),
            matched_peak_fraction=0.0,
            quality_passed=False,
            quality_reason=reason,
            details={"reason": reason},
        )

    if n_w < 4 or inten_np.size < 4:
        return _fail("spectrum_too_short")

    peaks, _baseline, _noise = detect_peaks_auto(
        wl_np, inten_np, threshold_factor=_CAL_THRESHOLD_FACTOR
    )
    if not peaks:
        return _fail("no_peaks_detected")
    peak_idx = np.asarray([p[0] for p in peaks], dtype=int)
    peak_wl = np.asarray([p[1] for p in peaks], dtype=float)
    peak_amp = np.maximum(inten_np[peak_idx], 1e-12)

    line_wl, line_strength = _build_reference_line_pool(
        atomic_db=atomic_db,
        elements=list(elements),
        wavelength_min=float(np.min(wl_np)) - _CAL_PAIR_WINDOW_NM,
        wavelength_max=float(np.max(wl_np)) + _CAL_PAIR_WINDOW_NM,
        max_lines_per_element=_CAL_MAX_LINES_PER_ELEMENT,
        min_aki_gk=_CAL_MIN_AKI_GK,
        reference_temperature_K=_CAL_REF_T_K,
    )
    if line_wl.size == 0:
        return _fail("no_reference_lines", n_peaks=peak_wl.size)
    # The kernel requires the line pool sorted ascending (host responsibility).
    order = np.argsort(line_wl)
    line_wl = line_wl[order]
    line_strength = line_strength[order]

    p_max = _next_pow2_min(peak_wl.size, 64)
    l_max = _next_pow2_min(line_wl.size, 64)
    w_max = _next_pow2_min(n_w, 64)

    def _pad(a, n, fill=0.0):
        out = np.full(n, fill, dtype=float)
        out[: a.size] = a
        return out

    _MODEL_ID = {
        "shift": _C.MODEL_SHIFT,
        "affine": _C.MODEL_AFFINE,
        "quadratic": _C.MODEL_QUADRATIC,
    }
    model_ids = tuple(_MODEL_ID[m] for m in candidate_models)

    kernel = _jitted_calibrate_axis_kernel(
        bool(apply_quality_gate), model_ids, _CAL_K_PAIR, int(h_affine), True, int(random_seed)
    )
    res = kernel(
        jnp.asarray(_pad(peak_wl, p_max)),
        jnp.asarray(_pad(peak_amp, p_max)),
        jnp.asarray(np.r_[np.ones(peak_wl.size, bool), np.zeros(p_max - peak_wl.size, bool)]),
        jnp.asarray(_pad(line_wl, l_max, fill=1e9)),
        jnp.asarray(_pad(line_strength, l_max)),
        jnp.asarray(np.r_[np.ones(line_wl.size, bool), np.zeros(l_max - line_wl.size, bool)]),
        jnp.asarray(_pad(wl_np, w_max, fill=float(wl_np[-1]))),
        jnp.asarray(np.r_[np.ones(n_w, bool), np.zeros(w_max - n_w, bool)]),
    )

    if not bool(np.asarray(res.success)):
        return _fail("no_valid_model_fit", n_peaks=peak_wl.size)

    model = _C.MODEL_STRINGS[int(np.asarray(res.model_id))]
    coef_full = np.asarray(res.coefficients)
    n_param = {"shift": 1, "affine": 2, "quadratic": 3}[model]
    coefficients = tuple(float(c) for c in coef_full[:n_param])

    # The kernel's quality gate ran on-device; recover the corrected axis on the
    # live region. When the gate is off (per-segment path) the kernel returns the
    # model correction; reproduce the reference convention (identity on failure).
    quality_passed = bool(np.asarray(res.quality_passed))
    if quality_passed and _is_monotonic_on_grid(model, coefficients, wl_np):
        corrected = np.asarray(_eval_model(wl_np, model, coefficients), dtype=float)
    else:
        corrected = wl_np.copy()

    # Inlier anchor wavelengths (measured peak wl of the robust inliers) — the
    # coverage gate reads these from details["inlier_anchor_wl_nm"].
    robust = np.asarray(res.robust_mask) if res.robust_mask is not None else None
    if robust is not None and robust.any():
        peak_id_grid = np.repeat(np.arange(p_max), _CAL_K_PAIR)
        sel_peak_ids = np.unique(peak_id_grid[robust])
        sel_peak_ids = sel_peak_ids[sel_peak_ids < peak_wl.size]
        inlier_anchor_wl = sorted(float(peak_wl[i]) for i in sel_peak_ids)
    else:
        inlier_anchor_wl = []

    reason = _C.REASON_STRINGS.get(int(np.asarray(res.reason_code)), "passed")
    details = {
        "peak_count": float(peak_wl.size),
        "line_pool_size": float(line_wl.size),
        "inlier_anchor_min_nm": (
            float(min(inlier_anchor_wl)) if inlier_anchor_wl else float("nan")
        ),
        "inlier_anchor_max_nm": (
            float(max(inlier_anchor_wl)) if inlier_anchor_wl else float("nan")
        ),
        "inlier_anchor_wl_nm": inlier_anchor_wl,
        "selected_model_bic": float(np.asarray(res.bic)),
        "quality_gate_enabled": bool(apply_quality_gate),
        "quality_passed": quality_passed,
        "quality_reason": reason,
    }

    return WavelengthCalibrationResult(
        success=True,
        model=model,
        coefficients=coefficients,
        corrected_wavelength=corrected,
        bic=float(np.asarray(res.bic)),
        rmse_nm=float(np.asarray(res.rmse_nm)),
        n_inliers=int(np.asarray(res.n_inliers)),
        n_peaks=int(peak_wl.size),
        n_candidates=int(peak_wl.size * _CAL_K_PAIR),
        matched_peak_fraction=float(np.asarray(res.matched_peak_fraction)),
        quality_passed=quality_passed,
        quality_reason=reason,
        details=details,
    )


def _ondevice_calibrate_segmented(wl, inten, atomic_db, elements, pipeline):
    """ON-DEVICE segmented wavelength calibration (J2 driver, kernel-backed core).

    Replaces the host ``calibrate_wavelength_axis_segmented`` delegation
    (:func:`_ld_calibrate`) with the SAME segmented driver structure (seams ->
    always-computed coverage-gated global fit -> per-segment fit with trust /
    coverage / global-disagreement gates -> global-offset fallback -> neighbour
    fallback -> seam-monotonicity restore -> revert-to-global gates) where every
    inner robust RANSAC fit runs on-device via :func:`_calibrate_axis_kernel_backed`
    (the parity-tested :func:`cflibs.jitpipe.calibrate.calibrate_axis_kernel`).
    The cheap, branchy orchestration (a handful of segments) reuses the reference
    helpers verbatim so seam detection, the gate sequencing, fallbacks, the
    seam-monotonicity cascade, and result assembly are byte-faithful — the ONLY
    on-device-vs-reference difference is the robust-fit core.

    This per-segment re-detection (each segment slice gets its OWN
    ``detect_peaks_auto`` + segment-range line pool, exactly as the reference
    ``_fit_one_segment`` does) is what closes the J2 §7 R8 model-class flip:
    feeding the kernel a per-segment candidate set instead of the global peaks
    masked to a window restores the reference's shift/affine decision on the
    ye6t BHVO-2 confounder (obs-set Jaccard back to 1.0). It is also FASTER than
    the monolithic ``calibrate_segmented_kernel`` — N+1 small-axis kernel fits
    (one global + one per segment, each ~2k samples) instead of vmapping two
    full-axis fits over SEG_max=8 padded slots.

    Returns ``(corrected_wavelength (N,), success, quality_passed)`` — the same
    triple :func:`run_front_end_ondevice` consumed from the reference
    ``WavelengthCalibrationResult``.
    """
    from cflibs.inversion.preprocess.wavelength_calibration import (
        _apply_neighbor_fallback,
        _build_segmented_result,
        _restore_seam_monotonicity,
        _revert_segmented_to_global,
        _segment_anchor_coverage,
        _segment_fit_trusted,
        detect_ccd_seams,
    )

    wl_np = np.asarray(wl, dtype=float)
    inten_np = np.asarray(inten, dtype=float)
    n_w = wl_np.size
    # Reference short-axis no-op (calibrate_wavelength_axis_segmented:1511).
    if n_w < 4 or inten_np.size < 4:
        return wl_np.copy(), False, False

    elements = list(elements)
    candidate_models = ("shift", "affine")
    sparse_models = ("shift",)
    seed = 42

    seams = detect_ccd_seams(wl_np, ratio_threshold=3.0, window=51)

    # --- always-computed global single-axis fit (coverage-gated) ---------------
    global_result = _calibrate_axis_kernel_backed(
        wl_np,
        inten_np,
        atomic_db,
        elements,
        candidate_models=candidate_models,
        apply_quality_gate=True,
        random_seed=seed,
    )
    # J2 §7 R8 model-selection tiebreak: the kernel's exhaustive/stratified search
    # is strictly stronger than the reference's 600 random RANSAC draws, so it can
    # find an under-anchored slope model the reference's weaker search never
    # samples — the ye6t hazard (a 53%-covered affine that extrapolates a wrong
    # ~0.1 nm slope across the red end and flips the Al doublet). The reference's
    # ``affine_coverage_gate`` only governs the segmented post-fit degrade, but the
    # dense-hull coverage check is the physically-correct guard against exactly
    # this class of over-strong slope fit, so we apply it to MODEL SELECTION
    # unconditionally (independent of ``affine_coverage_gate``): a slope model is
    # kept only when its dense inlier-anchor hull covers >= 60 % of the axis AND
    # its extrapolation drift stays within 1.0 local pixel; otherwise degrade to
    # shift — which is the model class the reference selects.
    if global_result.success and global_result.model != "shift":
        span_fraction, _extrap_nm, extrap_px = _segment_anchor_coverage(global_result, wl_np)
        if span_fraction < 0.6 or extrap_px > 1.0:
            global_result = _calibrate_axis_kernel_backed(
                wl_np,
                inten_np,
                atomic_db,
                elements,
                candidate_models=("shift",),
                apply_quality_gate=True,
                random_seed=seed,
            )

    if seams.size == 0:
        # aa9e: the single-segment (seam-free) regime is exactly where the on-device
        # global single-axis fit diverges from the reference. The kernel's
        # deterministic stratified RANSAC and the reference's 600-draw random RANSAC
        # resolve a sparse/multimodal anchor set to different registrations (model-
        # class flip on sparse synthetic; ~1.9 nm shift-mode flip on narrow-band real
        # minerals), shifting the corrected axis enough to flip marginal lines and the
        # solve/fail outcome — 102/289 synthetic axes + the aalto muscoviteE35/
        # adulariaE11 board failures (all seam-free). The discriminating signal is
        # structural (seam count), not anchor count (the regimes' anchor counts
        # overlap). Delegate the whole single-axis fit to the reference robust core
        # for parity. The multi-segment broadband path below (BHVO-2: 3-segment,
        # parity at 0.00025 nm) is untouched and keeps the on-device speed win.
        ref = _ld_calibrate(wl_np, inten_np, atomic_db, elements, pipeline)
        corrected = np.asarray(ref.corrected_wavelength, dtype=float)[:n_w]
        return corrected, bool(ref.success), bool(ref.quality_passed)

    # --- per-segment fits (re-detect peaks per slice; reference flow) ----------
    bounds = [0] + [int(s) + 1 for s in seams] + [int(n_w)]
    corrected = wl_np.copy()
    global_corrected = (
        np.asarray(global_result.corrected_wavelength, dtype=float)
        if global_result.success
        else wl_np
    )
    global_offset = global_corrected - wl_np

    seg_diag: list = []
    seg_status: list = []
    total_inliers = 0
    rmse_accum: list = []

    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        seg_wl = wl_np[a:b]
        seg_in = inten_np[a:b]
        n_pts = int(seg_wl.size)
        status = "global"
        seg_model = "none"
        seg_n_in = 0
        seg_rmse = float("inf")
        coverage_status = "not_applicable"
        coverage_extrap_nm = 0.0
        global_disagreement_nm = 0.0
        accepted_inliers = 0
        accepted_rmse = None

        if n_pts >= _CAL_MIN_SEGMENT_POINTS:
            models = sparse_models if n_pts < _CAL_SPARSE_SEGMENT_POINTS else candidate_models
            seg_cal = _calibrate_axis_kernel_backed(
                seg_wl,
                seg_in,
                atomic_db,
                elements,
                candidate_models=models,
                apply_quality_gate=False,
                random_seed=seed + i,
            )
            trusted = _segment_fit_trusted(
                seg_cal, _CAL_SEGMENT_MIN_INLIERS, _CAL_SEGMENT_MAX_RMSE_NM
            )
            # J2 §7 R8 model-selection tiebreak (see the global-fit note above):
            # the dense-hull coverage gate runs UNCONDITIONALLY on a winning slope
            # model so the kernel's stronger search cannot keep an under-anchored
            # affine the reference's weaker RANSAC missed (the ye6t seg-2 877 nm
            # Al-doublet flip). ``affine_coverage_gate`` no longer guards it.
            if trusted and seg_cal.model != "shift":
                seg_cal, coverage_status, coverage_extrap_nm = _segment_coverage_gate_kernel(
                    seg_cal, seg_wl, seg_in, atomic_db, elements, seed + i
                )
                trusted = _segment_fit_trusted(
                    seg_cal, _CAL_SEGMENT_MIN_INLIERS, _CAL_SEGMENT_MAX_RMSE_NM
                )
                if coverage_status == "degraded_to_shift" and not trusted:
                    coverage_status = "degraded_shift_untrusted"
            if trusted:
                seg_offset_med = float(np.median(seg_cal.corrected_wavelength - seg_wl))
                global_offset_med = float(np.median(global_offset[a:b]))
                global_disagreement_nm = abs(seg_offset_med - global_offset_med)
                if global_disagreement_nm > _CAL_SEGMENT_MAX_GLOBAL_DISAGREEMENT_NM:
                    trusted = False
            seg_model = seg_cal.model
            seg_n_in = int(seg_cal.n_inliers)
            seg_rmse = float(seg_cal.rmse_nm)
            if trusted:
                corrected[a:b] = seg_cal.corrected_wavelength
                status = "fit"
                accepted_inliers = seg_n_in
                accepted_rmse = seg_rmse

        if status != "fit":
            corrected[a:b] = seg_wl + global_offset[a:b]

        seg_diag.append(
            {
                "index": i,
                "wl_min": float(seg_wl.min()) if n_pts else 0.0,
                "wl_max": float(seg_wl.max()) if n_pts else 0.0,
                "n_points": n_pts,
                "model": seg_model,
                "n_inliers": seg_n_in,
                "rmse_nm": seg_rmse,
                "status": status,
                "coverage_gate": coverage_status,
                "coverage_extrapolation_nm": float(coverage_extrap_nm),
                "global_disagreement_nm": float(global_disagreement_nm),
            }
        )
        seg_status.append(status)
        if status == "fit":
            total_inliers += accepted_inliers
            if accepted_rmse is not None:
                rmse_accum.append(accepted_rmse)

    if not global_result.success:
        _apply_neighbor_fallback(bounds, wl_np, corrected, seg_diag, seg_status)

    n_clamped, max_clamp_nm, cumulative_shift = _restore_seam_monotonicity(bounds, corrected)

    if cumulative_shift > 0.5:
        rev = _revert_segmented_to_global(global_result, bounds, seams, "large_seam_shift")
        gc = np.asarray(rev.corrected_wavelength, dtype=float)[:n_w]
        return gc, bool(rev.success), bool(rev.quality_passed)
    if not bool(np.all(np.diff(corrected) > 0)):
        rev = _revert_segmented_to_global(global_result, bounds, seams, "residual_non_monotonic")
        gc = np.asarray(rev.corrected_wavelength, dtype=float)[:n_w]
        return gc, bool(rev.success), bool(rev.quality_passed)

    agg = _build_segmented_result(
        global_result,
        corrected,
        bounds,
        seams,
        seg_diag,
        seg_status,
        total_inliers,
        rmse_accum,
        n_clamped,
        max_clamp_nm,
        _CAL_SEGMENT_MIN_INLIERS,
        _CAL_SEGMENT_MAX_RMSE_NM,
    )
    out = np.asarray(agg.corrected_wavelength, dtype=float)[:n_w]
    return out, bool(agg.success), bool(agg.quality_passed)


def _segment_coverage_gate_kernel(seg_cal, seg_wl, seg_in, atomic_db, elements, random_seed):
    """Per-segment ye6t coverage gate (kernel-backed degrade-to-shift refit).

    Mirrors the reference :func:`_apply_segment_coverage_gate`: keep a slope
    model only when its dense inlier-anchor hull covers >= 60 % of the segment
    AND the implied correction drift past the hull stays within 1.0 local pixel;
    otherwise refit the segment with a pure ``shift`` model (the kernel-backed
    calibrator). Returns ``(seg_cal, coverage_status, extrap_nm)``.
    """
    from cflibs.inversion.preprocess.wavelength_calibration import _segment_anchor_coverage

    span_fraction, extrap_nm, extrap_px = _segment_anchor_coverage(seg_cal, seg_wl)
    if span_fraction >= 0.6 and extrap_px <= 1.0:
        return seg_cal, "passed", extrap_nm
    shift_cal = _calibrate_axis_kernel_backed(
        seg_wl,
        seg_in,
        atomic_db,
        elements,
        candidate_models=("shift",),
        apply_quality_gate=False,
        random_seed=random_seed,
    )
    return shift_cal, "degraded_to_shift", extrap_nm


def _ld_select(observations, resonance_lines, pipeline, exclude_resonance):
    """Reference ``LineSelector`` post-stage (delegated; see module note)."""
    from cflibs.inversion.physics.line_selection import LineSelector

    selector = LineSelector(
        min_snr=pipeline.min_snr,
        min_energy_spread_ev=pipeline.min_energy_spread_ev,
        min_lines_per_element=pipeline.min_lines_per_element,
        exclude_resonance=exclude_resonance,
        isolation_wavelength_nm=pipeline.isolation_wavelength_nm,
        max_lines_per_element=pipeline.max_lines_per_element,
    )
    return selector.select(observations, resonance_lines=resonance_lines).selected_lines


def _line_selector_scores(wavelength_nm, intensity, intensity_unc):
    """DEVICE kernel: per-obs SNR + isolation + composite score (LineSelector).

    Fixed-shape port of :meth:`LineSelector._score_line` / ``_compute_isolation``
    on the padded observation block (Wave-3b stage 3). Bit-faithful to the
    reference:

    * ``snr = intensity / intensity_uncertainty`` when the uncertainty is
      strictly positive, else ``100.0`` (``_score_line``);
    * ``isolation = 1 - exp(-min_sep / iso_scale)`` where ``min_sep`` is the
      distance to the nearest OTHER observation (``+inf`` -> isolation ``1.0``
      for a lone observation);
    * ``score = snr * (1/atomic_unc) * isolation`` with the default
      ``atomic_unc = 0.10`` (no per-line atomic-uncertainty dict in the pipeline
      path) — the ``1/0.10 = 10`` factor cancels in the per-element ranking but
      is kept for byte-faithfulness.

    The isolation scale + the per-element top-K cap stay host-resolved (closed
    over below); this kernel returns ``(snr, isolation, score)`` over ``(C,)``.
    """
    import jax.numpy as jnp

    wl = jnp.asarray(wavelength_nm, dtype=jnp.float64)
    inten = jnp.asarray(intensity, dtype=jnp.float64)
    unc = jnp.asarray(intensity_unc, dtype=jnp.float64)
    n = wl.shape[0]

    has_unc = unc > 0.0
    snr = jnp.where(has_unc, inten / jnp.where(has_unc, unc, 1.0), 100.0)

    # nearest-other-observation separation (exclude self via the diagonal).
    dwl = jnp.abs(wl[:, None] - wl[None, :])
    eye = jnp.eye(n, dtype=bool)
    dwl = jnp.where(eye, jnp.inf, dwl)
    min_sep = jnp.min(dwl, axis=1)  # +inf for a lone observation
    return snr, min_sep


def _ondevice_line_select(observations, resonance_lines, pipeline, exclude_resonance):
    """ON-DEVICE ``LineSelector`` post-stage (J-stage 3) via a fixed-shape kernel.

    Replaces the host :meth:`LineSelector.select` delegation (:func:`_ld_select`)
    with a device kernel (:func:`_line_selector_scores`) over the padded
    observation block + a host gather that applies the reference gate / per-element
    top-K cap in the EXACT reference order:

    1. score each observation on-device (SNR, isolation, composite score);
    2. partition (reject ``snr < min_snr``, resonance when ``exclude_resonance``,
       ``isolation < 0.5`` — the reference order, ``_partition_by_criteria``);
    3. group by element in first-appearance order (``defaultdict`` insertion
       order), stable-sort each group by ``-score``, take the top
       ``max_lines_per_element`` (``_select_per_element``).

    The selected-line ORDER (element-group order, score-desc within group) is the
    reference's; it feeds the downstream Stark candidate selection + solve, so it
    must match for end-to-end T/n_e parity. Returns the selected
    ``LineObservation`` list.
    """
    if not observations:
        return []

    n = len(observations)
    wl = np.asarray([o.wavelength_nm for o in observations], dtype=float)
    inten = np.asarray([o.intensity for o in observations], dtype=float)
    unc = np.asarray([o.intensity_uncertainty for o in observations], dtype=float)

    snr_j, min_sep_j = _line_selector_scores(wl, inten, unc)
    snr = np.asarray(snr_j)
    min_sep = np.asarray(min_sep_j)
    iso_scale = float(pipeline.isolation_wavelength_nm)
    # isolation = 1 - exp(-min_sep / iso_scale); a lone obs (min_sep=+inf) -> 1.0.
    isolation = np.where(np.isinf(min_sep), 1.0, 1.0 - np.exp(-min_sep / iso_scale))
    # atomic_unc default 0.10 (no per-line dict in the pipeline path).
    score = snr * (1.0 / 0.10) * isolation

    min_snr = float(pipeline.min_snr)
    max_lines = int(pipeline.max_lines_per_element)
    resonance_lines = resonance_lines or set()

    # --- partition (reference _partition_by_criteria order) -------------------
    accepted = np.zeros(n, dtype=bool)
    for i, o in enumerate(observations):
        key = (o.element, o.ionization_stage, o.wavelength_nm)
        is_res = key in resonance_lines
        if snr[i] < min_snr:
            continue
        if is_res and exclude_resonance:
            continue
        if isolation[i] < 0.5:
            continue
        accepted[i] = True

    # --- per-element group (first-appearance order) -> sort-desc -> top-K ------
    by_element: dict = {}
    for i, o in enumerate(observations):
        if accepted[i]:
            by_element.setdefault(o.element, []).append(i)

    selected: list = []
    for _el, idxs in by_element.items():
        # Stable sort by -score (Python's sort is stable; ties keep input order),
        # bit-identical to the reference ``elem_scores.sort(key=-score)``.
        idxs_sorted = sorted(idxs, key=lambda i: -score[i])
        n_select = min(len(idxs_sorted), max_lines)
        for i in idxs_sorted[:n_select]:
            selected.append(observations[i])
    return selected


def _ondevice_kdet_filter(
    peaks,
    transitions_by_element,
    *,
    shift_scan_nm,
    wl_step,
    tolerance_nm,
    shift_coherence_veto,
    kdet_min_candidates=2,
    coherence_min_lines=2,
):
    """ON-DEVICE kdet pre-filter (coherence branch) via ``kdet_keep_mask``.

    Wave-3b stage 2. The pipeline runs kdet with ``shift_coherence_veto=True``
    (the default), so ``_kdet_element_passes`` takes the coherence branch
    (``best_candidates >= max(kdet_min_candidates, coherence_min_lines)``;
    ``best_candidates`` = max-over-shift-grid in-tolerance candidate count). That
    branch is the parity-tested fixed-shape kernel
    :func:`cflibs.jitpipe.identify.kdet_keep_mask`. kdet runs on the FULL
    transition set (not the comb top-K), so the host builds a snapshot whose
    comb-wavelength axis IS each element's full transition wavelength list.

    The density-scaled score branch (``shift_coherence_veto=False``) dispatches to
    Rust + a density score the J3 kernel does not port; for that branch the caller
    keeps the reference delegation. Returns ``None`` when the on-device path does
    not apply (density branch / empty inputs) so the caller falls back.

    Mirrors the reference ``_apply_kdet_filter`` keep semantics: keep the filtered
    map when non-empty, else fall back to the full map (``kdet_filtered_all_elements``).
    """
    if not shift_coherence_veto:
        return None
    if not peaks or not transitions_by_element:
        return None

    import jax.numpy as jnp

    from cflibs.inversion.identify.line_detection import _build_shift_grid

    from cflibs.jitpipe.identify import FrontEndSnapshot, kdet_keep_mask

    elements = list(transitions_by_element.keys())
    n_e = len(elements)
    peak_wl = np.asarray([p[1] for p in peaks], dtype=float)
    n_p = peak_wl.size
    if n_p == 0:
        return None

    k_full = max((len(transitions_by_element[el]) for el in elements), default=1)
    e_max = _next_pow2(max(n_e, 1))
    k_max = _next_pow2(max(k_full, 1))
    p_max = _next_pow2(max(n_p, 1))

    comb_wl = np.zeros((e_max, k_max), dtype=np.float64)
    comb_mask = np.zeros((e_max, k_max), dtype=bool)
    element_mask = np.zeros(e_max, dtype=bool)
    for e, el in enumerate(elements):
        trans = transitions_by_element[el]
        element_mask[e] = True
        for k, t in enumerate(trans[:k_max]):
            comb_wl[e, k] = float(t.wavelength_nm)
            comb_mask[e, k] = True

    peak_wl_pad = np.zeros(p_max, dtype=np.float64)
    peak_wl_pad[:n_p] = peak_wl
    peak_mask = np.r_[np.ones(n_p, bool), np.zeros(p_max - n_p, bool)]

    zeros_ek = np.zeros((e_max, k_max), dtype=np.float64)
    snap = FrontEndSnapshot(
        peak_wavelength_nm=jnp.asarray(peak_wl_pad),
        peak_index=jnp.asarray(np.full(p_max, -1, dtype=np.int64)),
        peak_mask=jnp.asarray(peak_mask),
        comb_wavelength_nm=jnp.asarray(comb_wl),
        comb_E_k_ev=jnp.asarray(zeros_ek),
        comb_g_k=jnp.asarray(np.ones((e_max, k_max), dtype=np.float64)),
        comb_A_ki=jnp.asarray(np.ones((e_max, k_max), dtype=np.float64)),
        comb_E_i_ev=jnp.asarray(zeros_ek),
        comb_stage=jnp.asarray(np.zeros((e_max, k_max), dtype=np.int64)),
        comb_is_resonance=jnp.asarray(np.full((e_max, k_max), -1, dtype=np.int64)),
        comb_mask=jnp.asarray(comb_mask),
        element_mask=jnp.asarray(element_mask),
        full_n_lines=jnp.asarray(np.zeros(e_max, dtype=np.int64)),
    )

    shift_grid = _build_shift_grid(shift_scan_nm, None, wl_step, tolerance_nm)
    keep = np.asarray(
        kdet_keep_mask(
            snap,
            jnp.asarray(np.asarray(shift_grid, dtype=float)),
            tolerance_nm=jnp.float64(tolerance_nm),
            kdet_min_candidates=jnp.int32(int(kdet_min_candidates)),
            coherence_min_lines=jnp.int32(int(coherence_min_lines)),
        )
    )
    filtered = {el: transitions_by_element[el] for e, el in enumerate(elements) if keep[e]}
    # Reference keep policy: non-empty filtered map wins; else keep the full map.
    if filtered:
        return filtered
    return dict(transitions_by_element)


def _gather_observations(
    snap,
    ob,
    comb_transitions_by_element,
    elem_order,
    wavelength,
    intensity,
    half_width_px,
    wl_step,
    *,
    ground_state_threshold_ev,
):
    """Gather the on-device obs-build masks -> reference ``LineObservation`` list.

    J8 plan §2.C host gather: for each valid ``(element-slot, comb-slot)`` from
    :func:`cflibs.jitpipe.identify.build_observations`, run the DEVICE trapezoid
    intensity kernel (:func:`extract_intensity_trapezoid`) on the peak slot's
    sample index and build the :class:`LineObservation` from the comb-slot's
    transition (the host-ranked catalog row). The degenerate Gaussian-area
    fallback (non-finite / <=0 trapezoid integral) is delegated to the reference
    ``_build_observation`` — it fires only on over-subtracted baselines and is
    host-flagged in the J3 kernel (the on-device trapezoid covers the real path).
    """
    import jax.numpy as jnp

    from cflibs.inversion.common import LineObservation
    from cflibs.inversion.identify.line_detection import _build_observation
    from cflibs.jitpipe.identify import extract_intensity_trapezoid

    obs_valid = np.asarray(ob.obs_valid)
    obs_peak_slot = np.asarray(ob.obs_peak_slot)
    peak_index = np.asarray(snap.peak_index)
    wl = np.asarray(wavelength, dtype=float)
    inten = np.asarray(intensity, dtype=float)
    wl_j = jnp.asarray(wl)
    inten_j = jnp.asarray(inten)

    # Walk elements in the kernel's ranked (f1, matched)-desc order — the same
    # order the reference ``_collect_observations`` appends in (``element_order``
    # is ``argsort`` over the accepted-element composite key). The observation
    # ORDER feeds the downstream element grouping + Stark candidate selection, so
    # it must match the reference for end-to-end T/n_e parity.
    ranked_slots = [int(s) for s in np.asarray(ob.element_order)]

    observations: list = []
    resonance_lines: set = set()
    seen_keys: set = set()
    for e in ranked_slots:
        if e < 0 or e >= len(elem_order):
            continue
        el = elem_order[e]
        trans = comb_transitions_by_element[el]
        for k in range(min(len(trans), obs_valid.shape[1])):
            if not obs_valid[e, k]:
                continue
            slot = int(obs_peak_slot[e, k])
            if slot < 0:
                continue
            peak_idx = int(peak_index[slot])
            if peak_idx < 0:
                continue
            t = trans[k]
            key = (t.element, t.ionization_stage, t.wavelength_nm)
            if key in seen_keys:
                continue

            area, sigma = extract_intensity_trapezoid(
                jnp.int64(peak_idx), wl_j, inten_j, int(half_width_px), jnp.float64(wl_step)
            )
            area_f = float(area)
            if not np.isfinite(area_f) or area_f <= 0.0:
                # Degenerate-trapezoid Gaussian-area fallback -> reference.
                built = _build_observation(
                    t,
                    peak_idx,
                    wl,
                    inten,
                    int(half_width_px),
                    float(wl_step),
                    ground_state_threshold_ev,
                )
                if built is None:
                    continue
                obs, is_res = built
            else:
                is_res = bool(t.E_i_ev <= ground_state_threshold_ev)
                obs = LineObservation(
                    wavelength_nm=float(t.wavelength_nm),
                    intensity=area_f,
                    intensity_uncertainty=max(float(sigma), 1e-6),
                    element=t.element,
                    ionization_stage=t.ionization_stage,
                    E_k_ev=t.E_k_ev,
                    g_k=t.g_k,
                    A_ki=t.A_ki,
                    aki_uncertainty=t.aki_uncertainty,
                )
            seen_keys.add(key)
            observations.append(obs)
            if is_res:
                resonance_lines.add(key)
    return observations, resonance_lines


def _ondevice_stark_ne(wl_in, inten, selected, atomic_db, snapshot, pipeline) -> float | None:
    """ON-DEVICE Stark n_e diagnostic — composes the J6 ``stark.py`` kernels.

    The device-side replacement for the reference-delegated ``measure_stark_ne``
    call in :func:`run_front_end_ondevice`. It reproduces the reference two-pass
    control flow (``cflibs.inversion.physics.stark_ne.measure_stark_ne``,
    :483-606) by gathering the atomic-data-only scalars on the host (the
    ``get_stark_parameters_with_source`` 0.1 nm nearest-match, the Doppler width,
    the instrument FWHM) and routing every measurement step through the
    parity-tested fixed-shape kernels in :mod:`cflibs.jitpipe.stark`:

    * :func:`stark.select_stark_candidates` for the SNR / isolation / source-
      class gate ladder + score-descending rank (``top_k = candidate count`` so
      the kernel's built-in ``max_lines`` cap is disabled — the cap is applied
      *after* the multiplet/fit gates, as the reference breaks only after
      ``max_lines`` *successes*);
    * :func:`stark.multiplet_blend_mask` for the same-species multiplet-blend
      veto (against the snapshot line table, never a DB query in the loop);
    * :func:`stark.recenter_idx` + :func:`stark.extract_windows` for the raw
      window gather (on ``wl_in`` — the response-corrected, NOT calibrated axis,
      matching :func:`run_front_end_ondevice`'s reference-parity convention);
    * :func:`stark.measure_stark_ne_jit` for the vmapped LM Voigt fit, width-law
      inversion, QC gates, cohort trim, and median combine.

    The load-bearing detail is the **break-after-``max_lines``-successes**
    semantics (``stark_ne.py:541``): the reference fits candidates in score-
    descending order and stops only after collecting ``max_lines`` lines that
    pass *every* gate (multiplet / fit / poor-fit / unresolved / implausible
    each ``continue`` without counting). We reproduce this by fitting ALL
    gate-and-multiplet-survivors, then taking the first ``MAX_LINES`` survivors
    (in score-descending order) whose QC code is ``QC_OK`` as the success set,
    and re-running the cohort-trim + median combine over exactly that set.

    Parameters
    ----------
    wl_in : ndarray
        Response-corrected (NOT calibration-corrected) wavelength axis, nm. This
        is the reference Stark axis (``pipeline.py:887``).
    inten : ndarray
        Response-corrected intensity on ``wl_in``.
    selected : list[LineObservation]
        The selected observation set (the solver's input list).
    atomic_db : AtomicDatabase
        Reference DB (host-side ``get_stark_parameters_with_source`` +
        ``resolve_element_mass`` lookups only — never queried in a device loop).
    snapshot : PipelineSnapshot
        Host-built atomic snapshot (per-line wavelength / species / g_k / A_ki /
        E_k for the multiplet-blend mask).
    pipeline : AnalysisPipelineConfig
        Resolved front-end config (``resolving_power`` is the instrument-width
        source).

    Returns
    -------
    float or None
        The robust median electron density, cm^-3, or ``None`` when no line is
        usable (reference ``StarkNeDiagnostics.usable``: ``n_lines > 0`` and the
        median is finite).
    """
    import jax.numpy as jnp

    from cflibs.atomic.masses import resolve_element_mass
    from cflibs.core.constants import EV_TO_K
    from cflibs.inversion.physics.stark_ne import (
        LITERATURE_STARK_SOURCES,
        estimate_instrument_fwhm,
    )
    from cflibs.jitpipe import stark as _S
    from cflibs.radiation.profiles import doppler_width

    wl_in = np.asarray(wl_in, dtype=float)
    inten = np.asarray(inten, dtype=float)
    if wl_in.size < 5 or not selected:
        return None

    # Reference T floor + window/Stark temperature (``stark_ne.py:402,466``).
    T_K = 10000.0
    T_eV = max(T_K, 1000.0) / EV_TO_K

    resolving_power = pipeline.resolving_power
    have_rp = resolving_power is not None and float(resolving_power) > 0.0

    # --- instrument-width resolution ladder (reference ``measure_stark_ne``) ----
    # explicit FWHM -> resolving-power (lambda/R) -> data-driven narrowest-line
    # floor (``estimate_instrument_fwhm``). The floor is the PRODUCTION path for
    # ChemCam/SuperCam (their config carries no ``resolving_power``), so it is NOT
    # unreachable; it is pure host NumPy signal processing (no device kernel, no
    # atomic data) and we reuse the reference helper verbatim for bit-parity. A
    # candidate whose instrument width cannot be resolved (floor returns ``None``)
    # is gated out on-device by ``SEL_NO_INSTRUMENT_WIDTH`` (``instr = 0``).
    def _instr_fwhm(center_nm: float) -> float | None:
        if have_rp:
            return center_nm / float(resolving_power)
        return estimate_instrument_fwhm(wl_in, inten, selected, center_nm=center_nm)

    # --- PASS 1: host gather of atomic-data-only scalars per observation -------
    # The reference resolves the Stark params by the OBSERVATION wavelength with a
    # 0.1 nm nearest-match (SQL ORDER BY ABS(delta) LIMIT 1), NOT by the catalog
    # wavelength keyed in the snapshot; the two diverge when the obs wl is offset.
    # Keep the scalar Stark params from THIS DB call for bit-identical scalars.
    n = len(selected)
    intensity_arr = np.empty(n, dtype=float)
    intensity_unc_arr = np.empty(n, dtype=float)
    wl_obs_arr = np.empty(n, dtype=float)
    instr_arr = np.zeros(n, dtype=float)
    dopp_arr = np.zeros(n, dtype=float)
    w_ref_arr = np.zeros(n, dtype=float)
    alpha_arr = np.full(n, _S.DEFAULT_STARK_ALPHA, dtype=float)
    lit_arr = np.zeros(n, dtype=bool)
    res_arr = np.zeros(n, dtype=bool)
    is_pref_arr = np.zeros(n, dtype=bool)
    cand_line_index = np.zeros(n, dtype=np.int64)
    have_index = np.zeros(n, dtype=bool)

    # Canonical-diagnostic preference (x2.0 score bonus) — computed on the
    # OBSERVATION wavelength + (element, stage), bit-identical to the reference
    # ``stark_ne._preference_factor``. is_preferred drives candidate RANK, which
    # in turn drives the top_k cap and the break-after-successes success set, so
    # dropping it (the old jnp.zeros wiring) diverged the on-device candidate set
    # from the reference exactly for Ca II / Mg II / H-alpha diagnostics.
    from cflibs.inversion.physics.stark_ne import _preference_factor

    line_wl = np.asarray(snapshot.line_wavelength_nm, dtype=float)
    line_sp = np.asarray(snapshot.line_species_index, dtype=np.int64)
    species = list(snapshot.species)  # ((element, sp_num), ...)
    # Pre-resolve obs (element, stage) -> snapshot species index for the
    # same-species window restriction (cand_line_index argmin within species).
    species_of: dict = {}
    for si, (el, sp) in enumerate(species):
        species_of[(el, int(sp))] = si

    for i, obs in enumerate(selected):
        intensity_arr[i] = obs.intensity
        intensity_unc_arr[i] = obs.intensity_uncertainty
        wl_obs_arr[i] = obs.wavelength_nm

        # Positional 0.1 nm tolerance: the real ``AtomicDatabase`` method names
        # this ``tolerance_nm`` while ``measure_stark_ne`` passes it positionally
        # (stark_ne.py:487); call it positionally so any source signature works.
        w_ref, alpha, source, is_resonance = atomic_db.get_stark_parameters_with_source(
            obs.element, obs.ionization_stage, obs.wavelength_nm, 0.1
        )
        is_lit = source in LITERATURE_STARK_SOURCES and w_ref is not None and w_ref > 0
        lit_arr[i] = bool(is_lit)
        res_arr[i] = bool(is_resonance)
        if w_ref is not None and w_ref > 0:
            w_ref_arr[i] = float(w_ref)
        alpha_arr[i] = float(alpha) if alpha is not None else _S.DEFAULT_STARK_ALPHA

        # Instrument FWHM via the reference ladder (resolving-power or the
        # data-driven narrowest-line floor); ``None`` -> 0 (gated by
        # ``SEL_NO_INSTRUMENT_WIDTH``).
        instr = _instr_fwhm(obs.wavelength_nm)
        instr_arr[i] = float(instr) if (instr is not None and instr > 0) else 0.0

        mass = resolve_element_mass(obs.element, atomic_db)
        dopp_arr[i] = float(doppler_width(obs.wavelength_nm, T_eV, mass))

        # Reference-identical preference: matches the canonical-diagnostic table
        # on the observation wavelength (NOT the snapshot catalog wavelength),
        # within 0.3 nm and same (element, stage).
        is_pref_arr[i] = (
            _preference_factor(obs.element, obs.ionization_stage, obs.wavelength_nm) > 1.0
        )

        # Resolve obs -> snapshot line index for multiplet/preference: argmin
        # |line_wl - obs.wl| restricted to the SAME species AND |delta| < 0.1 nm.
        si = species_of.get((obs.element, int(obs.ionization_stage)))
        if si is not None and line_wl.size:
            same = line_sp == si
            if np.any(same):
                idx_pool = np.nonzero(same)[0]
                deltas = np.abs(line_wl[idx_pool] - obs.wavelength_nm)
                j = int(np.argmin(deltas))
                if deltas[j] < 0.1:
                    cand_line_index[i] = int(idx_pool[j])
                    have_index[i] = True
        if not have_index[i]:
            # No snapshot line within 0.1 nm of the same species -> the
            # multiplet/preference lookup cannot anchor; treat as not
            # literature-grade so the candidate is never fit (reference port
            # plan step 4). Park the index at 0 (its mask gates it out).
            lit_arr[i] = False

    candidate_mask = np.ones(n, dtype=bool)

    # --- candidate selection + ranking (top_k = C disables the pre-fit cap) ----
    sel = _S.select_stark_candidates(
        jnp.asarray(intensity_arr),
        jnp.asarray(intensity_unc_arr),
        jnp.asarray(wl_obs_arr),
        jnp.asarray(instr_arr),
        jnp.asarray(dopp_arr),
        jnp.asarray(lit_arr, dtype=bool),
        jnp.asarray(is_pref_arr, dtype=bool),  # x2.0 rank bonus for canonical diagnostics
        jnp.asarray(res_arr, dtype=bool),
        jnp.asarray(candidate_mask, dtype=bool),
        min_snr=5.0,
        isolation_factor=1.5,
        top_k=n,  # C, NOT 5 — disable the kernel's built-in max_lines cap.
    )
    gate_ok = np.asarray(sel.selected, dtype=bool)
    score = np.asarray(sel.score, dtype=float)
    iso = np.asarray(sel.isolation_nm, dtype=float)
    gauss = np.asarray(sel.gaussian_fwhm_nm, dtype=float)

    if not gate_ok.any():
        return None

    # --- multiplet-blend gate AFTER selection, BEFORE fit ---------------------
    # half = min(max(4*gauss, 0.3), max(iso/2, 2*gauss)) (stark_ne.py:545).
    half = np.minimum(np.maximum(4.0 * gauss, 0.3), np.maximum(iso / 2.0, 2.0 * gauss))
    blend = np.asarray(
        _S.multiplet_blend_mask(
            jnp.asarray(cand_line_index),
            jnp.asarray(half),
            snapshot,
            T_eV,
            strength_fraction=0.25,
        ),
        dtype=bool,
    )
    # A candidate with no resolvable snapshot anchor cannot be evaluated for a
    # blend; it was already forced not-literature-grade above (gate_ok False).
    gate2_ok = gate_ok & ~blend

    if not gate2_ok.any():
        return None

    # --- window gather (recenter + extract) on wl_in --------------------------
    # W=96 raw samples, wider than ADR-0004 row 8's nominal W=64: real spectra
    # are locally coarser-sampled than the synthetic design case, so 96 avoids
    # truncating the Voigt wings (a conservative, accuracy-preserving deviation).
    W = 96
    search_nm = np.maximum(0.5 * gauss, 0.15)
    center_idx = np.empty(n, dtype=np.int64)
    wl_j = jnp.asarray(wl_in)
    inten_j = jnp.asarray(inten)
    for i in range(n):
        center_idx[i] = int(
            _S.recenter_idx(wl_j, inten_j, float(wl_obs_arr[i]), float(search_nm[i]))
        )
    wl_win, inten_win, mask = _S.extract_windows(
        wl_j, inten_j, jnp.asarray(center_idx), jnp.asarray(half), W
    )
    center0 = wl_j[jnp.asarray(center_idx)]

    # --- fit ALL gate2_ok candidates (no pre-cap) -----------------------------
    res = _S.measure_stark_ne_jit(
        wl_win,
        inten_win,
        mask,
        center0,
        jnp.asarray(gauss),
        jnp.asarray(w_ref_arr),
        jnp.asarray(alpha_arr),
        jnp.asarray(gate2_ok, dtype=bool),
        T_K,
        max_fit_rel_rmse=0.25,
    )

    # --- BREAK-AFTER-MAX_LINES-SUCCESSES: cap the success set to the first 5 ---
    # QC_OK candidates in score-descending order, THEN cohort-trim + combine over
    # exactly those (cap precedes trim; ``measure_stark_ne_jit`` re-run with the
    # capped candidate_mask reproduces the reference success set + median).
    quality = np.asarray(res.quality)
    qc_ok = quality == _S.QC_OK  # post-trim valid set over all gate2_ok candidates
    # The kernel already cohort-trimmed; recover the raw QC_OK (pre-trim accepted)
    # success set so the cap matches the reference (cap precedes trim).
    cohort_trimmed = quality == _S.QC_COHORT_OUTLIER
    raw_ok = qc_ok | cohort_trimmed  # accepted by the per-line gate ladder

    order = np.argsort(-score, kind="stable")  # score-descending, stable tiebreak
    success_mask = np.zeros(n, dtype=bool)
    taken = 0
    for idx in order:
        if not gate2_ok[idx]:
            continue
        if raw_ok[idx]:
            success_mask[idx] = True
            taken += 1
            if taken >= _S.MAX_LINES:
                break

    if not success_mask.any():
        return None

    # Re-run the cohort-trim + median over EXACTLY the capped success set.
    res2 = _S.measure_stark_ne_jit(
        wl_win,
        inten_win,
        mask,
        center0,
        jnp.asarray(gauss),
        jnp.asarray(w_ref_arr),
        jnp.asarray(alpha_arr),
        jnp.asarray(success_mask, dtype=bool),
        T_K,
        max_fit_rel_rmse=0.25,
    )
    n_lines = int(res2.n_lines)
    ne_median = float(res2.ne_median)
    if n_lines >= 1 and np.isfinite(ne_median):
        return ne_median
    return None


def run_front_end_ondevice(wavelength, intensity, atomic_db, pipeline, snapshot) -> FrontEndResult:
    """ON-DEVICE front-end: J1 detect + J3 identify gate-stack run as JIT kernels.

    The Wave-3 deliverable: ``run_one``'s front-end runs the parity-tested JIT
    kernels (``cflibs.jitpipe.{detect,identify}``) instead of delegating
    byte-for-byte to the reference ``detect_and_select_lines``. The flow mirrors
    the reference stage order (``pipeline.py:806-916``):

        response (host) -> [segmented wavelength calibration (DEVICE)] ->
        [adaptive tolerances — reference] -> [catalog SQL + gA-Boltzmann comb
        ranking — host gather] -> J1 detect_peaks_detection (DEVICE) ->
        kdet pre-filter (DEVICE, coherence branch) -> J3 comb scan + shift select
        + veto + observation build (DEVICE) -> trapezoid intensity (DEVICE) ->
        LineSelector SNR/iso/score (DEVICE) -> Stark n_e diagnostic (DEVICE).

    The DEVICE stages are the J1/J3 fixed-shape kernels plus (Wave 3b) the kdet
    coherence keep-rule (``kdet_keep_mask``), the LineSelector score kernel
    (``_line_selector_scores``), and (this change) the J6 Stark n_e diagnostic
    (:func:`_ondevice_stark_ne`, composing the ``cflibs.jitpipe.stark`` kernels);
    segmented calibration is on-device via :func:`_ondevice_calibrate_segmented`
    (``_ld_calibrate`` is retained only as the single-segment fallback + parity
    oracle). The lone residual reference-delegated stage is the kdet density-score
    branch (dispatched to Rust in the reference, dead-by-default under the
    shift-coherence veto); the kernels are fed byte-identical inputs so the produced
    observation line-key set matches the
    reference ``detect_and_select_lines`` (the parity oracle). Never raises on
    zero observations (returns ``n_observations == 0`` for the caller — the J8
    plan §4 failure policy).

    Parameters
    ----------
    wavelength, intensity : ndarray
        Spectrum axes (nm, intensity).
    atomic_db : AtomicDatabase
        Reference DB (host catalog SQL + the reference-delegated stages).
    pipeline : AnalysisPipelineConfig
        Resolved front-end config (every front-end knob).
    snapshot : PipelineSnapshot
        Host-built atomic snapshot. The detect/identify kernels read the
        host-gathered comb/peak arrays, not the snapshot; the ON-DEVICE Stark
        stage (:func:`_ondevice_stark_ne`) consumes the snapshot per-line tables
        for its multiplet-blend mask.

    Returns
    -------
    FrontEndResult
    """
    import jax.numpy as jnp

    from cflibs.inversion.identify import line_detection as _ld

    from cflibs.jitpipe import identify as _J

    # NOTE: the detect/identify kernels read host-gathered arrays, not the
    # snapshot, but the ON-DEVICE Stark stage (:func:`_ondevice_stark_ne`)
    # consumes the snapshot per-line tables for the multiplet-blend mask.

    wl_in = np.asarray(wavelength, dtype=float)
    inten = np.asarray(intensity, dtype=float)
    if pipeline.response_curve:
        inten = inten * response_multiplier(wl_in, pipeline.response_curve)

    elements = list(pipeline.elements)
    empty = FrontEndResult(
        observations=[],
        elements=elements,
        ne_stark_cm3=None,
        diagnostics={},
        n_observations=0,
    )
    if wl_in.size == 0 or inten.size == 0 or not elements:
        return empty

    # --- Wavelength calibration (ON-DEVICE; sets the detection axis) -----------
    # The segmented calibration now runs on-device via the kernel-backed driver
    # (:func:`_ondevice_calibrate_segmented`): the reference segmented-driver
    # structure (seams / global fit / per-segment gates / fallbacks / monotonicity
    # restore / revert) with every robust RANSAC core routed through the J2
    # ``calibrate_axis_kernel``. Each segment re-detects its own peaks + line pool
    # (the reference ``_fit_one_segment`` behaviour), which closes the J2 §7 R8
    # model-class flip that diverged the old monolithic ``calibrate_segmented_kernel``
    # on the real ChemCam BHVO-2 confounder (the per-segment shift/affine decision
    # now matches the reference, restoring the ye6t Al-doublet observation set).
    # ``_ld_calibrate`` (byte-faithful reference) is retained for the regression
    # cross-check in the parity tests.
    shift_scan_nm = float(pipeline.global_shift_scan_nm)
    wl = wl_in
    if pipeline.wavelength_calibration:
        try:
            corrected, cal_success, cal_quality = _ondevice_calibrate_segmented(
                wl_in, inten, atomic_db, elements, pipeline
            )
        except Exception:  # pragma: no cover - defensive
            corrected, cal_success, cal_quality = None, False, False
        if corrected is not None and cal_success and cal_quality:
            wl = np.asarray(corrected, dtype=float)
            shift_scan_nm = float(pipeline.residual_shift_scan_nm)

    # --- Adaptive tolerances + sampling cap (reference parity) -----------------
    wl_min = float(np.min(wl))
    wl_max = float(np.max(wl))
    wl_step = _ld._estimate_wl_step(wl)
    lambda_mid = 0.5 * (wl_min + wl_max)
    tol_nm, peak_width_nm = _ld._resolve_adaptive_tolerances(
        pipeline.wavelength_tolerance_nm,
        pipeline.peak_width_nm,
        wl_step,
        lambda_mid,
        pipeline.resolving_power,
    )
    if wl_step > 0:
        from cflibs.inversion.pipeline import SAMPLING_TOLERANCE_PX, SAMPLING_WIDTH_PX

        if tol_nm is not None:
            tol_nm = min(tol_nm, SAMPLING_TOLERANCE_PX * wl_step)
        if peak_width_nm is not None:
            peak_width_nm = min(peak_width_nm, SAMPLING_WIDTH_PX * wl_step)

    # --- Catalog SQL + gA-Boltzmann ranking (HOST gather, J8 plan §2.B) --------
    transitions = _ld._load_transitions(
        atomic_db,
        elements,
        wavelength_min=wl_min,
        wavelength_max=wl_max,
        min_relative_intensity=pipeline.min_relative_intensity,
        top_k_per_element=pipeline.top_k_per_element,
    )
    if not transitions:
        return empty

    # --- J1 detect kernel (DEVICE): the detection-path peaks -------------------
    peaks, total_peaks, _trunc = _detect_peaks_ondevice(
        wl, inten, pipeline.min_peak_height, peak_width_nm
    )
    if total_peaks == 0:
        return empty
    half_width_px = max(int((peak_width_nm / max(wl_step, 1e-9)) / 2), 1)

    transitions_by_element: dict = {}
    for t in transitions:
        transitions_by_element.setdefault(t.element, []).append(t)

    # --- kdet pre-filter (ON-DEVICE coherence branch: J3 kdet_keep_mask) -------
    # Wave-3b stage 2. The pipeline default ``shift_coherence_veto=True`` takes
    # the coherence keep-rule, which is the parity-tested kernel
    # (:func:`_ondevice_kdet_filter`). The density-scaled score branch
    # (``shift_coherence_veto=False``: Rust + density) is not ported; for it the
    # on-device helper returns ``None`` and we fall back to the reference filter.
    warnings: list = []
    kept = _ondevice_kdet_filter(
        peaks,
        transitions_by_element,
        shift_scan_nm=shift_scan_nm,
        wl_step=wl_step,
        tolerance_nm=tol_nm,
        shift_coherence_veto=pipeline.shift_coherence_veto,
        kdet_min_candidates=2,
        coherence_min_lines=2,
    )
    if kept is None:
        kept = _ld._apply_kdet_filter(
            peaks=peaks,
            transitions_by_element=transitions_by_element,
            shift_scan_nm=shift_scan_nm,
            shift_step_nm=None,
            wavelength_tolerance_nm=tol_nm,
            wl_step=wl_step,
            kdet_min_score=0.05,
            kdet_min_candidates=2,
            kdet_rarity_power=0.5,
            kdet_weight_clip=(0.25, 4.0),
            use_jax_kdet=False,
            shift_coherence_veto=pipeline.shift_coherence_veto,
            coherence_min_lines=2,
            coherence_min_fraction=0.5,
            warnings=warnings,
        )
    transitions_by_element = kept

    comb_transitions_by_element = {
        el: _ld._select_comb_transitions(tr, 30) for el, tr in transitions_by_element.items()
    }

    # --- Build the J3 FrontEndSnapshot (HOST gather) ---------------------------
    n_e = len(comb_transitions_by_element)
    k_used = max((len(v) for v in comb_transitions_by_element.values()), default=1)
    e_max = _next_pow2(max(n_e, 1))
    k_comb = _next_pow2(max(k_used, 1))
    p_max = _next_pow2(max(total_peaks, 1))
    snap, elem_order = build_front_end_snapshot(
        peaks, comb_transitions_by_element, e_max=e_max, k_comb=k_comb, p_max=p_max
    )

    shift_grid = _ld._build_shift_grid(shift_scan_nm, None, wl_step, tol_nm)
    shift_grid_j = jnp.asarray(np.asarray(shift_grid, dtype=float))

    # --- J3 comb scan + shift selection (DEVICE) -------------------------------
    scores = _J.score_comb_grid(
        snap,
        shift_grid_j,
        total_peaks=jnp.int32(total_peaks),
        tolerance_nm=jnp.float64(tol_nm),
        comb_min_matches=jnp.int32(3),
        comb_min_precision=jnp.float64(0.02),
        comb_min_recall=jnp.float64(0.1),
        comb_max_missing_fraction=jnp.float64(0.85),
    )
    best_idx, fb_idx, best_has_pass, _ = _J.select_shifts(scores, shift_grid_j, snap.element_mask)
    accepted_mask, applied_idx = _J.select_accepted_mask(
        scores,
        best_idx,
        fb_idx,
        best_has_pass,
        snap.element_mask,
        comb_min_matches=jnp.int32(3),
        comb_fallback_max_elements=5,
    )
    applied_shift_nm = float(np.asarray(shift_grid)[int(applied_idx)])

    # --- J3 shift-coherence veto (DEVICE) --------------------------------------
    if pipeline.shift_coherence_veto:
        accepted_mask = _J.shift_coherence_veto(
            snap,
            accepted_mask,
            jnp.float64(applied_shift_nm),
            jnp.float64(tol_nm),
            min_coherent_lines=jnp.int32(2),
            min_coherent_fraction=jnp.float64(0.5),
        )

    accepted_np = np.asarray(accepted_mask)
    if not accepted_np.any():
        return empty

    # --- J3 observation build (DEVICE): ownership + gate + min-kept-bars --------
    f1_applied = jnp.asarray(np.asarray(scores.f1)[int(applied_idx)])
    ml_applied = jnp.asarray(np.asarray(scores.matched_lines)[int(applied_idx)])
    use_gate = bool(pipeline.line_residual_gate)
    if use_gate:
        center, _ = _J.pooled_consensus(
            snap, accepted_mask, jnp.float64(applied_shift_nm), jnp.float64(tol_nm)
        )
        band = jnp.float64(tol_nm / 3.0)
        gate_flag = jnp.asarray(True)
    else:
        center = jnp.float64(0.0)
        band = jnp.float64(np.inf)
        gate_flag = jnp.asarray(False)

    ob = _J.build_observations(
        snap,
        f1=f1_applied,
        matched_lines=ml_applied,
        accepted_mask=accepted_mask,
        shift_nm=jnp.float64(applied_shift_nm),
        tolerance_nm=jnp.float64(tol_nm),
        residual_center_nm=center,
        residual_band_nm=band,
        use_residual_gate=gate_flag,
        coherence_min_lines=jnp.int32(2),
        coherence_min_fraction=jnp.float64(0.5),
        residual_gate_min_kept_lines=jnp.int32(int(pipeline.min_lines_per_element)),
    )

    # --- Gather valid obs -> LineObservation list (DEVICE intensity) -----------
    observations, resonance_lines = _gather_observations(
        snap,
        ob,
        comb_transitions_by_element,
        elem_order,
        wl,
        inten,
        half_width_px,
        wl_step,
        ground_state_threshold_ev=0.1,
    )

    # --- LineSelector post-stage (ON-DEVICE: fixed-shape SNR/iso/score kernel) -
    # Wave-3b stage 3. The SNR / isolation / composite-score computation runs as
    # a device kernel on the padded observation block; the gate + per-element
    # top-K cap are applied in the reference order on the host
    # (:func:`_ondevice_line_select`).
    exclude_resonance = pipeline.exclude_resonance
    if exclude_resonance is None:
        exclude_resonance = False
    selected = _ondevice_line_select(observations, resonance_lines, pipeline, exclude_resonance)

    detected_elements = {o.element for o in observations}
    selected_elements = {o.element for o in selected}
    dropped: dict = {}
    for el in elements:
        if el not in detected_elements:
            dropped[el] = "detection"
        elif el not in selected_elements:
            dropped[el] = "selection"
    diagnostics = {
        "requested_elements": list(elements),
        "detected_elements": sorted(detected_elements),
        "selected_elements": sorted(selected_elements),
        "dropped_elements": dropped,
        "applied_shift_nm": applied_shift_nm,
    }

    # --- Stark n_e diagnostic (ON-DEVICE J6 kernels, reference fallback) --------
    # The reference ``run_pipeline`` measures Stark n_e on the ORIGINAL
    # (response-corrected but NOT calibration-corrected) axis ``wl_in`` — the
    # calibration is applied only inside ``detect_and_select_lines`` on its own
    # copy (``pipeline.py:887``). Use ``wl_in`` to bit-match the reference n_e
    # that pins the production solve.
    #
    # :func:`_ondevice_stark_ne` composes the parity-tested fixed-shape kernels
    # (candidate selection / multiplet-blend / window gather / vmapped LM fit /
    # width-law inversion / cohort-trim). It reproduces the reference on the
    # synthetic parity envelope (``tests/jitpipe/test_ondevice_stark_ne.py``), but
    # on REAL coarse-sampled, high-dynamic-range ChemCam spectra the fixed-20-iter
    # LM Voigt kernel (``cflibs.jitpipe.stark.fit_lorentz_fwhm_lm``) is numerically
    # fragile (NaN at ~1e12 intensity scale + non-left-packed ``extract_windows``
    # gathers) and its strict ``converged`` quality gate rejects otherwise-good
    # fits — so it yields no usable line where scipy's trust-region fitter
    # succeeds. The kernels are parity-frozen (bead 6apc forbids modifying them),
    # so the on-device path is wired as an OPPORTUNISTIC primary with a
    # bounded-failure fallback to the reference ``measure_stark_ne`` (the exact
    # path :func:`run_front_end` uses) whenever it returns no usable n_e. See bead
    # 6apc for the gap analysis + a follow-up to harden the LM kernel for
    # real-data parity (then this fallback can be dropped).
    ne_stark: float | None = None
    if pipeline.stark_ne and selected:
        try:
            ne_stark = _ondevice_stark_ne(wl_in, inten, selected, atomic_db, snapshot, pipeline)
        except Exception:  # pragma: no cover - defensive
            # Log (don't silently swallow) so a genuine kernel regression -- a
            # shape/dtype bug, not the expected numerical non-convergence that
            # returns None -- is observable in logs rather than masquerading as
            # a normal reference fallback.
            from cflibs.core.logging_config import get_logger

            get_logger("jitpipe.host").debug(
                "on-device Stark n_e raised; falling back to reference measure_stark_ne",
                exc_info=True,
            )
            ne_stark = None
        if ne_stark is None:
            from cflibs.inversion.physics.stark_ne import measure_stark_ne

            try:
                stark_result = measure_stark_ne(
                    wl_in, inten, selected, atomic_db, resolving_power=pipeline.resolving_power
                )
            except Exception:  # pragma: no cover - defensive
                stark_result = None
            if stark_result is not None and stark_result.usable:
                ne_stark = float(stark_result.ne_median_cm3)

    return FrontEndResult(
        observations=list(selected),
        elements=elements,
        ne_stark_cm3=ne_stark,
        diagnostics=diagnostics,
        n_observations=len(selected),
    )
