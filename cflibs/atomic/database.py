"""
Atomic database interface for loading and querying atomic data.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from cflibs.core.jax_runtime import AtomicSnapshot

from cflibs.atomic.structures import Transition, EnergyLevel, SpeciesPhysics, PartitionFunction
from cflibs.core.logging_config import get_logger
from cflibs.core.abc import AtomicDataSource
from cflibs.core.cache import cached_transitions, cached_ionization
from cflibs.core.pool import get_pool

logger = get_logger("atomic.database")


class AtomicDatabase(AtomicDataSource):
    """
    Interface to atomic data stored in SQLite database.

    The database should have the following tables:
    - `lines`: Spectral line data
    - `energy_levels`: Energy level data
    - `species_physics`: Ionization potentials and species properties
    - `partition_functions`: Partition function coefficients
    """

    db_path: str | Path

    def __init__(self, db_path: str):
        """
        Initialize database connection and verify schema.

        Parameters
        ----------
        db_path : str
            Path to SQLite database file
        """
        path = Path(db_path)
        if not path.exists():
            raise FileNotFoundError(f"Atomic database not found: {path}")

        self.db_path = path
        # Use connection pool for better performance
        try:
            self._pool = get_pool(str(path), max_connections=5)
            self._use_pool = True
        except Exception as e:
            # Fallback to direct connection if pool fails
            logger.warning(f"Failed to create connection pool, using direct connection: {e}")
            self.conn = sqlite3.connect(str(path))
            self._use_pool = False

        # Verify and migrate schema if needed
        self._check_and_migrate_schema()
        logger.info(f"Connected to atomic database: {path}")

    @contextmanager
    def _get_connection(self):
        """Yield a database connection from the pool or the direct connection."""
        if self._use_pool:
            with self._pool.get_connection() as conn:
                yield conn
        else:
            yield self.conn

    def _check_and_migrate_schema(self):
        """Check database schema and migrate if necessary."""
        try:
            with self._get_connection() as conn:
                self._perform_migration(conn)
        except Exception as e:
            logger.error(f"Schema migration failed: {e}")
            raise

    def _perform_migration(self, conn: sqlite3.Connection):
        """Perform the actual migration steps."""
        cursor = conn.cursor()

        # 1. Check lines table columns
        cursor.execute("PRAGMA table_info(lines)")
        columns = {row[1] for row in cursor.fetchall()}

        required_line_cols = {
            "stark_w": "REAL",
            "stark_alpha": "REAL",
            "stark_shift": "REAL",
            "is_resonance": "INTEGER",
            "aki_uncertainty": "REAL",
            "accuracy_grade": "TEXT",
        }

        valid_dtypes = {"REAL", "INTEGER", "TEXT", "BLOB", "NUMERIC"}
        for col, dtype in required_line_cols.items():
            if col not in columns:
                if not col.isidentifier():
                    raise ValueError(f"Invalid column name for migration: {col}")
                if dtype.upper() not in valid_dtypes:
                    raise ValueError(f"Invalid data type for migration: {dtype}")

                logger.info(f"Migrating schema: Adding {col} to lines table")
                cursor.execute(f"ALTER TABLE lines ADD COLUMN {col} {dtype}")

                # Backfill is_resonance if we just added it
                if col == "is_resonance":
                    logger.info("Backfilling is_resonance based on ei_ev")
                    # SQLite doesn't strictly support boolean, so 1/0
                    # Assuming ei_ev exists and is populated
                    cursor.execute("UPDATE lines SET is_resonance = 1 WHERE ei_ev < 0.01")
                    cursor.execute(
                        "UPDATE lines SET is_resonance = 0 WHERE ei_ev >= 0.01 OR ei_ev IS NULL"
                    )

        # 2. Check partition_functions table
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='partition_functions'"
        )
        if not cursor.fetchone():
            logger.info("Migrating schema: Creating partition_functions table")
            cursor.execute("""
                CREATE TABLE partition_functions (
                    element TEXT,
                    sp_num INTEGER,
                    a0 REAL,
                    a1 REAL,
                    a2 REAL,
                    a3 REAL,
                    a4 REAL,
                    t_min REAL,
                    t_max REAL,
                    source TEXT,
                    PRIMARY KEY (element, sp_num)
                )
            """)

        # 3. Check species_physics table for atomic_mass
        cursor.execute("PRAGMA table_info(species_physics)")
        physics_columns = {row[1] for row in cursor.fetchall()}

        if "atomic_mass" not in physics_columns:
            logger.info("Migrating schema: Adding atomic_mass to species_physics table")
            cursor.execute("ALTER TABLE species_physics ADD COLUMN atomic_mass REAL")

        # 4. Auto-populate energy_levels from lines table if empty
        cursor.execute("SELECT COUNT(*) FROM energy_levels")
        if cursor.fetchone()[0] == 0:
            cursor.execute("SELECT COUNT(*) FROM lines")
            n_lines = cursor.fetchone()[0]
            if n_lines > 0:
                logger.info("Populating energy_levels from lines table...")
                self._populate_energy_levels(cursor)

        # 5. Auto-populate species_physics with NIST IPs if empty
        cursor.execute("SELECT COUNT(*) FROM species_physics")
        if cursor.fetchone()[0] == 0:
            logger.info("Populating species_physics with NIST ionization potentials...")
            self._populate_species_physics(cursor)

        # 6. Auto-populate partition_functions with NIST-fitted Irwin coefficients if empty
        cursor.execute("SELECT COUNT(*) FROM partition_functions")
        if cursor.fetchone()[0] == 0:
            logger.info("Populating partition_functions with NIST Irwin coefficients...")
            self._populate_partition_functions(cursor)

        # 7. Backfill aki_uncertainty from accuracy_grade or heuristic where missing
        cursor.execute(
            "SELECT COUNT(*) FROM lines WHERE aki IS NOT NULL AND aki_uncertainty IS NULL"
        )
        n_missing = cursor.fetchone()[0]
        if n_missing > 0:
            logger.info("Backfilling %d lines missing aki_uncertainty...", n_missing)
            self._populate_aki_uncertainties(cursor)

        conn.commit()

    @staticmethod
    def _populate_energy_levels(cursor: sqlite3.Cursor):
        """Extract unique energy levels from the lines table."""
        # Lower levels (INSERT OR IGNORE for idempotent migration)
        cursor.execute("""
            INSERT OR IGNORE INTO energy_levels (element, sp_num, g_level, energy_ev)
            SELECT DISTINCT element, sp_num, CAST(gi AS INTEGER), ROUND(ei_ev, 8)
            FROM lines
            WHERE gi IS NOT NULL AND ei_ev IS NOT NULL
        """)
        # Upper levels (avoid duplicates)
        cursor.execute("""
            INSERT OR IGNORE INTO energy_levels (element, sp_num, g_level, energy_ev)
            SELECT DISTINCT element, sp_num, CAST(gk AS INTEGER), ROUND(ek_ev, 8)
            FROM lines
            WHERE gk IS NOT NULL AND ek_ev IS NOT NULL
        """)
        # Deduplicate by (element, sp_num, g_level, energy_ev) so distinct levels
        # (different g_level or energy) are preserved for partition-function sums
        cursor.execute("""
            DELETE FROM energy_levels
            WHERE rowid NOT IN (
                SELECT MIN(rowid)
                FROM energy_levels
                GROUP BY element, sp_num, g_level, ROUND(energy_ev, 8)
            )
        """)
        cursor.execute("SELECT COUNT(*) FROM energy_levels")
        n = cursor.fetchone()[0]
        logger.info(f"Populated {n} energy levels from lines table")

    @staticmethod
    def _populate_species_physics(cursor: sqlite3.Cursor):
        """Populate species_physics with NIST ionization potentials and atomic masses."""
        # NIST recommended values: (element, mass, IP_I, IP_II)
        # fmt: off
        from cflibs.atomic.reference_data import NIST_SPECIES_PHYSICS
        nist_data = NIST_SPECIES_PHYSICS
        # fmt: on
        for elem, mass, ip1, ip2 in nist_data:
            cursor.execute(
                "INSERT OR REPLACE INTO species_physics (element, sp_num, ip_ev, atomic_mass) "
                "VALUES (?,?,?,?)",
                (elem, 1, ip1, mass),
            )
            if ip2 is not None:
                cursor.execute(
                    "INSERT OR REPLACE INTO species_physics (element, sp_num, ip_ev, atomic_mass) "
                    "VALUES (?,?,?,?)",
                    (elem, 2, ip2, mass),
                )

    @staticmethod
    def _populate_partition_functions(cursor: sqlite3.Cursor):
        """Populate partition_functions with Irwin polynomial coefficients.

        Coefficients fit ln(U) = sum(a_n * (ln T)^n), valid for T in [2000, 20000] K.
        Computed from NIST ASD energy level summation with weighted least-squares
        emphasising standard test temperatures (5000, 10000, 15000, 20000 K).
        Max relative error at key temperatures < 0.35% for all 106 species.
        Covers 68 elements across stages I–III.
        """
        # (element, sp_num, a0, a1, a2, a3, a4)
        # fmt: off
        from cflibs.atomic.reference_data import NIST_PARTITION_COEFFICIENTS
        nist_coefficients = NIST_PARTITION_COEFFICIENTS
        # fmt: on
        for row in nist_coefficients:
            cursor.execute(
                "INSERT OR REPLACE INTO partition_functions "
                "(element, sp_num, a0, a1, a2, a3, a4, t_min, t_max, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 2000.0, 20000.0, 'NIST ASD fit')",
                row,
            )
        logger.info("Populated %d partition function coefficient sets", len(nist_coefficients))

    @staticmethod
    def _populate_aki_uncertainties(cursor: sqlite3.Cursor):
        """Assign aki_uncertainty using Kramida/NIST data or intensity heuristic.

        1. If accuracy_grade is present but numerical uncertainty is missing,
           map grade to sigma using NIST standards.
        2. For lines missing both numerical uncertainty and accuracy grade,
           assign estimates based on relative intensity as a proxy for quality.
        """
        from cflibs.atomic.reference_data import NIST_GRADE_UNCERTAINTY

        # 1. First, backfill numerical uncertainty from known accuracy grades where missing
        n_from_grade = 0
        for grade, unc in NIST_GRADE_UNCERTAINTY.items():
            cursor.execute(
                "UPDATE lines SET aki_uncertainty = ? "
                "WHERE aki IS NOT NULL AND accuracy_grade = ? AND aki_uncertainty IS NULL",
                (unc, grade),
            )
            n_from_grade += cursor.rowcount

        if n_from_grade > 0:
            logger.info("Assigned aki_uncertainty from accuracy_grade for %d lines", n_from_grade)

        # 2. Heuristic: assign based on relative intensity for lines missing both
        unc_b = NIST_GRADE_UNCERTAINTY["B"]
        unc_c = NIST_GRADE_UNCERTAINTY["C"]
        unc_d = NIST_GRADE_UNCERTAINTY["D"]
        unc_e = NIST_GRADE_UNCERTAINTY["E"]

        # rel_int > 100 → well-measured lines (B grade)
        cursor.execute(
            f"UPDATE lines SET aki_uncertainty = {unc_b}, accuracy_grade = 'B' "
            "WHERE aki IS NOT NULL AND rel_int IS NOT NULL AND rel_int > 100 "
            "AND aki_uncertainty IS NULL AND accuracy_grade IS NULL"
        )
        n_b = cursor.rowcount

        # rel_int 10-100 → moderate quality (C grade)
        cursor.execute(
            f"UPDATE lines SET aki_uncertainty = {unc_c}, accuracy_grade = 'C' "
            "WHERE aki IS NOT NULL AND rel_int IS NOT NULL AND rel_int > 10 "
            "AND aki_uncertainty IS NULL AND accuracy_grade IS NULL"
        )
        n_c = cursor.rowcount

        # rel_int 1-10 → weaker lines (D grade)
        cursor.execute(
            f"UPDATE lines SET aki_uncertainty = {unc_d}, accuracy_grade = 'D' "
            "WHERE aki IS NOT NULL AND rel_int IS NOT NULL AND rel_int >= 1 "
            "AND aki_uncertainty IS NULL AND accuracy_grade IS NULL"
        )
        n_d = cursor.rowcount

        # Everything else: worst case (E grade)
        cursor.execute(
            f"UPDATE lines SET aki_uncertainty = {unc_e}, accuracy_grade = 'E' "
            "WHERE aki IS NOT NULL AND aki_uncertainty IS NULL AND accuracy_grade IS NULL"
        )
        n_e = cursor.rowcount

        logger.info(
            "Heuristic assignment (missing both): B=%d, C=%d, D=%d, E=%d lines",
            n_b,
            n_c,
            n_d,
            n_e,
        )

    @cached_transitions
    def get_transitions(
        self,
        element: str,
        ionization_stage: int | None = None,
        wavelength_min: float | None = None,
        wavelength_max: float | None = None,
        min_relative_intensity: float | None = None,
    ) -> list[Transition]:
        """
        Get transitions for an element.

        Parameters
        ----------
        element : str
            Element symbol
        ionization_stage : int, optional
            Filter by ionization stage (1=neutral, 2=singly ionized, etc.)
        wavelength_min : float, optional
            Minimum wavelength in nm
        wavelength_max : float, optional
            Maximum wavelength in nm
        min_relative_intensity : float, optional
            Minimum relative intensity threshold

        Returns
        -------
        list[Transition]
            list of transition objects
        """
        # Check if new columns exist in the actual query execution (though schema check should have fixed it)
        # We select all relevant columns.
        query = """
            SELECT
                element, sp_num, wavelength_nm, aki, ek_ev, ei_ev,
                gk, gi, rel_int,
                stark_w, stark_alpha, stark_shift, is_resonance,
                aki_uncertainty, accuracy_grade
            FROM lines
            WHERE element = ?
        """
        params: list[object] = [element]

        if ionization_stage is not None:
            query += " AND sp_num = ?"
            params.append(ionization_stage)

        if wavelength_min is not None:
            query += " AND wavelength_nm >= ?"
            params.append(wavelength_min)

        if wavelength_max is not None:
            query += " AND wavelength_nm <= ?"
            params.append(wavelength_max)

        if min_relative_intensity is not None:
            query += " AND rel_int >= ?"
            params.append(min_relative_intensity)

        query += " ORDER BY wavelength_nm"

        try:
            with self._get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
        except Exception as e:
            logger.error(f"Error querying transitions: {e}")
            return []

        transitions = []
        for _, row in df.iterrows():
            # Handle potential missing columns if something went wrong, defaulting to None
            stark_w = (
                float(row["stark_w"]) if "stark_w" in row and pd.notna(row["stark_w"]) else None
            )
            stark_alpha = (
                float(row["stark_alpha"])
                if "stark_alpha" in row and pd.notna(row["stark_alpha"])
                else None
            )
            stark_shift = (
                float(row["stark_shift"])
                if "stark_shift" in row and pd.notna(row["stark_shift"])
                else None
            )
            is_resonance = (
                bool(row["is_resonance"])
                if "is_resonance" in row and pd.notna(row["is_resonance"])
                else False
            )

            aki_uncertainty = (
                float(row["aki_uncertainty"]) if pd.notna(row["aki_uncertainty"]) else None
            )
            accuracy_grade = str(row["accuracy_grade"]) if pd.notna(row["accuracy_grade"]) else None

            trans = Transition(
                element=row["element"],
                ionization_stage=int(row["sp_num"]),
                wavelength_nm=float(row["wavelength_nm"]),
                A_ki=float(row["aki"]),
                E_k_ev=float(row["ek_ev"]),
                E_i_ev=(0.0 if pd.isna(row.get("ei_ev", 0.0)) else float(row.get("ei_ev", 0.0))),
                g_k=int(row["gk"]),
                g_i=1 if pd.isna(row.get("gi", 1)) else int(row.get("gi", 1)),
                relative_intensity=(
                    float(row.get("rel_int", 0.0)) if pd.notna(row.get("rel_int")) else None
                ),
                stark_w=stark_w,
                stark_alpha=stark_alpha,
                stark_shift=stark_shift,
                is_resonance=is_resonance,
                aki_uncertainty=aki_uncertainty,
                accuracy_grade=accuracy_grade,
            )
            transitions.append(trans)

        logger.debug(f"Retrieved {len(transitions)} transitions for {element}")
        return transitions

    def get_energy_levels(self, element: str, ionization_stage: int) -> list[EnergyLevel]:
        """
        Get energy levels for a species.

        Parameters
        ----------
        element : str
            Element symbol
        ionization_stage : int
            Ionization stage

        Returns
        -------
        list[EnergyLevel]
            list of energy level objects
        """
        query = """
            SELECT g_level, energy_ev
            FROM energy_levels
            WHERE element = ? AND sp_num = ?
            ORDER BY energy_ev
        """
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(element, ionization_stage))

        levels = []
        for _, row in df.iterrows():
            level = EnergyLevel(
                element=element,
                ionization_stage=ionization_stage,
                energy_ev=float(row["energy_ev"]),
                g=int(row["g_level"]),
            )
            levels.append(level)

        logger.debug(f"Retrieved {len(levels)} energy levels for {element} {ionization_stage}")
        return levels

    @cached_ionization
    def get_ionization_potential(self, element: str, ionization_stage: int) -> float | None:
        """
        Get ionization potential for a species.

        Parameters
        ----------
        element : str
            Element symbol
        ionization_stage : int
            Ionization stage (1=neutral, 2=singly ionized, etc.)

        Returns
        -------
        float or None
            Ionization potential in eV, or None if not found
        """
        query = """
            SELECT ip_ev
            FROM species_physics
            WHERE element = ? AND sp_num = ?
        """
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(query, (element, ionization_stage))
            res = cur.fetchone()
            return float(res[0]) if res else None

    def get_atomic_mass(self, element: str) -> float | None:
        """
        Get atomic mass for an element.

        Parameters
        ----------
        element : str
            Element symbol

        Returns
        -------
        float or None
            Atomic mass in amu, or None if not found
        """
        # Usually atomic mass is per element, not per species, but stored in species_physics which is (element, sp_num).
        # We can grab it from any sp_num or specifically sp_num=0 or 1.
        # Let's query any record for this element.
        query = """
            SELECT atomic_mass
            FROM species_physics
            WHERE element = ? AND atomic_mass IS NOT NULL
            LIMIT 1
        """
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(query, (element,))
            res = cur.fetchone()
            return float(res[0]) if res else None

    def get_partition_coefficients(
        self, element: str, ionization_stage: int
    ) -> PartitionFunction | None:
        """
        Get partition function coefficients for a species.

        Parameters
        ----------
        element : str
            Element symbol
        ionization_stage : int
            Ionization stage

        Returns
        -------
        PartitionFunction or None
        """
        query = """
            SELECT a0, a1, a2, a3, a4, t_min, t_max, source
            FROM partition_functions
            WHERE element = ? AND sp_num = ?
        """
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(query, (element, ionization_stage))
            res = cur.fetchone()

        if not res:
            return None

        return PartitionFunction(
            element=element,
            ionization_stage=ionization_stage,
            coefficients=[res[0], res[1], res[2], res[3], res[4]],
            t_min=res[5],
            t_max=res[6],
            source=res[7],
        )

    def partition_function_for(self, element: str, ionization_stage: int):
        """Return an encapsulated :class:`PartitionFunctionProvider` for a species.

        Vending the provider — instead of the raw :class:`PartitionFunction`
        dataclass — concentrates the ``[t_min, t_max]`` clamp and the ``g0``
        floor inside the provider's ``.at()`` method.  Every CF-LIBS call
        site that consumes this method picks up the extrapolation guard
        automatically; previously each site had to remember to thread
        ``t_min``/``t_max``/``g0`` through :func:`polynomial_partition_function`
        and half of them silently didn't.

        Returns ``None`` when no row exists for ``(element, stage)``; callers
        should fall back to direct-summation or a default provider in that
        case.  See ``docs/architecture/2026-05-26-architecture-review.md``
        § Candidate 4 and ADR-0001 § B-P7.
        """
        from cflibs.plasma.partition import (
            PolynomialPartitionFunctionProvider,
            get_ground_state_g,
        )

        pf = self.get_partition_coefficients(element, ionization_stage)
        if pf is None:
            return None
        coeffs = list(pf.coefficients)
        coeffs += [0.0] * (5 - len(coeffs))
        return PolynomialPartitionFunctionProvider(
            element=element,
            ionization_stage=ionization_stage,
            coefficients=tuple(float(c) for c in coeffs[:5]),
            t_min=float(pf.t_min),
            t_max=float(pf.t_max),
            _g0=float(get_ground_state_g(self, element, ionization_stage)),
            source=str(pf.source or ""),
        )

    def get_species_physics(self, element: str, ionization_stage: int) -> SpeciesPhysics | None:
        """
        Get physical properties for a species.

        Parameters
        ----------
        element : str
            Element symbol
        ionization_stage : int
            Ionization stage

        Returns
        -------
        SpeciesPhysics or None
            Species physics object, or None if not found
        """
        query = """
            SELECT ip_ev, atomic_mass
            FROM species_physics
            WHERE element = ? AND sp_num = ?
        """
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(query, (element, ionization_stage))
            res = cur.fetchone()

        if not res:
            return None

        ip_ev = float(res[0])
        atomic_mass = float(res[1]) if res[1] is not None else None

        return SpeciesPhysics(
            element=element,
            ionization_stage=ionization_stage,
            ionization_potential_ev=ip_ev,
            atomic_mass=atomic_mass,
        )

    def get_stark_parameters(
        self, element: str, ionization_stage: int, wavelength_nm: float, tolerance_nm: float = 0.01
    ) -> tuple[float | None, float | None, float | None]:
        """
        Get Stark broadening parameters for a specific line.

        Parameters
        ----------
        element : str
            Element symbol
        ionization_stage : int
            Ionization stage
        wavelength_nm : float
            Target wavelength
        tolerance_nm : float
            Wavelength matching tolerance

        Returns
        -------
        tuple[float, float, float]
            (stark_w, stark_alpha, stark_shift) or (None, None, None)
        """
        query = """
            SELECT stark_w, stark_alpha, stark_shift
            FROM lines
            WHERE element = ? AND sp_num = ?
            AND ABS(wavelength_nm - ?) < ?
            ORDER BY ABS(wavelength_nm - ?) ASC
            LIMIT 1
        """
        params = (element, ionization_stage, wavelength_nm, tolerance_nm, wavelength_nm)

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(query, params)
            res = cur.fetchone()

        if not res:
            return (None, None, None)

        stark_w = float(res[0]) if res[0] is not None else None
        stark_alpha = float(res[1]) if res[1] is not None else None
        stark_shift = float(res[2]) if res[2] is not None else None

        return (stark_w, stark_alpha, stark_shift)

    def get_available_elements(self) -> list[str]:
        """Get list of elements available in the database."""
        query = "SELECT DISTINCT element FROM lines ORDER BY element"
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn)
            return list(df["element"].astype(str))

    def snapshot(
        self,
        *,
        elements: list[str],
        wavelength_range: tuple[float, float],
        min_relative_intensity: float = 0.0,
        pad_to_n_elements: int | None = None,
        include_levels: bool = False,
    ) -> "AtomicSnapshot":
        """Build a frozen :class:`AtomicSnapshot` for jit consumption.

        Replaces ad-hoc dict-building done by ``SpectrumModel``,
        ``batch_forward``, and Bayesian callers. Mirrors exojax's
        ``MDBSnapshot`` construction (ADR-0001 §5.2 C-P10) and ports the
        per-line array packing logic from
        ``cflibs/manifold/batch_forward.py::pack_atomic_data``.

        Parameters
        ----------
        elements : list[str]
            Element symbols. Order defines the species axis.
        wavelength_range : (float, float)
            Lower/upper wavelength bounds in nm.
        min_relative_intensity : float, optional
            Minimum relative intensity threshold; default 0.0 keeps all
            lines that have a non-NULL ``rel_int`` entry.
        pad_to_n_elements : int, optional
            If provided, pad the species axis up to this size with zero
            rows. Used by callers that pre-allocate fixed-size arrays.
        include_levels : bool, optional
            When True, populate the optional ``level_g``, ``level_E_ev``,
            ``level_mask`` arrays for direct-sum partition consumers.
            Default False.

        Returns
        -------
        AtomicSnapshot
            JAX-pytree-registered snapshot ready to flow through jit'd
            kernels without holding a live SQLite connection inside a trace.
        """
        from cflibs.core.jax_runtime import (
            AtomicSnapshot,
            HAS_JAX,
            jax_default_real_dtype,
        )

        if HAS_JAX:
            import jax.numpy as _xp

            _real_dtype = jax_default_real_dtype()
        else:  # pragma: no cover - fallback path mirrors numpy semantics
            _xp = np
            _real_dtype = np.float64

        wl_min, wl_max = wavelength_range
        if wl_min >= wl_max:
            raise ValueError(
                f"wavelength_range must be (low, high) with low < high; got {wavelength_range!r}"
            )

        species_keys: list[tuple[str, int]] = []
        ip_list: list[float] = []
        partition_rows: list[list[float]] = []
        # Per-species t_min / t_max / g0 carried alongside the polynomial
        # coefficients for the BatchedPartitionFunctionProvider.  Default
        # window (2000–25000 K) matches the production fit range used in
        # ``populate_partition_functions.py``; default g0 = 1.0 is the
        # conservative physical lower bound (every quantum level has g >= 1).
        partition_t_min_list: list[float] = []
        partition_t_max_list: list[float] = []
        partition_g0_list: list[float] = []
        level_g_rows: list[list[float]] = []
        level_E_rows: list[list[float]] = []

        from cflibs.plasma.partition import get_ground_state_g

        for element in elements:
            for stage in (1, 2):
                ip = self.get_ionization_potential(element, stage)
                if ip is None:
                    continue
                species_keys.append((element, stage))
                ip_list.append(float(ip))
                pf = self.get_partition_coefficients(element, stage)
                if pf is None:
                    partition_rows.append([float(np.log(2.0)), 0.0, 0.0, 0.0, 0.0])
                    partition_t_min_list.append(2000.0)
                    partition_t_max_list.append(25000.0)
                else:
                    coeffs = list(pf.coefficients)
                    coeffs += [0.0] * (5 - len(coeffs))
                    partition_rows.append([float(c) for c in coeffs[:5]])
                    # SQLite t_min/t_max columns can be NULL when only the
                    # coefficients were imported. Fall back to the same
                    # default window used in the no-partition-row branch so
                    # the snapshot stays serializable and consumers always
                    # receive concrete bounds.
                    pf_t_min = pf.t_min if pf.t_min is not None else 2000.0
                    pf_t_max = pf.t_max if pf.t_max is not None else 25000.0
                    partition_t_min_list.append(float(pf_t_min))
                    partition_t_max_list.append(float(pf_t_max))
                partition_g0_list.append(float(get_ground_state_g(self, element, stage)))
                if include_levels:
                    levels = self.get_energy_levels(element, stage)
                    level_g_rows.append([float(lev.g) for lev in levels])
                    level_E_rows.append([float(lev.energy_ev) for lev in levels])

        if pad_to_n_elements is not None and pad_to_n_elements > len(species_keys):
            pad = pad_to_n_elements - len(species_keys)
            species_keys.extend([("", 0)] * pad)
            ip_list.extend([0.0] * pad)
            partition_rows.extend([[0.0] * 5 for _ in range(pad)])
            partition_t_min_list.extend([2000.0] * pad)
            partition_t_max_list.extend([25000.0] * pad)
            partition_g0_list.extend([1.0] * pad)
            if include_levels:
                level_g_rows.extend([[] for _ in range(pad)])
                level_E_rows.extend([[] for _ in range(pad)])

        species_to_idx = {key: i for i, key in enumerate(species_keys)}
        wls: list[float] = []
        Akis: list[float] = []
        Eks: list[float] = []
        gks: list[float] = []
        Eis: list[float] = []
        gis: list[float] = []
        sp_idx: list[int] = []
        stark_ws: list[float] = []
        stark_alphas: list[float] = []

        for element in elements:
            for transition in self.get_transitions(
                element=element,
                wavelength_min=wl_min,
                wavelength_max=wl_max,
                min_relative_intensity=(
                    min_relative_intensity if min_relative_intensity > 0 else None
                ),
            ):
                key = (transition.element, int(transition.ionization_stage))
                if key not in species_to_idx:
                    continue
                wls.append(float(transition.wavelength_nm))
                Akis.append(float(transition.A_ki))
                Eks.append(float(transition.E_k_ev))
                gks.append(float(transition.g_k))
                Eis.append(float(transition.E_i_ev))
                gis.append(float(transition.g_i))
                sp_idx.append(species_to_idx[key])
                stark_ws.append(
                    float(transition.stark_w) if transition.stark_w is not None else 0.0
                )
                # 0.0 default means factor_T = (T/T_ref)^0 = 1.0 — the legacy
                # T-independent behaviour for lines without alpha coverage.
                stark_alphas.append(
                    float(transition.stark_alpha) if transition.stark_alpha is not None else 0.0
                )

        n_lines = len(wls)
        line_wavelengths_nm = _xp.asarray(wls, dtype=_real_dtype)
        line_A_ki = _xp.asarray(Akis, dtype=_real_dtype)
        line_E_k_ev = _xp.asarray(Eks, dtype=_real_dtype)
        line_g_k = _xp.asarray(gks, dtype=_real_dtype)
        line_E_i_ev = _xp.asarray(Eis, dtype=_real_dtype)
        line_g_i = _xp.asarray(gis, dtype=_real_dtype)
        line_species_index = _xp.asarray(sp_idx, dtype=_xp.int32)
        line_stark_w = _xp.asarray(stark_ws, dtype=_real_dtype)
        line_stark_alpha = _xp.asarray(stark_alphas, dtype=_real_dtype)
        line_natural_w = _xp.zeros(n_lines, dtype=_real_dtype)

        if partition_rows:
            partition_coeffs = _xp.asarray(partition_rows, dtype=_real_dtype)
            partition_t_min = _xp.asarray(partition_t_min_list, dtype=_real_dtype)
            partition_t_max = _xp.asarray(partition_t_max_list, dtype=_real_dtype)
            partition_g0 = _xp.asarray(partition_g0_list, dtype=_real_dtype)
        else:
            partition_coeffs = _xp.zeros((0, 5), dtype=_real_dtype)
            partition_t_min = _xp.zeros((0,), dtype=_real_dtype)
            partition_t_max = _xp.zeros((0,), dtype=_real_dtype)
            partition_g0 = _xp.zeros((0,), dtype=_real_dtype)
        ionization_potential_ev = _xp.asarray(ip_list, dtype=_real_dtype)

        level_g_out: Any
        level_E_ev_out: Any
        level_mask_out: Any
        if include_levels and level_g_rows:
            n_max = max((len(row) for row in level_g_rows), default=0)
            n_cols = max(n_max, 1)
            level_g_padded = np.zeros((len(level_g_rows), n_cols), dtype=np.float64)
            level_E_padded = np.zeros_like(level_g_padded)
            level_mask = np.zeros_like(level_g_padded, dtype=bool)
            for i, (gs, Es) in enumerate(zip(level_g_rows, level_E_rows)):
                level_g_padded[i, : len(gs)] = gs
                level_E_padded[i, : len(Es)] = Es
                level_mask[i, : len(gs)] = True
            level_g_out = _xp.asarray(level_g_padded)
            level_E_ev_out = _xp.asarray(level_E_padded)
            level_mask_out = _xp.asarray(level_mask)
        else:
            level_g_out = None
            level_E_ev_out = None
            level_mask_out = None

        return AtomicSnapshot(
            species=tuple(species_keys),
            line_wavelengths_nm=line_wavelengths_nm,
            line_A_ki=line_A_ki,
            line_E_k_ev=line_E_k_ev,
            line_g_k=line_g_k,
            line_E_i_ev=line_E_i_ev,
            line_g_i=line_g_i,
            line_species_index=line_species_index,
            line_stark_w=line_stark_w,
            line_natural_w=line_natural_w,
            line_stark_alpha=line_stark_alpha,
            partition_coeffs=partition_coeffs,
            ionization_potential_ev=ionization_potential_ev,
            level_g=level_g_out,
            level_E_ev=level_E_ev_out,
            level_mask=level_mask_out,
            partition_t_min=partition_t_min,
            partition_t_max=partition_t_max,
            partition_g0=partition_g0,
        )

    def close(self):
        """Close database connection."""
        if self._use_pool:
            # Note: Pool is shared, so we don't close it here
            # Use close_all_pools() if needed
            logger.debug("Database connection pool reference released")
        else:
            self.conn.close()
            logger.debug("Database connection closed")

    def __getstate__(self):
        """Pickle support: exclude connection/pool."""
        state = self.__dict__.copy()
        # Remove unpicklable entries
        state.pop("_pool", None)
        state.pop("conn", None)
        return state

    def __setstate__(self, state):
        """Unpickle support: restore connection/pool."""
        self.__dict__.update(state)
        # Re-initialize connection/pool
        try:
            self._pool = get_pool(str(self.db_path), max_connections=5)
            self._use_pool = True
        except Exception:
            self.conn = sqlite3.connect(str(self.db_path))
            self._use_pool = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Pool is managed globally, no cleanup needed
        pass
