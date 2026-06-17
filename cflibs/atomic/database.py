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
from cflibs.core.cache import cached_transitions, cached_ionization
from cflibs.core.pool import get_pool

logger = get_logger("atomic.database")


class AtomicDatabase:
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

        self._migrate_lines_columns(cursor)
        self._migrate_partition_functions_table(cursor)
        self._migrate_species_physics_columns(cursor)
        self._populate_energy_levels_if_empty(cursor)
        self._populate_species_physics_if_empty(cursor)
        self._populate_partition_functions_if_empty(cursor)
        self._backfill_aki_uncertainties_if_missing(cursor)

        conn.commit()

    @staticmethod
    def _migrate_lines_columns(cursor: sqlite3.Cursor):
        """Step 1: add any missing Stark/uncertainty columns to the lines table."""
        cursor.execute("PRAGMA table_info(lines)")
        columns = {row[1] for row in cursor.fetchall()}

        required_line_cols = {
            "stark_w": "REAL",
            "stark_alpha": "REAL",
            "stark_shift": "REAL",
            "is_resonance": "INTEGER",
            "aki_uncertainty": "REAL",
            "accuracy_grade": "TEXT",
            # Provenance of stark_w (e.g. 'stark_b' literature-grade vs
            # 'konjevic_lambda_sq_scaled' heuristic). NULL = unknown, which the
            # Stark n_e diagnostic treats as not literature-grade.
            "stark_w_source": "TEXT",
        }

        valid_dtypes = {"REAL", "INTEGER", "TEXT", "BLOB", "NUMERIC"}
        for col, dtype in required_line_cols.items():
            if col not in columns:
                AtomicDatabase._add_line_column(cursor, col, dtype, valid_dtypes)

    @staticmethod
    def _add_line_column(cursor: sqlite3.Cursor, col: str, dtype: str, valid_dtypes: set[str]):
        """Validate and add a single column to the lines table, backfilling as needed."""
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
            cursor.execute("UPDATE lines SET is_resonance = 0 WHERE ei_ev >= 0.01 OR ei_ev IS NULL")

    @staticmethod
    def _migrate_partition_functions_table(cursor: sqlite3.Cursor):
        """Step 2: create the partition_functions table if it does not exist."""
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

    @staticmethod
    def _migrate_species_physics_columns(cursor: sqlite3.Cursor):
        """Step 3: add the atomic_mass column to species_physics if missing."""
        cursor.execute("PRAGMA table_info(species_physics)")
        physics_columns = {row[1] for row in cursor.fetchall()}

        if "atomic_mass" not in physics_columns:
            logger.info("Migrating schema: Adding atomic_mass to species_physics table")
            cursor.execute("ALTER TABLE species_physics ADD COLUMN atomic_mass REAL")

    def _populate_energy_levels_if_empty(self, cursor: sqlite3.Cursor):
        """Step 4: auto-populate energy_levels from the lines table when empty."""
        cursor.execute("SELECT COUNT(*) FROM energy_levels")
        if cursor.fetchone()[0] == 0:
            cursor.execute("SELECT COUNT(*) FROM lines")
            n_lines = cursor.fetchone()[0]
            if n_lines > 0:
                logger.info("Populating energy_levels from lines table...")
                self._populate_energy_levels(cursor)

    def _populate_species_physics_if_empty(self, cursor: sqlite3.Cursor):
        """Step 5: auto-populate species_physics with NIST IPs when empty."""
        cursor.execute("SELECT COUNT(*) FROM species_physics")
        if cursor.fetchone()[0] == 0:
            logger.info("Populating species_physics with NIST ionization potentials...")
            self._populate_species_physics(cursor)

    def _populate_partition_functions_if_empty(self, cursor: sqlite3.Cursor):
        """Step 6: auto-populate partition_functions with Irwin coefficients when empty."""
        cursor.execute("SELECT COUNT(*) FROM partition_functions")
        if cursor.fetchone()[0] == 0:
            logger.info("Populating partition_functions with NIST Irwin coefficients...")
            self._populate_partition_functions(cursor)

    def _backfill_aki_uncertainties_if_missing(self, cursor: sqlite3.Cursor):
        """Step 7: backfill aki_uncertainty from accuracy_grade or heuristic where missing."""
        cursor.execute(
            "SELECT COUNT(*) FROM lines WHERE aki IS NOT NULL AND aki_uncertainty IS NULL"
        )
        n_missing = cursor.fetchone()[0]
        if n_missing > 0:
            logger.info("Backfilling %d lines missing aki_uncertainty...", n_missing)
            self._populate_aki_uncertainties(cursor)

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
            Minimum relative intensity (``rel_int``) threshold. WARNING: this is
            an *absolute* SQL floor on the NIST ``rel_int`` column, which is
            incomparable across elements/ion stages and NULL for many lines —
            a fixed floor silently deletes whole real elements whose tabulated
            rel_int is small or absent. For line detection prefer the
            element-relative top-K selection in
            ``cflibs.inversion.identify.line_detection._load_transitions``
            (``top_k_per_element``) and leave this ``None``.

        Returns
        -------
        list[Transition]
            list of transition objects
        """
        query, params = self._build_transitions_query(
            element,
            ionization_stage,
            wavelength_min,
            wavelength_max,
            min_relative_intensity,
        )

        try:
            with self._get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
        except Exception as e:
            logger.error(f"Error querying transitions: {e}")
            return []

        transitions = [self._row_to_transition(row) for _, row in df.iterrows()]

        logger.debug(f"Retrieved {len(transitions)} transitions for {element}")
        return transitions

    @staticmethod
    def _build_transitions_query(
        element: str,
        ionization_stage: int | None,
        wavelength_min: float | None,
        wavelength_max: float | None,
        min_relative_intensity: float | None,
    ) -> tuple[str, list[object]]:
        """Build the SQL query and bound parameters for :meth:`get_transitions`."""
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
        return query, params

    @staticmethod
    def _row_to_transition(row: "pd.Series") -> Transition:
        """Convert a single ``lines`` query row into a :class:`Transition`."""
        # Handle potential missing columns if something went wrong, defaulting to None
        stark_w = float(row["stark_w"]) if "stark_w" in row and pd.notna(row["stark_w"]) else None
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

        return Transition(
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

    def partition_spec_for(self, element: str, ionization_stage: int):
        """Return the single :class:`PartitionFunctionSpec` for a species.

        THE one source of partition-function coefficients + bounds + ``g0`` for
        this codebase.  Applies the locked policy in one place — *prefer the
        direct-sum FIT over energy levels when the species has tabulated levels;
        otherwise the stored polynomial fallback; always carry ``[t_min,
        t_max]`` and ``g0``* — and caches the (expensive) per-species fit so it
        runs compute-once (see
        :func:`cflibs.plasma.partition.derive_partition_spec`).

        Both adapters are built from this spec:

        * the CPU scalar adapter via :meth:`partition_function_for` (which calls
          ``spec.to_provider()``), and
        * the JAX batched adapter, whose static snapshot arrays bake the spec's
          ``coefficients`` / ``t_min`` / ``t_max`` / ``g0`` as plain floats (see
          :meth:`snapshot`).

        Returns ``None`` when the species has neither energy levels nor a stored
        polynomial row; callers decide the default.
        """
        from cflibs.plasma.partition import derive_partition_spec

        return derive_partition_spec(self, element, ionization_stage)

    def partition_function_for(self, element: str, ionization_stage: int):
        """Return an encapsulated :class:`PartitionFunctionProvider` for a species.

        Vending the provider — instead of the raw :class:`PartitionFunction`
        dataclass — concentrates the ``[t_min, t_max]`` clamp and the ``g0``
        floor inside the provider's ``.at()`` method.  Every CF-LIBS call
        site that consumes this method picks up the extrapolation guard
        automatically; previously each site had to remember to thread
        ``t_min``/``t_max``/``g0`` through :func:`polynomial_partition_function`
        and half of them silently didn't.

        The coefficients now come from :meth:`partition_spec_for`, i.e. the
        DIRECT-SUM FIT over energy levels when the species has tabulated levels
        (the reference the validation gate trusts), falling back to the stored
        polynomial only when levels are absent.  This makes the CPU scalar path
        consume the *same* coefficients the JAX batched adapter bakes into its
        snapshot (the PF-3/PF-4 unification, diagnosis § 2.1).

        Returns ``None`` when no levels and no stored row exist for
        ``(element, stage)``; callers should fall back to a default provider in
        that case.  See ``docs/architecture/2026-06-03-composition-pipeline-
        diagnosis.md`` § 2.1, ``2026-05-26-architecture-review.md`` § Candidate 4
        and ADR-0001 § B-P7.
        """
        spec = self.partition_spec_for(element, ionization_stage)
        if spec is None:
            return None
        return spec.to_provider()

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

    def get_stark_parameters_with_source(
        self, element: str, ionization_stage: int, wavelength_nm: float, tolerance_nm: float = 0.01
    ) -> tuple[float | None, float | None, str | None, bool | None]:
        """
        Get Stark width, alpha, the width's provenance and the resonance flag.

        The ``stark_w_source`` column records how the stored width was obtained
        (e.g. ``'stark_b'`` = literature-grade Stark-B / Sahal-Brechot tables;
        ``'konjevic_lambda_sq_scaled'`` = lambda^2-scaled heuristic). The
        Stark-broadening n_e diagnostic gates on this provenance: only
        literature-grade widths are trustworthy enough to *measure* n_e
        (Konjevic et al. 2002, J. Phys. Chem. Ref. Data 31, 819; Gigosos 2014,
        J. Phys. D 47, 343001).

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
        tuple[float | None, float | None, str | None, bool | None]
            (stark_w, stark_alpha, stark_w_source, is_resonance) or
            (None, None, None, None). ``stark_w`` is the electron-impact FWHM
            at n_e = 1e17 cm^-3, T = 10000 K (project-wide convention, see
            cflibs/radiation/stark.py). ``is_resonance`` lets the n_e
            diagnostic down-rank optically-thick-prone resonance lines whose
            self-absorption inflates the apparent width (biasing n_e high).
        """
        query = """
            SELECT stark_w, stark_alpha, stark_w_source, is_resonance
            FROM lines
            WHERE element = ? AND sp_num = ?
            AND ABS(wavelength_nm - ?) < ?
            ORDER BY ABS(wavelength_nm - ?) ASC
            LIMIT 1
        """
        params = (element, ionization_stage, wavelength_nm, tolerance_nm, wavelength_nm)

        try:
            with self._get_connection() as conn:
                cur = conn.cursor()
                cur.execute(query, params)
                res = cur.fetchone()
        except sqlite3.OperationalError:
            # Pre-provenance schema (no stark_w_source column): treat every
            # width as provenance-unknown so the diagnostic gates it out.
            return (None, None, None, None)

        if not res:
            return (None, None, None, None)

        stark_w = float(res[0]) if res[0] is not None else None
        stark_alpha = float(res[1]) if res[1] is not None else None
        stark_w_source = str(res[2]) if res[2] is not None else None
        is_resonance = bool(res[3]) if res[3] is not None else None

        return (stark_w, stark_alpha, stark_w_source, is_resonance)

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

        (
            species_keys,
            ip_list,
            partition_rows,
            partition_t_min_list,
            partition_t_max_list,
            partition_g0_list,
            level_g_rows,
            level_E_rows,
            partition_rows_iii,
            partition_t_min_iii_list,
            partition_t_max_iii_list,
            partition_g0_iii_list,
            from_direct_sum_list,
        ) = self._collect_species_partitions(elements, pad_to_n_elements, include_levels)

        species_to_idx = {key: i for i, key in enumerate(species_keys)}
        (
            wls,
            Akis,
            Eks,
            gks,
            Eis,
            gis,
            sp_idx,
            stark_ws,
            stark_alphas,
            stark_ds,
        ) = self._collect_line_arrays(
            elements, wl_min, wl_max, min_relative_intensity, species_to_idx
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
        line_stark_d = _xp.asarray(stark_ds, dtype=_real_dtype)
        line_natural_w = _xp.zeros(n_lines, dtype=_real_dtype)

        if partition_rows:
            partition_coeffs = _xp.asarray(partition_rows, dtype=_real_dtype)
            partition_t_min = _xp.asarray(partition_t_min_list, dtype=_real_dtype)
            partition_t_max = _xp.asarray(partition_t_max_list, dtype=_real_dtype)
            partition_g0 = _xp.asarray(partition_g0_list, dtype=_real_dtype)
            partition_coeffs_iii = _xp.asarray(partition_rows_iii, dtype=_real_dtype)
            partition_t_min_iii = _xp.asarray(partition_t_min_iii_list, dtype=_real_dtype)
            partition_t_max_iii = _xp.asarray(partition_t_max_iii_list, dtype=_real_dtype)
            partition_g0_iii = _xp.asarray(partition_g0_iii_list, dtype=_real_dtype)
            partition_from_direct_sum = _xp.asarray(from_direct_sum_list, dtype=_real_dtype)
        else:
            partition_coeffs = _xp.zeros((0, 5), dtype=_real_dtype)
            partition_t_min = _xp.zeros((0,), dtype=_real_dtype)
            partition_t_max = _xp.zeros((0,), dtype=_real_dtype)
            partition_g0 = _xp.zeros((0,), dtype=_real_dtype)
            partition_coeffs_iii = _xp.zeros((0, 5), dtype=_real_dtype)
            partition_t_min_iii = _xp.zeros((0,), dtype=_real_dtype)
            partition_t_max_iii = _xp.zeros((0,), dtype=_real_dtype)
            partition_g0_iii = _xp.zeros((0,), dtype=_real_dtype)
            partition_from_direct_sum = _xp.zeros((0,), dtype=_real_dtype)
        ionization_potential_ev = _xp.asarray(ip_list, dtype=_real_dtype)

        level_g_out, level_E_ev_out, level_mask_out = self._pad_level_arrays(
            level_g_rows, level_E_rows, include_levels, _xp
        )

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
            line_stark_d=line_stark_d,
            partition_coeffs=partition_coeffs,
            ionization_potential_ev=ionization_potential_ev,
            level_g=level_g_out,
            level_E_ev=level_E_ev_out,
            level_mask=level_mask_out,
            partition_t_min=partition_t_min,
            partition_t_max=partition_t_max,
            partition_g0=partition_g0,
            partition_coeffs_iii=partition_coeffs_iii,
            partition_t_min_iii=partition_t_min_iii,
            partition_t_max_iii=partition_t_max_iii,
            partition_g0_iii=partition_g0_iii,
            partition_from_direct_sum=partition_from_direct_sum,
        )

    def _collect_species_partitions(
        self,
        elements: list[str],
        pad_to_n_elements: int | None,
        include_levels: bool,
    ) -> tuple[
        list[tuple[str, int]],
        list[float],
        list[list[float]],
        list[float],
        list[float],
        list[float],
        list[list[float]],
        list[list[float]],
        list[list[float]],
        list[float],
        list[float],
        list[float],
        list[float],
    ]:
        """Gather per-species ionization/partition data for :meth:`snapshot`.

        Builds, in species-axis order, the keys, ionization potentials,
        partition-polynomial rows (with per-species ``t_min``/``t_max``/``g0``),
        and optional energy-level rows, then applies the optional zero-padding
        up to ``pad_to_n_elements``.

        Stage-III extension (bead CF-LIBS-improved-rs7e): every species row
        also carries its ELEMENT's stage-III partition spec (duplicated across
        the element's stage-I/II rows) plus a ``from_direct_sum`` flag for the
        row's own spec, so the three-stage Saha kernel can gather both with
        the species-axis indices it already uses.
        """
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
        partition_rows_iii: list[list[float]] = []
        partition_t_min_iii_list: list[float] = []
        partition_t_max_iii_list: list[float] = []
        partition_g0_iii_list: list[float] = []
        from_direct_sum_list: list[float] = []

        for element in elements:
            for stage in (1, 2):
                ip = self.get_ionization_potential(element, stage)
                if ip is None:
                    continue
                species_keys.append((element, stage))
                ip_list.append(float(ip))

                self._append_species_partition_row(
                    element,
                    stage,
                    partition_rows,
                    partition_t_min_list,
                    partition_t_max_list,
                    partition_g0_list,
                    from_direct_sum_list=from_direct_sum_list,
                )
                # Element's stage-III spec on every species row (the
                # ``derive_partition_spec`` cache makes the second per-element
                # call a hit, so this stays one fit per (element, 3)).
                self._append_species_partition_row(
                    element,
                    3,
                    partition_rows_iii,
                    partition_t_min_iii_list,
                    partition_t_max_iii_list,
                    partition_g0_iii_list,
                )
                if include_levels:
                    stage_levels = self.get_energy_levels(element, stage)
                    level_g_rows.append([float(lev.g) for lev in stage_levels])
                    level_E_rows.append([float(lev.energy_ev) for lev in stage_levels])

        self._pad_species_partitions(
            pad_to_n_elements,
            include_levels,
            species_keys,
            ip_list,
            partition_rows,
            partition_t_min_list,
            partition_t_max_list,
            partition_g0_list,
            level_g_rows,
            level_E_rows,
            partition_rows_iii,
            partition_t_min_iii_list,
            partition_t_max_iii_list,
            partition_g0_iii_list,
            from_direct_sum_list,
        )

        return (
            species_keys,
            ip_list,
            partition_rows,
            partition_t_min_list,
            partition_t_max_list,
            partition_g0_list,
            level_g_rows,
            level_E_rows,
            partition_rows_iii,
            partition_t_min_iii_list,
            partition_t_max_iii_list,
            partition_g0_iii_list,
            from_direct_sum_list,
        )

    def _append_species_partition_row(
        self,
        element: str,
        stage: int,
        partition_rows: list[list[float]],
        partition_t_min_list: list[float],
        partition_t_max_list: list[float],
        partition_g0_list: list[float],
        from_direct_sum_list: list[float] | None = None,
    ) -> None:
        """Append one species' partition-polynomial row for :meth:`snapshot`.

        Direct-sum-preferred partition coefficients (PF-3/PF-4 fix), via the
        SINGLE factory ``partition_spec_for``.  The JAX manifold / Bayesian
        kernels can only consume a polynomial (they vmap over plasma parameters
        and cannot hold a per-species variable-length level sum), so "prefer
        direct-sum" is a BUILD-TIME choice here: the factory fits the direct-sum
        to an ln-polynomial when energy levels exist (cached, compute-once), and
        only falls back to the stored polynomial when the level table is missing
        rows.  Baking the SAME spec the CPU scalar adapter uses
        (``partition_function_for``) into these static arrays is what makes the
        two adapters provably agree.  The kernel still evaluates the same guarded
        polynomial form, so jit / vmap are unaffected.
        """
        spec = self.partition_spec_for(element, stage)
        if spec is not None:
            partition_rows.append([float(c) for c in spec.coefficients])
            partition_t_min_list.append(float(spec.t_min))
            partition_t_max_list.append(float(spec.t_max))
            partition_g0_list.append(float(spec.g0))
            if from_direct_sum_list is not None:
                from_direct_sum_list.append(1.0 if spec.from_direct_sum else 0.0)
        else:
            # Neither energy levels nor a stored polynomial row: bake the
            # canonical fallback ladder's constant (closed-shell ions get
            # their exact U, e.g. Na II -> 1.0 instead of the old U = 2
            # placeholder) so the snapshot stays serializable and consumers
            # always receive concrete bounds.
            from cflibs.plasma.partition import canonical_partition_fallback

            u_fallback = canonical_partition_fallback(element, stage, self)
            partition_rows.append([float(np.log(u_fallback)), 0.0, 0.0, 0.0, 0.0])
            partition_t_min_list.append(2000.0)
            partition_t_max_list.append(25000.0)
            partition_g0_list.append(1.0)
            if from_direct_sum_list is not None:
                from_direct_sum_list.append(0.0)

    def _pad_species_partitions(
        self,
        pad_to_n_elements: int | None,
        include_levels: bool,
        species_keys: list[tuple[str, int]],
        ip_list: list[float],
        partition_rows: list[list[float]],
        partition_t_min_list: list[float],
        partition_t_max_list: list[float],
        partition_g0_list: list[float],
        level_g_rows: list[list[float]],
        level_E_rows: list[list[float]],
        partition_rows_iii: list[list[float]],
        partition_t_min_iii_list: list[float],
        partition_t_max_iii_list: list[float],
        partition_g0_iii_list: list[float],
        from_direct_sum_list: list[float],
    ) -> None:
        """Zero-pad the species-axis accumulators up to ``pad_to_n_elements``.

        Mutates the supplied lists in place; no-op when padding is not requested
        or the species axis already meets the target size.
        """
        if pad_to_n_elements is not None and pad_to_n_elements > len(species_keys):
            pad = pad_to_n_elements - len(species_keys)
            species_keys.extend([("", 0)] * pad)
            ip_list.extend([0.0] * pad)
            partition_rows.extend([[0.0] * 5 for _ in range(pad)])
            partition_t_min_list.extend([2000.0] * pad)
            partition_t_max_list.extend([25000.0] * pad)
            partition_g0_list.extend([1.0] * pad)
            partition_rows_iii.extend([[0.0] * 5 for _ in range(pad)])
            partition_t_min_iii_list.extend([2000.0] * pad)
            partition_t_max_iii_list.extend([25000.0] * pad)
            partition_g0_iii_list.extend([1.0] * pad)
            from_direct_sum_list.extend([0.0] * pad)
            if include_levels:
                level_g_rows.extend([[] for _ in range(pad)])
                level_E_rows.extend([[] for _ in range(pad)])

    def _collect_line_arrays(
        self,
        elements: list[str],
        wl_min: float,
        wl_max: float,
        min_relative_intensity: float,
        species_to_idx: dict[tuple[str, int], int],
    ) -> tuple[
        list[float],
        list[float],
        list[float],
        list[float],
        list[float],
        list[float],
        list[int],
        list[float],
        list[float],
        list[float],
    ]:
        """Pack the per-line column arrays for :meth:`snapshot`.

        Iterates transitions for every requested element within the wavelength
        window, dropping lines whose species is not in ``species_to_idx``, and
        returns the parallel per-line lists in line-axis order.
        """
        wls: list[float] = []
        Akis: list[float] = []
        Eks: list[float] = []
        gks: list[float] = []
        Eis: list[float] = []
        gis: list[float] = []
        sp_idx: list[int] = []
        stark_ws: list[float] = []
        stark_alphas: list[float] = []
        stark_ds: list[float] = []

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
                # Signed Stark shift of the line center at REF_NE = 1e17 cm^-3
                # (nm). Many lines have no catalogued shift — default 0.0 so
                # the forward kernel leaves them unmoved (no failure).
                # TODO: backfill ``lines.stark_shift`` from a published Stark
                # database (e.g. Stark-B / NIST) for elements where shift data
                # is missing; until then those lines carry a 0.0 shift.
                stark_ds.append(
                    float(transition.stark_shift) if transition.stark_shift is not None else 0.0
                )

        return (
            wls,
            Akis,
            Eks,
            gks,
            Eis,
            gis,
            sp_idx,
            stark_ws,
            stark_alphas,
            stark_ds,
        )

    @staticmethod
    def _pad_level_arrays(
        level_g_rows: list[list[float]],
        level_E_rows: list[list[float]],
        include_levels: bool,
        _xp: Any,
    ) -> tuple[Any, Any, Any]:
        """Pad ragged per-species energy-level rows into rectangular arrays.

        Returns ``(level_g, level_E_ev, level_mask)`` ready for the snapshot, or
        ``(None, None, None)`` when levels were not requested or none exist.
        """
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
            return (
                _xp.asarray(level_g_padded),
                _xp.asarray(level_E_padded),
                _xp.asarray(level_mask),
            )
        return (None, None, None)

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
