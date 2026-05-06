"""
Atomic database interface for loading and querying atomic data.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

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

        # 8. Add energy_uncertainty_ev and apply Ding 2024 Fe II refresh
        cursor.execute("PRAGMA table_info(lines)")
        line_cols = {row[1] for row in cursor.fetchall()}
        if "energy_uncertainty_ev" not in line_cols:
            logger.info("Migrating schema: Adding energy_uncertainty_ev to atomic tables")
            cursor.execute("ALTER TABLE lines ADD COLUMN energy_uncertainty_ev REAL")
            cursor.execute("ALTER TABLE energy_levels ADD COLUMN energy_uncertainty_ev REAL")
            # Default uncertainty for existing levels (1 meV)
            cursor.execute("UPDATE lines SET energy_uncertainty_ev = 0.001")
            cursor.execute("UPDATE energy_levels SET energy_uncertainty_ev = 0.001")

        # Apply Ding 2024 improvements (10x lower uncertainty for Fe II 4f/5d)
        logger.info("Applying Ding 2024 Fe II atomic data refresh")
        cursor.execute("""
            UPDATE lines 
            SET accuracy_grade = 'A', aki_uncertainty = 0.03, energy_uncertainty_ev = 0.0000012
            WHERE element = 'Fe' AND sp_num = 2 AND ek_ev BETWEEN 10.5 AND 12.0
        """)
        cursor.execute("""
            UPDATE energy_levels
            SET energy_uncertainty_ev = 0.0000012
            WHERE element = 'Fe' AND sp_num = 2 AND energy_ev BETWEEN 10.5 AND 12.0
        """)

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
        """Assign aki_uncertainty using heuristic based on relative intensity.

        Since our database was generated from NIST ASD without explicit accuracy
        grades, we assign uncertainty estimates based on line properties:
        - Lines with high relative intensity (>100) and from well-studied
          elements get lower uncertainty (~5-10%, grade B)
        - Lines with low relative intensity get higher uncertainty (~25%, grade C)
        - Lines with no relative intensity get 50% uncertainty (grade D)

        When explicit NIST accuracy grades are available (via future ingestion),
        those should override these heuristics.
        """
        from cflibs.atomic.reference_data import NIST_GRADE_UNCERTAINTY

        unc_b = NIST_GRADE_UNCERTAINTY["B"]
        unc_c = NIST_GRADE_UNCERTAINTY["C"]
        unc_d = NIST_GRADE_UNCERTAINTY["D"]
        unc_e = NIST_GRADE_UNCERTAINTY["E"]

        # Heuristic: assign based on relative intensity as a proxy for data quality
        # rel_int > 100 → well-measured lines (B grade)
        cursor.execute(
            f"UPDATE lines SET aki_uncertainty = {unc_b}, accuracy_grade = 'B' "
            "WHERE aki IS NOT NULL AND rel_int IS NOT NULL AND rel_int > 100"
        )
        n_b = cursor.rowcount

        # rel_int 10-100 → moderate quality (C grade)
        cursor.execute(
            f"UPDATE lines SET aki_uncertainty = {unc_c}, accuracy_grade = 'C' "
            "WHERE aki IS NOT NULL AND rel_int IS NOT NULL AND rel_int > 10 "
            "AND aki_uncertainty IS NULL"
        )
        n_c = cursor.rowcount

        # rel_int 1-10 → weaker lines (D grade)
        cursor.execute(
            f"UPDATE lines SET aki_uncertainty = {unc_d}, accuracy_grade = 'D' "
            "WHERE aki IS NOT NULL AND rel_int IS NOT NULL AND rel_int >= 1 "
            "AND aki_uncertainty IS NULL"
        )
        n_d = cursor.rowcount

        # Everything else: worst case (E grade)
        cursor.execute(
            f"UPDATE lines SET aki_uncertainty = {unc_e}, accuracy_grade = 'E' "
            "WHERE aki IS NOT NULL AND aki_uncertainty IS NULL"
        )
        n_e = cursor.rowcount

        logger.info(
            "Assigned aki_uncertainty: B=%d, C=%d, D=%d, E=%d lines",
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
                aki_uncertainty, accuracy_grade, energy_uncertainty_ev
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
            # Propagate energy uncertainty as an extra attribute if present
            if "energy_uncertainty_ev" in row and pd.notna(row["energy_uncertainty_ev"]):
                setattr(trans, "energy_uncertainty_ev", float(row["energy_uncertainty_ev"]))
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
            SELECT g_level, energy_ev, energy_uncertainty_ev
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
            if "energy_uncertainty_ev" in row and pd.notna(row["energy_uncertainty_ev"]):
                setattr(level, "energy_uncertainty_ev", float(row["energy_uncertainty_ev"]))
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
