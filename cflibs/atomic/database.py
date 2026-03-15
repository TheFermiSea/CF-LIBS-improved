"""
Atomic database interface for loading and querying atomic data.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
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

    db_path: Union[str, Path]

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
        }

        for col, dtype in required_line_cols.items():
            if col not in columns:
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
        nist_data = [
            ("H", 1.008, 13.598, None), ("He", 4.003, 24.587, 54.418),
            ("Li", 6.94, 5.392, 75.640), ("Be", 9.012, 9.323, 18.211),
            ("B", 10.81, 8.298, 25.155), ("C", 12.011, 11.260, 24.383),
            ("N", 14.007, 14.534, 29.601), ("O", 15.999, 13.618, 35.117),
            ("F", 18.998, 17.422, 34.970), ("Ne", 20.180, 21.564, 40.962),
            ("Na", 22.990, 5.139, 47.286), ("Mg", 24.305, 7.646, 15.035),
            ("Al", 26.982, 5.986, 18.828), ("Si", 28.085, 8.151, 16.345),
            ("P", 30.974, 10.486, 19.769), ("S", 32.06, 10.360, 23.337),
            ("Cl", 35.45, 12.967, 23.814), ("Ar", 39.948, 15.759, 27.629),
            ("K", 39.098, 4.341, 31.63), ("Ca", 40.078, 6.113, 11.871),
            ("Sc", 44.956, 6.561, 12.800), ("Ti", 47.867, 6.828, 13.575),
            ("V", 50.942, 6.746, 14.618), ("Cr", 51.996, 6.766, 16.498),
            ("Mn", 54.938, 7.434, 15.640), ("Fe", 55.845, 7.902, 16.187),
            ("Co", 58.933, 7.881, 17.083), ("Ni", 58.693, 7.639, 18.168),
            ("Cu", 63.546, 7.726, 20.292), ("Zn", 65.38, 9.394, 17.964),
            ("Ga", 69.723, 5.999, 20.514), ("Ge", 72.63, 7.899, 15.934),
            ("As", 74.922, 9.788, 18.633), ("Se", 78.971, 9.752, 21.19),
            ("Br", 79.904, 11.814, 21.8), ("Kr", 83.798, 13.999, 24.359),
            ("Rb", 85.468, 4.177, 27.285), ("Sr", 87.62, 5.695, 11.030),
            ("Y", 88.906, 6.217, 12.24), ("Zr", 91.224, 6.634, 13.13),
            ("Nb", 92.906, 6.759, 14.32), ("Mo", 95.95, 7.092, 16.16),
            ("Ru", 101.07, 7.361, 16.76), ("Rh", 102.91, 7.459, 18.08),
            ("Pd", 106.42, 8.337, 19.43), ("Ag", 107.87, 7.576, 21.49),
            ("Cd", 112.41, 8.993, 16.908), ("In", 114.82, 5.786, 18.869),
            ("Sn", 118.71, 7.344, 14.632), ("Sb", 121.76, 8.608, 16.53),
            ("Te", 127.60, 9.010, 18.60), ("I", 126.90, 10.451, 19.131),
            ("Xe", 131.29, 12.130, 20.975), ("Cs", 132.91, 3.894, 23.157),
            ("Ba", 137.33, 5.211, 10.004), ("La", 138.91, 5.577, 11.06),
            ("Ce", 140.12, 5.539, 10.85), ("Pr", 140.91, 5.473, 10.55),
            ("Nd", 144.24, 5.525, 10.72), ("Sm", 150.36, 5.643, 11.07),
            ("Eu", 151.96, 5.670, 11.241), ("Gd", 157.25, 6.150, 12.09),
            ("Tb", 158.93, 5.864, 11.52), ("Dy", 162.50, 5.939, 11.67),
            ("Ho", 164.93, 6.022, 11.80), ("Er", 167.26, 6.108, 11.93),
            ("Tm", 168.93, 6.184, 12.05), ("Yb", 173.05, 6.254, 12.176),
            ("Lu", 174.97, 5.426, 13.9), ("Hf", 178.49, 6.825, 14.9),
            ("Ta", 180.95, 7.549, 16.2), ("W", 183.84, 7.864, 16.1),
            ("Re", 186.21, 7.834, 16.6), ("Os", 190.23, 8.438, 17.0),
            ("Ir", 192.22, 8.967, 17.0), ("Pt", 195.08, 8.959, 18.563),
            ("Au", 196.97, 9.225, 20.5), ("Hg", 200.59, 10.437, 18.756),
            ("Tl", 204.38, 6.108, 20.428), ("Pb", 207.2, 7.416, 15.032),
            ("Bi", 208.98, 7.285, 16.69), ("Th", 232.04, 6.307, 11.5),
            ("U", 238.03, 6.194, 11.6),
        ]
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
        """Populate partition_functions with NIST-fitted Irwin polynomial coefficients.

        Coefficients fit ln(U) = sum(a_n * (ln T)^n), valid for T in [2000, 20000] K.
        Fitted from NIST ASD reference data with weighted least-squares emphasizing
        standard test temperatures (5000, 10000, 15000, 20000 K).
        Max relative error at key temperatures < 1% for all species.
        """
        # (element, sp_num, a0, a1, a2, a3, a4)
        # fmt: off
        nist_coefficients = [
            ("Al", 1,  1.72512062e+02, -7.96857411e+01,  1.39233241e+01, -1.07976298e+00,  3.13656271e-02),
            ("Al", 2,  3.92811672e+01, -1.88181146e+01,  3.37529329e+00, -2.68643989e-01,  8.00543990e-03),
            ("Cr", 1, -3.85560335e+03,  1.76236158e+03, -3.00569822e+02,  2.26740256e+01, -6.38121124e-01),
            ("Cr", 2, -1.29576764e+03,  6.05999478e+02, -1.05541464e+02,  8.11796155e+00, -2.32464318e-01),
            ("Cu", 1, -1.74984786e+02,  9.31566745e+01, -1.78694923e+01,  1.47397097e+00, -4.41140492e-02),
            ("Cu", 2, -4.74717257e+02,  2.00115917e+02, -3.09942034e+01,  2.07773945e+00, -5.03685874e-02),
            ("Fe", 1,  5.77562346e+02, -2.56294039e+02,  4.29128628e+01, -3.20333473e+00,  9.02181834e-02),
            ("Fe", 2,  6.41711030e+02, -2.87977905e+02,  4.85867335e+01, -3.63854155e+00,  1.02228113e-01),
            ("Fe", 3,  9.71363722e+00, -1.14939269e+00, -3.38885852e-02,  1.00066643e-02, -9.04068027e-05),
            ("Ni", 1, -1.59560549e+03,  7.29251309e+02, -1.24121200e+02,  9.33687258e+00, -2.61705772e-01),
            ("Ni", 2,  3.60830455e+02, -1.53326339e+02,  2.43158034e+01, -1.70189481e+00,  4.46444310e-02),
            ("Ti", 1,  2.13580563e+02, -8.10618895e+01,  1.11812419e+01, -6.47921733e-01,  1.32077684e-02),
            ("Ti", 2,  3.98005019e+02, -1.71211555e+02,  2.74939697e+01, -1.94207913e+00,  5.11932064e-02),
        ]
        # fmt: on
        for row in nist_coefficients:
            cursor.execute(
                "INSERT OR REPLACE INTO partition_functions "
                "(element, sp_num, a0, a1, a2, a3, a4, t_min, t_max, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 2000.0, 20000.0, 'NIST ASD fit')",
                row,
            )
        logger.info("Populated %d partition function coefficient sets", len(nist_coefficients))

    @cached_transitions
    def get_transitions(
        self,
        element: str,
        ionization_stage: Optional[int] = None,
        wavelength_min: Optional[float] = None,
        wavelength_max: Optional[float] = None,
        min_relative_intensity: Optional[float] = None,
    ) -> List[Transition]:
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
        List[Transition]
            List of transition objects
        """
        # Check if new columns exist in the actual query execution (though schema check should have fixed it)
        # We select all relevant columns.
        query = """
            SELECT 
                element, sp_num, wavelength_nm, aki, ek_ev, ei_ev, 
                gk, gi, rel_int,
                stark_w, stark_alpha, stark_shift, is_resonance
            FROM lines
            WHERE element = ?
        """
        params: List[Any] = [element]

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
            )
            transitions.append(trans)

        logger.debug(f"Retrieved {len(transitions)} transitions for {element}")
        return transitions

    def get_energy_levels(self, element: str, ionization_stage: int) -> List[EnergyLevel]:
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
        List[EnergyLevel]
            List of energy level objects
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
    def get_ionization_potential(self, element: str, ionization_stage: int) -> Optional[float]:
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

    def get_atomic_mass(self, element: str) -> Optional[float]:
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
    ) -> Optional[PartitionFunction]:
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

    def get_species_physics(self, element: str, ionization_stage: int) -> Optional[SpeciesPhysics]:
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
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
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
        Tuple[float, float, float]
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

    def get_available_elements(self) -> List[str]:
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