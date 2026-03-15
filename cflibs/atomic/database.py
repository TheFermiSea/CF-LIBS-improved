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
            "aki_uncertainty": "REAL",
            "accuracy_grade": "TEXT",
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
            cursor.execute(
                """
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
            """
            )

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
        cursor.execute(
            """
            INSERT OR IGNORE INTO energy_levels (element, sp_num, g_level, energy_ev)
            SELECT DISTINCT element, sp_num, CAST(gi AS INTEGER), ROUND(ei_ev, 8)
            FROM lines
            WHERE gi IS NOT NULL AND ei_ev IS NOT NULL
        """
        )
        # Upper levels (avoid duplicates)
        cursor.execute(
            """
            INSERT OR IGNORE INTO energy_levels (element, sp_num, g_level, energy_ev)
            SELECT DISTINCT element, sp_num, CAST(gk AS INTEGER), ROUND(ek_ev, 8)
            FROM lines
            WHERE gk IS NOT NULL AND ek_ev IS NOT NULL
        """
        )
        # Deduplicate by (element, sp_num, g_level, energy_ev) so distinct levels
        # (different g_level or energy) are preserved for partition-function sums
        cursor.execute(
            """
            DELETE FROM energy_levels
            WHERE rowid NOT IN (
                SELECT MIN(rowid)
                FROM energy_levels
                GROUP BY element, sp_num, g_level, ROUND(energy_ev, 8)
            )
        """
        )
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
        """Populate partition_functions with Irwin polynomial coefficients.

        Coefficients fit ln(U) = sum(a_n * (ln T)^n), valid for T in [2000, 20000] K.
        Computed from NIST ASD energy level summation with weighted least-squares
        emphasising standard test temperatures (5000, 10000, 15000, 20000 K).
        Max relative error at key temperatures < 0.35% for all 106 species.
        Covers 68 elements across stages I–III.
        """
        # (element, sp_num, a0, a1, a2, a3, a4)
        # fmt: off
        nist_coefficients = [
            ("Ag", 1,  1.024328980396e+02, -5.520072690331e+01,  1.107939770305e+01, -9.771022268156e-01,  3.200031343543e-02),
            ("Ag", 2,  3.337727834288e+02, -1.691187297254e+02,  3.200635127285e+01, -2.681848996155e+00,  8.395855366574e-02),
            ("Al", 2,  2.703248775587e+02, -1.288278497001e+02,  2.299779639351e+01, -1.822783479715e+00,  5.412541710376e-02),
            ("As", 1, -8.188457919357e+01,  4.597651698457e+01, -9.159369708480e+00,  7.839459289809e-01, -2.434816446908e-02),
            ("Au", 1,  2.219816405851e+02, -9.556767796454e+01,  1.543361300069e+01, -1.110299912356e+00,  3.026995408277e-02),
            ("B",  1,  2.295344062622e+02, -1.068241958690e+02,  1.875539807684e+01, -1.461048007748e+00,  4.261416130790e-02),
            ("Ba", 1,  7.203398383917e+02, -3.024090683010e+02,  4.684726223165e+01, -3.177604717500e+00,  8.002833239692e-02),
            ("Ba", 2,  4.031219245369e+02, -1.836741819128e+02,  3.115024115221e+01, -2.330345096904e+00,  6.514113083156e-02),
            ("Be", 2, -2.767290183521e+01,  8.432109153618e+00, -5.764176911949e-01, -2.940771066811e-02,  3.141685064197e-03),
            ("Bi", 1, -1.897082627499e+02,  8.773649856282e+01, -1.490897071284e+01,  1.107253506799e+00, -3.015928775856e-02),
            ("Br", 1,  6.430983545635e+02, -2.975829314409e+02,  5.163146793659e+01, -3.972734832279e+00,  1.143903343004e-01),
            ("C",  1,  6.015529265286e+02, -2.795044955991e+02,  4.876796116109e+01, -3.774602704732e+00,  1.093660525897e-01),
            ("C",  2,  9.827337881082e+01, -4.630675995006e+01,  8.268776445400e+00, -6.543083397969e-01,  1.936946867734e-02),
            ("Ca", 1, -3.750861193609e+02,  1.682257057197e+02, -2.807720033345e+01,  2.064648237466e+00, -5.636017015804e-02),
            ("Ca", 2, -2.649918385755e+02,  1.305255385506e+02, -2.373307840692e+01,  1.889863399786e+00, -5.547662070075e-02),
            ("Cd", 1, -2.618956751216e+02,  1.036874730896e+02, -1.465870173558e+01,  8.510453982340e-01, -1.590075658388e-02),
            ("Cd", 2,  3.698292119294e+02, -1.726735887444e+02,  3.023587883302e+01, -2.349144083328e+00,  6.833542624104e-02),
            ("Ce", 1, -2.261287460509e+02,  9.804818148226e+01, -1.591768610452e+01,  1.161528530962e+00, -3.186247097387e-02),
            ("Ce", 2,  7.316871549395e+00, -1.227010050350e+01,  3.402990040473e+00, -3.228553783642e-01,  1.042411896033e-02),
            ("Cl", 1,  4.448657682011e+02, -2.053734552559e+02,  3.558091293941e+01, -2.733513047809e+00,  7.857996823918e-02),
            ("Co", 1, -1.343040699415e+02,  5.543048096737e+01, -8.284517826954e+00,  5.381508586886e-01, -1.251069753417e-02),
            ("Co", 2,  2.188995156591e+01, -1.258146349121e+01,  2.706492044306e+00, -2.446001286705e-01,  8.258308728578e-03),
            ("Cr", 1, -1.203954096044e+03,  5.663621919151e+02, -9.918190946353e+01,  7.665103761710e+00, -2.201587605378e-01),
            ("Cr", 2, -5.893397367190e+02,  2.784715002680e+02, -4.878549548864e+01,  3.760371753943e+00, -1.073409585117e-01),
            ("Cs", 1, -4.979878687817e+02,  2.225626657995e+02, -3.695017626795e+01,  2.701636791740e+00, -7.329251616573e-02),
            ("Cu", 1, -2.717934465402e+02,  1.276902323719e+02, -2.214279394582e+01,  1.677600601923e+00, -4.658481253473e-02),
            ("Cu", 2, -1.572938540452e+03,  7.218261391426e+02, -1.235078555314e+02,  9.332109989303e+00, -2.624876026609e-01),
            ("Dy", 1, -1.461688327486e+02,  8.607323966703e+01, -1.782754752737e+01,  1.579308068492e+00, -5.055109541164e-02),
            ("Dy", 2, -2.656465050857e+02,  1.180973472726e+02, -1.934791360793e+01,  1.398482102310e+00, -3.744853366634e-02),
            ("Er", 1, -1.174696191196e+03,  5.396125535684e+02, -9.223909867862e+01,  6.964252093970e+00, -1.957963073411e-01),
            ("Er", 2, -3.451301665987e+02,  1.525421144322e+02, -2.481642222464e+01,  1.773956599923e+00, -4.688067483193e-02),
            ("Eu", 1, -2.020398880696e+02,  1.249823197668e+02, -2.666156887894e+01,  2.398591848711e+00, -7.753172841297e-02),
            ("Eu", 2, -3.502558679907e+02,  1.546239665172e+02, -2.522680134081e+01,  1.811948406259e+00, -4.819568972447e-02),
            ("Fe", 1, -1.138697834314e+03,  5.236273882168e+02, -8.956630893039e+01,  6.762754953465e+00, -1.898122805694e-01),
            ("Fe", 2, -3.134458487636e+02,  1.386359715710e+02, -2.259824248713e+01,  1.620600212037e+00, -4.285874426115e-02),
            ("Gd", 1, -3.484923987010e+02,  1.543890099000e+02, -2.529990129761e+01,  1.832556318809e+00, -4.937624360176e-02),
            ("Ge", 1,  1.136204658006e+02, -5.568559596591e+01,  1.015873389119e+01, -8.108567676097e-01,  2.408384888524e-02),
            ("Ge", 2,  1.031279282840e+02, -4.752645037632e+01,  8.250421647788e+00, -6.351848978427e-01,  1.830053741688e-02),
            ("Hf", 1, -2.803573122709e+02,  1.322376381234e+02, -2.329276075366e+01,  1.820292947605e+00, -5.287947905300e-02),
            ("Hf", 2,  6.306216004677e+01, -3.150190176212e+01,  5.877646613972e+00, -4.780198719587e-01,  1.445832991464e-02),
            ("Hg", 1,  9.829371575279e+02, -4.644931261240e+02,  8.219853682932e+01, -6.456629004558e+00,  1.899570167939e-01),
            ("Hg", 2,  3.765791080542e+02, -1.790569805858e+02,  3.195100939569e+01, -2.531381195104e+00,  7.513752399453e-02),
            ("Ho", 1, -7.176359955888e+02,  3.313406617567e+02, -5.684545730359e+01,  4.306890848344e+00, -1.213956415544e-01),
            ("Ho", 2, -5.083125680138e+01,  2.159540582710e+01, -3.153359709011e+00,  1.974246718555e-01, -4.360072400941e-03),
            ("I",  1,  1.408911575590e+03, -6.520719722349e+02,  1.130434489328e+02, -8.693490201496e+00,  2.503109585254e-01),
            ("In", 1, -4.011050278670e+01,  8.687116984303e+00,  2.353622469408e-01, -1.496402553200e-01,  8.155682733757e-03),
            ("In", 2,  4.649227923714e+02, -2.209296955311e+02,  3.931938125262e+01, -3.106387102170e+00,  9.192695025283e-02),
            ("Ir", 1,  8.252939908340e+01, -3.525770777176e+01,  5.694458994114e+00, -4.039863429357e-01,  1.088608133388e-02),
            ("Ir", 2,  1.170867157317e+02, -4.289118591253e+01,  5.637328266760e+00, -3.001865676258e-01,  5.254436412404e-03),
            ("Kr", 1,  3.133543782450e+02, -1.450501057012e+02,  2.511831281415e+01, -1.928777058479e+00,  5.541761595217e-02),
            ("La", 1, -3.855593027638e+02,  1.800214876745e+02, -3.146851983834e+01,  2.445774740538e+00, -7.085342473961e-02),
            ("La", 2, -3.017446327751e+02,  1.343601414496e+02, -2.230502895726e+01,  1.647878886948e+00, -4.548442368820e-02),
            ("Li", 1, -4.467715376641e+02,  2.080551145678e+02, -3.603908121420e+01,  2.753213278917e+00, -7.815228213084e-02),
            ("Lu", 1, -7.777757481415e+02,  3.531125176081e+02, -5.975905821795e+01,  4.472593436985e+00, -1.246498511923e-01),
            ("Lu", 2,  1.991502182185e+02, -7.131809046242e+01,  8.883374274865e+00, -4.368178340419e-01,  6.457100933397e-03),
            ("Mg", 1,  2.123269681363e+02, -1.046214843003e+02,  1.929881039159e+01, -1.579488449372e+00,  4.839411328728e-02),
            ("Mn", 1, -2.052473051588e+03,  9.485554466254e+02, -1.633303789401e+02,  1.242039555792e+01, -3.516155501796e-01),
            ("Mn", 2, -1.553105391816e+03,  7.101801383437e+02, -1.208570538645e+02,  9.074529187312e+00, -2.533223381680e-01),
            ("Mo", 1, -1.048606198987e+03,  5.095993841976e+02, -9.184097163015e+01,  7.279002640435e+00, -2.137007889257e-01),
            ("N",  1,  1.023778168377e+02, -4.673128371572e+01,  8.089294733527e+00, -6.209012913510e-01,  1.783192830988e-02),
            ("N",  2,  6.124091236065e+01, -2.894155797246e+01,  5.216936293580e+00, -4.150233608403e-01,  1.231518627450e-02),
            ("Na", 1, -4.671760321046e+02,  2.160220016793e+02, -3.717835364298e+01,  2.824240902420e+00, -7.980713862419e-02),
            ("Nd", 1, -2.563032129648e+02,  1.148319887509e+02, -1.911659529279e+01,  1.414231902712e+00, -3.904120711369e-02),
            ("Nd", 2, -4.604199497120e+02,  2.101202355475e+02, -3.582594991180e+01,  2.717122670662e+00, -7.690865303719e-02),
            ("Ni", 1, -3.813858726712e+02,  1.650955430980e+02, -2.631381343696e+01,  1.840138470472e+00, -4.739583283824e-02),
            ("Ni", 2,  4.662589085768e+02, -1.990380089951e+02,  3.170729946601e+01, -2.230323617665e+00,  5.871592350006e-02),
            ("O",  1, -2.041641362395e+02,  9.341329671317e+01, -1.584594263712e+01,  1.184687806461e+00, -3.284580315702e-02),
            ("P",  1,  5.242840743754e+01, -1.741283527673e+01,  2.046786139259e+00, -9.543041866347e-02,  1.491169201116e-03),
            ("Pb", 1,  4.591045123433e+02, -1.972041732799e+02,  3.149255984909e+01, -2.222444001810e+00,  5.882704397139e-02),
            ("Pd", 1,  6.324742994480e+02, -2.861069919439e+02,  4.809282167656e+01, -3.565834386955e+00,  9.875938577190e-02),
            ("Pr", 2,  9.695622530185e+01, -4.442586439771e+01,  7.515255603844e+00, -5.418082286410e-01,  1.434658828527e-02),
            ("Pt", 1, -1.177930653246e+02,  5.007163185772e+01, -7.645005359030e+00,  5.019751263459e-01, -1.163988775341e-02),
            ("Pt", 2,  3.970907597439e+01, -1.111586651673e+01,  8.112796068545e-01,  1.653268956103e-02, -2.247256851708e-03),
            ("Rb", 1, -9.865099711336e+02,  4.568781173107e+02, -7.882732417432e+01,  6.003678094173e+00, -1.700978157050e-01),
            ("Rh", 1, -2.764605029689e+01,  1.203391641684e+01, -1.876918184944e+00,  1.344117357653e-01, -3.513034521971e-03),
            ("Ru", 1, -1.224858676552e+02,  5.205993514574e+01, -8.070978024254e+00,  5.514243622098e-01, -1.384697248738e-02),
            ("Ru", 2, -3.983941131845e+01,  2.429575117740e+01, -4.980365886125e+00,  4.342955310592e-01, -1.363207538745e-02),
            ("S",  1,  1.337780859894e+03, -6.203515070966e+02,  1.077660564264e+02, -8.302808469165e+00,  2.393994962512e-01),
            ("Sb", 1,  1.064163773615e+02, -4.051549916720e+01,  5.668081498299e+00, -3.407106522613e-01,  7.515809044513e-03),
            ("Si", 1,  5.164291514507e+02, -2.416235830388e+02,  4.247825021279e+01, -3.314192428439e+00,  9.691242887967e-02),
            ("Si", 2,  3.404848761455e+02, -1.584189432381e+02,  2.764232103477e+01, -2.139669016954e+00,  6.199843304258e-02),
            ("Sm", 1, -2.441574753648e+02,  1.045039073022e+02, -1.667918064518e+01,  1.184826305621e+00, -3.143309367194e-02),
            ("Sm", 2, -5.650658583611e+01,  1.795848801037e+01, -1.785307890228e+00,  5.524944535138e-02,  4.122213812458e-04),
            ("Sn", 1,  2.852659196925e+02, -1.406561055034e+02,  2.566071221507e+01, -2.055115083902e+00,  6.131067883491e-02),
            ("Sn", 2,  5.199031002409e+02, -2.397245270188e+02,  4.127983624629e+01, -3.144328273336e+00,  8.954794110900e-02),
            ("Sr", 1, -9.660478604050e+02,  4.744998492169e+02, -8.646902993788e+01,  6.921337384278e+00, -2.050138828629e-01),
            ("Sr", 2, -1.401229656201e+02,  5.846173847460e+01, -8.829687971212e+00,  5.674837727156e-01, -1.277130613454e-02),
            ("Ta", 1, -2.868781722064e+01,  1.911667342059e+01, -4.404464080469e+00,  4.301558558009e-01, -1.475039169465e-02),
            ("Tb", 2, -2.386031297158e+02,  1.047362441475e+02, -1.686026348059e+01,  1.192874828461e+00, -3.117521473506e-02),
            ("Te", 1,  1.560762930915e+02, -6.980575847506e+01,  1.174907346711e+01, -8.741263507006e-01,  2.433237662601e-02),
            ("Ti", 1, -3.580171135905e+02,  1.811628072185e+02, -3.370959225683e+01,  2.749357642661e+00, -8.265017448823e-02),
            ("Ti", 2, -4.515581446701e+01,  2.253610353567e+01, -3.941777499539e+00,  3.048822654633e-01, -8.589431090376e-03),
            ("Tl", 1,  1.068219198683e+02, -4.721577293493e+01,  7.905663215316e+00, -5.948791410199e-01,  1.715266251096e-02),
            ("Tm", 1, -8.128767865819e+02,  4.062440981933e+02, -7.497380395813e+01,  6.066053846164e+00, -1.812920832755e-01),
            ("Tm", 2, -5.480658292351e+02,  2.449516410312e+02, -4.053196503148e+01,  2.954041677893e+00, -7.988296130592e-02),
            ("V",  1, -8.318092387181e+02,  3.890108338171e+02, -6.766967829435e+01,  5.201485166223e+00, -1.486885087896e-01),
            ("V",  2, -1.281892270347e+02,  6.223943075656e+01, -1.103067690524e+01,  8.604716087960e-01, -2.462816464344e-02),
            ("W",  1, -6.418137467827e+02,  2.917264151272e+02, -4.976367401832e+01,  3.774022057613e+00, -1.068302819468e-01),
            ("W",  2,  2.644959606395e+02, -1.192716479025e+02,  1.983942598094e+01, -1.439393022821e+00,  3.878126325013e-02),
            ("Xe", 1,  1.340354095553e+03, -6.212467196228e+02,  1.077291801682e+02, -8.284322104374e+00,  2.383914444461e-01),
            ("Y",  1, -2.931805946419e+02,  1.555212249927e+02, -3.003853833258e+01,  2.522409977682e+00, -7.760843409999e-02),
            ("Y",  2, -1.723104477797e+02,  7.724148703455e+01, -1.295583498635e+01,  9.716890592907e-01, -2.723696013098e-02),
            ("Yb", 1, -1.102174406541e+03,  5.015853412511e+02, -8.508280722497e+01,  6.371072243842e+00, -1.775236952879e-01),
            ("Yb", 2, -9.110819368128e+02,  4.091462437705e+02, -6.832927447865e+01,  5.028187573273e+00, -1.373747503587e-01),
            ("Zn", 1, -1.566314031981e+02,  5.787496898388e+01, -7.276636206364e+00,  3.304311347212e-01, -2.393354389745e-03),
            ("Zn", 2,  2.004159398509e+02, -9.364074685912e+01,  1.643608883510e+01, -1.280145543035e+00,  3.733393585391e-02),
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

    # NIST accuracy grade → fractional uncertainty mapping.
    # Reference: NIST ASD documentation, Kramida et al.
    # Grade AAA: ≤0.3%, AA: ≤1%, A: ≤3%, B: ≤10%, C: ≤25%, D: ≤50%, E: >50%
    NIST_GRADE_UNCERTAINTY = {
        "AAA": 0.003,
        "AA": 0.01,
        "A+": 0.02,
        "A": 0.03,
        "B+": 0.07,
        "B": 0.10,
        "C+": 0.18,
        "C": 0.25,
        "D+": 0.40,
        "D": 0.50,
        "E": 0.50,
    }

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
        # Heuristic: assign based on relative intensity as a proxy for data quality
        # rel_int > 100 → well-measured lines (B grade, 10%)
        cursor.execute(
            "UPDATE lines SET aki_uncertainty = 0.10, accuracy_grade = 'B' "
            "WHERE aki IS NOT NULL AND rel_int IS NOT NULL AND rel_int > 100"
        )
        n_b = cursor.rowcount

        # rel_int 10-100 → moderate quality (C grade, 25%)
        cursor.execute(
            "UPDATE lines SET aki_uncertainty = 0.25, accuracy_grade = 'C' "
            "WHERE aki IS NOT NULL AND rel_int IS NOT NULL AND rel_int > 10 "
            "AND aki_uncertainty IS NULL"
        )
        n_c = cursor.rowcount

        # rel_int 1-10 → weaker lines (D grade, 50%)
        cursor.execute(
            "UPDATE lines SET aki_uncertainty = 0.50, accuracy_grade = 'D' "
            "WHERE aki IS NOT NULL AND rel_int IS NOT NULL AND rel_int >= 1 "
            "AND aki_uncertainty IS NULL"
        )
        n_d = cursor.rowcount

        # Everything else: worst case (E grade, 50%)
        cursor.execute(
            "UPDATE lines SET aki_uncertainty = 0.50, accuracy_grade = 'E' "
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
                stark_w, stark_alpha, stark_shift, is_resonance,
                aki_uncertainty, accuracy_grade
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
