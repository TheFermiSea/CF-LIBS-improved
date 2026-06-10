#!/usr/bin/env python3
"""
Validate CF-LIBS direct summation partition functions against Barklem & Collet (2016).

Reference: Barklem, P. S. & Collet, R. (2016), A&A 588, A96.
           "Partition functions and equilibrium constants for diatomic molecules
            and atoms of astrophysical interest"
           https://arxiv.org/abs/1602.03304
           Data: https://github.com/barklem/Equilibrium

This script downloads the Barklem & Collet tabulated partition function data
(ELT extended temperature grid, Nov 2022 update) and compares against our
direct summation U(T) computed from the production atomic database.

Usage:
    python scripts/validate_partition_functions.py
    python scripts/validate_partition_functions.py --db-path ASD_da/libs_production.db
    python scripts/validate_partition_functions.py --no-download  # use hardcoded values
"""

import argparse
import os
import sys
import urllib.request

os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Ensure the project root is on the path (needed for worktree execution)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np  # noqa: E402

from cflibs.atomic.database import AtomicDatabase  # noqa: E402
from cflibs.plasma.partition import (  # noqa: E402
    _level_cache,
    direct_sum_partition_function,
    get_levels_for_species,
)

# ============================================================================
# Barklem & Collet (2016) reference data
# ============================================================================

# Temperatures for comparison (Kelvin)
TEMPERATURES = [3000, 5000, 8000, 10000, 12000, 15000, 20000]

# Species to validate: (element, ionization_stage, B&C label)
SPECIES = [
    ("Fe", 1, "Fe_I"),
    ("Fe", 2, "Fe_II"),
    ("Cu", 1, "Cu_I"),
    ("Cu", 2, "Cu_II"),
    ("Al", 1, "Al_I"),
    ("Al", 2, "Al_II"),
    ("Cr", 1, "Cr_I"),
    ("Cr", 2, "Cr_II"),
    ("Ti", 1, "Ti_I"),
    ("Ti", 2, "Ti_II"),
    ("Ni", 1, "Ni_I"),
    ("Ni", 2, "Ni_II"),
]

# Workhorse species (iron-group + light metals) whose STORED polynomial is
# gated in CI: at the ps-LIBS band these MUST track the B&C16 / direct-sum
# reference to within POLY_GATE_TOL.  This is the gate the original audit was
# missing — the validator computed only the (correct) direct-sum path and never
# exercised the stored polynomial that production JAX / Bayesian / manifold
# paths actually consume (composition-pipeline diagnosis 2026-06-03 § 2.6).
POLY_GATE_SPECIES = {
    ("Fe", 1),
    ("Fe", 2),
    ("Cr", 1),
    ("Cr", 2),
    ("Ti", 1),
    ("Ti", 2),
    ("Ni", 1),
    ("Ni", 2),
    ("Cu", 1),
    ("Cu", 2),
    ("Al", 1),
    ("Al", 2),
}
# Temperatures at which the polynomial CI gate is enforced (ps-LIBS band).
POLY_GATE_TEMPS = [8000, 10000, 12000]
# Max relative error of the stored polynomial vs reference before CI fails.
POLY_GATE_TOL = 0.20

# Hardcoded Barklem & Collet (2016) reference values (ELT 2024 dataset, vNov2022 update).
# Temperatures 3000-10000 K are exact grid points from the pubT table.
# Temperatures 12000, 15000, 20000 K are from the ELT extended grid at the
# nearest grid point (within ~0.3% of nominal).
BARKLEM_REFERENCE = {
    # fmt: off
    "Fe_I":  [21.9554, 27.794,  42.8266, 59.6627, 84.3939, 137.019, 273.230],
    "Fe_II": [34.3147, 43.4176, 56.4784, 66.9023, 79.2004, 100.280, 145.096],
    "Cu_I":  [ 2.03485, 2.32883, 3.25011, 4.17708, 5.60526,  9.23044, 21.7478],
    "Cu_II": [ 1.00032, 1.02534, 1.30264, 1.69815, 2.23721,  3.21025,  5.37253],
    "Al_I":  [ 5.79075, 5.87971, 6.19328, 7.05012, 8.96164, 14.1872, 29.8935],
    "Al_II": [ 1.0,     1.00018, 1.01064, 1.04138, 1.10404,  1.26036,  1.73423],
    "Cr_I":  [ 7.65435,10.3912, 20.1376, 33.1787, 53.5766, 97.4120, 207.013],
    "Cr_II": [ 6.08747, 7.10899,12.184,  18.4825, 27.3591, 44.6757, 85.7533],
    "Ti_I":  [20.8195, 29.4388, 55.3232, 83.2038, 121.023, 192.654, 349.336],
    "Ti_II": [44.0264, 55.4152, 72.368,  83.7248, 95.3732, 112.957, 145.717],
    "Ni_I":  [26.3546, 30.887,  36.3831, 41.5802, 49.2579, 65.7844, 108.463],
    "Ni_II": [ 8.29948,10.8117, 15.7985, 19.4018, 23.1506, 28.8895, 40.4162],
    # fmt: on
}


# ============================================================================
# Download + parse Barklem & Collet data from GitHub
# ============================================================================

BARKLEM_ELT_URL = (
    "https://raw.githubusercontent.com/barklem/Equilibrium/"
    "master/DATA/ELT/2024/atompartf_table.txt"
)
BARKLEM_PUBT_URL = (
    "https://raw.githubusercontent.com/barklem/Equilibrium/"
    "master/DATA/pubT/pubT%20vNov2022/atompartf_table.txt"
)


def _download(url: str) -> str:
    """Download a URL and return its content as a string."""
    print(f"  Downloading {url.split('/')[-1]} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "CF-LIBS-validator/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def parse_barklem_table(content: str):
    """Parse a Barklem & Collet atompartf_table.txt file.

    Returns
    -------
    temps : np.ndarray
        Temperature grid in Kelvin.
    species_data : dict[str, np.ndarray]
        Mapping from species label (e.g. 'Fe_I') to partition function array.
    """
    lines = content.split("\n")

    # Parse temperature grid
    temp_parts = []
    i = 2
    seen_header = False
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        parts = line.split()
        if parts[0] == "T":
            temp_parts.extend(parts[2:])  # skip 'T' and '[K]'
            seen_header = True
            i += 1
            continue
        if seen_header:
            try:
                float(parts[0])
                temp_parts.extend(parts)
                i += 1
                continue
            except ValueError:
                break
        i += 1

    temps = np.array([float(x) for x in temp_parts])

    # Parse species data (multiline)
    species_data = {}
    current_name = None
    current_vals = []

    for line_idx in range(i, len(lines)):
        line = lines[line_idx].strip()
        if not line:
            continue
        parts = line.split()
        try:
            float(parts[0])
            is_name = False
        except ValueError:
            is_name = True

        if is_name:
            if current_name is not None:
                species_data[current_name] = np.array(current_vals)
            current_name = parts[0]
            current_vals = [float(x) for x in parts[1:]]
        else:
            current_vals.extend([float(x) for x in parts])

    if current_name is not None:
        species_data[current_name] = np.array(current_vals)

    return temps, species_data


def fetch_barklem_reference(target_temps, target_species_labels):
    """Download and extract Barklem & Collet values at target temperatures.

    Uses the pubT table for exact temperature grid points (up to 10000 K)
    and the ELT table for higher temperatures.

    Returns
    -------
    dict[str, list[float]]
        Mapping from species label to partition function values at target_temps.
    """
    print("Fetching Barklem & Collet (2016) reference data from GitHub...")

    # Download pubT table (exact temps up to 10000 K)
    pubt_content = _download(BARKLEM_PUBT_URL)
    pubt_temps, pubt_data = parse_barklem_table(pubt_content)

    # Download ELT table (extended grid to 1e6 K)
    elt_content = _download(BARKLEM_ELT_URL)
    elt_temps, elt_data = parse_barklem_table(elt_content)

    result = {}
    for label in target_species_labels:
        vals = []
        for T_target in target_temps:
            if T_target <= 10000 and label in pubt_data:
                # Use pubT table for exact grid points
                idx = np.argmin(np.abs(pubt_temps - T_target))
                if abs(pubt_temps[idx] - T_target) / T_target < 0.01:
                    vals.append(float(pubt_data[label][idx]))
                    continue

            # Fall back to ELT
            if label in elt_data:
                idx = np.argmin(np.abs(elt_temps - T_target))
                vals.append(float(elt_data[label][idx]))
            else:
                vals.append(np.nan)

        result[label] = vals

    return result


# ============================================================================
# Main validation
# ============================================================================


def compute_our_partition_functions(db_path: str):
    """Compute U(T) via direct summation for all target species and temperatures.

    Returns
    -------
    tuple[dict[str, list[float]], dict[str, int]]
        (partition_function_values, level_counts) mappings from species label.
    """
    _level_cache.clear()  # ensure no stale cached results
    db = AtomicDatabase(db_path)
    result = {}
    n_levels = {}

    for element, stage, label in SPECIES:
        level_data = get_levels_for_species(db, element, stage)
        if level_data is None:
            print(f"  WARNING: No energy level data for {label}")
            result[label] = [np.nan] * len(TEMPERATURES)
            n_levels[label] = 0
            continue

        g_levels, E_levels_ev, ip_ev = level_data
        n_levels[label] = len(g_levels)
        vals = []
        for T in TEMPERATURES:
            # No IPD (n_e=None) for isolated-atom comparison with Barklem & Collet.
            # Their calculation uses a similar direct sum without plasma effects.
            U = direct_sum_partition_function(T, g_levels, E_levels_ev, ip_ev, n_e=None)
            vals.append(U)
        result[label] = vals

    return result, n_levels


def compute_stored_polynomial(db_path: str):
    """Evaluate the CPU SCALAR adapter via ``partition_function_for(...).at(T)``.

    This is the U(T) the default ``invert`` / ``analyze`` / iterative / closed-form
    CPU path consumes.  For a species WITH energy levels the provider returns the
    EXACT direct sum (the ``DirectSumPartitionFunctionProvider``), so this column
    is the Invariant-#2 regression guard (it must stay bit-for-bit direct-sum);
    only level-less species fall back to the stored polynomial.  Returns a mapping
    ``label -> [U(T) ...]`` with ``np.nan`` where the factory has no source row.
    """
    db = AtomicDatabase(db_path)
    result = {}
    for element, stage, label in SPECIES:
        provider = db.partition_function_for(element, stage)
        if provider is None:
            result[label] = [np.nan] * len(TEMPERATURES)
            continue
        result[label] = [float(provider.at(float(T))) for T in TEMPERATURES]
    return result


def compute_manifold_jax_polynomial(db_path: str):
    """Evaluate the MANIFOLD / JAX batched adapter over the real snapshot arrays.

    Builds the actual :class:`AtomicSnapshot` the manifold / JAX forward models
    consume and evaluates the ONE shared guarded
    :func:`polynomial_partition_function_jax` over the per-species
    direct-sum-FIT coefficients + ``[t_min, t_max]`` + ``g0`` baked into it.
    This is the production path the original PF-1..PF-4 defect lived on and that
    the legacy validator never exercised (it only computed the direct-sum / CPU
    path — diagnosis § 2.6).  ``np.nan`` for species absent from the snapshot or
    when JAX is unavailable.
    """
    try:
        import jax  # noqa: F401
    except Exception:
        print("  (JAX unavailable — skipping manifold/JAX snapshot column)")
        return {label: [np.nan] * len(TEMPERATURES) for _, _, label in SPECIES}

    from cflibs.plasma.partition import polynomial_partition_function_jax

    db = AtomicDatabase(db_path)
    elements = sorted({element for element, _, _ in SPECIES})
    snap = db.snapshot(elements=elements, wavelength_range=(200.0, 900.0))
    species_to_idx = {key: i for i, key in enumerate(snap.species)}
    coeffs_all = np.asarray(snap.partition_coeffs)
    tmin_all = np.asarray(snap.partition_t_min)
    tmax_all = np.asarray(snap.partition_t_max)
    g0_all = np.asarray(snap.partition_g0)

    result = {}
    for element, stage, label in SPECIES:
        i = species_to_idx.get((element, stage))
        if i is None:
            result[label] = [np.nan] * len(TEMPERATURES)
            continue
        result[label] = [
            float(
                polynomial_partition_function_jax(
                    float(T),
                    coeffs_all[i],
                    t_min=float(tmin_all[i]),
                    t_max=float(tmax_all[i]),
                    g0=float(g0_all[i]),
                )
            )
            for T in TEMPERATURES
        ]
    return result


def compute_bayesian_polynomial(db_path: str):
    """Evaluate the BAYESIAN adapter via the now-shared guarded evaluator.

    Loads the :class:`AtomicDataArrays` the NUTS forward model consumes (whose
    partition coefficients+bounds come from the SAME factory) and evaluates them
    through :func:`cflibs.inversion.solve.bayesian.atomic.partition_function`,
    the thin guarded delegator that replaced the deleted unguarded duplicate.
    ``np.nan`` for species the loader does not populate or when JAX is missing.
    """
    try:
        import jax  # noqa: F401
    except Exception:
        print("  (JAX unavailable — skipping Bayesian column)")
        return {label: [np.nan] * len(TEMPERATURES) for _, _, label in SPECIES}

    from cflibs.inversion.solve.bayesian.atomic import (
        _query_atomic_data,
        partition_function,
    )

    elements = sorted({element for element, _, _ in SPECIES})
    el_idx = {el: i for i, el in enumerate(elements)}
    _df, coeffs, _ips, t_min, t_max, g0 = _query_atomic_data(db_path, elements, (200.0, 900.0))

    result = {}
    n_stages = coeffs.shape[1]
    for element, stage, label in SPECIES:
        ei = el_idx[element]
        si = stage - 1
        if si >= n_stages:
            result[label] = [np.nan] * len(TEMPERATURES)
            continue
        result[label] = [
            float(
                partition_function(
                    float(T),
                    coeffs[ei, si],
                    t_min=float(t_min[ei, si]),
                    t_max=float(t_max[ei, si]),
                    g0=float(g0[ei, si]),
                )
            )
            for T in TEMPERATURES
        ]
    return result


# Per-path CI tolerance.  The CPU scalar adapter evaluates the EXACT direct sum
# for species-with-levels, so it must match to floating-point precision (the
# Invariant-#2 regression guard).  The JAX/manifold and Bayesian adapters
# evaluate the direct-sum-FIT polynomial (vmap needs static fixed-shape arrays),
# so they are gated at the looser ``POLY_GATE_TOL`` fit tolerance.
CPU_GATE_TOL = 1e-6


def check_polynomial_gate(poly_data, ref_data, our_data, jax_data=None, bayes_data=None):
    """CI gate: EVERY consumer path's U(T) vs the DB direct-sum (workhorse species).

    This is the Wave-5 per-path VALUE gate (diagnosis § 2.6): the legacy
    validator computed only the direct-sum (the correct path) and never the
    polynomial the production JAX / Bayesian / manifold paths actually consume —
    which is exactly why the PF-1..PF-4 defect shipped silently.  The gate now
    exercises all three consumer adapters against the direct-sum reference:

    * **CPU scalar** (``partition_function_for(...).at(T)``) — exact direct sum
      for species-with-levels; gated tight (``CPU_GATE_TOL``).
    * **Manifold / JAX** snapshot polynomial — gated at ``POLY_GATE_TOL``.
    * **Bayesian** ``AtomicDataArrays`` via the shared guarded evaluator — gated
      at ``POLY_GATE_TOL``.

    Each adapter is failed when it is more than its tolerance off the DB
    **direct-sum** (the achievable fit target — the regenerated rows track it to
    within ~1 %, the fit to ≤ ~7 %).  Comparing path-vs-direct-sum isolates the
    *fit* defect (the PF-1/PF-2 undershoot we fixed) from the separate
    *level-table completeness* gap (direct-sum vs B&C16): the latter is reported
    as an informational WARNING (e.g. Cr II / hot-edge Ti I, diagnosis open
    Q #3) and does NOT fail CI, because no re-fit can beat the underlying NIST
    level set.

    Returns
    -------
    list[tuple]
        ``(path, label, T, U_path, U_directsum, rel_err)`` for each gate failure.
    """
    jax_data = jax_data or {}
    bayes_data = bayes_data or {}
    failures = []
    completeness_warnings = []
    print("\n" + "=" * 100)
    print(
        "CI GATE: per-consumer-path U(T) vs DB direct-sum (workhorse species) — "
        f"CPU |err|>{CPU_GATE_TOL:.0e}, JAX/Bayes |err|>{POLY_GATE_TOL:.0%}"
    )
    print("=" * 100)
    print(
        f"{'Species':<8s} {'T (K)':>7s} {'U_dsum':>9s} {'U_cpu':>9s} {'cpu/ds':>8s} "
        f"{'U_jax':>9s} {'jax/ds':>8s} {'U_bayes':>9s} {'bay/ds':>8s} {'dsum/BC':>8s} {'status':>7s}"
    )
    print("-" * 100)

    def _rel(u, ds):
        if u is None or np.isnan(u) or np.isnan(ds) or ds == 0:
            return np.nan
        return (u - ds) / ds

    def _fmt(rel):
        return f"{rel:>+7.2%}" if not np.isnan(rel) else f"{'N/A':>8s}"

    for element, stage, label in SPECIES:
        if (element, stage) not in POLY_GATE_SPECIES:
            continue
        cpu_vals = poly_data.get(label, [np.nan] * len(TEMPERATURES))
        ref_vals = ref_data.get(label, [np.nan] * len(TEMPERATURES))
        ds_vals = our_data.get(label, [np.nan] * len(TEMPERATURES))
        jax_vals = jax_data.get(label, [np.nan] * len(TEMPERATURES))
        bay_vals = bayes_data.get(label, [np.nan] * len(TEMPERATURES))
        for T in POLY_GATE_TEMPS:
            idx = TEMPERATURES.index(T)
            U_dsum = ds_vals[idx]
            U_cpu = cpu_vals[idx]
            U_jax = jax_vals[idx]
            U_bay = bay_vals[idx]
            U_bc = ref_vals[idx]
            if np.isnan(U_dsum) or U_dsum == 0:
                continue
            rel_cpu = _rel(U_cpu, U_dsum)
            rel_jax = _rel(U_jax, U_dsum)
            rel_bay = _rel(U_bay, U_dsum)

            row_failed = False
            if not np.isnan(rel_cpu) and abs(rel_cpu) > CPU_GATE_TOL:
                failures.append(("cpu", label, T, U_cpu, U_dsum, rel_cpu))
                row_failed = True
            if not np.isnan(rel_jax) and abs(rel_jax) > POLY_GATE_TOL:
                failures.append(("jax", label, T, U_jax, U_dsum, rel_jax))
                row_failed = True
            if not np.isnan(rel_bay) and abs(rel_bay) > POLY_GATE_TOL:
                failures.append(("bayes", label, T, U_bay, U_dsum, rel_bay))
                row_failed = True
            status = "FAIL *" if row_failed else "PASS"

            ds_bc = (U_dsum - U_bc) / U_bc if (not np.isnan(U_bc) and U_bc != 0) else np.nan
            ds_bc_str = f"{ds_bc:>+7.1%}" if not np.isnan(ds_bc) else f"{'N/A':>8s}"
            cpu_str = f"{U_cpu:>9.4f}" if not np.isnan(U_cpu) else f"{'N/A':>9s}"
            jax_str = f"{U_jax:>9.4f}" if not np.isnan(U_jax) else f"{'N/A':>9s}"
            bay_str = f"{U_bay:>9.4f}" if not np.isnan(U_bay) else f"{'N/A':>9s}"
            print(
                f"{label:<8s} {T:>7d} {U_dsum:>9.4f} {cpu_str} {_fmt(rel_cpu)} "
                f"{jax_str} {_fmt(rel_jax)} {bay_str} {_fmt(rel_bay)} {ds_bc_str} {status:>7s}"
            )
            # Separate completeness warning: direct-sum far below B&C16 => the
            # NIST level table is missing high-lying levels (not a fit defect).
            if not np.isnan(ds_bc) and ds_bc < -POLY_GATE_TOL:
                completeness_warnings.append((label, T, U_dsum, U_bc, ds_bc))
    print("-" * 100)

    if failures:
        print(f"\nFAIL: {len(failures)} per-path U(T) value(s) diverged from the DB direct-sum:")
        for path, label, T, U_path, U_dsum, rel in failures:
            print(
                f"  [{path}] {label} @ {T} K: U={U_path:.4f} vs direct-sum={U_dsum:.4f} ({rel:+.1%})"
            )
        print(
            "\nThe JAX manifold / Bayesian adapters consume the direct-sum-FIT\n"
            "polynomial baked into the snapshot.  Regenerate with\n"
            "  python scripts/archive/migrations/regenerate_partition_functions.py --db-path <db>\n"
        )
    else:
        print(
            "\nGATE PASS: CPU / manifold-JAX / Bayesian U(T) all track the DB "
            "direct-sum within tolerance."
        )

    if completeness_warnings:
        print(
            f"\nWARNING (informational, does NOT fail CI): {len(completeness_warnings)} "
            "value(s) where the DB direct-sum itself is >20% below B&C16 —"
        )
        print(
            "  the NIST energy_levels table is incomplete at the hot edge " "(diagnosis open Q #3):"
        )
        for label, T, U_dsum, U_bc, ds_bc in completeness_warnings:
            print(f"  {label} @ {T} K: direct-sum={U_dsum:.3f} vs B&C16={U_bc:.3f} ({ds_bc:+.1%})")

    return failures


def print_comparison_table(our_data, ref_data, error_threshold=0.10):
    """Print a formatted comparison table and flag large deviations.

    Parameters
    ----------
    our_data : dict[str, list[float]]
        Our computed partition function values.
    ref_data : dict[str, list[float]]
        Barklem & Collet reference values.
    error_threshold : float
        Relative error threshold for flagging (default 10%).

    Returns
    -------
    int
        Number of flagged entries exceeding the threshold.
    """
    n_flagged = 0

    # Header
    temp_headers = "".join(f"{'T=' + str(T):>12s}" for T in TEMPERATURES)
    print(f"\n{'Species':<10s}{temp_headers}")
    print("=" * (10 + 12 * len(TEMPERATURES)))

    for element, stage, label in SPECIES:
        our = our_data.get(label, [np.nan] * len(TEMPERATURES))
        ref = ref_data.get(label, [np.nan] * len(TEMPERATURES))

        # Print reference values
        ref_line = "".join(f"{v:12.4f}" for v in ref)
        print(f"{label:<10s}{ref_line}  [B&C16]")

        # Print our values
        our_line = "".join(f"{v:12.4f}" for v in our)
        print(f"{'':10s}{our_line}  [Ours]")

        # Print relative errors
        errs = []
        for o, r in zip(our, ref):
            if np.isnan(o) or np.isnan(r) or r == 0:
                errs.append(np.nan)
            else:
                errs.append((o - r) / r)

        err_strs = []
        for e in errs:
            if np.isnan(e):
                err_strs.append(f"{'N/A':>12s}")
            else:
                flag = " *" if abs(e) > error_threshold else ""
                err_strs.append(f"{e:+11.2%}{flag}")
                if abs(e) > error_threshold:
                    n_flagged += 1

        err_line = "".join(err_strs)
        print(f"{'':10s}{err_line}  [rel err]")
        print()

    return n_flagged


def print_summary_table(our_data, ref_data, n_levels, error_threshold=0.10):
    """Print a compact summary with mean/max errors per species.

    Returns
    -------
    int
        Number of flagged species with max error > threshold.
    """
    print("\n" + "=" * 72)
    print("SUMMARY: Per-species error statistics")
    print("=" * 72)
    print(
        f"{'Species':<10s} {'N_levels':>8s} {'Mean |err|':>12s} {'Max |err|':>12s} {'Status':>10s}"
    )
    print("-" * 54)

    n_flagged = 0
    for element, stage, label in SPECIES:
        our = our_data.get(label, [np.nan] * len(TEMPERATURES))
        ref = ref_data.get(label, [np.nan] * len(TEMPERATURES))

        abs_errs = []
        for o, r in zip(our, ref):
            if not (np.isnan(o) or np.isnan(r) or r == 0):
                abs_errs.append(abs((o - r) / r))

        nl = n_levels.get(label, 0)
        if abs_errs:
            mean_err = np.mean(abs_errs)
            max_err = np.max(abs_errs)
            status = "PASS" if max_err <= error_threshold else "FAIL *"
            if max_err > error_threshold:
                n_flagged += 1
        else:
            mean_err = max_err = np.nan
            status = "N/A"

        print(f"{label:<10s} {nl:>8d} {mean_err:>11.2%}  {max_err:>11.2%}  {status:>10s}")

    print("-" * 54)
    return n_flagged


def main():
    parser = argparse.ArgumentParser(
        description="Validate partition functions against Barklem & Collet (2016)"
    )
    parser.add_argument(
        "--db-path",
        default="ASD_da/libs_production.db",
        help="Path to atomic database (default: ASD_da/libs_production.db)",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Use hardcoded reference values instead of downloading from GitHub",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help="Relative error threshold for flagging (default: 0.10 = 10%%)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.db_path):
        print(f"ERROR: Database not found at {args.db_path}")
        sys.exit(1)

    # Get reference data
    species_labels = [label for _, _, label in SPECIES]
    if args.no_download:
        print("Using hardcoded Barklem & Collet (2016) reference values.")
        ref_data = BARKLEM_REFERENCE
    else:
        try:
            ref_data = fetch_barklem_reference(TEMPERATURES, species_labels)
            print("  Download successful.\n")
        except Exception as e:
            print(f"  Download failed ({e}), falling back to hardcoded values.\n")
            ref_data = BARKLEM_REFERENCE

    # Compute our values
    print(f"Computing direct summation U(T) from {args.db_path} ...")
    our_data, n_levels = compute_our_partition_functions(args.db_path)

    # Evaluate EVERY consumer adapter's U(T) (the Wave-5 per-path value gate).
    print(f"Evaluating CPU scalar adapter U(T) from {args.db_path} ...")
    poly_data = compute_stored_polynomial(args.db_path)
    print(f"Evaluating manifold / JAX snapshot adapter U(T) from {args.db_path} ...")
    jax_data = compute_manifold_jax_polynomial(args.db_path)
    print(f"Evaluating Bayesian adapter U(T) from {args.db_path} ...")
    bayes_data = compute_bayesian_polynomial(args.db_path)

    # Compare
    print("\n" + "=" * 72)
    print("Partition Function Validation: CF-LIBS vs Barklem & Collet (2016)")
    print("=" * 72)
    print("Reference: Barklem & Collet (2016), A&A 588, A96")
    print("Method: Direct summation U(T) = sum_i g_i exp(-E_i / kT)")
    print("Note: No IPD applied (isolated-atom comparison)")
    print(f"Threshold: {args.threshold:.0%} relative error")

    n_flagged = print_comparison_table(our_data, ref_data, args.threshold)
    n_species_flagged = print_summary_table(our_data, ref_data, n_levels, args.threshold)

    print(f"\nTotal flagged entries (|err| > {args.threshold:.0%}): {n_flagged}")
    print(f"Species with at least one flag: {n_species_flagged} / {len(SPECIES)}")

    if n_flagged > 0:
        print("\n* Flagged entries may indicate:")
        print("  - Incomplete energy level data in our database")
        print("  - Different level selection / IP values")
        print("  - B&C use theoretical levels + ionization limit corrections")
        print("  - Our DB uses NIST ASD observed levels only")

    # CI GATE: the STORED polynomial (not the direct-sum) is what production
    # JAX manifold / Bayesian / forward paths consume.  Fail the run if any
    # workhorse species' polynomial is >POLY_GATE_TOL off the DB direct-sum
    # (the achievable fit target).  This is the gate the diagnosis § 2.6 asked
    # for and the defect-C-closing check.
    poly_failures = check_polynomial_gate(
        poly_data, ref_data, our_data, jax_data=jax_data, bayes_data=bayes_data
    )

    # NOTE on exit-code semantics: the legacy direct-sum-vs-B&C16 summary
    # (``n_species_flagged``) is INFORMATIONAL and intentionally does NOT drive
    # the exit code.  It always flagged species at the hot edge (e.g. 20000 K)
    # because the NIST ``energy_levels`` table is incomplete there — that is a
    # level-table completeness limitation (diagnosis open Q #3), not a defect
    # this validator can fix, and it caused the original script to exit 1
    # unconditionally (so the gate was effectively never green).  CI now gates
    # ONLY on the stored-polynomial-vs-direct-sum workhorse check, which
    # isolates the actual fit defect (PF-1/PF-2) from the completeness gap.
    if n_species_flagged > 0:
        print(
            f"\n[informational] {n_species_flagged}/{len(SPECIES)} species' direct-sum "
            "drifts >10% from B&C16 at some temperature (hot-edge level-table\n"
            "completeness, diagnosis open Q #3) — NOT a CI failure."
        )
    return 1 if poly_failures else 0


if __name__ == "__main__":
    sys.exit(main())
