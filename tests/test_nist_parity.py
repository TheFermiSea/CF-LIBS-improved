"""
NIST parity tests for CF-LIBS forward model.

Validates partition functions, ionization fractions, and spectral output
against NIST Atomic Spectra Database reference data.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from cflibs.atomic.database import AtomicDatabase
from cflibs.core.constants import EV_TO_K
from cflibs.plasma.saha_boltzmann import SahaBoltzmannSolver

DATA_DIR = Path(__file__).parent / "data" / "nist_reference"

# Elements with NIST partition function reference data
ELEMENTS = ["Fe", "Cu", "Al", "Ni", "Ti", "Cr"]

# Standard test temperatures
TEST_TEMPS_K = [5000, 10000, 15000, 20000]


def _load_partition_ref() -> dict:
    """Load NIST partition function reference values."""
    ref_file = DATA_DIR / "partition_functions.json"
    if not ref_file.exists():
        pytest.skip(f"Reference file not found: {ref_file}")
    with open(ref_file) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _load_ionization_ref() -> dict:
    """Load NIST ionization fraction reference values."""
    ref_file = DATA_DIR / "ionization_fractions.json"
    if not ref_file.exists():
        pytest.skip(f"Reference file not found: {ref_file}")
    with open(ref_file) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def production_db():
    """Session-scoped production database fixture.

    Tries several common paths for the production database.
    """
    candidates = [
        Path("libs_production.db"),
        Path("ASD_da/libs_production.db"),
        Path(__file__).parent.parent / "libs_production.db",
        Path(__file__).parent.parent / "ASD_da" / "libs_production.db",
    ]
    for path in candidates:
        if path.exists():
            return AtomicDatabase(str(path))
    pytest.skip("Production database not found; tried: " + ", ".join(str(p) for p in candidates))


@pytest.fixture(scope="session")
def solver(production_db):
    """Saha-Boltzmann solver using the production database."""
    return SahaBoltzmannSolver(production_db)


# ---------------------------------------------------------------------------
# Partition function tests
# ---------------------------------------------------------------------------


def _partition_function_cases():
    """Generate (element, stage, T_K) test cases from reference data."""
    ref = _load_partition_ref()
    cases = []
    for element in ELEMENTS:
        if element not in ref:
            continue
        for stage_str, temps in ref[element].items():
            stage = int(stage_str)
            for T_K in TEST_TEMPS_K:
                if str(T_K) in temps:
                    cases.append(
                        pytest.param(
                            element,
                            stage,
                            T_K,
                            temps[str(T_K)],
                            id=f"{element}_{'I'*stage}_{T_K}K",
                        )
                    )
    return cases


@pytest.mark.requires_db
@pytest.mark.parametrize("element,stage,T_K,nist_U", _partition_function_cases())
def test_partition_functions(solver, element, stage, T_K, nist_U):
    """Partition function U(T) within 5% of NIST reference."""
    T_eV = T_K / EV_TO_K
    our_U = solver.calculate_partition_function(element, stage, T_eV)
    rel_error = abs(our_U - nist_U) / nist_U
    assert rel_error < 0.05, (
        f"{element} {'I'*stage} at {T_K}K: U={our_U:.3f} vs NIST={nist_U:.3f} "
        f"({rel_error:.1%} error, limit 5%)"
    )


# ---------------------------------------------------------------------------
# Ionization fraction tests
# ---------------------------------------------------------------------------


def _ionization_fraction_cases():
    """Generate (element) test cases from reference data."""
    ref = _load_ionization_ref()
    cases = []
    for element in ELEMENTS:
        if element in ref:
            cases.append(pytest.param(element, id=element))
    return cases


@pytest.mark.requires_db
@pytest.mark.parametrize("element", _ionization_fraction_cases())
def test_ionization_fractions(solver, element):
    """Ionization fractions within 10% of NIST at standard LIBS conditions.

    Stages contributing < 1% are skipped.
    """
    ref = _load_ionization_ref()
    cond = ref[element]
    T_eV = cond["T_eV"]
    n_e = cond["n_e"]
    nist_fracs = cond["fractions"]

    our_fracs = solver.get_ionization_fractions(element, T_eV, n_e)

    for stage_str, nist_f in nist_fracs.items():
        stage = int(stage_str)
        if nist_f < 0.01:
            continue  # Skip negligible stages
        our_f = our_fracs.get(stage, 0.0)
        rel_error = abs(our_f - nist_f) / nist_f
        assert rel_error < 0.10, (
            f"{element} stage {stage} at T={T_eV} eV, ne={n_e:.0e}: "
            f"fraction={our_f:.4f} vs NIST={nist_f:.4f} ({rel_error:.1%} error, limit 10%)"
        )


# ---------------------------------------------------------------------------
# Spectral correlation tests
# ---------------------------------------------------------------------------


def _spectral_correlation_cases():
    """Generate test cases for elements with NIST reference spectra."""
    cases = []
    for element in ELEMENTS:
        csv_path = DATA_DIR / f"{element}_T1.0eV_ne1e+17_R1000.csv"
        if csv_path.exists():
            cases.append(pytest.param(element, csv_path, id=element))
    return cases


def _load_nist_spectrum(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load NIST stick spectrum from CSV."""
    import csv as csv_mod

    wavelengths = []
    strengths = []
    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            wavelengths.append(float(row["wavelength_nm"]))
            strengths.append(float(row["strength"]))
    return np.array(wavelengths), np.array(strengths)


def _broaden_stick_spectrum(
    line_wl: np.ndarray,
    line_strength: np.ndarray,
    grid: np.ndarray,
    resolving_power: float,
) -> np.ndarray:
    """Broaden stick spectrum onto wavelength grid using Gaussian profiles."""
    if len(line_wl) == 0:
        return np.zeros_like(grid)
    fwhm = np.maximum(line_wl / max(resolving_power, 1e-6), 1e-6)
    sigma = fwhm / 2.355
    x = (grid[:, None] - line_wl[None, :]) / sigma[None, :]
    return np.sum(line_strength[None, :] * np.exp(-0.5 * x * x), axis=1)


@pytest.mark.requires_db
@pytest.mark.slow
@pytest.mark.nist_parity
@pytest.mark.parametrize("element,csv_path", _spectral_correlation_cases())
def test_spectral_correlation(production_db, element, csv_path):
    """CF-LIBS synthetic spectrum correlates > 0.90 with NIST reference.

    Generates a CF-LIBS spectrum in NIST_PARITY broadening mode and compares
    against the NIST LIBS stick spectrum broadened onto the same grid.
    """
    from cflibs.instrument.model import InstrumentModel
    from cflibs.plasma.state import SingleZoneLTEPlasma
    from cflibs.radiation.profiles import BroadeningMode
    from cflibs.radiation.spectrum_model import SpectrumModel

    # Load NIST reference
    nist_wl, nist_strength = _load_nist_spectrum(csv_path)

    # Standard conditions matching NIST
    T_eV = 1.0
    n_e = 1e17
    R = 1000

    # Create plasma with this element
    plasma = SingleZoneLTEPlasma(
        T_e=T_eV * EV_TO_K,
        n_e=n_e,
        species={element: 1e15},
    )

    instrument = InstrumentModel.from_resolving_power(R)

    model = SpectrumModel(
        plasma=plasma,
        atomic_db=production_db,
        instrument=instrument,
        lambda_min=200.0,
        lambda_max=800.0,
        delta_lambda=0.1,
        broadening_mode=BroadeningMode.NIST_PARITY,
    )

    wl_cflibs, intensity_cflibs = model.compute_spectrum()

    # Broaden NIST stick spectrum onto same grid
    nist_broadened = _broaden_stick_spectrum(nist_wl, nist_strength, wl_cflibs, R)

    # Normalize both
    def _normalize(y):
        y = y - np.min(y)
        mx = np.max(y)
        return y / mx if mx > 0 else y

    norm_cflibs = _normalize(intensity_cflibs)
    norm_nist = _normalize(nist_broadened)

    # Pearson correlation
    if np.allclose(norm_cflibs, 0) or np.allclose(norm_nist, 0):
        corr = 0.0
    else:
        corr = float(np.corrcoef(norm_cflibs, norm_nist)[0, 1])
        corr = float(np.nan_to_num(corr, nan=0.0))

    assert corr > 0.90, (
        f"{element}: Pearson correlation {corr:.3f} < 0.90 "
        f"(CF-LIBS vs NIST at T={T_eV} eV, ne={n_e:.0e})"
    )
