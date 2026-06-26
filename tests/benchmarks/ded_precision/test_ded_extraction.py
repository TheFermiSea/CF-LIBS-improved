"""DB-backed: line selection + forward + extraction pipeline (DED-PLAN step 4)."""

from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.requires_db, pytest.mark.slow]


def _db_path():
    for c in (
        Path(__file__).resolve().parents[3] / "ASD_da" / "libs_production.db",
        Path(__file__).resolve().parents[3] / "libs_production.db",
    ):
        if c.exists():
            return str(c)
    return None


def test_ti64_line_list_has_spread():
    from cflibs.atomic.database import AtomicDatabase
    from tests.benchmarks.ded_precision.line_lists import build_alloy_line_list

    db_path = _db_path()
    if db_path is None:
        pytest.skip("libs_production.db not available")
    per_el = build_alloy_line_list(AtomicDatabase(db_path), "Ti-6Al-4V")
    for el in ("Ti", "Al", "V"):
        specs = per_el[el]
        assert len(specs) >= 3, f"{el}: only {len(specs)} lines"
        eks = [s.E_k_ev for s in specs]
        assert max(eks) - min(eks) >= 1.0, f"{el}: insufficient E_k spread"


def test_extraction_recovers_all_elements():
    from cflibs.atomic.database import AtomicDatabase
    from tests.benchmarks.ded_precision.alloy_definitions import (
        ALLOY_COMPOSITIONS,
        ALLOY_WINDOWS_NM,
        elements_of,
    )
    from tests.benchmarks.ded_precision.line_extractor import extract_line_intensities
    from tests.benchmarks.ded_precision.line_lists import build_alloy_line_list
    from tests.benchmarks.ded_precision.spectrum_generator import (
        clean_spectrum,
        default_grid,
        make_forward,
    )

    db_path = _db_path()
    if db_path is None:
        pytest.skip("libs_production.db not available")

    alloy = "Ti-6Al-4V"
    els = elements_of(alloy)
    wl = default_grid(ALLOY_WINDOWS_NM[alloy], step_nm=0.02)
    fwd = make_forward(db_path, els, wl, instrument_fwhm_nm=0.1)
    spec = clean_spectrum(fwd, ALLOY_COMPOSITIONS[alloy], els, T_K=11000.0, ne_cm3=1e17)
    assert np.all(np.isfinite(spec)) and spec.max() > 0

    per_el = build_alloy_line_list(AtomicDatabase(db_path), alloy)
    all_specs = [s for v in per_el.values() for s in v]
    obs = extract_line_intensities(wl, spec, all_specs, instrument_fwhm_nm=0.1)
    found = {o.element for o in obs}
    assert found == set(els), f"missing elements: {set(els) - found}"
    assert all(o.intensity > 0 for o in obs)
    # majority of curated lines should survive extraction on a clean spectrum
    assert len(obs) >= 0.6 * len(all_specs)
