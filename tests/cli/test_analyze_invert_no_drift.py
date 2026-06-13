"""
No-drift contract test for the unified ``analyze`` / ``invert`` line
detection+selection path (cflibs/cli/main.py).

History: the two CLI entry points DRIFTED. ``invert_cmd`` got a tuned
relative-intensity floor (100.0), a configured ``LineSelector``, and
self-absorption plumbing, while ``analyze_cmd`` still ran a bare
``detect_line_observations(...)`` (no floor) and a bare ``LineSelector()``.
On real BHVO-2 the default ``analyze`` path therefore produced a catastrophic
Na-dominated composition (RMSE 33.69 wt%, Na ~98%) while ``invert`` produced a
balanced estimate (RMSE ~14.83). The fix extracts a single shared helper,
``_detect_and_select_lines``, that both commands call.

This module keeps the goal-protecting piece of that history:

1. The default detection cascade does not admit the weak high-E_k Na Rydberg
   lines whose false matches drove the historical Na~98% blowup.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cflibs.cli.main import _detect_and_select_lines
from cflibs.io.spectrum import load_spectrum

_BHVO2_SPECTRUM = Path("data/bhvo2_usgs/chemcam_bhvo2_loc1_spectrum.csv")
_ELEMENTS = ["Si", "Ti", "Al", "Fe", "Mn", "Mg", "Ca", "Na", "K", "P"]
_NA_RYDBERG_NM = {413.1, 414.4, 417.2, 420.2, 421.6}


@pytest.mark.requires_db
@pytest.mark.integration
def test_default_path_admits_no_na_rydberg_lines(production_db):
    """
    The DEFAULT detection cascade (no absolute floor; element-relative top-K
    by gA-Boltzmann strength + shift-coherence veto) must not admit the weak
    high-E_k Na Rydberg lines (413-421 nm) whose false matches drove the
    catastrophic Na ~98% blowup on the old bare ``analyze`` path. The legacy
    floor (min_relative_intensity=100.0) must keep pruning them too.
    """
    if not _BHVO2_SPECTRUM.exists():
        pytest.skip("BHVO-2 spectrum not available")
    wl, inten = load_spectrum(str(_BHVO2_SPECTRUM))

    default_obs = _detect_and_select_lines(
        wl, inten, production_db, _ELEMENTS, min_relative_intensity=None
    )
    na_default = {round(o.wavelength_nm, 1) for o in default_obs if o.element == "Na"}
    assert not (na_default & _NA_RYDBERG_NM), (
        "The default detection cascade must suppress the weak high-E_k Na "
        f"Rydberg lines; survivors: {sorted(na_default & _NA_RYDBERG_NM)}."
    )

    with_floor = _detect_and_select_lines(
        wl, inten, production_db, _ELEMENTS, min_relative_intensity=100.0
    )
    na_with_floor = {round(o.wavelength_nm, 1) for o in with_floor if o.element == "Na"}
    assert not (na_with_floor & _NA_RYDBERG_NM), (
        "The legacy floor must prune the weak high-E_k Na Rydberg lines; "
        f"survivors: {sorted(na_with_floor & _NA_RYDBERG_NM)}."
    )

    # The cascade must not be trivially empty either: the default path keeps
    # at least as many real lines as the element-deleting legacy floor.
    assert len(default_obs) >= len(with_floor)
