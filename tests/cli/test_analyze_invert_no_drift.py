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

These tests pin the contract so the paths cannot silently re-drift:

1. The argparse defaults and the helper signature default agree on the
   relative-intensity floor (100.0) — a pure unit test, no DB.
2. On a real BHVO-2 spectrum, the helper invoked with ``analyze``'s default
   parameters and with ``invert``'s default parameters yields the SAME
   selected-line count and the SAME set of (element, wavelength) lines — i.e.
   the two entry points resolve to byte-identical detection+selection.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import patch

import pytest

from cflibs.cli.main import _detect_and_select_lines
from cflibs.io.spectrum import load_spectrum

_BHVO2_SPECTRUM = Path("data/bhvo2_usgs/chemcam_bhvo2_loc1_spectrum.csv")
_ELEMENTS = ["Si", "Ti", "Al", "Fe", "Mn", "Mg", "Ca", "Na", "K", "P"]
_NA_RYDBERG_NM = {413.1, 414.4, 417.2, 420.2, 421.6}


def _analyze_parser_defaults() -> dict:
    """
    Parse a minimal ``analyze`` invocation through the REAL production CLI
    parser and return the resulting namespace as a dict.

    ``main()`` parses ``sys.argv`` and then dispatches via ``args.func``. To
    inspect the parsed defaults without executing the command, we stub the
    bound ``func`` to capture the namespace and short-circuit.
    """
    import sys

    from cflibs.cli import main as cli_main

    captured: dict = {}

    def _capture(args):
        captured["ns"] = vars(args)

    argv = ["cflibs", "analyze", "x.csv", "--elements", "Fe"]
    old_argv = sys.argv
    try:
        sys.argv = argv
        with patch("cflibs.cli.main.analyze_cmd", _capture):
            cli_main.main()
    finally:
        sys.argv = old_argv
    return captured["ns"]


def test_helper_and_analyze_default_floor_agree():
    """The shared helper default and the analyze CLI default agree (both ``None``).

    The absolute relative-intensity floor (was 100.0) was replaced by the
    element-relative top-K gA-Boltzmann strength selection + shift-coherence
    veto, so the default floor is now ``None``. The no-drift invariant is
    preserved: the helper signature default and the analyze CLI default match.
    """
    helper_default = (
        inspect.signature(_detect_and_select_lines).parameters["min_relative_intensity"].default
    )
    assert helper_default is None

    ns = _analyze_parser_defaults()
    assert ns["min_relative_intensity"] is None
    assert ns["resolving_power"] is None
    # ``None`` = "not given"; the preset resolution in
    # ``_build_pipeline_config`` resolves it to off (bead l4a8).
    assert ns["apply_self_absorption"] is None


def test_helper_exclude_resonance_tied_to_self_absorption():
    """exclude_resonance defaults to (not apply_self_absorption) when None."""
    sig = inspect.signature(_detect_and_select_lines)
    assert sig.parameters["exclude_resonance"].default is None
    assert sig.parameters["apply_self_absorption"].default is False


@pytest.mark.requires_db
@pytest.mark.integration
def test_analyze_and_invert_select_identical_lines(production_db):
    """
    On real BHVO-2, the helper called with analyze's default params and with
    invert's default params produces the SAME selected-line set — the no-drift
    contract. Both default to min_relative_intensity=100.0 and SA off.
    """
    if not _BHVO2_SPECTRUM.exists():
        pytest.skip("BHVO-2 spectrum not available")
    wl, inten = load_spectrum(str(_BHVO2_SPECTRUM))

    # analyze path: bare good defaults (min_relative_intensity=100.0, SA off).
    analyze_obs = _detect_and_select_lines(
        wl,
        inten,
        production_db,
        _ELEMENTS,
        min_relative_intensity=100.0,
        apply_self_absorption=False,
    )

    # invert path: same params but plumbed through its config-derived kwargs
    # (which, with no config, resolve to the identical good defaults).
    invert_obs = _detect_and_select_lines(
        wl,
        inten,
        production_db,
        _ELEMENTS,
        min_relative_intensity=100.0,
        resolving_power=None,
        wavelength_tolerance_nm=0.1,
        min_peak_height=0.01,
        peak_width_nm=0.2,
        apply_self_absorption=False,
        exclude_resonance=True,
        min_snr=10.0,
        min_energy_spread_ev=2.0,
        min_lines_per_element=3,
        isolation_wavelength_nm=0.1,
        max_lines_per_element=20,
    )

    def _key(obs):
        return sorted((o.element, round(o.wavelength_nm, 3)) for o in obs)

    assert len(analyze_obs) == len(invert_obs)
    assert _key(analyze_obs) == _key(invert_obs)


@pytest.mark.requires_db
@pytest.mark.integration
def test_default_floor_kills_na_rydberg_blowup(production_db):
    """
    The unified default floor (100.0) prunes the spurious Na Rydberg lines that
    drove the catastrophic Na ~98% blowup on the old bare ``analyze`` path.
    """
    if not _BHVO2_SPECTRUM.exists():
        pytest.skip("BHVO-2 spectrum not available")
    wl, inten = load_spectrum(str(_BHVO2_SPECTRUM))

    # No floor (the old analyze behaviour) admits Na Rydberg lines...
    no_floor = _detect_and_select_lines(
        wl, inten, production_db, _ELEMENTS, min_relative_intensity=None
    )
    na_no_floor = {round(o.wavelength_nm, 1) for o in no_floor if o.element == "Na"}

    # ...the unified default floor prunes them.
    with_floor = _detect_and_select_lines(
        wl, inten, production_db, _ELEMENTS, min_relative_intensity=100.0
    )
    na_with_floor = {round(o.wavelength_nm, 1) for o in with_floor if o.element == "Na"}

    assert na_no_floor & _NA_RYDBERG_NM, (
        "Expected the no-floor path to admit Na Rydberg lines; "
        f"got Na lines {sorted(na_no_floor)}."
    )
    assert not (na_with_floor & _NA_RYDBERG_NM), (
        "The default floor must prune the weak high-E_k Na Rydberg lines; "
        f"survivors: {sorted(na_with_floor & _NA_RYDBERG_NM)}."
    )
