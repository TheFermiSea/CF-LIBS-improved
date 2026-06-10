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

The detection-cascade fix (PR #223, commit dfdb8a1) then RETIRED the absolute
relative-intensity floor: it deleted whole real elements (Mg/K and the Al I
394.4/396.2 resonance doublet all sit below rel_int 100). It was replaced by an
element-relative top-K gA-Boltzmann strength selection plus a shift-coherence
veto in ``detect_line_observations``. Those two gates also suppress the
Boltzmann-faint high-E_k Na Rydberg lines (413-421 nm) that drove the blowup,
so the absolute floor is no longer the pruning mechanism and the default
``min_relative_intensity`` is now ``None`` across helper, argparse, and
invert/analyze config-resolution paths.

These tests pin the contract so the paths cannot silently re-drift:

1. The argparse defaults and the helper signature default agree on the
   relative-intensity floor (now ``None``) — a pure unit test, no DB.
2. On a real BHVO-2 spectrum, the helper invoked with ``analyze``'s production
   default parameters and with ``invert``'s production default parameters
   yields the SAME selected-line count and the SAME set of (element,
   wavelength) lines — i.e. the two entry points resolve to byte-identical
   detection+selection.
3. On a real BHVO-2 spectrum, the default detection path (``min_relative_
   intensity=None``) does NOT admit the spurious Na Rydberg lines, because the
   gA-Boltzmann comb strength + shift-coherence veto prune them.
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
    assert ns["apply_self_absorption"] is False


def test_helper_exclude_resonance_tied_to_self_absorption():
    """exclude_resonance defaults to (not apply_self_absorption) when None."""
    sig = inspect.signature(_detect_and_select_lines)
    assert sig.parameters["exclude_resonance"].default is None
    assert sig.parameters["apply_self_absorption"].default is False


@pytest.mark.requires_db
@pytest.mark.integration
def test_analyze_and_invert_select_identical_lines(production_db):
    """
    On real BHVO-2, the helper called with ``analyze``'s production-default
    params and with ``invert``'s production-default params produces the SAME
    selected-line set — the no-drift contract.

    ``analyze_cmd`` passes only ``min_relative_intensity`` (None),
    ``resolving_power`` (None) and ``apply_self_absorption`` (False), letting
    everything else fall through to the helper signature defaults.
    ``invert_cmd`` resolves the same parameters from an empty ``analysis_cfg``
    via ``analysis_cfg.get(key, DEFAULT)`` where each DEFAULT equals the
    helper's own default — so with no config the two entry points pass
    byte-identical kwargs and must select the identical line set.
    """
    if not _BHVO2_SPECTRUM.exists():
        pytest.skip("BHVO-2 spectrum not available")
    wl, inten = load_spectrum(str(_BHVO2_SPECTRUM))

    analyze_obs = _detect_and_select_lines(
        wl,
        inten,
        production_db,
        _ELEMENTS,
        min_relative_intensity=None,
        resolving_power=None,
        apply_self_absorption=False,
    )

    invert_obs = _detect_and_select_lines(
        wl,
        inten,
        production_db,
        _ELEMENTS,
        min_relative_intensity=None,
        resolving_power=None,
        wavelength_tolerance_nm=0.1,
        min_peak_height=0.01,
        peak_width_nm=0.2,
        apply_self_absorption=False,
        exclude_resonance=None,
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
def test_default_path_prunes_na_rydberg_lines(production_db):
    """
    The default detection path (``min_relative_intensity=None``) must not
    admit the spurious high-E_k Na Rydberg lines (413-421 nm) that drove the
    catastrophic Na ~98% blowup. With the absolute floor retired (PR #223),
    the gA-Boltzmann top-K strength selection and shift-coherence veto are
    the pruning mechanism.
    """
    if not _BHVO2_SPECTRUM.exists():
        pytest.skip("BHVO-2 spectrum not available")
    wl, inten = load_spectrum(str(_BHVO2_SPECTRUM))

    default_obs = _detect_and_select_lines(
        wl, inten, production_db, _ELEMENTS, min_relative_intensity=None
    )
    na_default = {round(o.wavelength_nm, 1) for o in default_obs if o.element == "Na"}

    assert not (na_default & _NA_RYDBERG_NM), (
        "The default detection path must prune the weak high-E_k Na Rydberg lines; "
        f"survivors: {sorted(na_default & _NA_RYDBERG_NM)}."
    )
