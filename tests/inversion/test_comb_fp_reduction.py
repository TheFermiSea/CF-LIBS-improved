import numpy as np
from unittest.mock import MagicMock
from cflibs.inversion.identify.comb import CombIdentifier
from cflibs.atomic.structures import Transition

def test_mn_fp_rejection():
    """
    Verify that Mn (Tier-2 element) is rejected when only one line is present,
    even if min_active_teeth is set to 1.
    """
    # Mock atomic database with one Mn line and two Fe lines
    mock_db = MagicMock()
    
    mn_trans = Transition(
        element="Mn", ionization_stage=1, wavelength_nm=403.0,
        A_ki=1e8, g_k=6, g_i=4, E_k_ev=3.0, E_i_ev=0.0
    )
    fe_trans1 = Transition(
        element="Fe", ionization_stage=1, wavelength_nm=404.0,
        A_ki=1e8, g_k=6, g_i=4, E_k_ev=3.0, E_i_ev=0.0
    )
    fe_trans2 = Transition(
        element="Fe", ionization_stage=1, wavelength_nm=405.0,
        A_ki=1e8, g_k=6, g_i=4, E_k_ev=3.0, E_i_ev=0.0
    )
    
    def get_transitions(element, **kwargs):
        if element == "Mn":
            return [mn_trans]
        if element == "Fe":
            return [fe_trans1, fe_trans2]
        return []
        
    mock_db.get_transitions.side_effect = get_transitions
    mock_db.get_available_elements.return_value = ["Mn", "Fe"]
    
    # Create spectrum with a peak at Mn 403.0 and Fe 404.0, 405.0
    wavelength = np.linspace(400, 410, 1000)
    intensity = np.zeros_like(wavelength)
    
    def add_peak(wl, amp):
        # Use a wider window for the peak to ensure it's detected by the comb
        mask = np.abs(wavelength - wl) < 0.1
        intensity[mask] += amp * np.exp(-0.5 * ((wavelength[mask] - wl)/0.02)**2)

    add_peak(403.0, 100.0) # Mn line
    add_peak(404.0, 100.0) # Fe line 1
    add_peak(405.0, 100.0) # Fe line 2
    
    # Add some noise
    np.random.seed(42)
    intensity += np.random.normal(0, 1, len(intensity))
    
    # identifier with min_active_teeth=1 to test the override for Mn
    identifier = CombIdentifier(mock_db, min_active_teeth=1)
    result = identifier.identify(wavelength, intensity)
    
    detected_elements = [e.element for e in result.detected_elements]
    
    # Fe should be detected (2 lines)
    assert "Fe" in detected_elements
    
    # Mn should NOT be detected (only 1 line, and it's a Tier-2 element)
    assert "Mn" not in detected_elements

def test_na_k_fp_rejection():
    """Verify Na and K also require 2 lines."""
    mock_db = MagicMock()
    
    na_trans = Transition(
        element="Na", ionization_stage=1, wavelength_nm=589.0,
        A_ki=1e8, g_k=6, g_i=2, E_k_ev=3.0, E_i_ev=0.0
    )
    k_trans = Transition(
        element="K", ionization_stage=1, wavelength_nm=766.0,
        A_ki=1e8, g_k=6, g_i=2, E_k_ev=3.0, E_i_ev=0.0
    )
    
    def get_transitions(element, **kwargs):
        if element == "Na":
            return [na_trans]
        if element == "K":
            return [k_trans]
        return []
        
    mock_db.get_transitions.side_effect = get_transitions
    mock_db.get_available_elements.return_value = ["Na", "K"]
    
    wavelength = np.linspace(500, 800, 3000)
    intensity = np.zeros_like(wavelength)
    
    def add_peak(wl, amp):
        mask = np.abs(wavelength - wl) < 0.1
        intensity[mask] += amp * np.exp(-0.5 * ((wavelength[mask] - wl)/0.02)**2)

    add_peak(589.0, 100.0)
    add_peak(766.0, 100.0)
    
    identifier = CombIdentifier(mock_db, min_active_teeth=1)
    result = identifier.identify(wavelength, intensity)

    detected_elements = [e.element for e in result.detected_elements]
    assert "Na" not in detected_elements
    assert "K" not in detected_elements


# ---------------------------------------------------------------------------
# Per-tooth correlation floor for Tier-2 elements (CF-LIBS-improved-53x9)
#
# These tests cover the additional FP reduction layer landed in PR for
# CF-LIBS-improved-53x9: Mn/Na/K teeth must clear a stricter per-tooth
# correlation gate (default 0.7 vs the global 0.5). The change is
# scoped to those three elements only — non-Tier-2 elements (Fe, Ti,
# Si, ...) MUST be byte-identical to pre-PR behavior.
# ---------------------------------------------------------------------------


def _make_identifier_pair(mock_db, **kwargs):
    """Build a (strict, lax) pair of identifiers with otherwise identical kwargs."""
    strict = CombIdentifier(mock_db, strict_tier2=True, **kwargs)
    lax = CombIdentifier(mock_db, strict_tier2=False, **kwargs)
    return strict, lax


def _moderate_snr_spectrum(seed=7):
    """A spectrum with moderate-correlation peaks (~0.55-0.7) at four lines.

    The amplitude/noise ratio AND peak shape (wide gaussian vs the
    triangular template) are tuned so the matched-template Pearson
    correlation lands above the global threshold (0.5) but below the
    new Tier-2 threshold (0.7) — i.e. the lax path accepts the line,
    the strict path rejects it.

    Calibrated empirically: gaussian sigma=0.2 nm (wider than the
    triangular template peaks favor) + amplitude 30 + noise sigma 10
    consistently yields correlations in the 0.55-0.7 window across a
    range of seeds (verified during PR development for
    CF-LIBS-improved-53x9).
    """
    rng = np.random.default_rng(seed)
    wavelength = np.linspace(380, 800, 4000)
    intensity = np.zeros_like(wavelength)

    centers = (403.0, 404.0, 589.0, 766.0)
    for c in centers:
        intensity += 30.0 * np.exp(-0.5 * ((wavelength - c) / 0.2) ** 2)

    # Heavy noise — pushes off-line correlations down and pulls on-line
    # correlations into the moderate-SNR regime.
    intensity += rng.normal(0.0, 10.0, len(intensity))
    return wavelength, intensity


def test_tier2_strict_rejects_moderate_correlation_peaks():
    """With strict_tier2=True, Mn/Na/K teeth must clear the stricter floor.

    Construct a spectrum where matched-template Pearson correlation is
    in the moderate range (~0.55-0.65). The lax identifier (strict_tier2
    OFF) should accept Mn/Na/K teeth as active; the strict identifier
    should reject them because they fall below the 0.7 Tier-2 floor.
    """
    mn = Transition(
        element="Mn", ionization_stage=1, wavelength_nm=403.0,
        A_ki=1e8, g_k=6, g_i=4, E_k_ev=3.0, E_i_ev=0.0,
    )
    mn2 = Transition(
        element="Mn", ionization_stage=1, wavelength_nm=404.0,
        A_ki=1e8, g_k=6, g_i=4, E_k_ev=3.0, E_i_ev=0.0,
    )
    na = Transition(
        element="Na", ionization_stage=1, wavelength_nm=589.0,
        A_ki=1e8, g_k=6, g_i=2, E_k_ev=3.0, E_i_ev=0.0,
    )
    k = Transition(
        element="K", ionization_stage=1, wavelength_nm=766.0,
        A_ki=1e8, g_k=6, g_i=2, E_k_ev=3.0, E_i_ev=0.0,
    )

    mock_db = MagicMock()

    def get_transitions(element, **kwargs):
        if element == "Mn":
            return [mn, mn2]
        if element == "Na":
            return [na]
        if element == "K":
            return [k]
        return []

    mock_db.get_transitions.side_effect = get_transitions
    mock_db.get_available_elements.return_value = ["Mn", "Na", "K"]

    wavelength, intensity = _moderate_snr_spectrum()

    # tooth_activation_threshold=0.5 (global default), tier2 floor 0.7.
    strict, lax = _make_identifier_pair(
        mock_db,
        tooth_activation_threshold=0.5,
        tier2_tooth_activation_threshold=0.7,
    )

    strict_result = strict.identify(wavelength, intensity)
    lax_result = lax.identify(wavelength, intensity)

    def count_active(result, element):
        for e in result.all_elements:
            if e.element == element:
                return e.metadata.get("n_active_teeth", 0)
        return 0

    # The strict path must produce at most as many active Tier-2 teeth
    # as the lax path, with at least ONE element strictly reduced — the
    # fixture is calibrated so this is achievable.
    for elt in ("Mn", "Na", "K"):
        assert count_active(strict_result, elt) <= count_active(lax_result, elt), (
            f"strict_tier2 must never produce MORE active {elt} teeth than the "
            f"lax path: strict={count_active(strict_result, elt)} vs "
            f"lax={count_active(lax_result, elt)}"
        )

    total_strict = sum(count_active(strict_result, e) for e in ("Mn", "Na", "K"))
    total_lax = sum(count_active(lax_result, e) for e in ("Mn", "Na", "K"))
    assert total_strict < total_lax, (
        f"strict_tier2=True must reduce the active Mn/Na/K tooth count for "
        f"the moderate-SNR fixture, but got strict={total_strict} >= lax={total_lax}. "
        f"Check the fixture is calibrated to land in the 0.5-0.7 correlation "
        f"window or the activation threshold wiring."
    )


def test_tier2_strict_preserves_non_tier2_elements():
    """Non-Tier-2 elements (Fe, Ti, Si) must be byte-identical between strict and lax.

    Construct a spectrum with Fe/Ti/Si peaks at the same moderate-SNR
    level as the Tier-2 fixture. The strict and lax identifiers must
    return the same detected set, the same n_active_teeth, and the same
    fingerprint scores for these elements.
    """
    rng = np.random.default_rng(11)
    wavelength = np.linspace(280, 800, 4000)
    intensity = np.zeros_like(wavelength)

    # Fe, Ti, Si peaks at moderate SNR.
    centers = (388.0, 399.0, 251.0, 252.0, 365.0, 366.0)
    elements = ("Fe", "Fe", "Si", "Si", "Ti", "Ti")
    for c in centers:
        intensity += 50.0 * np.exp(-0.5 * ((wavelength - c) / 0.05) ** 2)
    intensity += rng.normal(0.0, 6.0, len(intensity))

    transitions = {}
    for elt, c in zip(elements, centers):
        transitions.setdefault(elt, []).append(
            Transition(
                element=elt, ionization_stage=1, wavelength_nm=c,
                A_ki=1e8, g_k=6, g_i=4, E_k_ev=3.0, E_i_ev=0.0,
            )
        )

    mock_db = MagicMock()

    def get_transitions(element, **kwargs):
        return transitions.get(element, [])

    mock_db.get_transitions.side_effect = get_transitions
    mock_db.get_available_elements.return_value = list(transitions.keys())

    strict, lax = _make_identifier_pair(
        mock_db,
        tooth_activation_threshold=0.5,
        tier2_tooth_activation_threshold=0.7,
    )

    strict_result = strict.identify(wavelength, intensity)
    lax_result = lax.identify(wavelength, intensity)

    # Build comparable per-element views.
    def index_by_element(result):
        return {e.element: e for e in result.all_elements}

    strict_by = index_by_element(strict_result)
    lax_by = index_by_element(lax_result)

    for elt in ("Fe", "Ti", "Si"):
        assert elt in strict_by and elt in lax_by, (
            f"{elt} not present in identification result (test fixture bug)"
        )
        # The active-tooth count, fingerprint score, detected flag, and
        # matched-line count must match between strict and lax for
        # non-Tier-2 elements.
        assert (
            strict_by[elt].metadata.get("n_active_teeth", 0)
            == lax_by[elt].metadata.get("n_active_teeth", 0)
        ), (
            f"n_active_teeth for {elt} diverged between strict and lax — "
            f"strict_tier2 must not change non-Tier-2 elements."
        )
        assert strict_by[elt].score == lax_by[elt].score, (
            f"score for {elt} diverged between strict and lax — "
            f"strict_tier2 must not change non-Tier-2 elements."
        )
        assert strict_by[elt].detected == lax_by[elt].detected, (
            f"detected flag for {elt} diverged between strict and lax."
        )
        assert strict_by[elt].n_matched_lines == lax_by[elt].n_matched_lines, (
            f"n_matched_lines for {elt} diverged between strict and lax."
        )


def test_tier2_strict_disabled_matches_baseline():
    """strict_tier2=False reverts to byte-identical pre-PR behavior.

    A test with Mn lines in the spectrum — when ``strict_tier2`` is OFF
    the identifier must use the global ``tooth_activation_threshold``
    for Mn (same as Fe), confirming the strict-tier2 path is opt-out
    rather than baked-in.
    """
    mn1 = Transition(
        element="Mn", ionization_stage=1, wavelength_nm=403.0,
        A_ki=1e8, g_k=6, g_i=4, E_k_ev=3.0, E_i_ev=0.0,
    )
    mn2 = Transition(
        element="Mn", ionization_stage=1, wavelength_nm=404.0,
        A_ki=1e8, g_k=6, g_i=4, E_k_ev=3.0, E_i_ev=0.0,
    )

    mock_db = MagicMock()

    def get_transitions(element, **kwargs):
        if element == "Mn":
            return [mn1, mn2]
        return []

    mock_db.get_transitions.side_effect = get_transitions
    mock_db.get_available_elements.return_value = ["Mn"]

    wavelength, intensity = _moderate_snr_spectrum(seed=3)

    # Opt-out path: strict_tier2=False AND the per-tooth threshold
    # explicit at the global default. Effective threshold for Mn is the
    # global 0.5 — same as Fe.
    id_lax = CombIdentifier(
        mock_db,
        strict_tier2=False,
        tooth_activation_threshold=0.5,
        tier2_tooth_activation_threshold=0.7,
    )
    assert id_lax._tier2_effective_activation_threshold("Mn") == 0.5
    assert id_lax._tier2_effective_activation_threshold("Fe") == 0.5

    id_strict = CombIdentifier(
        mock_db,
        strict_tier2=True,
        tooth_activation_threshold=0.5,
        tier2_tooth_activation_threshold=0.7,
    )
    assert id_strict._tier2_effective_activation_threshold("Mn") == 0.7
    assert id_strict._tier2_effective_activation_threshold("Na") == 0.7
    assert id_strict._tier2_effective_activation_threshold("K") == 0.7
    # Non-Tier-2 still gets the global threshold.
    assert id_strict._tier2_effective_activation_threshold("Fe") == 0.5
    assert id_strict._tier2_effective_activation_threshold("Ti") == 0.5
    assert id_strict._tier2_effective_activation_threshold("Si") == 0.5


def test_tier2_widen_only_never_tightens_global():
    """If global threshold > tier2 threshold, gate must NOT be loosened.

    Mirrors PR #154's widen-only pattern: ``max(global, tier2)``. Calling
    with global=0.9 and tier2=0.7 must yield effective threshold 0.9 for
    Mn/Na/K, NOT 0.7.
    """
    mock_db = MagicMock()
    mock_db.get_available_elements.return_value = []
    mock_db.get_transitions.return_value = []

    identifier = CombIdentifier(
        mock_db,
        strict_tier2=True,
        tooth_activation_threshold=0.9,
        tier2_tooth_activation_threshold=0.7,
    )
    assert identifier._tier2_effective_activation_threshold("Mn") == 0.9
    assert identifier._tier2_effective_activation_threshold("Na") == 0.9
    assert identifier._tier2_effective_activation_threshold("K") == 0.9


def test_tier2_defaults_are_strict():
    """Default constructor turns on strict_tier2 with threshold 0.7."""
    mock_db = MagicMock()
    mock_db.get_available_elements.return_value = []
    mock_db.get_transitions.return_value = []

    identifier = CombIdentifier(mock_db)
    assert identifier.strict_tier2 is True
    assert identifier.tier2_tooth_activation_threshold == 0.7
    # Global default lowered to 0.3 for improved recall (was 0.5).
    assert identifier.tooth_activation_threshold == 0.3
    # And the effective threshold for Mn/Na/K is 0.7.
    for elt in ("Mn", "Na", "K"):
        assert identifier._tier2_effective_activation_threshold(elt) == 0.7
    # Non-Tier-2 sees the global 0.5.
    for elt in ("Fe", "Ti", "Si", "Ca", "Al"):
        assert identifier._tier2_effective_activation_threshold(elt) == 0.3
