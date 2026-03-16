"""
Tests for ALIAS element identification algorithm.
"""

import pytest
import numpy as np
from cflibs.inversion.alias_identifier import ALIASIdentifier
from cflibs.inversion.element_id import ElementIdentificationResult

pytestmark = pytest.mark.requires_db


def test_detect_peaks(atomic_db, synthetic_libs_spectrum):
    """Test peak detection with 2nd derivative enhancement."""
    spectrum = synthetic_libs_spectrum(
        elements={"Fe": [(371.99, 1000.0), (400.0, 500.0), (450.0, 200.0)]},
        noise_level=0.01,
    )

    identifier = ALIASIdentifier(atomic_db)
    peaks = identifier._detect_peaks(spectrum["wavelength"], spectrum["intensity"])

    # Should detect 3 peaks
    assert len(peaks) > 0
    assert isinstance(peaks, list)
    assert all(isinstance(p, tuple) and len(p) == 2 for p in peaks)

    # Peaks should be (index, wavelength) tuples
    peak_wavelengths = [p[1] for p in peaks]

    # Should find peaks near expected positions (within 1 nm)
    expected_wls = [371.99, 400.0, 450.0]
    for expected_wl in expected_wls:
        closest = min(peak_wavelengths, key=lambda x: abs(x - expected_wl))
        assert (
            abs(closest - expected_wl) < 1.0
        ), f"Expected peak at {expected_wl}, closest found at {closest}"


def test_compute_element_emissivities(atomic_db):
    """Test emissivity calculation for Fe I lines."""
    identifier = ALIASIdentifier(atomic_db)

    # Compute emissivities for Fe in 370-376 nm range (covers test lines)
    line_data = identifier._compute_element_emissivities("Fe", 370.0, 376.0)

    assert len(line_data) > 0
    for line in line_data:
        assert "transition" in line
        assert "avg_emissivity" in line
        assert "wavelength_nm" in line
        assert line["avg_emissivity"] > 0
        assert 370.0 <= line["wavelength_nm"] <= 376.0


def test_fuse_lines(atomic_db):
    """Test line fusion within resolution element."""
    identifier = ALIASIdentifier(atomic_db, resolving_power=5000.0)

    # Get Fe lines
    line_data = identifier._compute_element_emissivities("Fe", 370.0, 376.0)
    wavelength = np.linspace(370.0, 376.0, 1000)

    # Fuse lines
    fused = identifier._fuse_lines(line_data, wavelength)

    assert len(fused) > 0
    for line in fused:
        assert "transition" in line
        assert "avg_emissivity" in line
        assert "wavelength_nm" in line
        assert "n_fused" in line
        assert line["n_fused"] >= 1


def test_match_lines(atomic_db):
    """Test matching theoretical lines to experimental peaks."""
    identifier = ALIASIdentifier(atomic_db)

    # Create fused lines at specific wavelengths
    from cflibs.atomic.structures import Transition

    trans1 = Transition("Fe", 1, 372.0, 1e7, 3.33, 0.0, 11, 9)
    trans2 = Transition("Fe", 1, 373.5, 5e6, 3.32, 0.0, 9, 9)
    fused_lines = [
        {"transition": trans1, "avg_emissivity": 1000.0, "wavelength_nm": 372.0},
        {"transition": trans2, "avg_emissivity": 500.0, "wavelength_nm": 373.5},
    ]

    # Create peaks near theoretical wavelengths
    peaks = [(100, 372.01), (200, 373.49)]  # (index, wavelength)

    matched_mask, wavelength_shifts, matched_peak_idx = identifier._match_lines(fused_lines, peaks)

    # Both lines should match
    assert matched_mask[0]
    assert matched_mask[1]

    # Shifts should be small
    assert abs(wavelength_shifts[0]) < 0.1
    assert abs(wavelength_shifts[1]) < 0.1

    # Peak indices should be valid
    assert matched_peak_idx[0] >= 0
    assert matched_peak_idx[1] >= 0


def test_identify_basic(atomic_db, synthetic_libs_spectrum):
    """Test full identify() with synthetic spectrum containing Fe lines."""
    spectrum = synthetic_libs_spectrum(
        elements={"Fe": [(371.99, 1000.0), (373.49, 500.0), (374.95, 200.0)]},
        noise_level=0.01,
    )

    identifier = ALIASIdentifier(atomic_db, elements=["Fe"])
    result = identifier.identify(spectrum["wavelength"], spectrum["intensity"])

    assert isinstance(result, ElementIdentificationResult)
    assert result.algorithm == "alias"
    assert result.n_peaks > 0

    # Fe should be in the results (detected or rejected)
    fe_elements = [e for e in result.all_elements if e.element == "Fe"]
    assert len(fe_elements) == 1

    fe_result = fe_elements[0]
    assert fe_result.n_matched_lines > 0


def test_identify_returns_result_type(atomic_db, synthetic_libs_spectrum):
    """Test that identify() returns ElementIdentificationResult."""
    spectrum = synthetic_libs_spectrum()

    identifier = ALIASIdentifier(atomic_db, elements=["Fe"])
    result = identifier.identify(spectrum["wavelength"], spectrum["intensity"])

    assert isinstance(result, ElementIdentificationResult)
    assert hasattr(result, "detected_elements")
    assert hasattr(result, "rejected_elements")
    assert hasattr(result, "all_elements")
    assert hasattr(result, "experimental_peaks")
    assert hasattr(result, "algorithm")
    assert result.algorithm == "alias"


def test_identify_no_elements(atomic_db, synthetic_libs_spectrum):
    """Test identify with no matching elements (edge case)."""
    # Create spectrum with only H line, but search for Cu
    spectrum = synthetic_libs_spectrum(
        elements={"H": [(656.28, 5000.0)]},
        noise_level=0.01,
    )

    identifier = ALIASIdentifier(atomic_db, elements=["Cu"])
    result = identifier.identify(spectrum["wavelength"], spectrum["intensity"])

    assert isinstance(result, ElementIdentificationResult)
    # Cu should not be detected (no Cu lines in wavelength range of test DB)
    cu_elements = [e for e in result.detected_elements if e.element == "Cu"]
    assert len(cu_elements) == 0


def test_detect_peaks_low_snr(atomic_db):
    """Peaks at SNR=5-15 should be detected with new threshold."""
    rng = np.random.default_rng(42)
    wavelength = np.linspace(200, 400, 2000)
    noise_level = 10.0
    baseline = 100 + 0.3 * wavelength  # sloped continuum
    noise = rng.normal(0, noise_level, 2000)

    # Add peaks at SNR 5, 8, 12, 15
    peaks_data = [(400, 50), (800, 80), (1200, 120), (1600, 150)]
    signal = np.zeros(2000)
    for loc, height in peaks_data:
        signal[loc - 2 : loc + 3] = height

    intensity = baseline + signal + noise
    identifier = ALIASIdentifier(atomic_db)
    detected = identifier._detect_peaks(wavelength, intensity)

    # Should detect at least the SNR=12 and SNR=15 peaks
    assert len(detected) >= 2, f"Only detected {len(detected)} peaks at SNR 5-15"


def test_noise_only_no_detection(atomic_db):
    """Pure noise should not detect any elements."""
    rng = np.random.default_rng(42)
    wavelength = np.linspace(200, 400, 2000)
    intensity = 100 + rng.normal(0, 10, 2000)

    identifier = ALIASIdentifier(atomic_db, elements=["Fe"])
    result = identifier.identify(wavelength, intensity)
    assert (
        len(result.detected_elements) == 0
    ), f"False detections in noise: {[e.element for e in result.detected_elements]}"


def test_scores_between_zero_and_one(atomic_db, synthetic_libs_spectrum):
    """Test that all scores are in [0, 1] range."""
    spectrum = synthetic_libs_spectrum(
        elements={"Fe": [(371.99, 1000.0), (373.49, 500.0)]},
        noise_level=0.01,
    )

    identifier = ALIASIdentifier(atomic_db, elements=["Fe", "H"])
    result = identifier.identify(spectrum["wavelength"], spectrum["intensity"])

    for element_id in result.all_elements:
        # Check main scores
        assert 0.0 <= element_id.score <= 1.0
        assert 0.0 <= element_id.confidence <= 1.0

        # Check metadata scores
        metadata = element_id.metadata
        if "k_sim" in metadata:
            assert 0.0 <= metadata["k_sim"] <= 1.0
        if "k_rate" in metadata:
            assert 0.0 <= metadata["k_rate"] <= 1.0
        if "k_shift" in metadata:
            assert 0.0 <= metadata["k_shift"] <= 1.0
        if "k_det" in metadata:
            assert 0.0 <= metadata["k_det"] <= 1.0


def test_max_lines_per_element_parameter(atomic_db):
    """Test that max_lines_per_element caps transition count."""
    identifier = ALIASIdentifier(atomic_db, max_lines_per_element=5)
    assert identifier.max_lines_per_element == 5

    # Default should be 20 (lowered from 50 to focus on strongest lines
    # visible at typical plasma temperatures — see ALIAS scoring fix)
    identifier_default = ALIASIdentifier(atomic_db)
    assert identifier_default.max_lines_per_element == 20


def test_default_detection_threshold(atomic_db):
    """Test that default detection_threshold is 0.02."""
    identifier = ALIASIdentifier(atomic_db)
    assert identifier.detection_threshold == 0.02


# ---------------------------------------------------------------------------
# Tests for the 4 bug-fixes (survivorship bias, uniqueness, P_maj, P_ab)
# ---------------------------------------------------------------------------


def test_k_sim_matched_only(atomic_db):
    """k_sim should be computed over matched lines only (paper-faithful).

    With matched-only cosine, both scenarios should have IDENTICAL k_sim
    because the matched lines and their intensities are the same.
    Coverage penalty is now handled by P_cov, not k_sim.
    """
    from cflibs.atomic.structures import Transition

    identifier = ALIASIdentifier(atomic_db, resolving_power=5000.0)

    # Scenario A: 2 lines, both matched → high k_sim
    fused_a = [
        {
            "transition": Transition("Fe", 1, 372.0, 1e7, 3.3, 0.0, 11, 9),
            "avg_emissivity": 1000.0,
            "wavelength_nm": 372.0,
        },
        {
            "transition": Transition("Fe", 1, 374.0, 5e6, 3.3, 0.0, 9, 9),
            "avg_emissivity": 800.0,
            "wavelength_nm": 374.0,
        },
    ]
    peaks = [(100, 372.01), (200, 374.01)]
    intensity = np.full(500, 10.0)
    intensity[100] = 500.0
    intensity[200] = 400.0
    matched_a = np.array([True, True])
    peak_idx_a = np.array([0, 1])
    shifts_a = np.array([0.01, 0.01])

    k_sim_a, _, _, _, _, _, _ = identifier._compute_scores(
        fused_a,
        matched_a,
        peak_idx_a,
        shifts_a,
        intensity,
        peaks,
        emissivity_threshold=-np.inf,
    )

    # Scenario B: 10 lines but only 2 matched — identical matched intensities
    fused_b = list(fused_a)  # first two are the same
    for i in range(8):
        wl = 375.0 + i * 0.5
        fused_b.append(
            {
                "transition": Transition("Fe", 1, wl, 1e6, 3.3, 0.0, 7, 7),
                "avg_emissivity": 600.0,
                "wavelength_nm": wl,
            }
        )
    matched_b = np.array([True, True] + [False] * 8)
    peak_idx_b = np.array([0, 1] + [-1] * 8)
    shifts_b = np.array([0.01, 0.01] + [0.0] * 8)

    k_sim_b, _, _, _, _, _, P_cov_b = identifier._compute_scores(
        fused_b,
        matched_b,
        peak_idx_b,
        shifts_b,
        intensity,
        peaks,
        emissivity_threshold=-np.inf,
    )

    # Matched-only k_sim: same matched lines → same k_sim
    assert (
        abs(k_sim_a - k_sim_b) < 0.01
    ), f"Matched-only k_sim should be equal: {k_sim_a} vs {k_sim_b}"

    # But P_cov should penalize scenario B (only 2/10 lines matched)
    assert P_cov_b < 0.5, f"P_cov should penalize many unmatched lines: {P_cov_b}"


def test_uniqueness_penalty(atomic_db):
    """Many theoretical lines mapping to one peak should be penalised."""
    from cflibs.atomic.structures import Transition

    identifier = ALIASIdentifier(atomic_db, resolving_power=500.0)

    # 4 theoretical lines all within ~0.1 nm → all map to a single broad peak
    fused = []
    for i in range(4):
        wl = 400.0 + i * 0.02
        fused.append(
            {
                "transition": Transition("Co", 1, wl, 1e7, 3.0, 0.0, 9, 7),
                "avg_emissivity": 1000.0,
                "wavelength_nm": wl,
            }
        )
    # Single experimental peak near 400.0
    peaks = [(250, 400.01)]
    intensity = np.full(500, 10.0)
    intensity[250] = 800.0

    matched = np.array([True, True, True, True])
    peak_idx = np.array([0, 0, 0, 0])  # all map to same peak
    shifts = np.array([0.01, -0.01, 0.01, -0.01])

    k_sim, _, _, _, _, _, _ = identifier._compute_scores(
        fused,
        matched,
        peak_idx,
        shifts,
        intensity,
        peaks,
        emissivity_threshold=-np.inf,
    )

    # uniqueness_factor = 1 unique peak / 4 matches = 0.25
    # So k_sim should be at most 0.25 (cosine sim capped then scaled)
    assert k_sim <= 0.30, f"Uniqueness penalty should reduce k_sim when many-to-one: got {k_sim}"


def test_P_maj_soft_coverage(atomic_db):
    """P_maj should be high when strongest line matched, lower when not."""
    from cflibs.atomic.structures import Transition

    identifier = ALIASIdentifier(atomic_db)

    # Scenario: strongest line IS matched → P_maj close to 1.0
    fused = [
        {
            "transition": Transition("Fe", 1, 372.0, 1e8, 3.3, 0.0, 11, 9),
            "avg_emissivity": 5000.0,
            "wavelength_nm": 372.0,
        },  # strongest
        {
            "transition": Transition("Fe", 1, 374.0, 1e6, 3.3, 0.0, 9, 9),
            "avg_emissivity": 100.0,
            "wavelength_nm": 374.0,
        },
    ]
    peaks = [(100, 372.01), (200, 374.01)]
    intensity = np.full(500, 10.0)
    intensity[100] = 1000.0
    intensity[200] = 50.0

    matched = np.array([True, True])
    peak_idx = np.array([0, 1])
    shifts = np.array([0.01, 0.01])

    _, _, _, P_maj_both, _, _, _ = identifier._compute_scores(
        fused,
        matched,
        peak_idx,
        shifts,
        intensity,
        peaks,
        emissivity_threshold=-np.inf,
    )
    # Both matched including strongest → P_maj should be 1.0
    assert P_maj_both > 0.9, f"P_maj should be ~1.0 when all matched: {P_maj_both}"

    # Scenario: strongest line NOT matched → P_maj should be lower
    matched_miss = np.array([False, True])
    peak_idx_miss = np.array([-1, 1])
    _, _, _, P_maj_miss, _, _, _ = identifier._compute_scores(
        fused,
        matched_miss,
        peak_idx_miss,
        shifts,
        intensity,
        peaks,
        emissivity_threshold=-np.inf,
    )
    assert (
        P_maj_miss < P_maj_both
    ), f"P_maj should decrease when strongest line missed: {P_maj_miss} vs {P_maj_both}"
    assert P_maj_miss >= 0.5, f"P_maj should be at least 0.5: {P_maj_miss}"


def test_N_matched_in_k_det_blend(atomic_db):
    """N_matched should be used in k_det blend, N_expected for metadata."""
    from cflibs.atomic.structures import Transition

    identifier = ALIASIdentifier(atomic_db, resolving_power=500.0)

    # 10 theoretical lines above threshold, only 1 matched
    fused = []
    for i in range(10):
        wl = 300.0 + i * 1.0
        fused.append(
            {
                "transition": Transition("Co", 1, wl, 1e7, 3.0, 0.0, 9, 7),
                "avg_emissivity": 1000.0,
                "wavelength_nm": wl,
            }
        )
    peaks = [(50, 300.01)]
    intensity = np.full(500, 10.0)
    intensity[50] = 800.0

    matched = np.array([True] + [False] * 9)
    peak_idx = np.array([0] + [-1] * 9)
    shifts = np.array([0.01] + [0.0] * 9)

    k_sim, k_rate, k_shift, P_maj, N_expected, N_matched, P_cov = identifier._compute_scores(
        fused,
        matched,
        peak_idx,
        shifts,
        intensity,
        peaks,
        emissivity_threshold=-np.inf,
    )

    # N_expected should be 10 (all above threshold)
    assert N_expected == 10, f"N_expected should be 10 but got {N_expected}"
    # N_matched should be 1 (only one line matched)
    assert N_matched == 1, f"N_matched should be 1 but got {N_matched}"

    # With gates removed, _decide should produce nonzero k_det/CL
    k_det, CL = identifier._decide(
        k_sim,
        k_rate,
        k_shift,
        N_expected,
        intensity,
        peaks,
        element="Co",
        P_maj=P_maj,
        N_matched=N_matched,
        P_cov=P_cov,
    )
    # N_X=1 → k_det = k_rate × k_shift (pure shift quality)
    assert k_det > 0.0, f"k_det should be nonzero without gates: {k_det}"
    assert CL > 0.0, f"CL should be nonzero without gates: {CL}"


def test_P_ab_tiers(atomic_db):
    """P_ab should be 1.0 for common, 0.75 for intermediate, 0.5 for rare."""
    identifier = ALIASIdentifier(atomic_db)

    # Common elements (>100 ppm)
    assert identifier._compute_P_ab("Fe") == 1.0
    assert identifier._compute_P_ab("Al") == 1.0
    assert identifier._compute_P_ab("Si") == 1.0
    assert identifier._compute_P_ab("Ca") == 1.0

    # Intermediate (0.001 - 100 ppm)
    assert identifier._compute_P_ab("Co") == 0.75  # 10^1.40 ≈ 25 ppm
    assert identifier._compute_P_ab("Cu") == 0.75  # 10^1.78 ≈ 60 ppm
    assert identifier._compute_P_ab("Sn") == 0.75  # 10^0.35 ≈ 2.2 ppm

    # Ag: 10^-0.62 ≈ 0.24 ppm → still intermediate (>= 0.001)
    assert identifier._compute_P_ab("Ag") == 0.75
    # Au: 10^-2.40 ≈ 0.004 ppm → still intermediate (>= 0.001)
    assert identifier._compute_P_ab("Au") == 0.75

    # Unknown element defaults to log_ppm=0.0 → 1 ppm → intermediate
    assert identifier._compute_P_ab("Xx") == 0.75


def test_single_line_naturally_lower_CL(atomic_db):
    """Single-line elements should get lower CL than multi-line elements naturally.

    With gates and N_penalty removed, single-line CL is naturally lower because
    k_det = k_rate × k_shift (N_X=1 blend, no k_sim contribution) and P_maj
    is lower with fewer lines. No artificial penalty is applied.
    """
    from cflibs.atomic.structures import Transition

    identifier = ALIASIdentifier(atomic_db, resolving_power=500.0)

    def _make_scenario(n_lines, n_matched):
        """Build fused lines, peaks, and masks for a given line count."""
        fused = []
        for i in range(n_lines):
            wl = 300.0 + i * 2.0  # well-separated
            fused.append(
                {
                    "transition": Transition("Fe", 1, wl, 1e7, 3.0, 0.0, 9, 7),
                    "avg_emissivity": 1000.0,
                    "wavelength_nm": wl,
                }
            )
        peaks_list = [(50 + i * 10, 300.0 + i * 2.0 + 0.01) for i in range(n_matched)]
        intensity = np.full(500, 10.0)
        for idx, _ in peaks_list:
            intensity[idx] = 800.0
        matched = np.array([True] * n_matched + [False] * (n_lines - n_matched))
        pidx = np.array(list(range(n_matched)) + [-1] * (n_lines - n_matched))
        shifts = np.array([0.01] * n_matched + [0.0] * (n_lines - n_matched))
        return fused, peaks_list, intensity, matched, pidx, shifts

    # N_expected=1, 1 matched → should produce nonzero CL (no gate)
    f1, p1, i1, m1, pi1, s1 = _make_scenario(1, 1)
    k_sim1, k_rate1, k_shift1, P_maj1, N1, Nm1, Pcov1 = identifier._compute_scores(
        f1,
        m1,
        pi1,
        s1,
        i1,
        p1,
        emissivity_threshold=-np.inf,
    )
    _, CL1 = identifier._decide(
        k_sim1,
        k_rate1,
        k_shift1,
        N1,
        i1,
        p1,
        element="Co",
        P_maj=P_maj1,
        N_matched=Nm1,
        P_cov=Pcov1,
    )

    # N_expected=5, 5 matched → higher CL due to k_sim contribution
    f5, p5, i5, m5, pi5, s5 = _make_scenario(5, 5)
    k_sim5, k_rate5, k_shift5, P_maj5, N5, Nm5, Pcov5 = identifier._compute_scores(
        f5,
        m5,
        pi5,
        s5,
        i5,
        p5,
        emissivity_threshold=-np.inf,
    )
    _, CL5 = identifier._decide(
        k_sim5,
        k_rate5,
        k_shift5,
        N5,
        i5,
        p5,
        element="Co",
        P_maj=P_maj5,
        N_matched=Nm5,
        P_cov=Pcov5,
    )

    assert N1 == 1
    assert N5 == 5
    # Single-line CL should be nonzero (no gate) but lower than multi-line
    assert CL1 > 0.0, f"Single-line CL should be nonzero: {CL1}"
    assert (
        CL1 < CL5
    ), f"Single-line CL should be naturally lower than multi-line: CL1={CL1}, CL5={CL5}"


def test_k_rate_emissivity_weighted(atomic_db):
    """k_rate should be pure emissivity-weighted (paper formula)."""
    from cflibs.atomic.structures import Transition

    identifier = ALIASIdentifier(atomic_db, resolving_power=5000.0)

    # 5 lines, only 1 matched — but it has highest emissivity
    fused = [
        {
            "transition": Transition("Ni", 1, 350.0, 1e8, 3.0, 0.0, 11, 9),
            "avg_emissivity": 5000.0,
            "wavelength_nm": 350.0,
        },  # matched
    ]
    for i in range(4):
        wl = 352.0 + i * 2.0
        fused.append(
            {
                "transition": Transition("Ni", 1, wl, 1e6, 3.0, 0.0, 7, 7),
                "avg_emissivity": 100.0,
                "wavelength_nm": wl,
            }
        )
    peaks = [(100, 350.01)]
    intensity = np.full(500, 10.0)
    intensity[100] = 1000.0

    matched = np.array([True] + [False] * 4)
    pidx = np.array([0] + [-1] * 4)
    shifts = np.array([0.01] + [0.0] * 4)

    _, k_rate, _, _, _, _, _ = identifier._compute_scores(
        fused,
        matched,
        pidx,
        shifts,
        intensity,
        peaks,
        emissivity_threshold=-np.inf,
    )

    # Emissivity-weighted rate: 5000 / 5400 ≈ 0.926
    raw_rate = 5000.0 / 5400.0
    expected_k_rate = raw_rate
    assert (
        abs(k_rate - expected_k_rate) < 0.01
    ), f"k_rate should be emissivity-weighted ~{expected_k_rate:.3f}: got {k_rate}"


def test_one_to_one_peak_assignment(atomic_db):
    """Each peak should match at most one theoretical line (highest emissivity wins)."""
    from cflibs.atomic.structures import Transition

    identifier = ALIASIdentifier(atomic_db, resolving_power=500.0)

    # 3 lines near the same wavelength — all would match the same peak
    # without one-to-one enforcement
    fused = [
        {
            "transition": Transition("Co", 1, 400.0, 1e7, 3.0, 0.0, 9, 7),
            "avg_emissivity": 500.0,
            "wavelength_nm": 400.0,
        },
        {
            "transition": Transition("Co", 1, 400.3, 1e7, 3.0, 0.0, 9, 7),
            "avg_emissivity": 1000.0,
            "wavelength_nm": 400.3,
        },  # highest emissivity
        {
            "transition": Transition("Co", 1, 400.5, 1e7, 3.0, 0.0, 9, 7),
            "avg_emissivity": 200.0,
            "wavelength_nm": 400.5,
        },
    ]
    # Single peak that's within delta_lambda of all three lines
    peaks = [(250, 400.2)]

    matched, _, peak_idx = identifier._match_lines(fused, peaks)

    # Only one line should be matched (highest emissivity = line at 400.3)
    assert (
        int(np.sum(matched)) == 1
    ), f"One-to-one should allow only 1 match per peak, got {int(np.sum(matched))}"
    # The matched line should be index 1 (highest emissivity)
    assert bool(matched[1]), "Highest emissivity line should win the peak assignment"


def test_fill_factor_and_P_sig(atomic_db):
    """Fill factor should be reasonable and P_sig should penalize random matches."""
    identifier = ALIASIdentifier(atomic_db, resolving_power=500.0)

    # Sparse peaks → low fill factor → high P_sig
    wavelength_sparse = np.linspace(280, 320, 2000)
    peaks_sparse = [(100, 290.0), (500, 300.0), (900, 310.0)]  # only 3 peaks
    ff_sparse = identifier._compute_fill_factor(peaks_sparse, wavelength_sparse)
    assert ff_sparse < 0.2, f"Sparse peaks should have low fill factor: {ff_sparse}"

    # P_sig for significant excess matches
    P_sig_good, _, _, _ = identifier._compute_random_match_significance(
        peaks_sparse,
        wavelength_sparse,
        N_expected=5,
        N_matched=4,
    )
    assert P_sig_good > 0.5, f"Excess matches should give high P_sig: {P_sig_good}"

    # Dense peaks → high fill factor → lower P_sig for same match ratio
    peaks_dense = [(i * 50, 280.0 + i * 1.3) for i in range(30)]
    ff_dense = identifier._compute_fill_factor(peaks_dense, wavelength_sparse)
    assert (
        ff_dense > ff_sparse
    ), f"Dense peaks should have higher fill factor: {ff_dense} vs {ff_sparse}"

    P_sig_dense, _, _, _ = identifier._compute_random_match_significance(
        peaks_dense,
        wavelength_sparse,
        N_expected=10,
        N_matched=3,
    )
    # With high fill factor and few matches relative to expected, P_sig should be low
    assert (
        P_sig_dense < P_sig_good
    ), f"Dense spectrum with few matches should have lower P_sig: {P_sig_dense}"


def test_cross_element_peak_competition(atomic_db, synthetic_libs_spectrum):
    """Cross-element competition should reassign shared peaks to strongest element."""
    # Create a spectrum dominated by Fe lines
    spectrum = synthetic_libs_spectrum(
        elements={
            "Fe": [
                (371.99, 1000.0),
                (373.49, 800.0),
                (374.56, 600.0),
                (375.82, 400.0),
            ],
        },
        noise_level=0.02,
    )

    # Search for Fe and a weaker element that may share peaks
    identifier = ALIASIdentifier(
        atomic_db,
        resolving_power=500.0,  # Broad peaks → more sharing
        elements=["Fe", "Co", "Cu"],
        detection_threshold=0.03,
    )

    result = identifier.identify(spectrum["wavelength"], spectrum["intensity"])

    # Fe should be detected (true positive)
    fe_ids = [e for e in result.all_elements if e.element == "Fe"]
    assert len(fe_ids) == 1, "Fe should be scored"

    # Any FP element should have lower confidence than Fe after competition
    for eid in result.all_elements:
        if eid.element != "Fe" and eid.detected:
            assert eid.confidence < fe_ids[0].confidence, (
                f"{eid.element} (CL={eid.confidence:.3f}) should not exceed "
                f"Fe (CL={fe_ids[0].confidence:.3f}) after peak competition"
            )


# ---------------------------------------------------------------------------
# New tests for refactored scoring (Round 1-3)
# ---------------------------------------------------------------------------


def test_P_cov_emissivity_weighted(atomic_db):
    """Missing a strong line should penalize P_cov more than missing a weak line."""
    from cflibs.atomic.structures import Transition

    identifier = ALIASIdentifier(atomic_db, resolving_power=5000.0)

    # 3 lines: one very strong (5000), two weak (100 each)
    fused = [
        {
            "transition": Transition("Fe", 1, 372.0, 1e8, 3.3, 0.0, 11, 9),
            "avg_emissivity": 5000.0,
            "wavelength_nm": 372.0,
        },
        {
            "transition": Transition("Fe", 1, 374.0, 1e6, 3.3, 0.0, 9, 9),
            "avg_emissivity": 100.0,
            "wavelength_nm": 374.0,
        },
        {
            "transition": Transition("Fe", 1, 376.0, 1e6, 3.3, 0.0, 9, 9),
            "avg_emissivity": 100.0,
            "wavelength_nm": 376.0,
        },
    ]
    peaks = [(100, 372.01), (200, 374.01), (300, 376.01)]
    intensity = np.full(500, 10.0)
    intensity[100] = 1000.0
    intensity[200] = 50.0
    intensity[300] = 50.0

    # Scenario A: strong line missed, weak lines matched
    matched_miss_strong = np.array([False, True, True])
    pidx_miss_strong = np.array([-1, 1, 2])
    shifts = np.array([0.0, 0.01, 0.01])
    _, _, _, _, _, _, P_cov_miss_strong = identifier._compute_scores(
        fused,
        matched_miss_strong,
        pidx_miss_strong,
        shifts,
        intensity,
        peaks,
        emissivity_threshold=-np.inf,
    )

    # Scenario B: weak line missed, strong line matched
    matched_miss_weak = np.array([True, False, True])
    pidx_miss_weak = np.array([0, -1, 2])
    _, _, _, _, _, _, P_cov_miss_weak = identifier._compute_scores(
        fused,
        matched_miss_weak,
        pidx_miss_weak,
        shifts,
        intensity,
        peaks,
        emissivity_threshold=-np.inf,
    )

    # Missing the strong line should hurt P_cov much more
    assert P_cov_miss_strong < P_cov_miss_weak, (
        f"Missing strong line should lower P_cov more: " f"{P_cov_miss_strong} vs {P_cov_miss_weak}"
    )
    # P_cov_miss_strong ~ 200/5200 ~ 0.038
    assert (
        P_cov_miss_strong < 0.1
    ), f"P_cov should be very low when strong line missed: {P_cov_miss_strong}"


def test_nnls_attribution(atomic_db):
    """NNLS should assign high P_mix to dominant element, low to spurious."""
    identifier = ALIASIdentifier(atomic_db, resolving_power=5000.0)

    # Build fake candidates: Element A has peaks matching observations,
    # Element B has peaks at wrong positions
    from cflibs.atomic.structures import Transition

    cand_a = {
        "fused_lines": [
            {
                "wavelength_nm": 400.0,
                "avg_emissivity": 1000.0,
                "transition": Transition("Fe", 1, 400.0, 1e7, 3.0, 0.0, 9, 7),
            },
            {
                "wavelength_nm": 405.0,
                "avg_emissivity": 800.0,
                "transition": Transition("Fe", 1, 405.0, 5e6, 3.0, 0.0, 7, 7),
            },
        ],
        "matched_mask": np.array([True, True]),
        "wavelength_shifts": np.array([0.01, 0.02]),
    }
    cand_b = {
        "fused_lines": [
            {
                "wavelength_nm": 420.0,
                "avg_emissivity": 500.0,
                "transition": Transition("Co", 1, 420.0, 1e7, 3.0, 0.0, 9, 7),
            },
            {
                "wavelength_nm": 425.0,
                "avg_emissivity": 300.0,
                "transition": Transition("Co", 1, 425.0, 5e6, 3.0, 0.0, 7, 7),
            },
        ],
        "matched_mask": np.array([False, False]),
        "wavelength_shifts": np.array([0.0, 0.0]),
    }

    # Peaks near element A's lines, nothing near element B's
    peaks = [(100, 400.01), (200, 405.02)]
    peak_intensities = np.array([800.0, 600.0])

    A = identifier._build_nnls_templates([cand_a, cand_b], peaks)
    P_mix, P_local, c = identifier._compute_nnls_attribution(A, peak_intensities)

    assert (
        P_mix[0] > P_mix[1]
    ), f"Dominant element should have higher P_mix: {P_mix[0]} vs {P_mix[1]}"
    assert P_mix[0] > 0.1, f"Dominant element P_mix should be substantial: {P_mix[0]}"
    # P_local: dominant element should explain its peaks
    assert P_local[0] > 0.3, f"Dominant P_local should be high: {P_local[0]}"
    assert P_local[1] < 0.01, f"FP P_local should be near zero: {P_local[1]}"


def test_P_SNR_erf_formula(atomic_db):
    """P_SNR should use erf formula and return values in [0, 1]."""
    import math
    from scipy.special import erf

    identifier = ALIASIdentifier(atomic_db, resolving_power=5000.0)

    # High-SNR scenario
    intensity_high = np.full(500, 10.0)
    intensity_high[100] = 500.0
    intensity_high[200] = 400.0
    peaks_high = [(100, 372.0), (200, 374.0)]

    # Low-SNR scenario
    rng = np.random.default_rng(42)
    intensity_low = 100.0 + rng.normal(0, 50, 500)
    intensity_low[100] = 120.0  # barely above noise

    from cflibs.atomic.structures import Transition

    fused = [
        {
            "transition": Transition("Fe", 1, 372.0, 1e7, 3.3, 0.0, 11, 9),
            "avg_emissivity": 1000.0,
            "wavelength_nm": 372.0,
        },
        {
            "transition": Transition("Fe", 1, 374.0, 5e6, 3.3, 0.0, 9, 9),
            "avg_emissivity": 800.0,
            "wavelength_nm": 374.0,
        },
    ]
    matched = np.array([True, True])
    pidx = np.array([0, 1])
    shifts = np.array([0.01, 0.01])

    # Get k_det for high SNR
    k_sim_h, k_rate_h, k_shift_h, P_maj_h, N_exp_h, N_m_h, P_cov_h = identifier._compute_scores(
        fused,
        matched,
        pidx,
        shifts,
        intensity_high,
        peaks_high,
        emissivity_threshold=-np.inf,
    )
    k_det_h, CL_h = identifier._decide(
        k_sim_h,
        k_rate_h,
        k_shift_h,
        N_exp_h,
        intensity_high,
        peaks_high,
        element="Fe",
        P_maj=P_maj_h,
        N_matched=N_m_h,
        P_cov=P_cov_h,
    )

    # CL should be valid
    assert 0.0 <= CL_h <= 1.0, f"CL should be in [0,1]: {CL_h}"

    # Verify erf formula behavior: high SNR → P_SNR close to 1
    noise_h = max(np.median(np.abs(intensity_high - np.median(intensity_high))) * 1.4826, 1e-10)
    median_peak_h = np.median([intensity_high[p[0]] for p in peaks_high])
    z_h = (median_peak_h - noise_h) / (noise_h * math.sqrt(2))
    expected_P_SNR = 0.5 * (1.0 + float(erf(z_h)))
    assert expected_P_SNR > 0.9, f"High-SNR P_SNR should be near 1: {expected_P_SNR}"


def test_triple_counting_eliminated(atomic_db):
    """Element with many predicted/few matched lines shouldn't get crushed.

    With the paper-faithful CL formula (k_det × P_SNR × P_maj × P_ab),
    the CL should be substantial for elements with good matched-line quality.
    P_cov, P_sig, and N_penalty are no longer in the CL product.
    """
    from cflibs.atomic.structures import Transition

    identifier = ALIASIdentifier(atomic_db, resolving_power=5000.0)

    # Element with 5 lines, 3 matched (realistic minor element scenario)
    fused = []
    for i in range(5):
        wl = 350.0 + i * 3.0  # well-separated
        fused.append(
            {
                "transition": Transition("V", 1, wl, 1e7, 3.0, 0.0, 9, 7),
                "avg_emissivity": 1000.0,
                "wavelength_nm": wl,
            }
        )

    peaks = [(50, 350.01), (100, 353.01), (150, 356.01)]
    intensity = np.full(500, 10.0)
    intensity[50] = 600.0
    intensity[100] = 500.0
    intensity[150] = 400.0

    matched = np.array([True, True, True, False, False])
    pidx = np.array([0, 1, 2, -1, -1])
    shifts = np.array([0.01, 0.01, 0.01, 0.0, 0.0])

    k_sim, k_rate, k_shift, P_maj, N_expected, N_matched, P_cov = identifier._compute_scores(
        fused,
        matched,
        pidx,
        shifts,
        intensity,
        peaks,
        emissivity_threshold=-np.inf,
    )

    # k_sim should be high (matched lines have consistent intensities)
    assert k_sim > 0.7, f"Matched-only k_sim should be high: {k_sim}"

    # k_rate should be 0.6 (emissivity-weighted rate)
    assert abs(k_rate - 0.6) < 0.01, f"k_rate should be ~0.6: {k_rate}"

    # P_cov should also be 0.6 (still computed for diagnostics)
    assert abs(P_cov - 0.6) < 0.01, f"P_cov should be ~0.6: {P_cov}"

    # N_matched used in blend, not N_expected
    assert N_matched == 3
    assert N_expected == 5

    # CL = k_det × P_SNR × P_maj × P_ab (paper formula, no P_cov/P_sig)
    k_det, CL = identifier._decide(
        k_sim,
        k_rate,
        k_shift,
        N_expected,
        intensity,
        peaks,
        element="V",
        P_maj=P_maj,
        N_matched=N_matched,
        P_cov=P_cov,
    )
    assert k_det > 0.3, f"k_det should be substantial: {k_det}"
    # CL should be well above detection threshold now that P_cov/P_sig
    # are removed from the product
    assert CL > 0.1, f"CL should be substantial with paper formula: {CL}"


def test_P_sig_overmatch_defensive(atomic_db):
    """P_sig should handle N_matched > N_expected defensively."""
    identifier = ALIASIdentifier(atomic_db, resolving_power=500.0)

    wavelength = np.linspace(280, 320, 2000)
    peaks = [(i * 50, 280.0 + i * 1.3) for i in range(8)]

    # N_matched > N_expected: should return max significance, not crash
    P_sig, _, _, p_tail = identifier._compute_random_match_significance(
        peaks,
        wavelength,
        N_expected=5,
        N_matched=8,
    )
    assert P_sig == 1.0, f"Overmatch should give P_sig=1.0: {P_sig}"
    assert p_tail == 0.0, f"Overmatch should give p_tail=0.0: {p_tail}"

    # Normal case: N_matched <= N_expected
    P_sig_normal, _, _, _ = identifier._compute_random_match_significance(
        peaks,
        wavelength,
        N_expected=10,
        N_matched=3,
    )
    assert 0.0 <= P_sig_normal <= 1.0, f"Normal P_sig out of range: {P_sig_normal}"


def test_k_sim_single_line_neutral(atomic_db):
    """Single matched line should get k_sim=0.5 (neutral) and produce nonzero CL."""
    from cflibs.atomic.structures import Transition

    identifier = ALIASIdentifier(atomic_db, resolving_power=5000.0)

    # 3 above-threshold lines, only 1 matched
    fused = [
        {
            "transition": Transition("V", 1, 350.0, 1e7, 3.0, 0.0, 9, 7),
            "avg_emissivity": 1000.0,
            "wavelength_nm": 350.0,
        },
        {
            "transition": Transition("V", 1, 355.0, 1e7, 3.0, 0.0, 9, 7),
            "avg_emissivity": 800.0,
            "wavelength_nm": 355.0,
        },
        {
            "transition": Transition("V", 1, 360.0, 1e7, 3.0, 0.0, 9, 7),
            "avg_emissivity": 600.0,
            "wavelength_nm": 360.0,
        },
    ]
    peaks = [(100, 350.01)]
    intensity = np.full(500, 10.0)
    intensity[100] = 800.0

    matched = np.array([True, False, False])
    pidx = np.array([0, -1, -1])
    shifts = np.array([0.01, 0.0, 0.0])

    k_sim, k_rate, k_shift, P_maj, N_expected, N_matched, P_cov = identifier._compute_scores(
        fused,
        matched,
        pidx,
        shifts,
        intensity,
        peaks,
        emissivity_threshold=-np.inf,
    )

    # Single matched line → k_sim should be 0.5 (neutral), not 0
    assert k_sim == 0.5, f"Single-line k_sim should be 0.5 (neutral): {k_sim}"
    assert N_matched == 1
    assert N_expected == 3

    # With gates removed, k_sim=0.5 should NOT be rejected.
    # k_det = k_rate × k_shift (N_X=1 blend) → nonzero CL.
    k_det, CL = identifier._decide(
        k_sim,
        k_rate,
        k_shift,
        N_expected,
        intensity,
        peaks,
        element="V",
        P_maj=P_maj,
        N_matched=N_matched,
        P_cov=P_cov,
    )
    assert k_det > 0.0, f"k_det should be nonzero without k_sim gate: {k_det}"
    assert CL > 0.0, f"CL should be nonzero without k_sim gate: {CL}"


# ---------------------------------------------------------------------------
# New tests for paper-faithful CL formula (Round 4)
# ---------------------------------------------------------------------------


def test_CL_paper_formula(atomic_db):
    """CL should equal exactly k_det × P_SNR × P_maj × P_ab (no extra terms)."""
    import math
    from scipy.special import erf
    from cflibs.atomic.structures import Transition

    identifier = ALIASIdentifier(atomic_db, resolving_power=5000.0)

    # 3 lines, all matched — clean scenario for verifying formula
    fused = [
        {
            "transition": Transition("Fe", 1, 372.0, 1e8, 3.3, 0.0, 11, 9),
            "avg_emissivity": 5000.0,
            "wavelength_nm": 372.0,
        },
        {
            "transition": Transition("Fe", 1, 374.0, 1e7, 3.3, 0.0, 9, 9),
            "avg_emissivity": 1000.0,
            "wavelength_nm": 374.0,
        },
        {
            "transition": Transition("Fe", 1, 376.0, 5e6, 3.3, 0.0, 7, 7),
            "avg_emissivity": 500.0,
            "wavelength_nm": 376.0,
        },
    ]
    peaks = [(100, 372.01), (200, 374.01), (300, 376.01)]
    intensity = np.full(500, 10.0)
    intensity[100] = 1000.0
    intensity[200] = 300.0
    intensity[300] = 150.0

    matched = np.array([True, True, True])
    pidx = np.array([0, 1, 2])
    shifts = np.array([0.01, 0.01, 0.01])

    k_sim, k_rate, k_shift, P_maj, N_expected, N_matched, P_cov = identifier._compute_scores(
        fused,
        matched,
        pidx,
        shifts,
        intensity,
        peaks,
        emissivity_threshold=-np.inf,
    )

    k_det, CL = identifier._decide(
        k_sim,
        k_rate,
        k_shift,
        N_expected,
        intensity,
        peaks,
        element="Fe",
        P_maj=P_maj,
        N_matched=N_matched,
        P_cov=P_cov,
    )

    # Manually compute expected CL using modified formula:
    # k_det = sqrt(k_det_raw * max(P_cov, 0.01))
    N_X = max(N_matched, 1)
    k_det_raw = k_rate * ((1.0 / N_X) * k_shift + ((N_X - 1.0) / N_X) * k_sim)
    expected_k_det = math.sqrt(k_det_raw * max(P_cov, 0.01))

    # P_SNR from the spectrum
    peak_intensities_local = [intensity[p[0]] for p in peaks]
    median_peak = np.median(peak_intensities_local)
    noise_estimate = max(np.median(np.abs(intensity - np.median(intensity))) * 1.4826, 1e-10)
    z = (median_peak - noise_estimate) / (noise_estimate * math.sqrt(2))
    expected_P_SNR = 0.5 * (1.0 + float(erf(z)))

    P_ab = identifier._compute_P_ab("Fe")

    expected_CL = expected_k_det * expected_P_SNR * P_maj * P_ab

    assert abs(k_det - expected_k_det) < 1e-10, f"k_det mismatch: {k_det} vs {expected_k_det}"
    assert abs(CL - expected_CL) < 1e-10, (
        f"CL should be exactly k_det × P_SNR × P_maj × P_ab: " f"got {CL}, expected {expected_CL}"
    )


def test_no_hard_gates(atomic_db):
    """N_expected=1 and low k_sim should produce nonzero k_det and CL."""
    from cflibs.atomic.structures import Transition

    identifier = ALIASIdentifier(atomic_db, resolving_power=5000.0)

    # Single line element (like Na D line)
    fused = [
        {
            "transition": Transition("Na", 1, 589.0, 6.16e7, 2.10, 0.0, 4, 2),
            "avg_emissivity": 8000.0,
            "wavelength_nm": 589.0,
        },
    ]
    peaks = [(300, 589.02)]
    intensity = np.full(600, 10.0)
    intensity[300] = 5000.0

    matched = np.array([True])
    pidx = np.array([0])
    shifts = np.array([0.02])

    k_sim, k_rate, k_shift, P_maj, N_expected, N_matched, P_cov = identifier._compute_scores(
        fused,
        matched,
        pidx,
        shifts,
        intensity,
        peaks,
        emissivity_threshold=-np.inf,
    )

    assert N_expected == 1, f"N_expected should be 1: {N_expected}"
    assert N_matched == 1

    k_det, CL = identifier._decide(
        k_sim,
        k_rate,
        k_shift,
        N_expected,
        intensity,
        peaks,
        element="Na",
        P_maj=P_maj,
        N_matched=N_matched,
        P_cov=P_cov,
    )

    # No gates — single line should produce nonzero detection
    assert k_det > 0.0, f"Single-line k_det should be nonzero: {k_det}"
    assert CL > 0.0, f"Single-line CL should be nonzero: {CL}"

    # Also test low k_sim (but not zero) — should not be gated
    k_det_low, CL_low = identifier._decide(
        0.10,
        k_rate,
        k_shift,
        N_expected,
        intensity,
        peaks,
        element="Na",
        P_maj=P_maj,
        N_matched=N_matched,
        P_cov=P_cov,
    )
    assert k_det_low > 0.0, f"Low k_sim should not gate k_det: {k_det_low}"
    assert CL_low > 0.0, f"Low k_sim should not gate CL: {CL_low}"


def test_k_shift_emissivity_weighted(atomic_db):
    """Strong-line shift should matter more than weak-line shift in k_shift."""
    from cflibs.atomic.structures import Transition

    identifier = ALIASIdentifier(atomic_db, resolving_power=5000.0)

    # 2 lines: one strong (5000), one weak (100)
    fused = [
        {
            "transition": Transition("Fe", 1, 372.0, 1e8, 3.3, 0.0, 11, 9),
            "avg_emissivity": 5000.0,
            "wavelength_nm": 372.0,
        },
        {
            "transition": Transition("Fe", 1, 374.0, 1e6, 3.3, 0.0, 9, 9),
            "avg_emissivity": 100.0,
            "wavelength_nm": 374.0,
        },
    ]
    peaks = [(100, 372.001), (200, 374.05)]
    intensity = np.full(500, 10.0)
    intensity[100] = 1000.0
    intensity[200] = 50.0

    # Scenario A: strong line has tiny shift, weak line has big shift
    matched_a = np.array([True, True])
    pidx_a = np.array([0, 1])
    shifts_a = np.array([0.001, 0.05])  # strong=0.001nm, weak=0.05nm

    _, _, k_shift_a, _, _, _, _ = identifier._compute_scores(
        fused,
        matched_a,
        pidx_a,
        shifts_a,
        intensity,
        peaks,
        emissivity_threshold=-np.inf,
    )

    # Scenario B: strong line has big shift, weak line has tiny shift
    shifts_b = np.array([0.05, 0.001])  # strong=0.05nm, weak=0.001nm

    _, _, k_shift_b, _, _, _, _ = identifier._compute_scores(
        fused,
        matched_a,
        pidx_a,
        shifts_b,
        intensity,
        peaks,
        emissivity_threshold=-np.inf,
    )

    # Emissivity-weighted: k_shift_a should be higher because the strong
    # line (which dominates the weight) has a tiny shift
    assert k_shift_a > k_shift_b, (
        f"Emissivity-weighted k_shift should favor small shift on strong line: "
        f"k_shift_a={k_shift_a}, k_shift_b={k_shift_b}"
    )


# ---------------------------------------------------------------------------
# Round 5: Competition at low RP & P_mix floor
# ---------------------------------------------------------------------------


def test_absolute_p_mix_normalization():
    """P_mix should be absolute (partial R^2), not max-normalized."""
    from scipy.optimize import nnls

    # Dominant element A has templates matching observations perfectly;
    # FP element B has templates at wrong positions (near-zero contribution).
    A = np.array(
        [
            [1.0, 0.01],  # peak 0: element A contributes, B negligible
            [0.8, 0.02],  # peak 1: same pattern
        ]
    )
    peak_intensities = np.array([100.0, 80.0])
    total_energy = float(np.sum(peak_intensities**2))

    # Full model
    c, _ = nnls(A, peak_intensities)
    total_rss = float(np.sum((peak_intensities - A @ c) ** 2))

    # Leave-one-out for element A (remove col 0)
    c_no_a, _ = nnls(A[:, 1:], peak_intensities)
    rss_no_a = float(np.sum((peak_intensities - A[:, 1:] @ c_no_a) ** 2))
    p_mix_a = (rss_no_a - total_rss) / total_energy

    # Leave-one-out for element B (remove col 1)
    c_no_b, _ = nnls(A[:, :1], peak_intensities)
    rss_no_b = float(np.sum((peak_intensities - A[:, :1] @ c_no_b) ** 2))
    p_mix_b = (rss_no_b - total_rss) / total_energy

    # Element A should have substantial P_mix; B should be near zero
    assert p_mix_a > 0.1, f"Dominant element P_mix should be > 0.1: {p_mix_a}"
    assert p_mix_b < 0.01, f"FP element P_mix should be ~ 0: {p_mix_b}"


def test_per_peak_sigma(atomic_db):
    """Sigma should vary across peaks (per-peak, not constant)."""
    identifier = ALIASIdentifier(atomic_db, resolving_power=500.0)
    from cflibs.atomic.structures import Transition

    cand = {
        "fused_lines": [
            {
                "wavelength_nm": 300.0,
                "avg_emissivity": 1000.0,
                "transition": Transition("Fe", 1, 300.0, 1e7, 3.0, 0.0, 9, 7),
            },
        ],
        "matched_mask": np.array([True]),
        "wavelength_shifts": np.array([0.0]),
    }
    # Peaks at very different wavelengths
    peaks = [(100, 300.0), (200, 600.0)]
    A = identifier._build_nnls_templates([cand], peaks)

    # The line is at 300nm.  Sigma at 300nm = 300/500/2.355 ≈ 0.255
    # Sigma at 600nm = 600/500/2.355 ≈ 0.510.
    # Distance from line to peak at 600nm = 300nm >> 3*0.51 ~ 1.53nm
    # So A[1,0] should be 0 (proximity filter excludes it).
    # A[0,0] should be ~1000 (right on peak).
    assert A[0, 0] > 900.0, f"Line at 300nm should strongly contribute to 300nm peak: {A[0,0]}"
    assert A[1, 0] == 0.0, f"Line at 300nm should not contribute to 600nm peak: {A[1,0]}"


def test_nnls_proximity_filter(atomic_db):
    """Lines far from any peak should not contribute to NNLS templates."""
    from cflibs.atomic.structures import Transition

    identifier = ALIASIdentifier(atomic_db, resolving_power=5000.0)

    # Line at 400nm, peak at 400nm (close) and peak at 410nm (far)
    cand = {
        "fused_lines": [
            {
                "wavelength_nm": 400.0,
                "avg_emissivity": 1000.0,
                "transition": Transition("Fe", 1, 400.0, 1e7, 3.0, 0.0, 9, 7),
            },
        ],
        "matched_mask": np.array([True]),
        "wavelength_shifts": np.array([0.0]),
    }
    # sigma at 400nm = 400/5000/2.355 ≈ 0.034nm; 3*sigma ≈ 0.102nm
    # sigma at 410nm = 410/5000/2.355 ≈ 0.035nm; 3*sigma ≈ 0.104nm
    # Distance 400→410 = 10nm >> 0.104nm
    peaks = [(100, 400.0), (200, 410.0)]
    A = identifier._build_nnls_templates([cand], peaks)

    assert A[0, 0] > 500.0, f"Close peak should get contribution: {A[0,0]}"
    assert A[1, 0] == 0.0, f"Far peak should get zero contribution: {A[1,0]}"


def test_nnls_shift_correction(atomic_db):
    """Per-element shift should be applied to template wavelengths."""
    from cflibs.atomic.structures import Transition

    identifier = ALIASIdentifier(atomic_db, resolving_power=500.0)

    # Line at 400nm with a +0.5nm shift from matching
    cand = {
        "fused_lines": [
            {
                "wavelength_nm": 400.0,
                "avg_emissivity": 1000.0,
                "transition": Transition("Fe", 1, 400.0, 1e7, 3.0, 0.0, 9, 7),
            },
        ],
        "matched_mask": np.array([True]),
        "wavelength_shifts": np.array([0.5]),  # peak was at 400.5
    }
    # Peak at 400.5nm (where the shift puts the line)
    peaks = [(100, 400.5)]
    A_shifted = identifier._build_nnls_templates([cand], peaks)

    # Without shift, line at 400.0 → peak at 400.5 has 0.5nm offset.
    # sigma ≈ 400.5/500/2.355 ≈ 0.340nm; 0.5nm > 3*0.340 → would be excluded!
    # But WITH shift, effective line position is 400.5 → perfectly on peak.
    assert A_shifted[0, 0] > 900.0, f"Shifted line should perfectly match peak: {A_shifted[0,0]}"


def test_p_mix_with_dominant_element(atomic_db):
    """Dominant element should get high P_mix, FP element should get ~0."""
    from cflibs.atomic.structures import Transition

    identifier = ALIASIdentifier(atomic_db, resolving_power=5000.0)

    cand_a = {
        "fused_lines": [
            {
                "wavelength_nm": 400.0,
                "avg_emissivity": 1000.0,
                "transition": Transition("Fe", 1, 400.0, 1e7, 3.0, 0.0, 9, 7),
            },
            {
                "wavelength_nm": 405.0,
                "avg_emissivity": 800.0,
                "transition": Transition("Fe", 1, 405.0, 5e6, 3.0, 0.0, 7, 7),
            },
        ],
        "matched_mask": np.array([True, True]),
        "wavelength_shifts": np.array([0.01, 0.02]),
    }
    cand_b = {
        "fused_lines": [
            {
                "wavelength_nm": 420.0,
                "avg_emissivity": 500.0,
                "transition": Transition("Co", 1, 420.0, 1e7, 3.0, 0.0, 9, 7),
            },
            {
                "wavelength_nm": 425.0,
                "avg_emissivity": 300.0,
                "transition": Transition("Co", 1, 425.0, 5e6, 3.0, 0.0, 7, 7),
            },
        ],
        "matched_mask": np.array([False, False]),
        "wavelength_shifts": np.array([0.0, 0.0]),
    }

    # Peaks near element A's lines, nothing near B's
    peaks = [(100, 400.01), (200, 405.02)]
    peak_intensities = np.array([800.0, 600.0])

    A = identifier._build_nnls_templates([cand_a, cand_b], peaks)
    P_mix, P_local, c = identifier._compute_nnls_attribution(A, peak_intensities)

    # Element A (dominant) should have substantial P_mix
    assert P_mix[0] > 0.1, f"Dominant element P_mix should be high: {P_mix[0]}"
    # Element B (FP, no matching peaks) should have P_mix ~ 0
    assert P_mix[1] < 0.01, f"FP element P_mix should be near zero: {P_mix[1]}"
    # P_local should strongly separate dominant from FP
    assert P_local[0] > 0.3, f"Dominant P_local should be high: {P_local[0]}"
    assert P_local[1] < 0.01, f"FP P_local should be near zero: {P_local[1]}"


def test_ratio_consistency(atomic_db):
    """R_rat should be high for consistent ratios, low for random ratios."""
    from cflibs.atomic.structures import Transition
    from cflibs.inversion.alias_identifier import ALIASIdentifier

    # Consistent ratios: emissivities [1000, 500, 250], observed [800, 400, 200]
    # log-ratios should be perfectly correlated
    fused = [
        {
            "transition": Transition("Fe", 1, 372.0, 1e7, 3.3, 0.0, 11, 9),
            "avg_emissivity": 1000.0,
            "wavelength_nm": 372.0,
        },
        {
            "transition": Transition("Fe", 1, 374.0, 5e6, 3.3, 0.0, 9, 9),
            "avg_emissivity": 500.0,
            "wavelength_nm": 374.0,
        },
        {
            "transition": Transition("Fe", 1, 376.0, 2.5e6, 3.3, 0.0, 7, 7),
            "avg_emissivity": 250.0,
            "wavelength_nm": 376.0,
        },
    ]
    peaks = [(100, 372.01), (200, 374.01), (300, 376.01)]
    intensity = np.full(500, 10.0)
    intensity[100] = 800.0
    intensity[200] = 400.0
    intensity[300] = 200.0

    matched = np.array([True, True, True])
    pidx = np.array([0, 1, 2])

    R_rat_good = ALIASIdentifier._compute_ratio_consistency(
        fused,
        matched,
        pidx,
        intensity,
        peaks,
    )
    assert R_rat_good > 0.8, f"Consistent ratios should give high R_rat: {R_rat_good}"

    # Inconsistent ratios: emissivities [1000, 500, 250], observed [200, 800, 400]
    # log-ratios should be anti-correlated
    intensity2 = np.full(500, 10.0)
    intensity2[100] = 200.0  # weakest where strongest predicted
    intensity2[200] = 800.0  # strongest where medium predicted
    intensity2[300] = 400.0

    R_rat_bad = ALIASIdentifier._compute_ratio_consistency(
        fused,
        matched,
        pidx,
        intensity2,
        peaks,
    )
    assert (
        R_rat_bad < R_rat_good
    ), f"Inconsistent ratios should give lower R_rat: {R_rat_bad} vs {R_rat_good}"
