"""REGRESSION GUARD — paper-faithful ALIAS/Comb identification contract.

Element identification has regressed THREE times by silently re-introducing
homegrown decision/scoring logic that deviates from the source papers
(Noël et al. 2025 for ALIAS; Gajarska et al. 2024 for Comb). This file pins the
exact invariants whose violation caused those regressions, so re-introduction
fails CI immediately. See docs/audit/2026-06-16-alias-faithful-fix.md.

These guards are intentionally structural (fast, deterministic, no corpus/real-DB
dependency): they assert the removed gates STAY removed and the paper formulas
stay in place. Behavioural validation (P/R/F1) lives in the synthetic-corpus
benchmark; the unit contract lives here.
"""

import inspect

import pytest

from cflibs.inversion.identify.alias import ALIAS_PRESETS, ALIASIdentifier
from cflibs.inversion.identify.comb import CombIdentifier
from cflibs.inversion.identify.hybrid import HybridIdentifier

pytestmark = pytest.mark.unit

# The paper's confidence threshold C_th (Noël §3.8 eq 6) lives on the k_det
# scale (default 0.5 strict / 0.4 recall). The legacy CL-deflated decision used
# floor values like 0.01–0.10; on the k_det scale those are "accept everything",
# silently neutering every "strict" path. Any detection_threshold below this
# bound is a re-introduced legacy CL floor.
_MIN_CTH = 0.3


def _code_only(src: str) -> str:
    """Source with comments stripped, so guards match real code — not the
    explanatory comments that intentionally name the removed logic."""
    return "\n".join(line.split("#", 1)[0] for line in src.splitlines())


# --------------------------------------------------------------------------
# ALIAS — peak detection default (paper §3.1: find_peaks on the -2nd derivative)
# --------------------------------------------------------------------------
def test_alias_peak_mode_default_is_second_derivative(atomic_db):
    """The paper §3.1 second-derivative detector must remain the default.
    Reverting to the old intensity-domain detector regressed recall."""
    assert ALIASIdentifier(atomic_db).peak_mode == "second_derivative"
    # legacy mode remains available; bogus values are rejected loudly.
    assert ALIASIdentifier(atomic_db, peak_mode="intensity").peak_mode == "intensity"
    with pytest.raises(ValueError):
        ALIASIdentifier(atomic_db, peak_mode="nope")


# --------------------------------------------------------------------------
# ALIAS — decision is k_det > C_th (eq 6), NOT the homegrown CL product
# --------------------------------------------------------------------------
def test_alias_decision_uses_kdet_not_cl():
    """detected must be `k_det > self.detection_threshold` (Noël §3.8 eq 6).
    The CL product (crustal-abundance P_ab, Boltzmann-R², NNLS, winner-relative)
    must NOT be the decision variable — that made detection non-monotonic in
    similarity (accepted low-sim confounders, rejected the true element)."""
    src = inspect.getsource(ALIASIdentifier._build_element_id)
    assert (
        "k_det > self.detection_threshold" in src
    ), "ALIAS decision must be k_det > C_th (paper eq 6), not CL-based"


def test_alias_winner_relative_cl_gate_stays_disabled():
    """The winner-relative-CL gate (reject if CL < max_CL*frac) is not in the
    paper and coupled every element's fate to the strongest one; it must stay
    out of the identify() decision path."""
    src = _code_only(inspect.getsource(ALIASIdentifier.identify))
    assert (
        "self._apply_relative_cl_gate(" not in src
    ), "winner-relative-CL gate must remain disabled in identify()"


def test_alias_ksim_has_no_deflators():
    """k_sim must be the bare cosine (paper eq 3): no uniqueness_factor and no
    self-absorption damping — both crushed the dominant element."""
    sim_src = _code_only(inspect.getsource(ALIASIdentifier._compute_k_sim))
    assert "uniqueness_factor" not in sim_src, "k_sim uniqueness deflator must stay removed"
    vec_src = _code_only(inspect.getsource(ALIASIdentifier._collect_k_sim_vectors))
    assert (
        "sa_damping" not in vec_src and "self_absorption_damping" not in vec_src
    ), "k_sim self-absorption damping must stay removed (paper eq 3 is a bare cosine)"


# --------------------------------------------------------------------------
# ALIAS — detection_threshold is the paper C_th, NOT a legacy CL floor
# --------------------------------------------------------------------------
def test_alias_init_default_threshold_is_paper_cth(atomic_db):
    """The strict/recall __init__ defaults must sit on the k_det C_th scale.
    Reverting either to a legacy CL floor (<0.3) makes ALIAS accept everything."""
    assert ALIASIdentifier(atomic_db).detection_threshold >= _MIN_CTH
    assert ALIASIdentifier(atomic_db, high_recall=True).detection_threshold >= _MIN_CTH


@pytest.mark.parametrize("name", sorted(ALIAS_PRESETS))
def test_alias_presets_threshold_on_cth_scale(name):
    """No ALIAS_PRESETS cocktail may pin a legacy CL-floor detection_threshold.
    'strict' once pinned 0.02 — "accept everything" on the k_det scale, turning
    the precision-king preset into the most permissive path (Codex P2)."""
    dt = ALIAS_PRESETS[name].get("detection_threshold")
    if dt is not None:
        assert dt >= _MIN_CTH, (
            f"ALIAS_PRESETS[{name!r}].detection_threshold={dt} is a legacy CL "
            f"floor; the decision is k_det > C_th (paper eq 6), expected >= {_MIN_CTH}"
        )


def test_hybrid_alias_confirmation_threshold_on_cth_scale():
    """HybridIdentifier's ALIAS confirmation stage must default to a paper C_th,
    not a legacy CL floor — 0.05 once neutered the Stage-2 confirmation (Codex P2)."""
    default = (
        inspect.signature(HybridIdentifier.__init__).parameters["alias_detection_threshold"].default
    )
    assert default >= _MIN_CTH, (
        f"HybridIdentifier.alias_detection_threshold default={default} is a legacy "
        f"CL floor; ALIAS detects when k_det > C_th, expected >= {_MIN_CTH}"
    )


# --------------------------------------------------------------------------
# Comb — fingerprint = mean over ACTIVE lines (Gajarska §2.2.3), gate removed
# --------------------------------------------------------------------------
def test_comb_fingerprint_is_mean_over_active(atomic_db):
    """Fingerprint must be the mean of ACTIVE-tooth correlations, not
    sum/min(total_teeth, k). The capped denominator systematically depressed
    line-rich elements (3 active teeth @0.9 scored 0.27 instead of 0.9)."""
    comb = CombIdentifier(atomic_db)
    teeth = [
        {"active": True, "best_correlation": 0.9},
        {"active": True, "best_correlation": 0.9},
        {"active": True, "best_correlation": 0.9},
    ] + [{"active": False, "best_correlation": 0.0} for _ in range(7)]
    # 3 active @0.9 over 10 total teeth -> mean over active = 0.9 (not 0.27).
    assert comb._compute_fingerprint(teeth) == pytest.approx(0.9, abs=1e-9)


def test_comb_relative_median_gate_stays_removed():
    """The homegrown relative-median rejection gate (reject score < scale*median)
    must NOT be wired into spectral matching — it could demand a score above the
    max achievable (1.0) and rejected even the top true element ('detects
    nothing on pure-Fe')."""
    src = _code_only(inspect.getsource(CombIdentifier._match_spectra))
    assert (
        "self._apply_relative_threshold(" not in src
    ), "Comb relative-median rejection gate must remain removed from _match_spectra"
