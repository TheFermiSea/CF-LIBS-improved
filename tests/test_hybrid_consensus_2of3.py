"""
Tests for the opt-in 2-of-3 majority-vote consensus identifier.

The bd (CF-LIBS-improved-7nmw) calls for a precision-improving variant of
the existing alias/comb/correlation union: an element is reported only when
at least 2 of the 3 line-matchers agree. The existing union semantics must
remain byte-identical — no defaults changed, no behavior implicitly
altered.

Two invariants live here:

1. *Default preservation*. The pre-existing :class:`HybridIdentifier`
   (NNLS+ALIAS two-stage) is untouched. Its default ``require_both=True``
   continues to produce intersection semantics, and ``require_both=False``
   continues to produce union semantics.

2. *Consensus correctness*. A synthetic 3-identifier scenario with one
   detector seeing Fe alone and two detectors seeing Mg produces:
   union (3-way OR) → {Fe, Mg}; 2-of-3 → {Mg}; 1-of-3 → {Fe, Mg};
   3-of-3 → {} (unanimous would require all three).
"""

from __future__ import annotations

from typing import List

import pytest

from cflibs.inversion.element_id import (
    ElementIdentification,
    ElementIdentificationResult,
)
from cflibs.inversion.identify.hybrid_consensus import (
    HybridConsensusIdentifier,
    consensus_detected_elements,
)


# --------------------------------------------------------------------- #
# Helpers                                                               #
# --------------------------------------------------------------------- #
def _make_eid(element: str, detected: bool, score: float = 0.5) -> ElementIdentification:
    return ElementIdentification(
        element=element,
        detected=detected,
        score=score,
        confidence=score,
        n_matched_lines=2 if detected else 0,
        n_total_lines=5,
        matched_lines=[],
        unmatched_lines=[],
        metadata={},
    )


def _make_result(
    detected_set: set, all_elements: List[str], algorithm: str = "fake"
) -> ElementIdentificationResult:
    eids = [_make_eid(e, e in detected_set) for e in all_elements]
    return ElementIdentificationResult(
        detected_elements=[e for e in eids if e.detected],
        rejected_elements=[e for e in eids if not e.detected],
        all_elements=eids,
        experimental_peaks=[],
        n_peaks=0,
        n_matched_peaks=0,
        n_unmatched_peaks=0,
        algorithm=algorithm,
        parameters={},
    )


class _StubIdentifier:
    """Identifier that returns a pre-canned result regardless of input."""

    def __init__(self, result: ElementIdentificationResult):
        self._result = result

    def identify(self, wavelength, intensity):  # noqa: ARG002
        return self._result


# --------------------------------------------------------------------- #
# Invariant 1: existing HybridIdentifier defaults are untouched         #
# --------------------------------------------------------------------- #
class TestHybridIdentifierDefaultPreservation:
    """
    Pin the constructor surface of the existing :class:`HybridIdentifier`
    so subsequent edits cannot silently flip its default semantics. We do
    NOT instantiate or run it (that would pull in AtomicDatabase /
    BasisLibrary fixtures); we only assert the signature contract.
    """

    def test_require_both_default_is_true(self):
        """Default require_both must remain True (intersection semantics)."""
        import inspect

        from cflibs.inversion.identify.hybrid import HybridIdentifier

        sig = inspect.signature(HybridIdentifier.__init__)
        param = sig.parameters.get("require_both")
        assert param is not None, "HybridIdentifier must keep its require_both kwarg"
        assert param.default is True, (
            "Default for require_both flipped — the leaderboard baseline "
            "(F1=0.6880 for hybrid_union) is keyed off the existing default. "
            "Do NOT silently change this."
        )

    def test_hybrid_identifier_module_exports_match_legacy_path(self):
        """The shim path used by the benchmark must still resolve."""
        from cflibs.inversion import hybrid_identifier as shim
        from cflibs.inversion.identify import hybrid as canonical

        assert shim.HybridIdentifier is canonical.HybridIdentifier

    def test_new_consensus_class_does_not_replace_hybrid_identifier(self):
        """HybridConsensusIdentifier is a *separate* class, not a subclass."""
        from cflibs.inversion.identify.hybrid import HybridIdentifier

        assert HybridConsensusIdentifier is not HybridIdentifier
        assert not issubclass(HybridConsensusIdentifier, HybridIdentifier)


# --------------------------------------------------------------------- #
# Invariant 2: consensus voting correctness                             #
# --------------------------------------------------------------------- #
class TestConsensusVoting:
    """
    Synthetic 3-identifier scenario:
      - alias detects {Fe}
      - comb detects {Mg}
      - correlation detects {Mg}

    Expected by vote count:
      - 1-of-3 (union):  {Fe, Mg}
      - 2-of-3 (default): {Mg}
      - 3-of-3 (unanim.):  {}
    """

    @pytest.fixture
    def universe(self) -> List[str]:
        return ["Fe", "Mg"]

    @pytest.fixture
    def per_identifier_results(self, universe):
        alias = _make_result({"Fe"}, universe, algorithm="alias")
        comb = _make_result({"Mg"}, universe, algorithm="comb")
        correlation = _make_result({"Mg"}, universe, algorithm="correlation")
        return [alias, comb, correlation]

    @pytest.fixture
    def stub_identifiers(self, per_identifier_results):
        return [_StubIdentifier(r) for r in per_identifier_results]

    def test_2of3_majority_keeps_only_mg(self, stub_identifiers, universe):
        """2-of-3 vote suppresses the Fe singleton."""
        identifier = HybridConsensusIdentifier(
            stub_identifiers, elements=universe, min_agreeing=2
        )
        result = identifier.identify(wavelength=None, intensity=None)
        detected = {eid.element for eid in result.detected_elements}
        assert detected == {"Mg"}
        # Fe is correctly marked as rejected with vote_count=1.
        fe_eid = next(eid for eid in result.all_elements if eid.element == "Fe")
        assert fe_eid.detected is False
        assert fe_eid.metadata["vote_count"] == 1
        assert fe_eid.metadata["votes_by"] == {
            "alias": True,
            "comb": False,
            "correlation": False,
        }
        # Mg has 2 agreeing votes.
        mg_eid = next(eid for eid in result.all_elements if eid.element == "Mg")
        assert mg_eid.metadata["vote_count"] == 2
        assert mg_eid.metadata["votes_by"] == {
            "alias": False,
            "comb": True,
            "correlation": True,
        }

    def test_union_via_min_agreeing_one(self, stub_identifiers, universe):
        """min_agreeing=1 reproduces union semantics (Fe AND Mg both pass)."""
        identifier = HybridConsensusIdentifier(
            stub_identifiers, elements=universe, min_agreeing=1
        )
        result = identifier.identify(wavelength=None, intensity=None)
        detected = {eid.element for eid in result.detected_elements}
        assert detected == {"Fe", "Mg"}

    def test_unanimous_3of3_rejects_everything(self, stub_identifiers, universe):
        """min_agreeing=3 requires all identifiers to agree — none do."""
        identifier = HybridConsensusIdentifier(
            stub_identifiers, elements=universe, min_agreeing=3
        )
        result = identifier.identify(wavelength=None, intensity=None)
        detected = {eid.element for eid in result.detected_elements}
        assert detected == set()

    def test_combine_path_matches_identify_path(self, stub_identifiers, universe, per_identifier_results):
        """``combine`` (pre-computed results) and ``identify`` (run-then-combine) must agree."""
        identifier = HybridConsensusIdentifier(
            stub_identifiers, elements=universe, min_agreeing=2
        )
        from_identify = identifier.identify(wavelength=None, intensity=None)
        from_combine = identifier.combine(per_identifier_results)
        assert {e.element for e in from_identify.detected_elements} == {
            e.element for e in from_combine.detected_elements
        }

    def test_consensus_detected_elements_helper(self, per_identifier_results):
        """The standalone helper matches the class-based decision."""
        union = consensus_detected_elements(per_identifier_results, min_agreeing=1)
        majority = consensus_detected_elements(per_identifier_results, min_agreeing=2)
        unanimous = consensus_detected_elements(per_identifier_results, min_agreeing=3)
        assert union == {"Fe", "Mg"}
        assert majority == {"Mg"}
        assert unanimous == set()

    def test_default_min_agreeing_is_two(self):
        """The default must be 2 (the 2-of-3 confirmation rule)."""
        import inspect

        sig = inspect.signature(HybridConsensusIdentifier.__init__)
        assert sig.parameters["min_agreeing"].default == 2

    def test_algorithm_string_encodes_vote_rule(self, stub_identifiers, universe):
        """Algorithm name records the consensus rule for downstream consumers."""
        identifier = HybridConsensusIdentifier(
            stub_identifiers, elements=universe, min_agreeing=2
        )
        result = identifier.identify(wavelength=None, intensity=None)
        assert result.algorithm == "hybrid_consensus_2of3"

    def test_score_is_mean_of_voting_identifiers(self, universe):
        """Score aggregation = mean across identifiers that voted detected."""
        # Build per-id results with distinct scores for Mg.
        def _mg_eid(score: float, detected: bool) -> ElementIdentification:
            return ElementIdentification(
                element="Mg",
                detected=detected,
                score=score,
                confidence=score,
                n_matched_lines=2 if detected else 0,
                n_total_lines=5,
                matched_lines=[],
                unmatched_lines=[],
                metadata={},
            )

        def _wrap(mg_eid_inst, alg):
            fe_eid = _make_eid("Fe", False)
            all_eids = [fe_eid, mg_eid_inst]
            return ElementIdentificationResult(
                detected_elements=[e for e in all_eids if e.detected],
                rejected_elements=[e for e in all_eids if not e.detected],
                all_elements=all_eids,
                experimental_peaks=[],
                n_peaks=0,
                n_matched_peaks=0,
                n_unmatched_peaks=0,
                algorithm=alg,
                parameters={},
            )

        per_results = [
            _wrap(_mg_eid(0.4, True), "alias"),
            _wrap(_mg_eid(0.8, True), "comb"),
            _wrap(_mg_eid(0.1, False), "correlation"),
        ]
        stubs = [_StubIdentifier(r) for r in per_results]
        identifier = HybridConsensusIdentifier(stubs, elements=universe, min_agreeing=2)
        result = identifier.identify(wavelength=None, intensity=None)
        mg_eid = next(e for e in result.all_elements if e.element == "Mg")
        # Mean of the *voting* scores (alias=0.4, comb=0.8) → 0.6.
        # Correlation's 0.1 must NOT be included because it didn't vote yes.
        assert mg_eid.score == pytest.approx(0.6)


# --------------------------------------------------------------------- #
# Constructor validation                                                #
# --------------------------------------------------------------------- #
class TestConstructorValidation:
    """Guard against misconfiguration that would silently degrade results."""

    def test_rejects_single_identifier(self):
        with pytest.raises(ValueError, match="at least 2"):
            HybridConsensusIdentifier([_StubIdentifier(_make_result(set(), ["Fe"]))], elements=["Fe"])

    def test_rejects_min_agreeing_above_count(self):
        stubs = [
            _StubIdentifier(_make_result(set(), ["Fe"])),
            _StubIdentifier(_make_result(set(), ["Fe"])),
        ]
        with pytest.raises(ValueError, match="cannot exceed"):
            HybridConsensusIdentifier(stubs, elements=["Fe"], min_agreeing=3)

    def test_rejects_zero_min_agreeing(self):
        stubs = [
            _StubIdentifier(_make_result(set(), ["Fe"])),
            _StubIdentifier(_make_result(set(), ["Fe"])),
        ]
        with pytest.raises(ValueError, match="must be >= 1"):
            HybridConsensusIdentifier(stubs, elements=["Fe"], min_agreeing=0)

    def test_names_length_must_match(self):
        stubs = [
            _StubIdentifier(_make_result(set(), ["Fe"])),
            _StubIdentifier(_make_result(set(), ["Fe"])),
            _StubIdentifier(_make_result(set(), ["Fe"])),
        ]
        with pytest.raises(ValueError, match="names length"):
            HybridConsensusIdentifier(stubs, elements=["Fe"], names=["a", "b"])

    def test_default_names_for_three_identifiers(self):
        stubs = [_StubIdentifier(_make_result(set(), ["Fe"])) for _ in range(3)]
        identifier = HybridConsensusIdentifier(stubs, elements=["Fe"])
        assert identifier.names == ["alias", "comb", "correlation"]

    def test_default_names_for_non_three(self):
        stubs = [_StubIdentifier(_make_result(set(), ["Fe"])) for _ in range(4)]
        identifier = HybridConsensusIdentifier(stubs, elements=["Fe"], min_agreeing=3)
        assert identifier.names == ["id0", "id1", "id2", "id3"]


# --------------------------------------------------------------------- #
# Invariant 3: weighted-confidence voting (bead jbfg follow-up)         #
# --------------------------------------------------------------------- #
class TestWeightedConsensusVoting:
    """
    Detective B's structural fix: with binary 2-of-N voting, a strong
    voter like NNLS (Phase 4 F1=0.399) is treated identically to a weak
    voter like comb (F1=0.014). The weighted-confidence path applies
    per-voter weights ∝ standalone F1 and uses an absolute
    ``weight_threshold``; with NNLS at w=0.46 and threshold=0.40, NNLS
    can pass an element alone — recovering the NNLS-only TPs that the
    binary rule discards.
    """

    @pytest.fixture
    def universe(self):
        return ["Fe", "Si", "Mg"]

    @pytest.fixture
    def per_identifier_results(self, universe):
        # NNLS uniquely sees Fe (strong voter); ALIAS uniquely sees Mg;
        # ALIAS+comb both see Si (weak coalition).
        alias = _make_result({"Si", "Mg"}, universe, algorithm="alias")
        comb = _make_result({"Si"}, universe, algorithm="comb")
        correlation = _make_result(set(), universe, algorithm="correlation")
        nnls = _make_result({"Fe"}, universe, algorithm="nnls")
        return [alias, comb, correlation, nnls]

    @pytest.fixture
    def stubs(self, per_identifier_results):
        return [_StubIdentifier(r) for r in per_identifier_results]

    def test_binary_mode_unchanged_when_no_weights(self, stubs, universe):
        """voter_weights=None → identical to existing binary rule."""
        binary = HybridConsensusIdentifier(
            stubs,
            elements=universe,
            min_agreeing=2,
            names=["alias", "comb", "correlation", "nnls"],
        )
        result = binary.identify(wavelength=None, intensity=None)
        detected = {e.element for e in result.detected_elements}
        # Si: alias+comb agree (2 votes) → pass.
        # Fe: only NNLS → 1 vote, fails.
        # Mg: only ALIAS → 1 vote, fails.
        assert detected == {"Si"}

    def test_weighted_lets_nnls_pass_alone_at_threshold_0_40(self, stubs, universe):
        """NNLS-only Fe now passes via w_nnls=0.46 ≥ threshold=0.40."""
        weighted = HybridConsensusIdentifier(
            stubs,
            elements=universe,
            names=["alias", "comb", "correlation", "nnls"],
            voter_weights={
                "alias": 0.30, "comb": 0.12, "correlation": 0.12, "nnls": 0.46,
            },
            weight_threshold=0.40,
        )
        result = weighted.identify(wavelength=None, intensity=None)
        detected = {e.element for e in result.detected_elements}
        # Fe: nnls alone = 0.46 ≥ 0.40 → pass.
        # Si: alias+comb = 0.42 ≥ 0.40 → pass.
        # Mg: alias alone = 0.30 < 0.40 → fail.
        assert detected == {"Fe", "Si"}

    def test_weighted_metadata_records_score_and_weights(self, stubs, universe):
        weighted = HybridConsensusIdentifier(
            stubs,
            elements=universe,
            names=["alias", "comb", "correlation", "nnls"],
            voter_weights={
                "alias": 0.30, "comb": 0.12, "correlation": 0.12, "nnls": 0.46,
            },
            weight_threshold=0.40,
        )
        result = weighted.identify(wavelength=None, intensity=None)
        fe_eid = next(e for e in result.all_elements if e.element == "Fe")
        assert fe_eid.metadata["weighted_score"] == pytest.approx(0.46)
        assert fe_eid.metadata["weight_threshold"] == 0.40
        assert fe_eid.metadata["consensus_mode"].startswith("weighted_")
        assert result.algorithm.startswith("hybrid_consensus_weighted_")

    def test_weighted_rejects_unknown_voter_name(self, stubs, universe):
        with pytest.raises(ValueError, match="voter_weights contains names"):
            HybridConsensusIdentifier(
                stubs,
                elements=universe,
                names=["alias", "comb", "correlation", "nnls"],
                voter_weights={"bogus_voter": 1.0},
            )

    def test_workflow_registry_includes_hybrid_consensus_weighted(self):
        """The new workflow is registered in build_id_workflow_registry."""
        from cflibs.benchmark.unified import build_id_workflow_registry

        reg = build_id_workflow_registry(quick=True)
        assert "hybrid_consensus_weighted" in reg
