"""Tests for cflibs.inversion.candidate_prefilter.select_candidate_elements."""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pytest

from cflibs.inversion.candidate_prefilter import (
    _extract_element_symbol,
    select_candidate_elements,
)


@dataclass
class _FakeEID:
    element: str
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class _FakeResult:
    all_elements: List[_FakeEID]


class _FakeIdentifier:
    def __init__(self, results_by_T):
        self.results_by_T = results_by_T
        self.fallback_T_K = 8000.0
        self.fallback_ne_cm3 = 1e17
        self.basis_index = object()

    def identify(self, wavelength, intensity):
        records = self.results_by_T.get(self.fallback_T_K, [])
        return _FakeResult(
            all_elements=[
                _FakeEID(element=el, metadata={"nnls_snr": snr, "nnls_coefficient": coeff})
                for (el, snr, coeff) in records
            ]
        )


class TestExtractElementSymbol:
    def test_neutral(self):
        assert _extract_element_symbol("Fe") == "Fe"

    def test_ionized(self):
        assert _extract_element_symbol("Fe II") == "Fe"

    def test_single_letter(self):
        assert _extract_element_symbol("H I") == "H"

    def test_strips_whitespace(self):
        assert _extract_element_symbol("  Ca  ") == "Ca"


class TestSelectCandidateElements:
    def _wl_int(self):
        return np.linspace(200.0, 400.0, 256), np.ones(256)

    def test_basic_ranking_by_coefficient(self):
        ident = _FakeIdentifier({
            8000.0: [
                ("Fe I", 10.0, 5.0),
                ("Ca I", 10.0, 3.0),
                ("Mg I", 10.0, 1.0),
            ]
        })
        wl, intens = self._wl_int()
        result = select_candidate_elements(
            ident, wl, intens, k_max=5, multi_t_offsets=[]
        )
        assert result == ["Fe", "Ca", "Mg"]

    def test_snr_gate_filters_noise(self):
        ident = _FakeIdentifier({
            8000.0: [
                ("Fe I", 10.0, 5.0),
                ("Ca I", 2.0, 100.0),
            ]
        })
        wl, intens = self._wl_int()
        result = select_candidate_elements(
            ident, wl, intens, min_snr=3.0, k_max=5, multi_t_offsets=[]
        )
        assert "Fe" in result
        assert "Ca" not in result

    def test_zero_or_negative_coefficient_filtered(self):
        ident = _FakeIdentifier({
            8000.0: [
                ("Fe I", 10.0, 5.0),
                ("Ca I", 10.0, 0.0),
                ("Mg I", 10.0, -1.0),
            ]
        })
        wl, intens = self._wl_int()
        result = select_candidate_elements(ident, wl, intens, multi_t_offsets=[])
        assert "Fe" in result
        assert "Ca" not in result
        assert "Mg" not in result

    def test_aggregates_ionization_stages_max(self):
        ident = _FakeIdentifier({
            8000.0: [
                ("Fe I", 10.0, 2.0),
                ("Fe II", 10.0, 7.0),
                ("Ca I", 10.0, 3.0),
            ]
        })
        wl, intens = self._wl_int()
        result = select_candidate_elements(ident, wl, intens, multi_t_offsets=[])
        assert result[0] == "Fe"
        assert "Ca" in result
        assert result.count("Fe") == 1

    def test_force_include_always_present(self):
        ident = _FakeIdentifier({8000.0: [("Fe I", 10.0, 5.0)]})
        wl, intens = self._wl_int()
        result = select_candidate_elements(
            ident, wl, intens, force_include=["Au"], multi_t_offsets=[]
        )
        assert "Au" in result
        assert "Fe" in result

    def test_k_max_caps_selection(self):
        symbols = [
            "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
            "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        ]
        records = [(s, 10.0, float(20 - i)) for i, s in enumerate(symbols)]
        ident = _FakeIdentifier({8000.0: records})
        wl, intens = self._wl_int()
        result = select_candidate_elements(
            ident, wl, intens, k_max=5, multi_t_offsets=[]
        )
        assert len(result) == 5

    def test_k_min_pads_from_rejected(self):
        ident = _FakeIdentifier({
            8000.0: [
                ("Fe I", 10.0, 100.0),
                ("Ca I", 10.0, 1e-10),
                ("Mg I", 10.0, 1e-10),
                ("Na I", 10.0, 1e-10),
            ]
        })
        wl, intens = self._wl_int()
        result = select_candidate_elements(
            ident, wl, intens, k_min=3, k_max=15, coeff_ratio=0.5, multi_t_offsets=[]
        )
        assert len(result) >= 3

    def test_empty_results_returns_force_include(self):
        ident = _FakeIdentifier({8000.0: []})
        wl, intens = self._wl_int()
        result = select_candidate_elements(
            ident, wl, intens, force_include=["Fe"], multi_t_offsets=[]
        )
        assert result == ["Fe"]

    def test_empty_results_no_force_returns_empty(self):
        ident = _FakeIdentifier({8000.0: []})
        wl, intens = self._wl_int()
        result = select_candidate_elements(ident, wl, intens, multi_t_offsets=[])
        assert result == []

    def test_invalid_k_min_exceeds_k_max(self):
        ident = _FakeIdentifier({8000.0: []})
        wl, intens = self._wl_int()
        with pytest.raises(ValueError, match="k_min"):
            select_candidate_elements(ident, wl, intens, k_min=10, k_max=5)

    def test_invalid_k_max_zero(self):
        ident = _FakeIdentifier({8000.0: []})
        wl, intens = self._wl_int()
        with pytest.raises(ValueError, match="k_max"):
            select_candidate_elements(ident, wl, intens, k_max=0)

    def test_multi_t_union(self):
        ident = _FakeIdentifier({
            8000.0: [("Fe I", 10.0, 5.0)],
            6500.0: [("Ca I", 10.0, 4.0)],
            9500.0: [("Mg I", 10.0, 3.0)],
        })
        wl, intens = self._wl_int()
        result = select_candidate_elements(
            ident, wl, intens, multi_t_offsets=[-1500.0, 1500.0]
        )
        assert "Fe" in result
        assert "Ca" in result
        assert "Mg" in result

    def test_force_include_does_not_count_against_k_max_first(self):
        symbols = [
            "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
            "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        ]
        records = [(s, 10.0, float(20 - i)) for i, s in enumerate(symbols)]
        ident = _FakeIdentifier({8000.0: records})
        wl, intens = self._wl_int()
        result = select_candidate_elements(
            ident, wl, intens, force_include=["Au", "Pt"], k_max=5, multi_t_offsets=[]
        )
        assert "Au" in result
        assert "Pt" in result
        assert len(result) == 5
