"""
DB-free tests for BasisIndex per-element neighbor voting (#8c).

These use a synthetic, in-memory multi-element "library" stub so they run in
the default gate (no ``requires_db`` marker). The key correctness property
under test: ``estimate_plasma_params`` restricts the (T, n_e) vote to a single
element family instead of mixing chemically distinct neighbors whose optima
differ.
"""

import numpy as np
import pytest

pytest.importorskip("faiss", reason="faiss-cpu required for BasisIndex tests")

from cflibs.manifold.basis_index import BasisIndex  # noqa: E402


class _FakeLibrary:
    """Minimal stand-in exposing the attributes ``build_from_library`` reads."""

    def __init__(self, elements, params, spectra):
        # spectra shape: (n_el, n_grid, n_pix)
        self.elements = list(elements)
        self._params = np.asarray(params, dtype=np.float64)  # (n_grid, 2)
        self._spectra = np.asarray(spectra, dtype=np.float64)

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    @property
    def n_grid(self) -> int:
        return self._params.shape[0]


def _two_element_library(n_pix: int = 64):
    """Two elements with DISJOINT peak locations and DIFFERENT (T, ne) optima.

    Element A peaks in the low-index half of the grid; element B peaks in the
    high-index half. A grid spans (T, ne) so that A's brightest point sits at a
    distinct (T, ne) from B's. A query that looks like element A must recover
    A's (T, ne), not a blend dragged toward B.
    """
    T_vals = np.array([6000.0, 10000.0])
    ne_vals = np.array([1e16, 1e17])
    params = np.array(
        [[t, n] for t in T_vals for n in ne_vals], dtype=np.float64
    )  # (4, 2): rows = (T,ne) in (6000,1e16),(6000,1e17),(10000,1e16),(10000,1e17)
    n_grid = params.shape[0]

    wl = np.arange(n_pix)
    spectra = np.zeros((2, n_grid, n_pix), dtype=np.float64)

    def gauss(center, sigma):
        return np.exp(-0.5 * ((wl - center) / sigma) ** 2)

    # Element A ("Aa"): peak near index 16; brightest/sharpest at grid row 0
    # (T=6000, ne=1e16). Other rows are shifted/broadened so they are clearly
    # farther in embedding space.
    a_centers = [16.0, 18.0, 22.0, 26.0]
    a_sigmas = [1.5, 2.5, 3.5, 4.5]
    # Element B ("Bb"): peak near index 48; brightest/sharpest at grid row 3
    # (T=10000, ne=1e17).
    b_centers = [40.0, 44.0, 47.0, 48.0]
    b_sigmas = [4.5, 3.5, 2.5, 1.5]
    for g in range(n_grid):
        sa = gauss(a_centers[g], a_sigmas[g])
        sb = gauss(b_centers[g], b_sigmas[g])
        spectra[0, g] = sa / sa.sum()
        spectra[1, g] = sb / sb.sum()

    return _FakeLibrary(["Aa", "Bb"], params, spectra), wl


class TestPerElementVoting:
    def test_query_recovers_target_element(self):
        lib, wl = _two_element_library()
        idx = BasisIndex(n_components=4)
        idx.build_from_library(lib)

        # Query identical to element Aa at grid row 0 (T=6000, ne=1e16).
        query = lib._spectra[0, 0].copy()
        T_est, ne_est, details = idx.estimate_plasma_params(query, k=8)

        # The vote must be restricted to the dominant element family, Aa.
        assert details["target_elements"] == ["Aa"]
        # Recovered (T, ne) must be Aa's bright point, NOT blended toward Bb's
        # high-T / high-ne optimum.
        assert T_est == pytest.approx(6000.0)
        assert ne_est == pytest.approx(1e16)

    def test_explicit_elements_restrict_vote(self):
        lib, wl = _two_element_library()
        idx = BasisIndex(n_components=4)
        idx.build_from_library(lib)

        # A neutral query; force the vote onto element Bb explicitly.
        query = 0.5 * lib._spectra[0, 0] + 0.5 * lib._spectra[1, 3]
        T_est, ne_est, details = idx.estimate_plasma_params(query, k=8, elements=["Bb"])

        assert details["target_elements"] == ["Bb"]
        # Restricted to Bb, all neighbors used for the median are Bb vectors.
        assert set(details["per_element_estimate"].keys()) == {"Bb"}

    def test_unrestricted_mixed_bag_is_avoided(self):
        """Without per-element restriction a 50/50 mixed query would blend the
        two elements' optima; the per-element vote must instead lock onto one
        element family and return that family's grid point exactly."""
        lib, wl = _two_element_library()
        idx = BasisIndex(n_components=4)
        idx.build_from_library(lib)

        query = 0.5 * lib._spectra[0, 0] + 0.5 * lib._spectra[1, 3]
        T_est, ne_est, details = idx.estimate_plasma_params(query, k=8)

        # Exactly one element family chosen.
        assert len(details["target_elements"]) == 1
        chosen = details["target_elements"][0]
        # The returned estimate equals that family's per-element estimate (no
        # cross-element averaging).
        assert (T_est, ne_est) == pytest.approx(details["per_element_estimate"][chosen])


class TestResolveTargetElements:
    def test_uses_provided_present_elements(self):
        from collections import Counter

        votes = Counter({"Fe": 5, "Cu": 3, "Ni": 1})
        assert BasisIndex._resolve_target_elements(["Cu", "Ni"], votes) == ["Cu", "Ni"]

    def test_filters_absent_provided_elements(self):
        from collections import Counter

        votes = Counter({"Fe": 5, "Cu": 3})
        # "Zz" is not among the neighbors; it is filtered out.
        assert BasisIndex._resolve_target_elements(["Zz", "Fe"], votes) == ["Fe"]

    def test_falls_back_to_dominant_when_none_provided(self):
        from collections import Counter

        votes = Counter({"Fe": 5, "Cu": 3})
        assert BasisIndex._resolve_target_elements(None, votes) == ["Fe"]

    def test_falls_back_to_dominant_when_all_provided_absent(self):
        from collections import Counter

        votes = Counter({"Fe": 5, "Cu": 3})
        assert BasisIndex._resolve_target_elements(["Zz"], votes) == ["Fe"]
