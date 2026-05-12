"""Tests for the JAX NNLS / sparse-NNLS / attribution / template-build /
P_SNR helpers in :mod:`cflibs.inversion.identify.alias`.

Each test compares the JAX path to its CPU counterpart and asserts
numerical agreement at the tolerance required by the consultation
synthesis in ``docs/jax-port/alias-consultation.md``:

* Standard NNLS on well-conditioned templates: rtol 1e-5 vs ``scipy.optimize.nnls``.
* Standard NNLS on correlated columns: objective gap at most 1e-6.
* Sparse elastic-net NNLS: objective gap at most 1e-4 (the L-BFGS-B baseline
  is itself approximate, and we allow the projected-gradient solution to
  shift mass across redundant columns).
* Attribution (P_mix, P_local): rtol 1e-4 on well-conditioned inputs.
* Template build: rtol 1e-10 (the two paths perform the same arithmetic).
* P_SNR: rtol 1e-6 (only differs by float-precision arithmetic between
  ``scipy.special.erf`` and ``jax.scipy.special.erf``).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import nnls
from scipy.optimize import minimize

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from cflibs.inversion.identify.alias import (  # noqa: E402
    ALIASIdentifier,
    compute_nnls_attribution_jax,
    compute_p_snr_jax,
    solve_nnls_jax,
    solve_sparse_nnls_jax,
)

pytestmark = [pytest.mark.requires_jax, pytest.mark.unit]


# ---------- Helpers ----------------------------------------------------------


def _random_well_conditioned_template(
    n_peaks: int = 30, n_cands: int = 12, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Random non-negative ``(A, b)`` problem that mimics LIBS Gaussian
    templates after the proximity filter. Columns are independent, so the
    NNLS solution is unique."""
    rng = np.random.default_rng(seed)
    A = rng.uniform(0, 1, (n_peaks, n_cands))
    true_c = rng.uniform(0, 5, n_cands)
    # Knock a few coefficients to zero so the active set is non-trivial.
    true_c[rng.integers(0, n_cands, size=2)] = 0.0
    b = A @ true_c + 0.01 * rng.standard_normal(n_peaks)
    b = np.maximum(b, 0.0)
    return A, b


def _correlated_template(
    n_peaks: int = 30, n_cands: int = 12, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Same shape, but with several columns highly correlated (LIBS lines that
    blend at low resolving power)."""
    A, b = _random_well_conditioned_template(n_peaks, n_cands, seed=seed)
    rng = np.random.default_rng(seed + 1000)
    # Make columns 1, 5, 9 strongly correlated with their neighbors.
    A[:, 1] = 0.9 * A[:, 0] + 0.1 * rng.standard_normal(n_peaks)
    A[:, 5] = 0.85 * A[:, 4] + 0.15 * rng.standard_normal(n_peaks)
    A[:, 9] = 0.8 * A[:, 8] + 0.2 * rng.standard_normal(n_peaks)
    return A, b


# ---------- Tests: solve_nnls_jax -------------------------------------------


class TestSolveNNLSJax:
    """FISTA NNLS vs ``scipy.optimize.nnls``."""

    def test_well_conditioned_matches_scipy_to_rtol_1e_5(self) -> None:
        for seed in range(5):
            A, b = _random_well_conditioned_template(seed=seed)
            c_scipy, _ = nnls(A, b)
            c_jax = solve_nnls_jax(A, b, max_iter=1000)
            assert c_jax.shape == c_scipy.shape
            # Coefficient agreement: rtol 1e-5 where scipy is non-zero.
            mask = c_scipy > 1e-8
            if np.any(mask):
                rel_err = np.abs(c_jax[mask] - c_scipy[mask]) / c_scipy[mask]
                assert rel_err.max() < 1e-5, f"seed={seed} max rel err = {rel_err.max()}"
            # Coefficients that scipy set to 0 should also be ~0 in JAX.
            assert np.max(c_jax[~mask]) < 1e-5

    def test_correlated_columns_match_objective(self) -> None:
        """With correlated columns the minimizer is non-unique; both
        solvers must hit the same objective value."""
        for seed in range(5):
            A, b = _correlated_template(seed=seed)
            c_scipy, _ = nnls(A, b)
            c_jax = solve_nnls_jax(A, b, max_iter=1000)
            obj_scipy = 0.5 * np.linalg.norm(A @ c_scipy - b) ** 2
            obj_jax = 0.5 * np.linalg.norm(A @ c_jax - b) ** 2
            # JAX objective should be no worse than scipy by more than 1e-6.
            assert obj_jax - obj_scipy < 1e-6, (
                f"seed={seed} obj_jax={obj_jax} obj_scipy={obj_scipy}"
            )
            # Non-negativity is mandatory.
            assert np.all(c_jax >= 0.0)

    def test_zero_rhs_returns_zero(self) -> None:
        A, _ = _random_well_conditioned_template()
        c = solve_nnls_jax(A, np.zeros(A.shape[0]))
        assert np.all(c == 0.0)

    def test_single_column(self) -> None:
        A = np.array([[1.0], [1.0], [1.0]])
        b = np.array([2.0, 2.0, 2.0])
        c = solve_nnls_jax(A, b, max_iter=500)
        np.testing.assert_allclose(c[0], 2.0, rtol=1e-6)

    def test_ridge_term(self) -> None:
        """A positive ridge stabilizes correlated columns toward each other."""
        A, b = _correlated_template(seed=42)
        c0 = solve_nnls_jax(A, b, max_iter=2000, l2=0.0)
        c_ridge = solve_nnls_jax(A, b, max_iter=2000, l2=0.1)
        # Adding L2 should not blow up — coefficients stay non-negative and
        # objective increases monotonically with l2.
        assert np.all(c_ridge >= 0)
        obj0 = 0.5 * np.linalg.norm(A @ c0 - b) ** 2
        obj_ridge = 0.5 * np.linalg.norm(A @ c_ridge - b) ** 2 + 0.05 * np.dot(c_ridge, c_ridge)
        assert obj_ridge >= obj0 - 1e-9


# ---------- Tests: solve_sparse_nnls_jax ------------------------------------


def _cpu_sparse_nnls(A, b, alpha=0.01, l1_ratio=0.9):
    """Reference CPU implementation: same L-BFGS-B path as
    ``ALIASIdentifier._compute_sparse_nnls_scores``."""
    col_norms = np.linalg.norm(A, axis=0)
    col_norms_safe = np.where(col_norms == 0, 1.0, col_norms)
    A_norm = A / col_norms_safe
    l1_weight = float(alpha * l1_ratio)
    l2_weight = float(alpha * (1.0 - l1_ratio))

    def loss_grad(x):
        r = A_norm @ x - b
        loss = 0.5 * float(r @ r) + l1_weight * float(np.sum(x)) + 0.5 * l2_weight * float(x @ x)
        grad = A_norm.T @ r + l1_weight + l2_weight * x
        return loss, grad

    res = minimize(
        loss_grad,
        x0=np.zeros(A.shape[1]),
        jac=True,
        bounds=[(0.0, None)] * A.shape[1],
        method="L-BFGS-B",
        options={"maxiter": 2000},
    )
    sparse_c = np.asarray(res.x) / col_norms_safe
    residual = float(np.linalg.norm(b - A @ sparse_c))
    return sparse_c, residual


class TestSolveSparseNNLSJax:
    def test_matches_cpu_objective(self) -> None:
        """JAX and L-BFGS-B reach the same elastic-net objective."""
        for seed in range(5):
            A, b = _random_well_conditioned_template(seed=seed)
            c_cpu, _ = _cpu_sparse_nnls(A, b, alpha=0.01, l1_ratio=0.9)
            c_jax, _ = solve_sparse_nnls_jax(A, b, alpha=0.01, l1_ratio=0.9)
            # Both solutions must be non-negative.
            assert np.all(c_jax >= 0)
            assert np.all(c_cpu >= -1e-10)

            def en_obj(c):
                col_norms = np.linalg.norm(A, axis=0)
                col_norms_safe = np.where(col_norms == 0, 1.0, col_norms)
                A_norm = A / col_norms_safe
                xn = c * col_norms_safe
                r = A_norm @ xn - b
                return (
                    0.5 * np.dot(r, r) + 0.01 * 0.9 * np.sum(xn) + 0.5 * 0.01 * 0.1 * np.dot(xn, xn)
                )

            assert en_obj(c_jax) - en_obj(c_cpu) < 1e-4

    def test_sparser_with_higher_alpha(self) -> None:
        A, b = _random_well_conditioned_template(seed=1)
        c_lo, _ = solve_sparse_nnls_jax(A, b, alpha=0.0, l1_ratio=0.9)
        c_hi, _ = solve_sparse_nnls_jax(A, b, alpha=1.0, l1_ratio=0.9)
        # Higher alpha => more sparsity => lower sum.
        assert np.sum(c_hi) <= np.sum(c_lo) + 1e-6

    def test_residual_correctness(self) -> None:
        """Returned residual matches the un-normalized convention."""
        A, b = _random_well_conditioned_template(seed=2)
        c, res = solve_sparse_nnls_jax(A, b, alpha=0.01)
        np.testing.assert_allclose(res, np.linalg.norm(b - A @ c), rtol=1e-12)


# ---------- Tests: compute_nnls_attribution_jax -----------------------------


def _cpu_attribution(A, b):
    """Reference CPU implementation matching ``_compute_nnls_attribution``."""
    n_cands = A.shape[1]
    if n_cands == 0 or np.all(A == 0):
        return np.ones(n_cands), np.ones(n_cands), np.zeros(n_cands)
    c, _ = nnls(A, b)
    total_rss = float(np.sum((b - A @ c) ** 2))
    total_energy = float(np.sum(b**2))
    if total_energy == 0:
        return np.ones(n_cands), np.ones(n_cands), c

    P_mix = np.zeros(n_cands)
    for j in range(n_cands):
        A_red = np.delete(A, j, axis=1)
        if A_red.shape[1] == 0:
            P_mix[j] = 1.0
            continue
        c_red, _ = nnls(A_red, b)
        rss_without = float(np.sum((b - A_red @ c_red) ** 2))
        P_mix[j] = (rss_without - total_rss) / total_energy

    P_local = np.zeros(n_cands)
    for j in range(n_cands):
        claimed = A[:, j] > 1e-6
        if not np.any(claimed):
            continue
        obs = np.sum(b[claimed])
        if obs <= 0:
            continue
        elem = np.sum(A[claimed, j] * c[j])
        P_local[j] = float(np.clip(elem / obs, 0.0, 1.0))
    return P_mix, P_local, c


class TestComputeNNLSAttributionJax:
    def test_well_conditioned_match(self) -> None:
        """P_mix, P_local, c match CPU implementation."""
        for seed in range(3):
            A, b = _random_well_conditioned_template(seed=seed)
            P_mix_cpu, P_local_cpu, c_cpu = _cpu_attribution(A, b)
            P_mix_jax, P_local_jax, c_jax = compute_nnls_attribution_jax(A, b)
            np.testing.assert_allclose(c_jax, c_cpu, rtol=1e-4, atol=1e-6)
            np.testing.assert_allclose(P_local_jax, P_local_cpu, rtol=1e-4, atol=1e-6)
            np.testing.assert_allclose(P_mix_jax, P_mix_cpu, rtol=1e-4, atol=1e-6)

    def test_all_zero_template_returns_ones(self) -> None:
        """Edge case: zero template matrix returns the ``np.ones`` sentinel
        used by the CPU code."""
        A = np.zeros((10, 4))
        b = np.ones(10)
        P_mix, P_local, c = compute_nnls_attribution_jax(A, b)
        np.testing.assert_array_equal(P_mix, np.ones(4))
        np.testing.assert_array_equal(P_local, np.ones(4))
        np.testing.assert_array_equal(c, np.zeros(4))

    def test_empty_candidate_set(self) -> None:
        A = np.zeros((10, 0))
        b = np.ones(10)
        P_mix, P_local, c = compute_nnls_attribution_jax(A, b)
        assert P_mix.shape == (0,)
        assert P_local.shape == (0,)
        assert c.shape == (0,)

    def test_zero_energy_signal(self) -> None:
        A = np.random.default_rng(0).uniform(0, 1, (8, 3))
        b = np.zeros(8)
        P_mix, P_local, c = compute_nnls_attribution_jax(A, b)
        np.testing.assert_array_equal(P_mix, np.ones(3))
        np.testing.assert_array_equal(P_local, np.ones(3))


# ---------- Tests: build_nnls_templates_jax ---------------------------------


class TestBuildTemplatesJax:
    """JAX template builder vs CPU loop, via ALIASIdentifier dispatch."""

    def _make_identifier(self, use_jax: bool = False) -> ALIASIdentifier:
        class _StubDB:
            def get_available_elements(self):
                return ["Fe"]

        return ALIASIdentifier(
            atomic_db=_StubDB(),
            resolving_power=5000.0,
            use_jax_template_build=use_jax,
        )

    def _make_candidates(self, rng: np.random.Generator):
        """Build a representative set of candidate dicts that the template
        builder can consume."""
        n_cands = 4
        candidates = []
        for c_idx in range(n_cands):
            n_lines = rng.integers(3, 8)
            fused_lines = [
                {
                    "wavelength_nm": float(rng.uniform(400.0, 700.0)),
                    "avg_emissivity": float(rng.uniform(0.1, 10.0)),
                }
                for _ in range(n_lines)
            ]
            n_matched = rng.integers(1, n_lines + 1)
            matched_mask = np.zeros(n_lines, dtype=bool)
            matched_mask[:n_matched] = True
            wavelength_shifts = np.zeros(n_lines)
            wavelength_shifts[:n_matched] = rng.uniform(-0.05, 0.05, n_matched)
            candidates.append(
                {
                    "fused_lines": fused_lines,
                    "matched_mask": matched_mask,
                    "wavelength_shifts": wavelength_shifts,
                }
            )
        return candidates

    def test_exact_arithmetic_agreement(self) -> None:
        """The JAX builder performs the same arithmetic in a different order
        and must therefore agree to near-machine precision."""
        rng = np.random.default_rng(1234)
        candidates = self._make_candidates(rng)
        peaks = [(i, float(wl)) for i, wl in enumerate(np.linspace(420, 680, 20))]

        cpu_id = self._make_identifier(use_jax=False)
        jax_id = self._make_identifier(use_jax=True)

        A_cpu = cpu_id._build_nnls_templates(candidates, peaks)
        A_jax = jax_id._build_nnls_templates_jax_wrapper(candidates, peaks)
        np.testing.assert_allclose(A_jax, A_cpu, rtol=1e-12, atol=1e-14)

    def test_empty_inputs(self) -> None:
        cpu_id = self._make_identifier(use_jax=False)
        jax_id = self._make_identifier(use_jax=True)
        peaks_empty: list = []
        cands_empty: list = []
        A_cpu = cpu_id._build_nnls_templates(cands_empty, peaks_empty)
        A_jax = jax_id._build_nnls_templates_jax_wrapper(cands_empty, peaks_empty)
        assert A_cpu.shape == A_jax.shape == (0, 0)

    def test_no_candidates_with_peaks(self) -> None:
        jax_id = self._make_identifier(use_jax=True)
        peaks = [(0, 500.0), (1, 510.0)]
        A = jax_id._build_nnls_templates_jax_wrapper([], peaks)
        assert A.shape == (2, 0)


# ---------- Tests: compute_p_snr_jax ----------------------------------------


class TestComputePSNRJax:
    """JAX P_SNR vs the static CPU computation."""

    def test_matches_cpu(self) -> None:
        rng = np.random.default_rng(0)
        intensity = rng.uniform(0, 10, 200)
        peaks = [(int(i), float(i)) for i in rng.choice(200, 12, replace=False)]
        peak_idx = np.array([p[0] for p in peaks], dtype=np.int32)
        p_snr_cpu = ALIASIdentifier._compute_p_snr(intensity, peaks)
        p_snr_jax = compute_p_snr_jax(intensity, peak_idx)
        np.testing.assert_allclose(p_snr_jax, p_snr_cpu, rtol=1e-6, atol=1e-8)

    def test_no_peaks_returns_neutral(self) -> None:
        intensity = np.linspace(0, 1, 50)
        peak_idx = np.array([], dtype=np.int32)
        assert compute_p_snr_jax(intensity, peak_idx) == 0.5

    def test_single_peak(self) -> None:
        intensity = np.ones(20)
        intensity[5] = 10.0  # one bright peak
        peak_idx = np.array([5], dtype=np.int32)
        # JAX path matches CPU formula
        p_snr_jax = compute_p_snr_jax(intensity, peak_idx)
        p_snr_cpu = ALIASIdentifier._compute_p_snr(intensity, [(5, 5.0)])
        np.testing.assert_allclose(p_snr_jax, p_snr_cpu, rtol=1e-6)


# ---------- Tests: ALIASIdentifier opt-in flags -----------------------------


class TestALIASIdentifierJaxFlags:
    def test_all_flags_accepted(self) -> None:
        class _StubDB:
            def get_available_elements(self):
                return ["Fe"]

        ident = ALIASIdentifier(
            atomic_db=_StubDB(),
            use_jax_boltzmann_fit=True,
            use_jax_nnls=True,
            use_jax_p_snr=True,
            use_jax_template_build=True,
        )
        assert ident.use_jax_boltzmann_fit
        assert ident.use_jax_nnls
        assert ident.use_jax_p_snr
        assert ident.use_jax_template_build

    def test_default_is_cpu(self) -> None:
        class _StubDB:
            def get_available_elements(self):
                return ["Fe"]

        ident = ALIASIdentifier(atomic_db=_StubDB())
        assert not ident.use_jax_boltzmann_fit
        assert not ident.use_jax_nnls
        assert not ident.use_jax_p_snr
        assert not ident.use_jax_template_build


# ---------- Tests: end-to-end ALIAS run with JAX flags -----------------------


class TestEndToEndJaxIdentify:
    """Running ``identify()`` with the JAX flags toggled must yield element
    detections consistent with the CPU run.

    These are *integration* checks — the exact CL values can drift slightly
    because NNLS non-uniqueness can shift mass across degenerate solutions,
    but the set of identified elements should be identical.
    """

    def test_jax_nnls_identifies_same_elements(self, atomic_db, synthetic_libs_spectrum) -> None:
        spectrum = synthetic_libs_spectrum(
            elements={
                "Fe": [(371.99, 1000.0), (373.49, 500.0), (374.95, 200.0)],
                "Ca": [(393.37, 800.0), (396.85, 600.0)],
            },
            noise_level=0.01,
        )
        wl = spectrum["wavelength"]
        intensity = spectrum["intensity"]

        cpu = ALIASIdentifier(
            atomic_db=atomic_db,
            resolving_power=5000.0,
            elements=["Fe", "Ca", "Mg", "Cu"],
        )
        gpu = ALIASIdentifier(
            atomic_db=atomic_db,
            resolving_power=5000.0,
            elements=["Fe", "Ca", "Mg", "Cu"],
            use_jax_nnls=True,
            use_jax_p_snr=True,
            use_jax_template_build=True,
        )
        res_cpu = cpu.identify(wl, intensity)
        res_gpu = gpu.identify(wl, intensity)

        # Set of detected elements must match.
        det_cpu = {e.element for e in res_cpu.detected_elements}
        det_gpu = {e.element for e in res_gpu.detected_elements}
        assert det_cpu == det_gpu, f"cpu={det_cpu} jax={det_gpu}"

        # CL ordering should be the same on the elements that both detected.
        by_cpu = sorted(res_cpu.detected_elements, key=lambda e: -e.confidence_level)
        by_gpu = sorted(res_gpu.detected_elements, key=lambda e: -e.confidence_level)
        assert [e.element for e in by_cpu] == [e.element for e in by_gpu]
