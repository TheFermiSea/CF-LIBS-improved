"""Strict / no-fallback mode for the Bayesian solver area.

Covers the high-severity masked-failure sites audited in
``cflibs/inversion/solve/bayesian/*`` and ``candidate_prefilter.py``:

* prefilter empty candidate set / missing NNLS metadata / sub-k_min / multi-T
  offset failure -> ``PrefilterError`` (strict) vs current behaviour (default),
* NUTS convergence/divergence gate -> ``BayesianConvergenceError``,
* nested-sampling forward-model exception -> ``OptimizerFailure`` (strict) vs
  legacy ``-inf`` + counter (default),
* atomic-data load failure / zero ionization potentials -> ``MissingAtomicData``.

Every test asserts the contract from BOTH sides: (i) strict OFF is byte-identical
to current production, (ii) strict ON raises / flags the masked failure. The
tests use light fakes/monkeypatch — no full DB, no XLA compile, no real MCMC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pytest

from cflibs.inversion.candidate_prefilter import (
    PrefilterError,
    select_candidate_elements,
)
from cflibs.inversion.common.strict import MissingAtomicData, OptimizerFailure
from cflibs.inversion.solve.bayesian import atomic as atomic_mod
from cflibs.inversion.solve.bayesian.atomic import _load_physics_arrays


# --------------------------------------------------------------------------- #
# Prefilter fakes                                                             #
# --------------------------------------------------------------------------- #
@dataclass
class _FakeEID:
    element: str
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class _FakeResult:
    all_elements: List[_FakeEID]


class _FakeIdentifier:
    """Minimal SpectralNNLSIdentifier stand-in for the prefilter."""

    def __init__(self, eids: List[_FakeEID], *, raise_on_offset: bool = False):
        self._eids = eids
        self.fallback_T_K = 8000.0
        self.fallback_ne_cm3 = 1e17
        self.estimated_T_K = 8000.0
        self.estimated_ne_cm3 = 1e17
        self.basis_index = object()  # non-None -> base path
        self._raise_on_offset = raise_on_offset

    def identify(self, wavelength, intensity):
        # The prefilter forces basis_index=None on each deepcopied offset id.
        if self._raise_on_offset and self.basis_index is None:
            raise RuntimeError("offset NNLS evaluation blew up")
        return _FakeResult(all_elements=[_FakeEID(e.element, dict(e.metadata)) for e in self._eids])


def _wl_int(n: int = 32):
    return np.linspace(200.0, 400.0, n), np.ones(n)


def _eid(element: str, snr: float, coeff: float) -> _FakeEID:
    return _FakeEID(element=element, metadata={"nnls_snr": snr, "nnls_coefficient": coeff})


# --------------------------------------------------------------------------- #
# Prefilter — strict off == production, strict on raises                       #
# --------------------------------------------------------------------------- #
def test_prefilter_strict_off_matches_strict_on_healthy():
    """Healthy spectrum: strict ON returns the SAME selection as strict OFF."""
    ident = _FakeIdentifier([_eid("Fe I", 10.0, 5.0), _eid("Cu I", 8.0, 3.0)])
    wl, inten = _wl_int()
    kwargs = dict(k_min=1, multi_t_offsets=[])

    off = select_candidate_elements(ident, wl, inten, strict=False, **kwargs)
    on = select_candidate_elements(ident, wl, inten, strict=True, **kwargs)

    assert off == ["Fe", "Cu"]
    assert on == off  # strict adds no behavioural change on a clean case


def test_prefilter_empty_strict_raises_default_returns_forced():
    """No element passes the SNR gate (site 5, high)."""
    ident = _FakeIdentifier([_eid("Fe I", 0.5, 5.0), _eid("Cu I", 0.4, 3.0)])  # snr < 3
    wl, inten = _wl_int()

    # Default: silent collapse to force_include-or-[].
    assert select_candidate_elements(ident, wl, inten, strict=False, multi_t_offsets=[]) == []

    # Strict: loud PrefilterError carrying the diagnostic provenance.
    with pytest.raises(PrefilterError) as exc:
        select_candidate_elements(ident, wl, inten, strict=True, multi_t_offsets=[])
    diag = exc.value.diagnostics
    assert diag is not None
    assert diag.extra["max_observed_nnls_snr"] == pytest.approx(0.5)
    assert diag.extra["min_snr_threshold"] == pytest.approx(3.0)


def test_prefilter_missing_metadata_strict_raises():
    """Identifier omits NNLS metadata (site 8, medium contract break)."""
    ident = _FakeIdentifier([_FakeEID(element="Fe I", metadata={})])
    wl, inten = _wl_int()

    # Default: element silently dropped (0.0 defaults) -> empty -> [].
    assert select_candidate_elements(ident, wl, inten, strict=False, multi_t_offsets=[]) == []

    with pytest.raises(PrefilterError, match="metadata missing"):
        select_candidate_elements(ident, wl, inten, strict=True, multi_t_offsets=[])


def test_prefilter_sub_kmin_strict_raises_default_pads():
    """Fewer than k_min survivors (site 9, medium): default pads, strict refuses."""
    ident = _FakeIdentifier([_eid("Fe I", 10.0, 5.0)])  # only one above gate
    wl, inten = _wl_int()

    # Default: returns the single survivor (pool too small to reach k_min=3).
    assert select_candidate_elements(
        ident, wl, inten, strict=False, k_min=3, multi_t_offsets=[]
    ) == ["Fe"]

    with pytest.raises(PrefilterError, match="k_min"):
        select_candidate_elements(ident, wl, inten, strict=True, k_min=3, multi_t_offsets=[])


def test_prefilter_multi_t_offset_failure_strict_raises():
    """A multi-T offset NNLS evaluation raises (site 10, medium)."""
    ident = _FakeIdentifier([_eid("Fe I", 10.0, 5.0)], raise_on_offset=True)
    wl, inten = _wl_int()

    # Default: offset silently skipped; base survivor returned.
    assert select_candidate_elements(
        ident, wl, inten, strict=False, k_min=1, multi_t_offsets=[1500.0]
    ) == ["Fe"]

    with pytest.raises(PrefilterError, match="Multi-T offset"):
        select_candidate_elements(ident, wl, inten, strict=True, k_min=1, multi_t_offsets=[1500.0])


def test_prefilter_strict_resolves_from_env(monkeypatch):
    """strict=None resolves CFLIBS_NO_FALLBACK (foundation contract)."""
    ident = _FakeIdentifier([_eid("Fe I", 0.5, 5.0)])  # below gate -> empty
    wl, inten = _wl_int()
    monkeypatch.setenv("CFLIBS_NO_FALLBACK", "1")
    with pytest.raises(PrefilterError):
        select_candidate_elements(ident, wl, inten, multi_t_offsets=[])  # strict unset -> env


# --------------------------------------------------------------------------- #
# NUTS convergence / divergence gate                                          #
# --------------------------------------------------------------------------- #
def test_convergence_gate_converged_no_raise():
    from cflibs.inversion.solve.bayesian.priors import ConvergenceStatus
    from cflibs.inversion.solve.bayesian.samplers import _strict_convergence_gate

    diag = _strict_convergence_gate(
        status=ConvergenceStatus.CONVERGED,
        r_hat={"T_eV": 1.0},
        ess={"T_eV": 500.0},
        n_divergences=0,
        num_samples=1000,
        solver="bayesian.mcmc",
        strict=True,
    )
    assert diag.converged is True
    assert diag.failure_reason is None


@pytest.mark.parametrize("status_name", ["NOT_CONVERGED", "UNKNOWN"])
def test_convergence_gate_bad_status_strict_raises(status_name):
    from cflibs.inversion.solve.bayesian.priors import ConvergenceStatus
    from cflibs.inversion.solve.bayesian.samplers import (
        BayesianConvergenceError,
        _strict_convergence_gate,
    )

    status = getattr(ConvergenceStatus, status_name)
    with pytest.raises(BayesianConvergenceError) as exc:
        _strict_convergence_gate(
            status=status,
            r_hat={"T_eV": 1.5},
            ess={"T_eV": 12.0},
            n_divergences=0,
            num_samples=1000,
            solver="bayesian.mcmc",
            strict=True,
        )
    assert exc.value.status == status
    assert exc.value.max_rhat == pytest.approx(1.5)


def test_convergence_gate_divergences_strict_raises():
    from cflibs.inversion.solve.bayesian.priors import ConvergenceStatus
    from cflibs.inversion.solve.bayesian.samplers import (
        BayesianConvergenceError,
        _strict_convergence_gate,
    )

    # Converged R-hat/ESS but divergent transitions present -> still untrustworthy.
    with pytest.raises(BayesianConvergenceError) as exc:
        _strict_convergence_gate(
            status=ConvergenceStatus.CONVERGED,
            r_hat={"T_eV": 1.0},
            ess={"T_eV": 500.0},
            n_divergences=7,
            num_samples=1000,
            solver="bayesian.mcmc",
            strict=True,
        )
    assert exc.value.n_divergences == 7


def test_convergence_gate_strict_off_records_no_raise():
    from cflibs.inversion.solve.bayesian.priors import ConvergenceStatus
    from cflibs.inversion.solve.bayesian.samplers import _strict_convergence_gate

    diag = _strict_convergence_gate(
        status=ConvergenceStatus.NOT_CONVERGED,
        r_hat={"T_eV": 1.5},
        ess={"T_eV": 12.0},
        n_divergences=3,
        num_samples=1000,
        solver="bayesian.mcmc",
        strict=False,  # visibility-only: record, never raise
    )
    assert diag.converged is False
    assert diag.failure_reason is not None
    assert diag.extra["n_divergences"] == 3


# --------------------------------------------------------------------------- #
# Nested sampling — split except (forward-model bug vs zero-prob region)       #
# --------------------------------------------------------------------------- #
class _FakeForwardModel:
    def __init__(self, fn):
        self.elements = ["Fe", "Cu"]
        self._fn = fn

    def forward_numpy(self, T_eV, log_ne, concentrations):
        return self._fn(T_eV, log_ne, concentrations)


def _make_nested(monkeypatch, fn, *, strict):
    from cflibs.inversion.solve.bayesian import samplers as samplers_mod

    monkeypatch.setattr(samplers_mod, "HAS_DYNESTY", True)
    return samplers_mod.NestedSampler(_FakeForwardModel(fn), strict=strict)


def _raise_fn(*_a, **_k):
    raise RuntimeError("forward model DB/shape failure (a real bug)")


def test_nested_loglike_exception_strict_raises(monkeypatch):
    sampler = _make_nested(monkeypatch, _raise_fn, strict=True)
    params = np.array([0.8, 17.0, 0.5])
    with pytest.raises(OptimizerFailure, match="forward model raised"):
        sampler._log_likelihood(params, np.ones(32))


def test_nested_loglike_exception_default_returns_neginf(monkeypatch):
    sampler = _make_nested(monkeypatch, _raise_fn, strict=False)
    params = np.array([0.8, 17.0, 0.5])
    val = sampler._log_likelihood(params, np.ones(32))
    assert val == -np.inf
    assert sampler._n_loglike_exceptions == 1  # counted, not raised


def test_nested_loglike_healthy_identical_in_both_modes(monkeypatch):
    """Valid params: strict flag does not change the numeric log-likelihood."""
    obs = np.ones(32)
    fn = lambda T, ln, c: np.ones(32)  # noqa: E731 - tiny test fixture
    params = np.array([0.8, 17.0, 0.5])

    off = _make_nested(monkeypatch, fn, strict=False)._log_likelihood(params, obs)
    on = _make_nested(monkeypatch, fn, strict=True)._log_likelihood(params, obs)
    assert np.isfinite(off)
    assert on == off


# --------------------------------------------------------------------------- #
# Atomic-data loader — refuse crude-U / zero-IP substitution                   #
# --------------------------------------------------------------------------- #
class _RaisingConn:
    def cursor(self):
        raise RuntimeError("DB connection broken")


class _FakeCursor:
    def execute(self, sql, params=None):
        return None

    def fetchall(self):
        return []  # no ionization potentials loaded


class _EmptyConn:
    def cursor(self):
        return _FakeCursor()


def _empty_physics_arrays(n_el: int = 2, max_stages: int = 3):
    ips = np.zeros((n_el, max_stages), dtype=np.float32)
    coeffs = np.zeros((n_el, max_stages, 5), dtype=np.float32)
    t_min = np.full((n_el, max_stages), 2000.0, dtype=np.float32)
    t_max = np.full((n_el, max_stages), 25000.0, dtype=np.float32)
    g0 = np.ones((n_el, max_stages), dtype=np.float32)
    return ips, coeffs, t_min, t_max, g0


def test_atomic_load_failure_strict_raises():
    """Wholesale physics-load failure (site 7, high)."""
    ips, coeffs, t_min, t_max, g0 = _empty_physics_arrays()
    el_map = {"Fe": 0, "Cu": 1}
    with pytest.raises(MissingAtomicData):
        _load_physics_arrays(
            "x.db",
            _RaisingConn(),
            ["Fe", "Cu"],
            "?,?",
            el_map,
            3,
            ips,
            coeffs,
            t_min,
            t_max,
            g0,
            strict=True,
        )


def test_atomic_load_failure_default_warns_keeps_defaults():
    """Default path: failure swallowed, zero-IP defaults preserved (byte-identical)."""
    ips, coeffs, t_min, t_max, g0 = _empty_physics_arrays()
    el_map = {"Fe": 0, "Cu": 1}
    _load_physics_arrays(
        "x.db",
        _RaisingConn(),
        ["Fe", "Cu"],
        "?,?",
        el_map,
        3,
        ips,
        coeffs,
        t_min,
        t_max,
        g0,
        strict=False,
    )
    assert np.all(ips == 0.0)  # crude default left in place, no raise


def test_atomic_all_zero_ip_strict_raises(monkeypatch):
    """Load 'succeeds' but leaves all IPs zero -> Saha factor invalid (site 7)."""
    monkeypatch.setattr(atomic_mod, "_apply_partition_factory_overrides", lambda *a, **k: None)
    ips, coeffs, t_min, t_max, g0 = _empty_physics_arrays()
    el_map = {"Fe": 0, "Cu": 1}
    with pytest.raises(MissingAtomicData, match="zero"):
        _load_physics_arrays(
            "x.db",
            _EmptyConn(),
            ["Fe", "Cu"],
            "?,?",
            el_map,
            3,
            ips,
            coeffs,
            t_min,
            t_max,
            g0,
            strict=True,
        )


def test_atomic_all_zero_ip_default_no_raise(monkeypatch):
    monkeypatch.setattr(atomic_mod, "_apply_partition_factory_overrides", lambda *a, **k: None)
    ips, coeffs, t_min, t_max, g0 = _empty_physics_arrays()
    el_map = {"Fe": 0, "Cu": 1}
    _load_physics_arrays(
        "x.db",
        _EmptyConn(),
        ["Fe", "Cu"],
        "?,?",
        el_map,
        3,
        ips,
        coeffs,
        t_min,
        t_max,
        g0,
        strict=False,
    )
    assert np.all(ips == 0.0)
