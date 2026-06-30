"""Strict / no-fallback exploratory mode for CF-LIBS solvers.

During exploration, silent fallbacks (warm-start substitution, default atomic
constants, clamps, swallowed exceptions, plausibility-reverts) hide *which*
solver/combination is actually failing and *why*. This module provides the
machinery to make failures honest:

* a typed :class:`SolverFailure` hierarchy,
* a :class:`SolveDiagnostics` record that every solve populates (so failures are
  visible even when not raising), and
* verified-condition **gate** helpers whose pass/fail criteria are the runtime
  image of the machine-checked theorems in the ``cflibs-formal`` Lean
  development (each gate cites its theorem). In strict mode a failed gate raises;
  otherwise it is recorded on the diagnostics and the caller's existing
  (production) behaviour is preserved.

Strict mode is **off by default** — ``resolve_strict()`` returns ``False`` unless
explicitly enabled (argument or ``CFLIBS_NO_FALLBACK`` env var), so production
paths are byte-identical. See ``cflibs-formal/docs/SOLVER_FORMALIZATION_GAPS.md``
for where a refuse-to-report is a heuristic awaiting a theorem (Tier 1/2) vs an
already-proven criterion (Tier 3).

Physics-only: NumPy + stdlib only; no banned ML imports.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "SolverFailure",
    "OptimizerFailure",
    "NotConverged",
    "NonIdentifiable",
    "MissingAtomicData",
    "IllConditioned",
    "NonPhysicalResult",
    "LTEViolation",
    "UnobservedStage",
    "GateResult",
    "SolveDiagnostics",
    "resolve_strict",
    "require_positive",
    "require_distinct_energy",
    "require_simplex",
    "require_atomic_data",
    "require_ion_stage_observed",
    "require_boltzmann_conditioning",
]

_ENV_FLAG = "CFLIBS_NO_FALLBACK"


# --------------------------------------------------------------------------- #
# Typed failure hierarchy                                                      #
# --------------------------------------------------------------------------- #
class SolverFailure(Exception):
    """Base class for an honest, surfaced solver failure (strict mode).

    Carries an optional :class:`SolveDiagnostics` so the caller sees the full
    provenance (which gate failed, the measured values, the objective) instead
    of an opaque substituted result.
    """

    def __init__(self, message: str, diagnostics: "Optional[SolveDiagnostics]" = None):
        super().__init__(message)
        self.diagnostics = diagnostics


class OptimizerFailure(SolverFailure):
    """The optimizer crashed, diverged, or returned a non-finite/failed status.

    Replaces ``except Exception: return warm_start`` and the ignored
    ``res.success``/``res.status`` paths.
    """


class NotConverged(SolverFailure):
    """The optimizer ran but did not converge (no move / no improvement / maxiter)."""


class NonIdentifiable(SolverFailure):
    """The quantity is not uniquely recoverable from the available observations.

    Grounded in ``Identifiability.lean`` / ``Inverse.lean`` /
    ``SahaInverse.lean``: e.g. degenerate Boltzmann lever arm, one-line-per-species
    composition, or n_e without an observed ion stage.
    """


class MissingAtomicData(SolverFailure):
    """A required atomic constant (IP, A_ki, g, partition U) is absent.

    Grounded in ``density_identifiability`` (the de-normalization constant must be
    strictly positive and real): a missing constant means the per-species density
    is *not recoverable*, so strict mode refuses rather than substituting a
    default (the ``IP=15.0 eV`` / crude-``U`` fallbacks). See gap #2/#7/#23.
    """


class IllConditioned(SolverFailure):
    """The fit is ill-conditioned beyond the verified error budget.

    Grounded in ``olsSlope_noise_gain`` / ``requiredEnergySpread_sufficient``
    (``ErrorBudget.lean``): ``ss_e`` too small / noise-gain too large for the
    target temperature accuracy.
    """


class NonPhysicalResult(SolverFailure):
    """A reported quantity violates a proven invariant (positivity / simplex closure).

    Grounded in ``lineIntensity_pos`` and ``composition_sum_one`` /
    ``composition_mem_stdSimplex`` (``Closure.lean``).
    """


class LTEViolation(SolverFailure):
    """The LTE precondition (assumed by every Saha/Boltzmann theorem) is unmet.

    Grounded in ``mcWhirterBound`` (``StarkBroadening.lean``) — LTE is a
    hypothesis, so the criterion *is* the runtime gate (gap #24).
    """


class UnobservedStage(SolverFailure):
    """n_e was inferred without a genuinely observed ion-stage line.

    Grounded in ``saha_joint_identifiability`` (needs both a neutral pair and an
    observed ion line); single-stage / imputed-ratio n_e is outside the verified
    envelope (gap #25) — condemns the pressure-balance n_e fallback.
    """


# --------------------------------------------------------------------------- #
# Diagnostics record                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class GateResult:
    """Outcome of one verified-condition gate."""

    name: str
    passed: bool
    theorem: str
    detail: str = ""
    values: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SolveDiagnostics:
    """Provenance of a single solve, populated in *both* strict and non-strict mode.

    In non-strict mode this is the visibility layer (failures are recorded, not
    hidden); in strict mode a failed gate also raises the matching
    :class:`SolverFailure`.
    """

    solver: str = ""
    strict: bool = False
    converged: Optional[bool] = None
    adopted: Optional[bool] = None
    failure_reason: Optional[str] = None
    objective_initial: Optional[float] = None
    objective_final: Optional[float] = None
    optimizer_status: Optional[Any] = None
    optimizer_success: Optional[bool] = None
    gates: List[GateResult] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def record(self, gate: GateResult) -> GateResult:
        self.gates.append(gate)
        if not gate.passed and self.failure_reason is None:
            self.failure_reason = f"{gate.name}: {gate.detail}"
        return gate

    @property
    def failed_gates(self) -> List[GateResult]:
        return [g for g in self.gates if not g.passed]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "solver": self.solver,
            "strict": self.strict,
            "converged": self.converged,
            "adopted": self.adopted,
            "failure_reason": self.failure_reason,
            "objective_initial": self.objective_initial,
            "objective_final": self.objective_final,
            "optimizer_status": self.optimizer_status,
            "optimizer_success": self.optimizer_success,
            "gates": [
                {"name": g.name, "passed": g.passed, "theorem": g.theorem,
                 "detail": g.detail, "values": g.values}
                for g in self.gates
            ],
            "extra": self.extra,
        }


# --------------------------------------------------------------------------- #
# Flag resolution                                                             #
# --------------------------------------------------------------------------- #
def resolve_strict(explicit: Optional[bool] = None) -> bool:
    """Resolve the strict / no-fallback flag.

    Precedence: explicit argument (if not ``None``) > ``CFLIBS_NO_FALLBACK`` env
    var > ``False``. Default ``False`` keeps production behaviour byte-identical.
    """
    if explicit is not None:
        return bool(explicit)
    return os.environ.get(_ENV_FLAG, "").strip().lower() in {"1", "true", "yes", "on"}


def _fail(
    exc_type: type, name: str, theorem: str, detail: str, values: Dict[str, Any],
    *, strict: bool, diagnostics: "Optional[SolveDiagnostics]",
) -> GateResult:
    """Record a failed gate; raise the typed failure iff strict."""
    gate = GateResult(name=name, passed=False, theorem=theorem, detail=detail, values=values)
    if diagnostics is not None:
        diagnostics.record(gate)
    if strict:
        raise exc_type(f"[{name}] {detail}  (verified by {theorem})", diagnostics)
    return gate


def _ok(name: str, theorem: str, values: Dict[str, Any],
        diagnostics: "Optional[SolveDiagnostics]") -> GateResult:
    gate = GateResult(name=name, passed=True, theorem=theorem, values=values)
    if diagnostics is not None:
        diagnostics.gates.append(gate)
    return gate


# --------------------------------------------------------------------------- #
# Verified-condition gates (each cites its cflibs-formal theorem)             #
# --------------------------------------------------------------------------- #
def require_positive(
    values: Sequence[float], what: str, *, strict: bool,
    diagnostics: Optional[SolveDiagnostics] = None,
) -> GateResult:
    """Every value must be finite and strictly positive (no clamping / ε-flooring).

    Theorem: ``lineIntensity_pos`` (ForwardMap.lean) — the forward observable and
    its de-normalization factors are strictly positive; a non-positive intensity
    or ``g*A`` means the log/Boltzmann step is invalid, not something to clamp.
    """
    arr = np.asarray(values, dtype=float)
    bad = ~np.isfinite(arr) | (arr <= 0.0)
    if np.any(bad):
        return _fail(
            NonPhysicalResult, f"positive:{what}", "lineIntensity_pos",
            f"{int(bad.sum())}/{arr.size} {what} are non-finite or <= 0",
            {"n_bad": int(bad.sum()), "n": int(arr.size), "min": float(np.nanmin(arr)) if arr.size else None},
            strict=strict, diagnostics=diagnostics,
        )
    return _ok(f"positive:{what}", "lineIntensity_pos", {"n": int(arr.size)}, diagnostics)


def require_distinct_energy(
    upper_energies_ev: Sequence[float], *, min_spread_ev: float, strict: bool,
    diagnostics: Optional[SolveDiagnostics] = None,
) -> GateResult:
    """The Boltzmann lever arm must be non-degenerate (energy spread above a floor).

    Theorems: ``temperature_from_two_lines`` / ``temperature_identifiability``
    (Identifiability.lean) — T is recoverable only from distinct upper-level
    energies; near-equal energies make the slope T-independent (non-identifiable),
    which is the converse the degenerate ``ss_e==0 -> R^2=1`` fit hides.
    """
    e = np.asarray(upper_energies_ev, dtype=float)
    spread = float(e.max() - e.min()) if e.size >= 2 else 0.0
    ss_e = float(np.sum((e - e.mean()) ** 2)) if e.size >= 2 else 0.0
    if e.size < 2 or spread < min_spread_ev or ss_e <= 0.0:
        return _fail(
            NonIdentifiable, "distinct_energy", "temperature_identifiability",
            f"energy spread {spread:.4g} eV (ss_e={ss_e:.4g}) below floor {min_spread_ev:.4g} eV "
            f"with n={e.size} lines -> T non-identifiable",
            {"spread_ev": spread, "ss_e": ss_e, "n_lines": int(e.size), "min_spread_ev": min_spread_ev},
            strict=strict, diagnostics=diagnostics,
        )
    return _ok("distinct_energy", "temperature_identifiability",
               {"spread_ev": spread, "ss_e": ss_e, "n_lines": int(e.size)}, diagnostics)


def require_simplex(
    composition: Sequence[float], *, tol: float = 1e-6, strict: bool,
    diagnostics: Optional[SolveDiagnostics] = None,
) -> GateResult:
    """A reported composition must lie on the probability simplex (sum 1, each in [0,1]).

    Theorems: ``composition_sum_one`` / ``composition_mem_stdSimplex``
    (Closure.lean) — a vector outside the simplex signals a failed solve and must
    NOT be silently renormalised away.
    """
    c = np.asarray(composition, dtype=float)
    s = float(c.sum())
    out_of_range = bool(np.any(c < -tol) or np.any(c > 1.0 + tol))
    if (not np.all(np.isfinite(c))) or abs(s - 1.0) > tol or out_of_range:
        return _fail(
            NonPhysicalResult, "simplex", "composition_sum_one",
            f"composition off-simplex: sum={s:.6g} (|sum-1|={abs(s-1.0):.2g}>tol), "
            f"out_of_range={out_of_range}",
            {"sum": s, "min": float(c.min()) if c.size else None, "max": float(c.max()) if c.size else None},
            strict=strict, diagnostics=diagnostics,
        )
    return _ok("simplex", "composition_sum_one", {"sum": s}, diagnostics)


def require_atomic_data(
    name: str, value: Optional[float], element: str, *, strict: bool,
    diagnostics: Optional[SolveDiagnostics] = None,
) -> GateResult:
    """A required atomic constant must be present and strictly positive — no default.

    Theorem: ``density_identifiability`` (Identifiability.lean) — the per-species
    de-normalization constant c=Fcal*A_u*g_u*exp(...)/U(T) must be strictly
    positive and real; a missing/None/<=0 constant means N is *not recoverable*.
    Condemns the ``IP=15.0 eV`` and crude-``U`` substitutions (gaps #2/#7/#23).
    """
    if value is None or not np.isfinite(value) or value <= 0.0:
        return _fail(
            MissingAtomicData, f"atomic_data:{name}", "density_identifiability",
            f"required atomic constant '{name}' for {element} is missing/non-positive "
            f"({value!r}) -> density not recoverable; refusing default substitution",
            {"constant": name, "element": element, "value": value},
            strict=strict, diagnostics=diagnostics,
        )
    return _ok(f"atomic_data:{name}", "density_identifiability",
               {"constant": name, "element": element, "value": float(value)}, diagnostics)


def require_ion_stage_observed(
    element: str, n_observed_ion_lines: int, *, strict: bool,
    diagnostics: Optional[SolveDiagnostics] = None,
) -> GateResult:
    """n_e from a Saha stage ratio needs a genuinely observed ion-stage line.

    Theorem: ``saha_joint_identifiability`` (SahaInverse.lean) — joint (T, n_e)
    recovery requires both a neutral pair AND >=1 observed ion line; inferring
    n_e from a single stage or an imputed ratio is non-identifiable. Condemns the
    pressure-balance n_e fallback (gap #25).
    """
    if n_observed_ion_lines < 1:
        return _fail(
            UnobservedStage, "ion_stage_observed", "saha_joint_identifiability",
            f"{element}: 0 observed ion-stage lines -> n_e not identifiable from "
            f"Saha-Boltzmann; refusing imputed/pressure-balance substitution",
            {"element": element, "n_observed_ion_lines": int(n_observed_ion_lines)},
            strict=strict, diagnostics=diagnostics,
        )
    return _ok("ion_stage_observed", "saha_joint_identifiability",
               {"element": element, "n_observed_ion_lines": int(n_observed_ion_lines)}, diagnostics)


def require_boltzmann_conditioning(
    upper_energies_ev: Sequence[float], *, snr: float, target_rel_temp_err: float,
    temperature_k: float, strict: bool, diagnostics: Optional[SolveDiagnostics] = None,
) -> GateResult:
    """The Boltzmann slope fit must meet the verified energy-spread / SNR budget.

    Theorems: ``olsSlope_noise_gain`` (noise gain = 1/ss_e) and
    ``requiredEnergySpread_sufficient`` (ss_e >= eps^2 * n / tau_beta^2) and
    ``temp_rel_error_eq`` (ErrorBudget.lean). Mirrors the shipped
    ``error_budget.py``. Fails when the selected lines cannot achieve the target
    temperature accuracy given the per-line SNR.
    """
    e = np.asarray(upper_energies_ev, dtype=float)
    n = int(e.size)
    ss_e = float(np.sum((e - e.mean()) ** 2)) if n >= 2 else 0.0
    if ss_e <= 0.0 or n < 2:
        return _fail(
            IllConditioned, "boltzmann_conditioning", "olsSlope_noise_gain",
            f"degenerate Boltzmann fit: ss_e={ss_e:.4g}, n={n} (noise gain infinite; "
            f"R^2 is meaningless here)",
            {"ss_e": ss_e, "n_lines": n},
            strict=strict, diagnostics=diagnostics,
        )
    # tau_beta is the allowed |Delta beta|; beta = 1/(kB*T). Convert the target
    # relative-T error to a tau_beta via temp_rel_error_eq: sigma_T/T = kB*T*|dbeta|.
    from cflibs.core.constants import KB_EV  # local import: keep module import-leaf

    eps = 1.0 / max(snr, 1e-12)
    tau_beta = target_rel_temp_err / max(KB_EV * temperature_k, 1e-30)
    required_ss_e = (eps ** 2) * n / max(tau_beta ** 2, 1e-300)
    if ss_e < required_ss_e:
        return _fail(
            IllConditioned, "boltzmann_conditioning", "requiredEnergySpread_sufficient",
            f"ss_e={ss_e:.4g} < required {required_ss_e:.4g} for sigma_T/T<={target_rel_temp_err:.3g} "
            f"at SNR={snr:.3g}, n={n} -> temperature accuracy target unreachable",
            {"ss_e": ss_e, "required_ss_e": required_ss_e, "snr": snr, "n_lines": n,
             "target_rel_temp_err": target_rel_temp_err},
            strict=strict, diagnostics=diagnostics,
        )
    return _ok("boltzmann_conditioning", "requiredEnergySpread_sufficient",
               {"ss_e": ss_e, "required_ss_e": required_ss_e, "n_lines": n}, diagnostics)
