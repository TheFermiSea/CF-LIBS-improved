"""Identifiability guards / refuse-to-report preconditions for CF-LIBS inversion.

Before a selector or solver trusts an inversion, it must first ask: *is the quantity
I am about to report even identifiable from the data I have?* CF-LIBS inversion is a
parameter-recovery problem, and each recovered parameter (temperature T, per-element
density Nₛ → composition, and the self-absorption optical depth τ) has a structural
precondition WITHOUT which the inverse map is not well-posed — distinct parameter values
produce the same observable, so no estimator can resolve them. Reporting a number anyway
is reporting noise dressed as physics; the honest action is to *refuse to report* and
name the failed precondition so the caller can fix the measurement or pick another route.

Each guard mirrors a PROVEN identifiability theorem in the companion Lean spec
(``CflibsFormal``) and returns a tiny ``IdentifiabilityResult(identifiable, reason)``:

1. ``temperature_identifiable`` — ``Identifiability.temperature_identifiability``: the
   Boltzmann-plot slope −1/(kB·T) is identifiable from one species iff there are ≥2 lines
   with DISTINCT upper-level energies E_k. One line, or all-equal E_k, gives a vertical/
   degenerate plot (zero energy spread, ssE = 0) → infinite slope variance → T not
   recoverable from that species alone (use a shared-T multi-species fit, or flag).

2. ``composition_identifiable`` — ``Identifiability.general_identifiability`` /
   ``compositionIdentifiability``: per-element number density (hence the closure
   composition) is identifiable iff EVERY element contributes ≥1 usable line AND a
   temperature anchor exists (the partition function and Boltzmann factor both need T).
   A missing element has density 0/0; a missing T anchor leaves the intercept→density map
   undetermined.

3. ``self_absorption_identifiable`` — ``Identifiability.selfAbsorption_breaks_identifiability``
   / ``cogRatio_injOn``: from a SINGLE optically-thick line with UNKNOWN τ, the pair
   (N, τ) is NOT identifiable — the measured intensity I = Fcal·A·n·SA(τ) admits a whole
   curve of (N, τ) that produce the same I (the (N,τ) alias). It becomes identifiable when
   τ is known a-priori (external constraint) OR when ≥2 lines of the same species at
   DISTINCT optical depth are available, because the curve-of-growth intensity RATIO is
   injective in τ (``cogRatio_injOn``) and pins τ, then N.

4. ``refuse_to_report`` — combines the above into ONE decision a selector/solver consults:
   returns ``'identifiable'`` when every precondition holds, else a flag string naming the
   FIRST failed precondition (``'temperature_not_identifiable'`` /
   ``'composition_not_identifiable'`` / ``'self_absorption_not_identifiable'``).

Physics-only: pure stdlib (no numpy needed, no ML, no DB). Energies compared with a small
absolute tolerance so floating-point-equal levels count as degenerate.
"""

from __future__ import annotations

from typing import Iterable, Mapping, NamedTuple, Sequence

# Two upper-level energies closer than this (eV) are treated as the SAME level for the
# purpose of energy spread — a Boltzmann plot over them has no usable lever arm.
_E_DEGENERACY_TOL_EV = 1e-9


class IdentifiabilityResult(NamedTuple):
    """Outcome of a single identifiability guard.

    Attributes
    ----------
    identifiable : bool
        ``True`` iff the structural precondition for recovering the quantity holds.
    reason : str
        Human-readable justification. On failure it names the violated precondition and
        the recommended route; it always cites the governing theorem.
    """

    identifiable: bool
    reason: str


def _distinct_energy_count(energies: Iterable[float], tol: float = _E_DEGENERACY_TOL_EV) -> int:
    """Number of pairwise-distinct upper-level energies (within ``tol`` eV).

    Levels within ``tol`` of one another collapse to a single representative, so a set of
    numerically-equal energies counts as ONE distinct energy (zero lever arm).
    """
    reps: list[float] = []
    for e in energies:
        ev = float(e)
        if not any(abs(ev - r) <= tol for r in reps):
            reps.append(ev)
    return len(reps)


def temperature_identifiable(
    upper_level_energies_ev: Sequence[float],
    tol: float = _E_DEGENERACY_TOL_EV,
) -> IdentifiabilityResult:
    """Is plasma temperature identifiable from ONE species' lines?

    The Boltzmann-plot slope is −1/(kB·T); recovering it requires the regression to have a
    non-zero energy lever arm, i.e. at least two lines whose upper-level energies E_k
    DIFFER. With one line, or with all upper levels equal, the energy spread ssE = 0, the
    slope is undetermined (0/0) and the slope variance is infinite — T cannot be recovered
    from this species alone. (``Identifiability.temperature_identifiability``.)

    Parameters
    ----------
    upper_level_energies_ev : sequence of float
        Upper-level energies E_k (eV) of the species' observed lines.
    tol : float, optional
        Absolute eV tolerance below which two energies count as the same level.

    Returns
    -------
    IdentifiabilityResult
        ``identifiable=True`` iff ≥2 distinct E_k; otherwise a reason recommending a
        shared-T multi-species fit (or flagging the species).
    """
    energies = list(upper_level_energies_ev)
    n = len(energies)
    if n < 2:
        return IdentifiabilityResult(
            False,
            f"temperature_identifiability: only {n} line(s) for this species; the "
            "Boltzmann-plot slope -1/(kB*T) needs >=2 lines with distinct upper-level "
            "energies. Cannot get T from one species alone -> use a shared-T multi-species "
            "fit or flag the species.",
        )
    distinct = _distinct_energy_count(energies, tol=tol)
    if distinct < 2:
        return IdentifiabilityResult(
            False,
            f"temperature_identifiability: {n} lines but all upper-level energies are "
            f"equal (within {tol:g} eV), so the energy spread ssE = 0 and the slope is "
            "degenerate (0/0). Cannot get T from one species alone -> use a shared-T "
            "multi-species fit or flag the species.",
        )
    return IdentifiabilityResult(
        True,
        f"temperature_identifiability: {distinct} distinct upper-level energies over {n} "
        "lines give a non-zero energy lever arm; the Boltzmann-plot slope -1/(kB*T) is "
        "identifiable.",
    )


def composition_identifiable(
    lines_by_element: Mapping[str, int],
    has_temperature_anchor: bool,
) -> IdentifiabilityResult:
    """Is the per-element composition identifiable?

    Closure composition Cₛ = Nₛ/ΣN is identifiable iff EVERY element contributes at least
    one usable line AND a temperature anchor is available. Each element's number density is
    recovered from a Boltzmann intercept via Nₛ = Uₛ(T)·exp(intercept)/Fcal — which needs
    BOTH a line for that element (otherwise its density is undetermined, 0/0) AND a value of
    T (the partition function Uₛ(T) and the Boltzmann factor are functions of T). A missing
    element or a missing T anchor breaks well-posedness.
    (``Identifiability.general_identifiability`` / ``compositionIdentifiability``.)

    Parameters
    ----------
    lines_by_element : mapping str -> int
        Number of usable lines observed per element.
    has_temperature_anchor : bool
        Whether a temperature is available (e.g. from a per-species Boltzmann fit that IS
        identifiable, or an externally imposed / shared-T value).

    Returns
    -------
    IdentifiabilityResult
        ``identifiable=True`` iff every element has >=1 line AND a T anchor exists.
    """
    if not lines_by_element:
        return IdentifiabilityResult(
            False,
            "compositionIdentifiability: no elements supplied; composition over an empty "
            "set is undefined. Provide >=1 line per element to be quantified.",
        )
    missing = sorted(sym for sym, n in lines_by_element.items() if int(n) < 1)
    if missing:
        return IdentifiabilityResult(
            False,
            "compositionIdentifiability: element(s) "
            f"{missing} have 0 usable lines; their number density is undetermined (0/0), so "
            "the closure composition is not identifiable. Acquire >=1 line per element or "
            "drop it from the closure set.",
        )
    if not has_temperature_anchor:
        return IdentifiabilityResult(
            False,
            "compositionIdentifiability: no temperature anchor; the intercept->density map "
            "Ns = Us(T)*exp(intercept)/Fcal needs T (partition function + Boltzmann factor "
            "depend on T). Establish T (per-species or shared-T fit) before reporting "
            "composition.",
        )
    n_elems = len(lines_by_element)
    return IdentifiabilityResult(
        True,
        f"compositionIdentifiability: all {n_elems} element(s) have >=1 line and a "
        "temperature anchor is present; per-element density and the closure composition are "
        "identifiable.",
    )


def self_absorption_identifiable(
    n_lines_same_species: int,
    tau_known: bool,
    n_distinct_optical_depth: int | None = None,
) -> IdentifiabilityResult:
    """Is the self-absorption optical depth τ (hence the absorption-corrected density)
    identifiable?

    A single optically-thick line with UNKNOWN τ does NOT identify (N, τ): the observable
    I = Fcal·A·n·SA(τ), with SA(τ)=(1−e^{−τ})/τ, is matched by an entire curve of (N, τ)
    pairs — increase τ (more self-absorption) and increase N to compensate. This is the
    (N, τ) alias (``Identifiability.selfAbsorption_breaks_identifiability``).

    It becomes identifiable when EITHER:
      * τ is known a-priori (an external constraint collapses the curve to a point), OR
      * ≥2 lines of the same species at DISTINCT optical depth are available — the
        curve-of-growth intensity RATIO is strictly monotone (injective) in τ
        (``cogRatio_injOn``), so the ratio pins τ, and then N follows.

    Parameters
    ----------
    n_lines_same_species : int
        Number of lines of the species available for the self-absorption diagnosis.
    tau_known : bool
        Whether τ is externally known / constrained.
    n_distinct_optical_depth : int or None, optional
        Number of those lines at DISTINCT optical depth. Defaults to
        ``n_lines_same_species`` (assume distinct τ when not specified). A curve-of-growth
        ratio needs ≥2 distinct-τ lines to be informative.

    Returns
    -------
    IdentifiabilityResult
        ``identifiable=True`` iff τ is known, or ≥2 same-species lines at distinct optical
        depth exist; otherwise the (N, τ) alias makes it unidentifiable.
    """
    n = int(n_lines_same_species)
    n_distinct = n if n_distinct_optical_depth is None else int(n_distinct_optical_depth)

    if tau_known:
        return IdentifiabilityResult(
            True,
            "selfAbsorption: tau is externally known, collapsing the (N, tau) curve to a "
            "point; the absorption-corrected density N = I/(Fcal*A*pop*SA(tau)) is "
            "identifiable. Route: apply the known-tau escape-factor correction.",
        )
    if n >= 2 and n_distinct >= 2:
        return IdentifiabilityResult(
            True,
            f"selfAbsorption: {n_distinct} same-species lines at distinct optical depth; "
            "the curve-of-growth intensity ratio is injective in tau (cogRatio_injOn), so "
            "tau is identifiable and N follows. Route: curve-of-growth / CDSB ratio fit.",
        )
    if n <= 1:
        return IdentifiabilityResult(
            False,
            f"selfAbsorption_breaks_identifiability: a single thick line ({n} line) with "
            "unknown tau cannot resolve the (N, tau) alias -- I = Fcal*A*n*SA(tau) admits a "
            "whole curve of (N, tau). Route: supply a known tau, or add a second "
            "same-species line at a distinct optical depth (curve-of-growth ratio).",
        )
    return IdentifiabilityResult(
        False,
        f"selfAbsorption_breaks_identifiability: {n} lines but only {n_distinct} distinct "
        "optical depth(s); without distinct-tau lines the curve-of-growth ratio is "
        "uninformative and the (N, tau) alias remains. Route: supply a known tau, or add a "
        "same-species line at a distinct optical depth.",
    )


def refuse_to_report(
    *,
    upper_level_energies_ev: Sequence[float] | None = None,
    has_temperature_anchor: bool | None = None,
    lines_by_element: Mapping[str, int] | None = None,
    self_absorption_n_lines: int | None = None,
    self_absorption_tau_known: bool = False,
    self_absorption_n_distinct_optical_depth: int | None = None,
) -> IdentifiabilityResult:
    """Single refuse-to-report gate combining every applicable identifiability guard.

    A selector/solver calls this BEFORE trusting an inversion. Each precondition is checked
    only when its inputs are supplied (so a caller can gate a temperature-only, a
    composition-only, or a full inversion). The FIRST failing precondition short-circuits
    and its flag name is reported, so the caller knows exactly what to fix.

    ``identifiable=True`` with reason ``'identifiable'`` means every supplied precondition
    holds. On failure, ``reason`` is one of the flag strings
    ``'temperature_not_identifiable'`` / ``'composition_not_identifiable'`` /
    ``'self_absorption_not_identifiable'`` followed by the underlying guard's explanation.

    Parameters
    ----------
    upper_level_energies_ev : sequence of float, optional
        Triggers the temperature guard. If a temperature anchor is being DERIVED from these
        lines and they fail, composition is not checked (it would have no anchor anyway).
    has_temperature_anchor : bool, optional
        Whether a T anchor exists for the composition guard. If ``None`` and
        ``upper_level_energies_ev`` is supplied, it is inferred from whether temperature is
        itself identifiable.
    lines_by_element : mapping str -> int, optional
        Triggers the composition guard (lines per element).
    self_absorption_n_lines : int, optional
        Triggers the self-absorption guard (same-species line count).
    self_absorption_tau_known : bool, optional
        Whether τ is externally known (self-absorption guard).
    self_absorption_n_distinct_optical_depth : int or None, optional
        Distinct-optical-depth line count (self-absorption guard).

    Returns
    -------
    IdentifiabilityResult
        ``identifiable=True, reason='identifiable'`` iff all supplied guards pass; else the
        first failure's flag + explanation.
    """
    checks: list[str] = []

    temperature_ok: bool | None = None
    if upper_level_energies_ev is not None:
        t_res = temperature_identifiable(upper_level_energies_ev)
        temperature_ok = t_res.identifiable
        if not t_res.identifiable:
            return IdentifiabilityResult(False, f"temperature_not_identifiable: {t_res.reason}")
        checks.append("temperature")

    if lines_by_element is not None:
        anchor = has_temperature_anchor
        if anchor is None:
            # Infer the anchor from the temperature guard if it ran; else assume absent.
            anchor = bool(temperature_ok)
        c_res = composition_identifiable(lines_by_element, anchor)
        if not c_res.identifiable:
            return IdentifiabilityResult(False, f"composition_not_identifiable: {c_res.reason}")
        checks.append("composition")

    if self_absorption_n_lines is not None:
        s_res = self_absorption_identifiable(
            self_absorption_n_lines,
            self_absorption_tau_known,
            self_absorption_n_distinct_optical_depth,
        )
        if not s_res.identifiable:
            return IdentifiabilityResult(False, f"self_absorption_not_identifiable: {s_res.reason}")
        checks.append("self_absorption")

    if not checks:
        return IdentifiabilityResult(
            False,
            "refuse_to_report: no identifiability preconditions were supplied; nothing was "
            "checked, so the inversion cannot be trusted. Provide at least one of "
            "upper_level_energies_ev / lines_by_element / self_absorption_n_lines.",
        )

    return IdentifiabilityResult(
        True,
        "identifiable",
    )
