"""Property-based tests of the verified CF-LIBS invariants over RANDOMIZED valid inputs.

The companion test ``tests/oracle/test_spec_regression.py`` pins our estimators against the
machine-verified Lean spec on the FOUR fixed fixtures. This file generalizes those same
PROVEN theorems to a *seeded random ensemble* (``random.Random``; no Hypothesis / external
dependency) of valid multi-element configs, so a regression that only happens to be correct
on the canonical ternary alloy is still caught.

Each test below instantiates an invariant that is a proven theorem in ``cflibs-formal``:

1. CLOSURE (``Closure.composition_sum_one`` / ``mem_stdSimplex``)
2. CALIBRATION-FREE (``Classic.classic_calibration_free``)
3. ROUND-TRIP (``Classic.classicDensity_recovers`` / ``classic_sound``)
4. PER-ELEMENT U_s (``Closure.composition_sum_one`` over heterogeneous U_s; + the
   negative: a shared/dropped U_s corrupts closure)
5. OLS soundness + agrees-classic (``Alt.olsDensity_recovers`` / ``leastSquares_sound`` /
   ``leastSquares_agrees_classic``)
6. SELF-ABSORPTION (``Alt.selfAbsorbed_sound`` + downward-bias direction)
7. SAHA antitone (``Saha.electronDensity_antitone``): ``n_e = S/R`` strictly decreasing in R

Convention bridge (identical to ``test_spec_regression.py``): the spec is dimensionless
(kB = T = Fcal = 1, photon-rate forward ``I_spec = Fcal·A·N·g·e^{-E/kBT}/U``, ordinate
``ln(I_spec/(gA))``). Our pipeline uses the energy ordinate ``ln(I_our·λ/(gA))``. We feed
``I_our = I_spec/λ`` with DISTINCT per-line λ so ``ln(I_our·λ/(gA)) = ln(I_spec/(gA))``
reproduces the verified ordinate EXACTLY while still exercising our λ path (λ=1 would be
blind to a λ-handling bug).
"""

from __future__ import annotations

import math
import random

from cflibs.inversion.common.data_structures import LineObservation
from cflibs.inversion.physics.boltzmann import BoltzmannPlotFitter
from cflibs.inversion.physics.closure import ClosureEquation
from cflibs.inversion.physics.self_absorption import _escape_factor

# Spec convention: dimensionless kB = T = 1.
KB = 1.0
T = 1.0
RTOL = 1e-6
N_CASES = 50  # seeded ensemble size; keeps total runtime well under 60s.


# ---------------------------------------------------------------- helpers (bridge)
def _approx(a: float, b: float, rtol: float = RTOL) -> bool:
    return abs(a - b) <= rtol * max(1.0, abs(b))


def _partition(g, E) -> float:
    """U(T) = Σ g_k exp(-E_k/(kB·T)); matches the spec's partition_function."""
    return sum(gk * math.exp(-Ek / (KB * T)) for gk, Ek in zip(g, E))


def _line_lambda_nm(k: int) -> float:
    """Distinct, realistic per-line wavelength so the test exercises our λ path."""
    return 300.0 + 80.0 * k  # 300, 380, 460, ... nm (all distinct)


def _spec_intensities(el, fcal: float = 1.0) -> list[float]:
    """Spec forward map I_spec[k] = Fcal·A_k·N·g_k·exp(-E_k/(kB·T)) / U."""
    U = _partition(el["g"], el["E"])
    return [
        fcal * el["A"][k] * el["N"] * el["g"][k] * math.exp(-el["E"][k] / (KB * T)) / U
        for k in range(len(el["g"]))
    ]


def _observations(el, intensities):
    """Build LineObservations applying the photon->energy bridge I_our = I_spec/λ with
    DISTINCT per-line λ (so ln(I_our·λ/gA) == spec ln(I_spec/gA), λ-bug-sensitive)."""
    return [
        LineObservation(
            wavelength_nm=_line_lambda_nm(k),
            intensity=float(intensities[k]) / _line_lambda_nm(k),  # I_spec/λ
            intensity_uncertainty=1e-9,  # noise-free synthetic forward
            element=el["sym"],
            ionization_stage=1,
            E_k_ev=float(el["E"][k]),
            g_k=int(el["g"][k]),
            A_ki=float(el["A"][k]),
        )
        for k in range(len(el["g"]))
    ]


def _fit(observations):
    return BoltzmannPlotFitter().fit(observations)


def _composition_map(result) -> dict:
    """Extract {element: fraction} from a ClosureResult across its possible shapes."""
    for attr in ("concentrations", "composition", "number_fractions", "fractions"):
        val = getattr(result, attr, None)
        if isinstance(val, dict):
            return {k: float(v) for k, v in val.items()}
    raise AssertionError(f"ClosureResult exposes no composition dict (fields: {dir(result)})")


def _density_from_fit(el, intensities) -> float:
    """OLS-recovered density N = U·exp(intercept) under the I_spec/λ bridge (Fcal=1)."""
    res = _fit(_observations(el, intensities))
    return _partition(el["g"], el["E"]) * math.exp(res.intercept)


# ---------------------------------------------------------------- random valid config
def _rand_element(rng: random.Random, sym: str) -> dict:
    """A valid element: >=2 lines, DISTINCT strictly-increasing energies (spread for the
    slope), positive integer g, positive A, positive density N, a tau in [0, 2]."""
    n_lines = rng.randint(2, 5)
    g = [rng.randint(1, 9) for _ in range(n_lines)]
    # strictly increasing, distinct energies (E[0]=0 ground), guarantees energy spread.
    E = [0.0]
    for _ in range(n_lines - 1):
        E.append(E[-1] + rng.uniform(0.4, 1.5))
    A = [rng.uniform(0.1, 1.0) for _ in range(n_lines)]
    N = rng.uniform(0.5, 10.0)
    u = rng.randrange(n_lines)
    tau = rng.uniform(0.0, 2.0)
    return {"sym": sym, "g": g, "E": E, "A": A, "N": N, "u": u, "tau": tau}


def _rand_alloy(rng: random.Random, n_elements: int | None = None) -> list[dict]:
    """>=2 elements with DIFFERENT (g, E, A) so each has its own partition function U_s.
    Re-rolls until at least two elements have distinct U_s (the per-element-U point)."""
    n = n_elements if n_elements is not None else rng.randint(2, 4)
    while True:
        els = [_rand_element(rng, f"El-{i}") for i in range(n)]
        partitions = [_partition(e["g"], e["E"]) for e in els]
        # require at least two DISTINCT partition functions (heterogeneous U_s).
        if any(not _approx(partitions[0], p) for p in partitions[1:]):
            return els


def _true_composition(els) -> list[float]:
    total = sum(e["N"] for e in els)
    return [e["N"] / total for e in els]


# ================================================================ PROPERTY 1: CLOSURE
def test_property_closure_simplex_and_truth():
    """Recovered composition lies in the standard simplex (0<=C_s<=1, Σ=1) AND equals the
    true mole fractions over a random ensemble of heterogeneous alloys
    (Closure.composition_sum_one / mem_stdSimplex / classic_sound)."""
    rng = random.Random(20240601)
    for _ in range(N_CASES):
        els = _rand_alloy(rng)
        intercepts = {e["sym"]: _fit(_observations(e, _spec_intensities(e))).intercept for e in els}
        partitions = {e["sym"]: _partition(e["g"], e["E"]) for e in els}
        comp = _composition_map(ClosureEquation().apply_standard(intercepts, partitions))
        total = sum(comp.values())
        assert _approx(total, 1.0), f"composition sums to {total}, not 1"
        truth = _true_composition(els)
        for i, e in enumerate(els):
            c = comp[e["sym"]]
            assert -RTOL <= c <= 1.0 + RTOL, f"{e['sym']}: C_s={c} outside [0,1]"
            assert _approx(c, truth[i]), f"{e['sym']}: recovered {c} != true {truth[i]}"


# ================================================================ PROPERTY 2: CALIBRATION-FREE
def test_property_calibration_free_invariance():
    """Scaling Fcal (≡ scaling all intensities) leaves the recovered composition unchanged
    over a random ensemble (Classic.classic_calibration_free)."""
    rng = random.Random(20240602)
    for case in range(N_CASES):
        els = _rand_alloy(rng)
        scale = rng.uniform(2.0, 1000.0)
        partitions = {e["sym"]: _partition(e["g"], e["E"]) for e in els}
        base = {e["sym"]: _fit(_observations(e, _spec_intensities(e))).intercept for e in els}
        scaled = {
            e["sym"]: _fit(_observations(e, _spec_intensities(e, fcal=scale))).intercept
            for e in els
        }
        cb = _composition_map(ClosureEquation().apply_standard(base, partitions))
        cs = _composition_map(ClosureEquation().apply_standard(scaled, partitions))
        for sym in cb:
            assert _approx(
                cb[sym], cs[sym]
            ), f"case {case} {sym}: not Fcal-invariant ({cb[sym]} vs {cs[sym]}, scale={scale})"


# ================================================================ PROPERTY 3: ROUND-TRIP
def test_property_round_trip_density():
    """Forward (I_spec) then invert (OLS Boltzmann plot) recovers the true density N per
    element over a random ensemble (Classic.classicDensity_recovers / classic_sound)."""
    rng = random.Random(20240603)
    for _ in range(N_CASES):
        els = _rand_alloy(rng)
        for e in els:
            n_rec = _density_from_fit(e, _spec_intensities(e))
            assert _approx(n_rec, e["N"]), f"{e['sym']}: recovered N {n_rec} != {e['N']}"


# ================================================================ PROPERTY 4: PER-ELEMENT U_s
def test_property_per_element_partition_required():
    """Elements with DISTINCT partition functions recover correct composition WITH their own
    U_s; and dropping/sharing a single U_s CORRUPTS the composition (the negative — a test
    that only ever passes proves nothing) (Closure.composition_sum_one)."""
    rng = random.Random(20240604)
    corrupted_ever = False
    for _ in range(N_CASES):
        els = _rand_alloy(rng)
        intercepts = {e["sym"]: _fit(_observations(e, _spec_intensities(e))).intercept for e in els}
        partitions = {e["sym"]: _partition(e["g"], e["E"]) for e in els}
        truth = _true_composition(els)

        # CORRECT: each element's own U_s -> recovers truth.
        good = _composition_map(ClosureEquation().apply_standard(intercepts, partitions))
        for i, e in enumerate(els):
            assert _approx(good[e["sym"]], truth[i]), f"{e['sym']}: own-U_s closure wrong"

        # BUG: share one U across all elements -> at least one fraction must go wrong.
        shared_U = {e["sym"]: 1.0 for e in els}
        bad = _composition_map(ClosureEquation().apply_standard(intercepts, shared_U))
        if any(not _approx(bad[e["sym"]], truth[i]) for i, e in enumerate(els)):
            corrupted_ever = True
    assert (
        corrupted_ever
    ), "sharing a single U_s never corrupted closure — oracle not discriminating"


# ================================================================ PROPERTY 5: OLS soundness
def test_property_ols_sound_and_agrees_classic():
    """OLS over ALL lines recovers N per element (leastSquares_sound) AND the OLS composition
    equals the closure/classic composition on noise-free data (leastSquares_agrees_classic)."""
    rng = random.Random(20240605)
    for _ in range(N_CASES):
        els = _rand_alloy(rng)
        # OLS density per element (uses all lines via the Boltzmann-plot intercept).
        ols_dens = [_density_from_fit(e, _spec_intensities(e)) for e in els]
        for d, e in zip(ols_dens, els):
            assert _approx(d, e["N"]), f"{e['sym']}: OLS N {d} != {e['N']}"
        total = sum(ols_dens)
        ols_comp = [d / total for d in ols_dens]
        # closure composition over the same fits must agree with the OLS-density composition.
        intercepts = {e["sym"]: _fit(_observations(e, _spec_intensities(e))).intercept for e in els}
        partitions = {e["sym"]: _partition(e["g"], e["E"]) for e in els}
        closure_comp = _composition_map(ClosureEquation().apply_standard(intercepts, partitions))
        for i, e in enumerate(els):
            assert _approx(
                ols_comp[i], closure_comp[e["sym"]]
            ), f"{e['sym']}: OLS comp {ols_comp[i]} != closure {closure_comp[e['sym']]}"


# ================================================================ PROPERTY 6: SELF-ABSORPTION
def test_property_self_absorption_corrects_bias():
    """For tau>0, the escape factor SA(tau)=(1-e^-tau)/tau is in (0,1); an optically-thick
    spectrum thin*SA biases the recovered density strictly BELOW truth, and dividing by
    SA(tau) recovers N exactly (Alt.selfAbsorbed_sound + bias direction)."""
    rng = random.Random(20240606)
    cases = 0
    for _ in range(N_CASES):
        els = _rand_alloy(rng)
        for e in els:
            tau = e["tau"]
            sa = _escape_factor(tau)
            expected = 1.0 if tau == 0 else (1 - math.exp(-tau)) / tau
            assert _approx(sa, expected), f"escape factor wrong at tau={tau}"
            if tau <= 1e-9:
                continue  # SA ~ 1: no measurable bias to assert.
            cases += 1
            assert 0.0 < sa < 1.0, f"SA(tau={tau})={sa} must be in (0,1)"
            thin = _spec_intensities(e)
            # uniform thick = thin*SA on every line: slope unchanged, intercept shifts ln(SA).
            thick = [x * sa for x in thin]
            n_uncorrected = _density_from_fit(e, thick)
            assert _approx(n_uncorrected, e["N"] * sa), f"{e['sym']}: thick density != N·SA"
            assert n_uncorrected < e["N"], f"{e['sym']}: self-absorption must bias DOWN"
            # correcting each line by /SA recovers truth.
            corrected = [x / sa for x in thick]
            n_corrected = _density_from_fit(e, corrected)
            assert _approx(
                n_corrected, e["N"]
            ), f"{e['sym']}: corrected N {n_corrected} != {e['N']}"
    assert cases > 0, "no tau>0 cases exercised the self-absorption correction"


# ================================================================ PROPERTY 7: SAHA antitone
def _saha_factor(me: float, h: float, chi: float, gZ, EZ, gZ1, EZ1) -> float:
    """S(T) = 2·(U_{Z+1}/U_Z)·(2π m_e kB T / h^2)^{3/2}·exp(-chi/(kB T)) (spec saha_factor)."""
    bracket = 2.0 * math.pi * me * KB * T / (h * h)
    return (
        2.0 * (_partition(gZ1, EZ1) / _partition(gZ, EZ)) * bracket**1.5 * math.exp(-chi / (KB * T))
    )


def test_property_saha_antitone_in_ratio():
    """n_e = S(T)/R is strictly DECREASING in the stage ratio R (Saha.electronDensity_antitone):
    for the same S(T), a larger ion/neutral ratio R implies a smaller electron density."""
    rng = random.Random(20240607)
    for _ in range(N_CASES):
        # random two-stage config: positive g, distinct energies, positive chi.
        n_neu, n_ion = rng.randint(2, 4), rng.randint(2, 4)
        gZ = [rng.randint(1, 9) for _ in range(n_neu)]
        EZ = [0.0] + [rng.uniform(0.4, 2.0) for _ in range(n_neu - 1)]
        gZ1 = [rng.randint(1, 9) for _ in range(n_ion)]
        EZ1 = [0.0] + [rng.uniform(0.4, 2.0) for _ in range(n_ion - 1)]
        me, h, chi = rng.uniform(0.5, 2.0), rng.uniform(0.5, 2.0), rng.uniform(0.5, 3.0)
        S = _saha_factor(me, h, chi, gZ, EZ, gZ1, EZ1)
        assert S > 0.0, "Saha factor must be positive"
        # strictly increasing ratios -> strictly decreasing n_e = S/R.
        ratios = sorted(rng.uniform(0.05, 5.0) for _ in range(6))
        # de-dup so the comparison is strict.
        ratios = [r for i, r in enumerate(ratios) if i == 0 or ratios[i] > ratios[i - 1] + 1e-9]
        ne = [S / R for R in ratios]
        for a, b in zip(ne, ne[1:]):
            assert b < a, f"n_e not strictly antitone in R: {b} !< {a} (S={S})"
        # spot-check the Saha law identity R·n_e == S(T).
        for R, n in zip(ratios, ne):
            assert _approx(R * n, S), f"Saha law R·n_e != S(T): {R * n} vs {S}"
