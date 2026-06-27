"""
Shared CF-LIBS analysis pipeline: configuration resolution and execution.

This module is the SINGLE source of truth for the detection -> selection ->
iterative-solve pipeline behind the ``cflibs analyze``/``invert``/``batch``
CLI commands AND the benchmark harnesses under ``scripts/`` (e.g.
``scripts/measure_bhvo2_presence.py``). It was extracted from
``cflibs/cli/main.py`` (bead vj82) because the BHVO-2 gate script used to
hand-build its own pipeline: two harnesses produced incomparable baselines
for the *same* knobs (CLI RMSE ~5.8 wt% vs script 4.03 wt% on ChemCam
BHVO-2), making every default-tuning decision ambiguous.

Anything that measures the CF-LIBS pipeline must build its configuration
through :func:`build_pipeline_config` and execute through
:func:`run_pipeline`. Knobs where harnesses have historically diverged
(per-spectrum wavelength calibration, the shift-coherence veto) are explicit
fields on :class:`AnalysisPipelineConfig` so any difference between two runs
is visible in the resolved config rather than buried in wiring.

The CLI keeps thin backward-compatible aliases (``_build_pipeline_config``,
``_detect_and_select_lines``, ``_run_pipeline``) in ``cflibs.cli.main``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from cflibs.core.logging_config import get_logger
from cflibs.inversion.physics.self_absorption_observable import normalize_self_absorption_mode

logger = get_logger("inversion.pipeline")


# ---------------------------------------------------------------------------
# Shared analyze/invert/batch pipeline configuration (bead l4a8)
# ---------------------------------------------------------------------------

#: Closure modes accepted by the iterative solver / CLI.
CLOSURE_MODES = ("standard", "matrix", "oxide", "ilr", "pwlr", "dirichlet_residual")

#: Solver backends the unified pipeline supports end-to-end (peak-based
#: ``iterative`` plus the full-spectrum ``joint``/``bayesian`` fits).
SOLVER_BACKENDS = ("iterative", "joint", "bayesian")
#: All solver names ``build_pipeline_config`` accepts. ``closed_form`` is an
#: additional peak-based backend; ``coarse_to_fine`` is reserved for the
#: manifold path (``_dispatch_solver`` raises NotImplementedError until wired).
_VALID_SOLVERS = SOLVER_BACKENDS + ("closed_form", "coarse_to_fine")

#: Preset bundles for the accuracy-critical solver knobs, measured on real
#: ChemCam BHVO-2 (docs/audit/2026-06-09-overhaul/04-pipeline-defaults.md):
#: the default ``analyze`` path scored RMSE 10.29 wt% (Fe 39 vs certified
#: 8.6) while ``--saha-boltzmann-graph --closure-mode oxide`` scored 4.03.
#: ``geological`` (the default) IS that measured-best configuration;
#: ``metallic`` keeps the pooled SB-graph but uses the standard closure
#: (oxide stoichiometry is wrong physics for alloys); ``raw`` reproduces the
#: legacy defaults for comparison runs.
#:
#: The three axis-alignment knobs (bead ye6t, measured on ChemCam BHVO-2
#: loc1: RMSE 4.50 -> 2.25 wt%) default to the FIXED behaviour in the
#: production presets and to the legacy behaviour in ``raw``:
#: ``residual_shift_scan_nm=0`` (a quality-passed calibration must not be
#: re-broken by a match-count-greedy mop-up scan that measurably rides its
#: window edge), ``affine_coverage_gate=True`` (never extrapolate a fitted
#: dispersion slope past its inlier anchors), and ``line_residual_gate=True``
#: (drop individual matched lines incoherent with the consensus residual).
ANALYSIS_PRESETS = {
    "geological": {
        "saha_boltzmann_graph": True,
        "closure_mode": "oxide",
        "stark_ne": True,
        "residual_shift_scan_nm": 0.0,
        "affine_coverage_gate": True,
        "line_residual_gate": True,
    },
    "metallic": {
        "saha_boltzmann_graph": True,
        "closure_mode": "standard",
        "stark_ne": True,
        "residual_shift_scan_nm": 0.0,
        "affine_coverage_gate": True,
        "line_residual_gate": True,
    },
    # DED constrained-known-element-set preset (DED-PLAN Edit B). Tuned to keep
    # a small fixed element set (e.g. {Ti,Al,V} for Ti-6Al-4V) intact end to
    # end: metallic sum-to-one closure (no oxide), Stark n_e to break the
    # T/n_e degeneracy, the degeneracy guard armed for K>=2, and the per-element
    # selection floors relaxed so a faint minor element (Al/V at extreme drift)
    # is never dropped. ``constrained_elements`` (the hard no-drop switch) is
    # added to this preset in DED-PLAN step 7 once the field exists.
    "metallic_ded": {
        "saha_boltzmann_graph": True,
        "closure_mode": "standard",
        "stark_ne": True,
        "residual_shift_scan_nm": 0.0,
        "affine_coverage_gate": True,
        "line_residual_gate": True,
        "degeneracy_min_elements": 2,
        "min_lines_per_element": 1,
        "min_snr": 5.0,
        "min_energy_spread_ev": 1.5,
        "degeneracy_dominance_threshold": 0.95,
        "max_lines_per_element": 30,
        "top_k_per_element": 40,
        # DED runs on a known/constrained element set and track composition drift;
        # the post-loop Cristoforetti reliability annotation is not consumed, so
        # skip the ~26-34% re-fit (T/n_e/composition are unaffected).
        "assess_quality": False,
    },
    "raw": {
        "saha_boltzmann_graph": False,
        "closure_mode": "standard",
        "stark_ne": False,
        "residual_shift_scan_nm": 0.05,
        "affine_coverage_gate": False,
        "line_residual_gate": False,
    },
}

DEFAULT_ANALYSIS_PRESET = "geological"

#: Sampling-resolution cap factors for the shared detection entry
#: (bead CF-LIBS-improved-qitd). The default matching tolerance (0.1 nm) and
#: integration width (0.2 nm) are absolute constants that silently assume a
#: moderate-resolution spectrometer (~0.05-0.1 nm/pixel). On a high-resolution
#: instrument that samples much more finely (e.g. the Silva 2022 tropical-soil
#: echelle data at ~0.011 nm/pixel) a 0.1 nm tolerance spans ~9 sampling steps,
#: so the +/-tolerance match window covers a large fraction of the densely-peaked
#: axis and comb matching degenerates to random coincidence: every dense-catalog
#: confounder accrues matches, no real element can clear the
#: ``matched / total_peaks`` precision gate, the comb falls back to nearest and
#: the shift-coherence veto removes the spurious survivors -- yielding
#: "No usable spectral lines detected for inversion" on every spectrum. The cap
#: ties the tolerance/width to the instrument's actual sampling step. It is
#: applied with ``min`` (tighten-only): a coarsely-sampled instrument's
#: ``factor * wl_step`` exceeds the legacy constant, so the cap is inert and its
#: behaviour is byte-identical; only a finely-sampled instrument is brought down
#: to its sampling-resolution scale. The 2:4 px ratio preserves the legacy
#: 0.1:0.2 nm tolerance:width ratio.
SAMPLING_TOLERANCE_PX = 2.0
SAMPLING_WIDTH_PX = 4.0


@dataclass
class AnalysisPipelineConfig:
    """Fully resolved configuration for the shared CF-LIBS analysis pipeline.

    One instance describes everything ``analyze``, ``invert`` and ``batch``
    feed into detection, selection and the iterative solver. Building it
    through :func:`build_pipeline_config` is what guarantees the three
    entry points cannot drift apart (the pre-fix ``batch`` wiring kept the
    exact raw-detection + bare-``LineSelector`` path whose drift caused the
    Na=98% blowup).
    """

    preset: str
    elements: list
    # Detection + selection knobs (mirror ``detect_and_select_lines``).
    min_relative_intensity: Optional[float] = None
    top_k_per_element: Optional[int] = 60
    resolving_power: Optional[float] = None
    wavelength_tolerance_nm: float = 0.1
    min_peak_height: float = 0.01
    peak_width_nm: float = 0.2
    #: Self-absorption mode: ``'off'`` or ``'observable'`` (bead 0jvr).
    #: Booleans are accepted on input and normalized (True -> 'observable').
    apply_self_absorption: str = "off"
    exclude_resonance: Optional[bool] = None
    min_snr: float = 10.0
    min_energy_spread_ev: float = 2.0
    min_lines_per_element: int = 3
    isolation_wavelength_nm: float = 0.1
    max_lines_per_element: int = 20
    grade_aware_selection: bool = False
    #: Target relative temperature accuracy σ_T/T (gated, default None=off). When set, the
    #: line-selection SNR/spread/min-lines gates are DERIVED from this target via the verified
    #: ErrorBudget (cflibs.inversion.physics.derived_thresholds), replacing the tuned magic
    #: numbers. ~0.10 reproduces the legacy min_snr=10.
    target_sigma_t: Optional[float] = None
    #: Representative plasma T (K) for the σ_T -> slope-target conversion; only used with target_sigma_t.
    plasma_temperature_K: float = 10000.0
    #: Reliability-ranked selection (gated): when the per-element max-lines cap binds, keep
    #: the max-energy-spread subset (best T-conditioning, twoLineBeta_stable_sharp) instead
    #: of the top-scored lines. No-op when the cap does not bind.
    reliability_ranked_selection: bool = False
    wavelength_calibration: bool = True
    shift_coherence_veto: bool = True
    #: Residual comb shift-scan half-width (nm) AFTER a quality-passed
    #: calibration (bead ye6t). Default 0: the fitted axis is trusted; the
    #: legacy 0.05 mop-up measurably rode its window edge and admitted
    #: contaminated matches (Al I 892.356). The full 0.5 nm global scan still
    #: runs whenever calibration is skipped or fails its quality gate.
    residual_shift_scan_nm: float = 0.0
    #: Global comb shift-scan half-width (nm) used whenever per-spectrum
    #: wavelength calibration is disabled, skipped, or fails its quality gate
    #: (the legacy single-constant scan). A quality-passed calibration replaces
    #: it with ``residual_shift_scan_nm``. Formerly a magic
    #: ``detection_overrides["shift_scan_nm"]`` key; now a first-class field
    #: resolved and logged like every other knob.
    global_shift_scan_nm: float = 0.5
    #: Degrade per-segment affine calibration fits to pure shift when the
    #: inlier anchors do not cover the segment (bead ye6t): never extrapolate
    #: a dispersion slope past its anchors.
    affine_coverage_gate: bool = True
    #: Drop individual matched lines whose residual is incoherent with the
    #: consensus residual shift (bead ye6t): kills contaminated matches and
    #: lucky-coherent FP line sets that survive the element-level veto.
    line_residual_gate: bool = True
    #: Build the wavelength-calibration reference line pool ONCE over the full
    #: axis and slice it per segment (flag ``CFLIBS_CALIB_POOL_CACHE``), instead
    #: of re-querying SQLite + re-ranking per segment. Parity-exact with the
    #: per-segment build; default ``False`` (also honours the env var so the
    #: benchmark can toggle without editing config).
    calib_pool_cache: bool = False
    #: Optional path to a spectral-response curve E(lambda); identity when None
    #: (audit 02-F5 — ChemCam CCS data arrive response-corrected upstream).
    response_curve: Optional[str] = None
    #: Deterministic Hough coarse-dispersion seed for the RANSAC wavelength
    #: calibrator (RASCAL-style, arXiv:1912.05883). ``None`` (default) defers to
    #: the ``CFLIBS_HOUGH_CALIB`` env var (off when unset), keeping the
    #: calibration path byte-identical to the legacy RNG-only RANSAC. When
    #: enabled a consensus affine seed warm-starts RANSAC and the loop runs a
    #: small polishing pass instead of the full cold-start budget (faster; NOT
    #: parity-exact -> benchmark-gated).
    hough_calib_seed: Optional[bool] = None
    # Iterative-solver knobs (mirror ``IterativeCFLIBSSolver``).
    max_iterations: int = 20
    t_tolerance_k: float = 100.0
    ne_tolerance_frac: float = 0.1
    pressure_pa: float = 101325.0
    boltzmann_weight_cap: float = 5.0
    min_boltzmann_r2: float = 0.3
    saha_boltzmann_graph: bool = True
    closure_mode: str = "oxide"
    closure_kwargs: dict = field(default_factory=dict)
    matrix_element: Optional[str] = None
    oxide_elements: Optional[list] = None
    #: Stark-broadening n_e diagnostic (bead pxex): measure n_e from observed
    #: literature-grade line widths; falls back (with warning) when none qualify.
    stark_ne: bool = True
    # --- Solver dispatch (unified solver seam) ---
    #: Which solver implementation run_pipeline dispatches to. Peak-based
    #: solvers ({"iterative", "closed_form"}) consume the detected+identified
    #: ``observations`` list; full-spectrum solvers ({"joint", "bayesian"})
    #: consume the raw wavelength+intensity spectrum via a JAX forward model.
    #: ``"coarse_to_fine"`` is reserved (needs a prebuilt manifold) and raises
    #: NotImplementedError for now. The solver axis is orthogonal to
    #: ``pipeline_impl`` — select it via ``config_overrides={"solver": ...}``.
    solver: str = "iterative"
    #: Solver-specific sub-config passed verbatim to the dispatched solver's
    #: factory branch (e.g. ClosedFormConfig fields, joint loss_type/n_starts,
    #: bayesian num_samples/priors). Read only inside ``_dispatch_solver`` so
    #: unknown joint/bayesian keys never trip build_pipeline_config's whitelist.
    solver_overrides: dict = field(default_factory=dict)
    # --- Iterative-solver knobs not previously sweepable (lifted onto the
    #     config so they ride config_overrides / the Optuna knob space) ---
    #: Apply ionization-potential depression in the Saha balance.
    apply_ipd: bool = False
    #: Fit a two-region (core + corona) Boltzmann plane.
    two_region: bool = False
    #: Weight Boltzmann ordinates by A_ki transition-probability uncertainty.
    aki_uncertainty_weighting: bool = True
    #: Fraction of closure mass a single element may soak before the solve is
    #: flagged as a degenerate composition.
    degeneracy_dominance_threshold: float = 0.8
    #: Minimum candidate-element count for the degeneracy guard to fire.
    degeneracy_min_elements: int = 4
    #: Errors-in-variables (orthogonal distance regression) Boltzmann / Saha-
    #: Boltzmann slope fit (Boggs & Rodgers 1990): accounts for E_k-axis
    #: uncertainty, removing the OLS regression-dilution bias on T. Default off
    #: mirrors the standard weighted-OLS fit (Track B B1; benchmark-gated).
    use_odr: bool = False
    #: Scalar 1-sigma E_k uncertainty (eV) for the ODR fit when per-line E_k
    #: uncertainties are unavailable; 0.0 degenerates ODR to weighted OLS.
    odr_x_uncertainty: float = 0.0
    #: Run the post-loop Cristoforetti reliability re-fit (perf knob). Default
    #: True preserves the M7 refuse-to-report annotation (quality_flag /
    #: saha_boltzmann_consistency / inter_element_t_std_frac / overall_reliable
    #: with real values). When False the solver skips the per-element Boltzmann
    #: re-fit + U_I/U_II re-eval (~26-34% of a solve) and emits the SAME keys with
    #: conservative unknown/NaN values; T/n_e/composition are byte-identical. The
    #: ``metallic_ded`` preset sets this False (drift-tracking on a known element
    #: set does not consume the annotation).
    assess_quality: bool = True
    #: Enable the adaptive RANSAC early-exit rule in robust wavelength
    #: calibration (prototype, default off; flag ``CFLIBS_RANSAC_EARLY_EXIT``).
    #: The sampling loop stops once the inlier-count plateaus (no improvement
    #: for ``patience`` iters) or the standard RANSAC confidence bound is met,
    #: instead of always running the full fixed iteration budget. PARITY-
    #: AFFECTING on hard low-inlier cases (a late lucky sample can be skipped),
    #: so it is benchmark-gated. Default ``False`` reproduces the legacy loop.
    #: The active runtime toggle is the env var (read inside the calibrator) so
    #: the benchmark can flip it without editing code; this field declares the
    #: default-off knob and is resolved/logged like every other knob.
    ransac_early_exit: bool = False
    #: Extra keyword overrides passed verbatim to ``detect_line_observations``
    #: (Campaign 1 knob plumbing, docs/audit/2026-06-10-goalfirst/
    #: optimization-program-design.md §3.1-B). Plain data only — keys must be
    #: ``detect_line_observations`` parameter names (e.g. ``comb_min_matches``,
    #: ``kdet_min_score``); values override the pipeline-derived kwargs.
    #: ``shift_scan_nm`` is NOT accepted here: the global scan width is the
    #: first-class ``global_shift_scan_nm`` field above. Empty in production.
    detection_overrides: dict = field(default_factory=dict)


#: Sentinel marking "knob not provided at this tier". Unlike ``None`` it lets
#: an explicit ``None`` flow through the resolution chain — several knobs use
#: ``None`` as a meaningful value (``wavelength_tolerance_nm=None`` = adaptive
#: R-derived tolerance, ``min_relative_intensity=None`` = floor disabled).
_UNSET: Any = object()

#: ``AnalysisPipelineConfig`` fields settable through the ``overrides`` tier
#: of :func:`build_pipeline_config`. ``elements`` is the run's identity, not
#: a knob.
_OVERRIDABLE_FIELDS = frozenset(AnalysisPipelineConfig.__dataclass_fields__) - {"elements"}


def _resolve(*values):
    """Return the first provided value (skipping ``_UNSET``; all-unset -> None).

    Tier encoding: dict-backed tiers (``overrides``, YAML ``analysis_cfg``)
    pass ``dict.get(key, _UNSET)`` so key *presence* decides — an explicit
    ``None`` value is honoured. The CLI-flag tier cannot express ``None``
    (argparse uses ``None`` for "flag not given"), so flags are wrapped with
    :func:`_flag` which maps ``None`` to ``_UNSET``.
    """
    for value in values:
        if value is not _UNSET:
            return value
    return None


def _flag(value):
    """CLI-flag tier adapter: argparse ``None`` means "flag not given"."""
    return _UNSET if value is None else value


def build_pipeline_config(
    elements,
    *,
    preset: Optional[str] = None,
    analysis_cfg: Optional[dict] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    saha_boltzmann_graph: Optional[bool] = None,
    closure_mode: Optional[str] = None,
    apply_self_absorption: Optional["bool | str"] = None,
    min_relative_intensity: Optional[float] = None,
    resolving_power: Optional[float] = None,
    wavelength_tolerance_nm: Optional[float] = None,
    min_peak_height: Optional[float] = None,
    peak_width_nm: Optional[float] = None,
    exclude_resonance: Optional[bool] = None,
    wavelength_calibration: Optional[bool] = None,
    shift_coherence_veto: Optional[bool] = None,
    residual_shift_scan_nm: Optional[float] = None,
    affine_coverage_gate: Optional[bool] = None,
    line_residual_gate: Optional[bool] = None,
    calib_pool_cache: Optional[bool] = None,
    response_curve: Optional[str] = None,
    stark_ne: Optional[bool] = None,
    hough_calib_seed: Optional[bool] = None,
) -> AnalysisPipelineConfig:
    """Resolve the shared pipeline configuration for analyze/invert/batch.

    Precedence per knob, highest first:

    1. ``overrides`` — explicit field overrides keyed by
       :class:`AnalysisPipelineConfig` field name (campaign/benchmark
       tooling, e.g. the goal-metric scoreboard). Key *presence* decides, so
       an explicit ``None`` (= adaptive/disabled for several knobs) is
       expressible. Unknown keys fail fast: a typo must not silently
       evaluate the production default.
    2. explicit CLI flags (the keyword arguments; ``None`` = not given),
    3. YAML ``analysis.*`` keys (``analysis_cfg``, used by ``invert``;
       key presence decides, so an explicit ``null`` is an explicit None),
    4. the resolved preset bundle (``--preset`` / ``analysis.preset``,
       default ``geological``),
    5. built-in defaults.

    Every tier — including ``overrides`` — passes through the same
    validation (closure mode, preset name) and normalization
    (``apply_self_absorption``). The resolved preset and every knob are
    logged at INFO *after* all tiers land, so each run records exactly the
    configuration that produced its numbers.
    """
    cfg = dict(analysis_cfg or {})
    ov = dict(overrides or {})
    unknown = set(ov) - _OVERRIDABLE_FIELDS
    if unknown:
        raise ValueError(
            f"AnalysisPipelineConfig has no knob(s) {sorted(unknown)}. "
            f"Overridable fields: {sorted(_OVERRIDABLE_FIELDS)}"
        )

    preset_name = _resolve(
        ov.get("preset", _UNSET), _flag(preset), cfg.get("preset", _UNSET), DEFAULT_ANALYSIS_PRESET
    )
    if preset_name not in ANALYSIS_PRESETS:
        raise ValueError(
            f"Unknown analysis preset {preset_name!r}. "
            f"Valid presets: {sorted(ANALYSIS_PRESETS)}"
        )
    preset_knobs = ANALYSIS_PRESETS[preset_name]

    def knob(key, flag_value, *fallbacks):
        """Resolve one knob: overrides > CLI flag > YAML > preset > fallbacks.

        The preset bundle is consulted as documented tier 4 for *every* knob
        (not only the handful wired with an explicit ``preset_knobs[...]``
        fallback), so a preset can set any config field. Backward-compatible:
        presets that omit a key yield ``_UNSET`` here and fall through to the
        built-in default exactly as before.
        """
        return _resolve(
            ov.get(key, _UNSET),
            _flag(flag_value),
            cfg.get(key, _UNSET),
            preset_knobs.get(key, _UNSET),
            *fallbacks,
        )

    resolved_closure_mode = knob("closure_mode", closure_mode, preset_knobs["closure_mode"])
    if resolved_closure_mode not in CLOSURE_MODES:
        raise ValueError(
            f"Unknown closure mode {resolved_closure_mode!r}. "
            f"Valid modes: {list(CLOSURE_MODES)}"
        )

    solver_name = str(knob("solver", None, "iterative"))
    if solver_name not in _VALID_SOLVERS:
        raise ValueError(
            f"Unknown solver backend {solver_name!r}; choose from {sorted(_VALID_SOLVERS)}."
        )

    pipeline = AnalysisPipelineConfig(
        preset=preset_name,
        elements=list(elements),
        min_relative_intensity=knob("min_relative_intensity", min_relative_intensity),
        top_k_per_element=knob("top_k_per_element", None, 60),
        resolving_power=knob("resolving_power", resolving_power),
        wavelength_tolerance_nm=knob("wavelength_tolerance_nm", wavelength_tolerance_nm, 0.1),
        min_peak_height=knob("min_peak_height", min_peak_height, 0.01),
        peak_width_nm=knob("peak_width_nm", peak_width_nm, 0.2),
        apply_self_absorption=normalize_self_absorption_mode(
            knob("apply_self_absorption", apply_self_absorption, False)
        ),
        exclude_resonance=knob("exclude_resonance", exclude_resonance),
        min_snr=knob("min_snr", None, 10.0),
        min_energy_spread_ev=knob("min_energy_spread_ev", None, 2.0),
        min_lines_per_element=knob("min_lines_per_element", None, 3),
        isolation_wavelength_nm=knob("isolation_wavelength_nm", None, 0.1),
        max_lines_per_element=knob("max_lines_per_element", None, 20),
        grade_aware_selection=bool(knob("grade_aware_selection", None, False)),
        target_sigma_t=knob("target_sigma_t", None, None),
        plasma_temperature_K=knob("plasma_temperature_K", None, 10000.0),
        reliability_ranked_selection=bool(knob("reliability_ranked_selection", None, False)),
        wavelength_calibration=bool(knob("wavelength_calibration", wavelength_calibration, True)),
        shift_coherence_veto=bool(knob("shift_coherence_veto", shift_coherence_veto, True)),
        residual_shift_scan_nm=float(
            knob(
                "residual_shift_scan_nm",
                residual_shift_scan_nm,
                preset_knobs["residual_shift_scan_nm"],
            )
        ),
        global_shift_scan_nm=float(knob("global_shift_scan_nm", None, 0.5)),
        affine_coverage_gate=bool(
            knob("affine_coverage_gate", affine_coverage_gate, preset_knobs["affine_coverage_gate"])
        ),
        line_residual_gate=bool(
            knob("line_residual_gate", line_residual_gate, preset_knobs["line_residual_gate"])
        ),
        calib_pool_cache=bool(knob("calib_pool_cache", calib_pool_cache, False)),
        response_curve=knob("response_curve", response_curve),
        hough_calib_seed=knob("hough_calib_seed", hough_calib_seed, None),
        max_iterations=knob("max_iterations", None, 20),
        t_tolerance_k=knob("t_tolerance_k", None, 100.0),
        ne_tolerance_frac=knob("ne_tolerance_frac", None, 0.1),
        pressure_pa=_resolve(
            ov.get("pressure_pa", _UNSET),
            cfg.get("pressure_pa", None) or cfg.get("pressure", 101325.0),
        ),
        boltzmann_weight_cap=knob("boltzmann_weight_cap", None, 5.0),
        min_boltzmann_r2=knob("min_boltzmann_r2", None, 0.3),
        saha_boltzmann_graph=bool(
            knob("saha_boltzmann_graph", saha_boltzmann_graph, preset_knobs["saha_boltzmann_graph"])
        ),
        closure_mode=resolved_closure_mode,
        closure_kwargs=dict(knob("closure_kwargs", None, {})),
        matrix_element=knob("matrix_element", None),
        oxide_elements=knob("oxide_elements", None),
        stark_ne=bool(knob("stark_ne", stark_ne, preset_knobs["stark_ne"])),
        solver=solver_name,
        solver_overrides=dict(ov.get("solver_overrides", None) or {}),
        apply_ipd=bool(knob("apply_ipd", None, False)),
        two_region=bool(knob("two_region", None, False)),
        aki_uncertainty_weighting=bool(knob("aki_uncertainty_weighting", None, True)),
        degeneracy_dominance_threshold=float(knob("degeneracy_dominance_threshold", None, 0.8)),
        degeneracy_min_elements=int(knob("degeneracy_min_elements", None, 4)),
        use_odr=bool(knob("use_odr", None, False)),
        odr_x_uncertainty=float(knob("odr_x_uncertainty", None, 0.0)),
        assess_quality=bool(knob("assess_quality", None, True)),
        ransac_early_exit=bool(knob("ransac_early_exit", None, True)),
        detection_overrides=dict(ov.get("detection_overrides", None) or {}),
    )
    _log_pipeline_config(pipeline)
    return pipeline


def _log_pipeline_config(pipeline: AnalysisPipelineConfig) -> None:
    """Log the resolved preset and every pipeline knob at INFO."""
    logger.info(
        "Resolved analysis preset '%s': saha_boltzmann_graph=%s, closure_mode=%s, "
        "stark_ne=%s, "
        "apply_self_absorption=%s, exclude_resonance=%s, min_relative_intensity=%s, "
        "top_k_per_element=%s, resolving_power=%s, wavelength_calibration=%s, "
        "shift_coherence_veto=%s, residual_shift_scan_nm=%s, global_shift_scan_nm=%s, "
        "affine_coverage_gate=%s, line_residual_gate=%s, "
        "wavelength_tolerance_nm=%s, min_peak_height=%s, peak_width_nm=%s, "
        "min_snr=%s, min_energy_spread_ev=%s, min_lines_per_element=%s, "
        "max_lines_per_element=%s, isolation_wavelength_nm=%s, max_iterations=%s, "
        "t_tolerance_k=%s, ne_tolerance_frac=%s, pressure_pa=%s, "
        "boltzmann_weight_cap=%s, min_boltzmann_r2=%s, response_curve=%s, elements=%s, "
        "detection_overrides=%s",
        pipeline.preset,
        pipeline.saha_boltzmann_graph,
        pipeline.closure_mode,
        pipeline.stark_ne,
        pipeline.apply_self_absorption,
        pipeline.exclude_resonance,
        pipeline.min_relative_intensity,
        pipeline.top_k_per_element,
        pipeline.resolving_power,
        pipeline.wavelength_calibration,
        pipeline.shift_coherence_veto,
        pipeline.residual_shift_scan_nm,
        pipeline.global_shift_scan_nm,
        pipeline.affine_coverage_gate,
        pipeline.line_residual_gate,
        pipeline.wavelength_tolerance_nm,
        pipeline.min_peak_height,
        pipeline.peak_width_nm,
        pipeline.min_snr,
        pipeline.min_energy_spread_ev,
        pipeline.min_lines_per_element,
        pipeline.max_lines_per_element,
        pipeline.isolation_wavelength_nm,
        pipeline.max_iterations,
        pipeline.t_tolerance_k,
        pipeline.ne_tolerance_frac,
        pipeline.pressure_pa,
        pipeline.boltzmann_weight_cap,
        pipeline.min_boltzmann_r2,
        pipeline.response_curve,
        pipeline.elements,
        pipeline.detection_overrides,
    )


def detect_and_select_lines(
    wavelength,
    intensity,
    atomic_db,
    elements,
    *,
    min_relative_intensity: float | None = None,
    top_k_per_element: int | None = 60,
    resolving_power: float | None = None,
    wavelength_tolerance_nm: float = 0.1,
    min_peak_height: float = 0.01,
    peak_width_nm: float = 0.2,
    #: Self-absorption mode: ``'off'`` or ``'observable'`` (bead 0jvr).
    #: Booleans are accepted on input and normalized (True -> 'observable').
    apply_self_absorption: str = "off",
    exclude_resonance: bool | None = None,
    min_snr: float = 10.0,
    min_energy_spread_ev: float = 2.0,
    min_lines_per_element: int = 3,
    isolation_wavelength_nm: float = 0.1,
    max_lines_per_element: int = 20,
    wavelength_calibration: bool = True,
    shift_coherence_veto: bool = True,
    residual_shift_scan_nm: float = 0.0,
    global_shift_scan_nm: float = 0.5,
    affine_coverage_gate: bool = True,
    line_residual_gate: bool = True,
    calib_pool_cache: bool = False,
    hough_calib_seed: Optional[bool] = None,
    ransac_early_exit: Optional[bool] = None,
    grade_aware_selection: bool = False,
    target_sigma_t: Optional[float] = None,
    plasma_temperature_K: float = 10000.0,
    reliability_ranked_selection: bool = False,
    detection_overrides: Optional[dict] = None,
    return_diagnostics: bool = False,
):
    """
    Detect spectral lines and apply the line-selection quality gate.

    This is the single shared detection+selection path for both the
    ``invert`` and ``analyze`` CLI entry points. Keeping it in one helper
    prevents the two paths from drifting: a prior drift left ``analyze`` with
    no relative-intensity floor and a bare ``LineSelector()``, which admitted
    spurious weak Na Rydberg lines and produced a catastrophic Na-dominated
    composition (RMSE 33.69 wt%, Na ~98 % on BHVO-2). Both paths now resolve
    to identical detection-floor + selector behaviour for the same inputs.

    The defaults here are the *good* defaults for the ``invert``/``analyze``
    paths: no absolute relative-intensity floor (replaced by element-relative
    top-K + a shift-coherence veto in ``detect_line_observations``), the tuned
    ``LineSelector`` gates, self-absorption off. ``exclude_resonance`` now
    defaults (``None``) to ``False`` — resonance lines are the brightest,
    most persistent LIBS lines and the only detectable lines for some majors
    (Al I 394.4/396.2 nm), so they are kept; the gA-Boltzmann comb strength and
    coherence veto guard against the spurious weak-line matches that resonance
    exclusion + the rel_int floor used to protect against. With
    self-absorption on the solver *corrects* resonance lines (Aragón &
    Aguilera 2008 §7).

    Parameters
    ----------
    wavelength, intensity : np.ndarray
        Spectrum axes (nm, intensity).
    atomic_db : AtomicDatabase
        Atomic database instance.
    elements : list[str]
        Elements to match against.
    min_relative_intensity : float or None
        Absolute relative-intensity floor for database lines. Default ``None``
        (disabled): the floor deletes whole real elements whose tabulated
        rel_int is small or NULL, so detection now bounds richness with
        ``top_k_per_element`` instead. Pass a number to re-enable the legacy
        floor.
    top_k_per_element : int or None
        Keep only each element's ``K`` strongest in-band lines (by gA-Boltzmann
        strength) instead of an absolute rel_int floor (default 60). ``None``
        keeps all in-band lines.
    resolving_power : float or None
        Instrument resolving power; enables adaptive tolerance/width when set.
    apply_self_absorption : bool
        Whether the downstream solver applies the curve-of-growth correction.
        Controls the default ``exclude_resonance`` tie.
    exclude_resonance : bool or None
        Override the resonance-exclusion policy. ``None`` ties it to
        ``not apply_self_absorption``.
    wavelength_calibration : bool
        If True (default), run robust per-spectrum/per-segment wavelength
        calibration (:func:`calibrate_wavelength_axis_segmented`) before
        detection. This replaces the single-constant global comb shift-scan's
        job for instruments with a per-spectrum or piecewise dispersion error;
        the global scan is reduced to a small residual mop-up. When calibration
        cannot be confidently estimated, it is skipped and the full global
        shift-scan is used (legacy behaviour preserved).
    shift_coherence_veto : bool
        If True (default), reject accepted elements whose matched lines do not
        agree on the single instrument residual shift (see
        ``detect_line_observations``). Exposed as a pipeline knob (bead vj82)
        so benchmark harnesses can ablate it explicitly instead of forking the
        pipeline wiring.
    residual_shift_scan_nm : float
        Half-width (nm) of the residual comb shift-scan applied *after* a
        quality-passed wavelength calibration. Default ``0.0`` (no residual
        scan, bead ye6t): the fitted axis is trusted as-is. The legacy 0.05 nm
        "mop-up" measurably rode its window edge in every configuration tested
        (its objective maximizes match count), displacing every element's
        residual cluster and admitting contaminated matches up to
        ``tolerance + 0.05`` away (Al I 892.356: +10.4 wt% Al on BHVO-2 loc1).
        When calibration is skipped or fails its quality gate the full
        ``global_shift_scan_nm`` scan still runs — this knob only controls
        the post-calibration scan.
    global_shift_scan_nm : float
        Half-width (nm) of the legacy single-constant global comb shift-scan
        used whenever calibration is disabled, skipped, or fails its quality
        gate (default 0.5). A quality-passed calibration replaces it with
        ``residual_shift_scan_nm``.
    affine_coverage_gate : bool
        Degrade per-segment affine calibration fits to a pure shift when the
        inlier anchors do not cover the segment (see
        :func:`calibrate_wavelength_axis_segmented`). Default True (bead ye6t).
    line_residual_gate : bool
        Drop individual matched lines whose residual is incoherent with the
        consensus residual shift (see ``detect_line_observations``). Default
        True (bead ye6t).
    calib_pool_cache : bool
        Build the wavelength-calibration reference line pool once over the full
        axis and slice it per segment (flag ``CFLIBS_CALIB_POOL_CACHE``; see
        :func:`calibrate_wavelength_axis_segmented`). Parity-exact; default
        False. Also honoured via the ``CFLIBS_CALIB_POOL_CACHE`` env var.
    hough_calib_seed : bool or None
        Deterministic Hough coarse-dispersion warm start for the RANSAC
        calibrator (flag ``CFLIBS_HOUGH_CALIB``; RASCAL-style). ``None`` (default)
        defers to the env var (off when unset). NOT parity-exact ->
        benchmark-gated.
    ransac_early_exit : bool or None
        Adaptive RANSAC early-exit on inlier-count plateau / confidence bound
        (flag ``CFLIBS_RANSAC_EARLY_EXIT``). ``None`` (default) defers to the env
        var (off when unset). Parity-affecting -> benchmark-gated.
    detection_overrides : dict or None
        Extra keyword overrides forwarded verbatim to
        ``detect_line_observations`` (Campaign 1 knob plumbing; see
        :class:`AnalysisPipelineConfig.detection_overrides`). ``shift_scan_nm``
        is rejected here — the global scan width is the first-class
        ``global_shift_scan_nm`` parameter (a verbatim forward would clobber
        the post-calibration residual scan as well, which the old magic key
        never did).
    return_diagnostics : bool
        When True, also return a diagnostics dict recording which requested
        elements were dropped and at which stage (``detection`` = no matched
        lines; ``selection`` = matched lines failed the quality gate). This
        feeds the CLI trust report: requested-but-dropped elements used to
        vanish silently while the run still printed ``converged=True``.

    Returns
    -------
    list or (list, dict)
        The selected ``LineObservation`` list (``selection.selected_lines``);
        with ``return_diagnostics=True``, a ``(selected_lines, diagnostics)``
        tuple.
    """
    # Imports are kept function-local to preserve this module's light
    # import-time contract: the cross-package symbols below are lazy to avoid
    # circular imports (see module docstring), and ``time``/``numpy`` are kept
    # alongside them so that merely importing this module does not pull numpy.
    import time

    import numpy as np

    from cflibs.inversion.identify.line_detection import detect_line_observations
    from cflibs.inversion.physics.line_selection import LineSelector
    from cflibs.inversion.preprocess.wavelength_calibration import (
        calibrate_wavelength_axis_segmented,
    )

    calibration_s = 0.0

    # Robust per-spectrum wavelength calibration BEFORE detection. Real LIBS
    # instruments carry a per-spectrum (and, for stitched multi-channel
    # spectrometers, per-channel) dispersion error that the single global comb
    # shift-scan inside ``detect_line_observations`` cannot represent: it can
    # only apply one constant offset, so on a piecewise-dispersion instrument it
    # compromises and leaves real majors mis-aligned beyond tolerance (measured
    # on BHVO-2: ChemCam Al 396.152 misses by 0.109 nm; CSA abandons Si vs Mg).
    # ``calibrate_wavelength_axis_segmented`` detects detector seams and fits each
    # channel independently (degrading to a single global affine when there are
    # no seams), aligning the axis robustly. After calibration only a small
    # residual ``shift_scan_nm`` is needed to mop up sub-tolerance jitter.
    overrides = dict(detection_overrides or {})
    if "shift_scan_nm" in overrides:
        raise ValueError(
            "detection_overrides['shift_scan_nm'] is no longer accepted: the global "
            "comb shift-scan width is the first-class 'global_shift_scan_nm' pipeline "
            "knob (AnalysisPipelineConfig.global_shift_scan_nm)."
        )
    # Legacy global scan width (used when calibration is disabled or fails its
    # quality gate); a quality-passed calibration replaces it with the
    # residual_shift_scan_nm mop-up below.
    shift_scan_nm = float(global_shift_scan_nm)
    if wavelength_calibration:
        _cal_t0 = time.perf_counter()
        try:
            cal = calibrate_wavelength_axis_segmented(
                wavelength=np.asarray(wavelength, dtype=float),
                intensity=np.asarray(intensity, dtype=float),
                atomic_db=atomic_db,
                elements=elements,
                affine_coverage_gate=affine_coverage_gate,
                calib_pool_cache=calib_pool_cache,
                hough_calib_seed=hough_calib_seed,
                ransac_early_exit=ransac_early_exit,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Wavelength calibration failed ({exc!r}); using raw axis.")
            cal = None
        if cal is not None and cal.success and cal.quality_passed:
            wavelength = cal.corrected_wavelength
            shift_scan_nm = residual_shift_scan_nm
            logger.info(
                "Applied wavelength calibration: model=%s segments=%s "
                "n_inliers=%d rmse=%.4f nm (residual shift-scan +/-%.3f nm)",
                cal.model,
                cal.details.get("segments", 1),
                cal.n_inliers,
                cal.rmse_nm,
                residual_shift_scan_nm,
            )
        else:
            reason = cal.quality_reason if cal is not None else "exception"
            logger.info(
                "Wavelength calibration not applied (%s); "
                "falling back to global comb shift-scan.",
                reason,
            )
        calibration_s = time.perf_counter() - _cal_t0

    # Sampling-resolution cap (bead CF-LIBS-improved-qitd): tighten the matching
    # tolerance / integration width toward the instrument's actual sampling step
    # so a high-resolution, finely-sampled spectrometer is not matched with a
    # tolerance many sampling steps wide (which degenerates to random coincidence
    # on the densely-peaked axis and crashes detection with "No usable spectral
    # lines"). Tighten-only via ``min``: coarsely-sampled instruments sit below
    # the cap and are unchanged. See SAMPLING_TOLERANCE_PX / SAMPLING_WIDTH_PX.
    wl_arr = np.asarray(wavelength, dtype=float)
    if wl_arr.size >= 2:
        diffs = np.diff(wl_arr)
        diffs = diffs[np.isfinite(diffs)]
        wl_step = float(np.median(diffs)) if diffs.size else 0.0
        if wl_step > 0:
            if wavelength_tolerance_nm is not None:
                wavelength_tolerance_nm = min(
                    wavelength_tolerance_nm, SAMPLING_TOLERANCE_PX * wl_step
                )
            if peak_width_nm is not None:
                peak_width_nm = min(peak_width_nm, SAMPLING_WIDTH_PX * wl_step)

    detect_kwargs = dict(
        wavelength=wavelength,
        intensity=intensity,
        atomic_db=atomic_db,
        elements=elements,
        wavelength_tolerance_nm=wavelength_tolerance_nm,
        resolving_power=resolving_power,
        min_peak_height=min_peak_height,
        peak_width_nm=peak_width_nm,
        min_relative_intensity=min_relative_intensity,
        top_k_per_element=top_k_per_element,
        shift_scan_nm=shift_scan_nm,
        shift_coherence_veto=shift_coherence_veto,
        line_residual_gate=line_residual_gate,
        residual_gate_min_kept_lines=min_lines_per_element,
    )
    # Campaign-1 knob plumbing: remaining overrides win over the pipeline-
    # derived kwargs (an unknown key fails fast as TypeError below).
    detect_kwargs.update(overrides)
    detection = detect_line_observations(**detect_kwargs)

    for warning in detection.warnings:
        logger.warning(f"Line detection warning: {warning}")

    # Resonance lines are the brightest, most persistent lines in a LIBS
    # plasma (low E_k, large gA) and are the *only* detectable lines for some
    # majors (e.g. the Al I 394.4/396.2 nm doublet). Keep them by default so
    # those elements reach the solver. The legacy tie to
    # ``not apply_self_absorption`` dropped them whenever self-absorption
    # correction was off, deleting whole elements at the selection gate; the
    # gA-Boltzmann comb strength + shift-coherence veto now guard against the
    # spurious weak-line matches that exclusion was protecting against.
    if exclude_resonance is None:
        exclude_resonance = False

    selector = LineSelector(
        min_snr=min_snr,
        min_energy_spread_ev=min_energy_spread_ev,
        min_lines_per_element=min_lines_per_element,
        exclude_resonance=exclude_resonance,
        isolation_wavelength_nm=isolation_wavelength_nm,
        max_lines_per_element=max_lines_per_element,
        target_sigma_t=target_sigma_t,
        plasma_temperature_K=plasma_temperature_K,
        reliability_ranked_selection=reliability_ranked_selection,
    )

    # Lever 1B (grade-aware selection, gated default-off): feed grade-derived A_ki
    # uncertainties into the selector so the per-element top-N prefers high-grade
    # (A/B) lines over D/U. Without this the selector defaults every line to 0.10
    # (grade-blind), so a completeness DB's many D/U lines pollute the analytical
    # set. Unknown grade (aki_uncertainty None) -> worst (1.0), NOT the optimistic
    # 0.10 default, so 'U' lines are downweighted rather than treated as accurate.
    atomic_uncertainties: Optional[dict] = None
    if grade_aware_selection:
        atomic_uncertainties = {}
        for o in detection.observations:
            unc = o.aki_uncertainty
            if unc is None or not math.isfinite(unc) or unc <= 0.0:
                unc = 1.0
            atomic_uncertainties[(o.element, o.ionization_stage, o.wavelength_nm)] = float(unc)

    selection = selector.select(
        detection.observations,
        resonance_lines=detection.resonance_lines,
        atomic_uncertainties=atomic_uncertainties,
    )
    if not return_diagnostics:
        return selection.selected_lines

    detected_elements = {o.element for o in detection.observations}
    selected_elements = {o.element for o in selection.selected_lines}
    dropped: dict = {}
    for el in elements:
        if el not in detected_elements:
            dropped[el] = "detection"
        elif el not in selected_elements:
            dropped[el] = "selection"
    diagnostics = {
        "requested_elements": list(elements),
        "detected_elements": sorted(detected_elements),
        "selected_elements": sorted(selected_elements),
        "dropped_elements": dropped,
        # Per-line residual-gate diagnostics (bead ye6t): consensus residual,
        # band, and the contaminated matches dropped at observation build.
        "residual_gate": detection.residual_gate,
        "applied_shift_nm": detection.applied_shift_nm,
        # Wall-clock spent in the wavelength-calibration stage (bead A1
        # scoreboard); ``run_pipeline`` folds this into ``stage_timings``.
        "calibration_s": calibration_s,
    }
    return selection.selected_lines, diagnostics


def _finalize_closure_kwargs(pipeline: AnalysisPipelineConfig, observations) -> dict:
    """Build the closure kwargs for a solve, applying matrix/oxide defaults."""
    closure_kwargs = dict(pipeline.closure_kwargs)
    if pipeline.closure_mode == "matrix" and pipeline.matrix_element is not None:
        closure_kwargs.setdefault("matrix_element", pipeline.matrix_element)
    if pipeline.closure_mode == "oxide":
        if pipeline.oxide_elements is not None:
            closure_kwargs.setdefault("oxide_elements", pipeline.oxide_elements)
        # Default geological molar-oxygen stoichiometry (O atoms per cation) when
        # the config does not supply an explicit oxide_stoichiometry map.
        if "oxide_stoichiometry" not in closure_kwargs:
            from cflibs.inversion.physics.closure import default_oxide_stoichiometry

            els = [o.element for o in observations]
            closure_kwargs["oxide_stoichiometry"] = default_oxide_stoichiometry(els)
    return closure_kwargs


def _solve_analyze_result(
    solver,
    observations,
    closure_mode: str,
    closure_kwargs: dict,
    uncertainty_mode: str,
    stark_diagnostics=None,
):
    """Run the solver for ``analyze`` honouring the requested uncertainty mode."""
    if uncertainty_mode == "analytical":
        try:
            return solver.solve_with_uncertainty(
                observations,
                closure_mode=closure_mode,
                stark_diagnostics=stark_diagnostics,
                **closure_kwargs,
            )
        except ImportError:
            logger.warning(
                "uncertainties package not installed; falling back to solve() without UQ. "
                "Install with: pip install uncertainties"
            )
            return solver.solve(
                observations,
                closure_mode=closure_mode,
                stark_diagnostics=stark_diagnostics,
                **closure_kwargs,
            )
    elif uncertainty_mode == "mc":
        from cflibs.inversion.physics.uncertainty import MonteCarloUQ

        mc = MonteCarloUQ(solver, n_samples=200)
        mc_result = mc.run(observations)
        result = solver.solve(
            observations,
            closure_mode=closure_mode,
            stark_diagnostics=stark_diagnostics,
            **closure_kwargs,
        )
        return result.__class__(
            temperature_K=mc_result.T_mean,
            temperature_uncertainty_K=mc_result.T_std,
            electron_density_cm3=mc_result.ne_mean,
            concentrations=mc_result.concentrations_mean,
            concentration_uncertainties=mc_result.concentrations_std,
            iterations=result.iterations,
            converged=result.converged,
            quality_metrics=result.quality_metrics,
            overall_reliable=getattr(result, "overall_reliable", False),
        )
    else:
        return solver.solve(
            observations,
            closure_mode=closure_mode,
            stark_diagnostics=stark_diagnostics,
            **closure_kwargs,
        )


# ---------------------------------------------------------------------------
# Unified solver dispatch (the single solver-selection fork)
# ---------------------------------------------------------------------------

#: Closure modes the closed-form solver accepts (it constructs the closure at
#: __init__ time, not as a solve() kwarg; ``matrix`` needs a global element
#: absent per-spectrum, and the ILR/PWLR/dirichlet variants are not part of
#: its closed-form regression). The factory clamps anything else to standard.
_CLOSED_FORM_CLOSURES = ("standard", "oxide")


def _number_to_mass_fractions(concentrations: dict) -> dict:
    """Convert a number/atom-fraction simplex to mass fractions.

    The Saha-Boltzmann forward model (and the Dirichlet MCMC prior / softmax
    joint closure that feed it) treat ``concentrations`` as **number/mole
    fractions**. The benchmark scoreboard scores ``result.concentrations``
    directly as mass fractions (``predicted_wt = 100 * c``), so a full-spectrum
    solver's number-fraction output MUST be converted here or every wt% RMSE is
    silently mis-scored.

    ``C_mass_i = C_i * AW_i / sum_j (C_j * AW_j)`` with atomic weights from
    :data:`cflibs.atomic.masses.STANDARD_ATOMIC_MASSES`.
    """
    from cflibs.atomic.masses import STANDARD_ATOMIC_MASSES

    weighted = {
        el: max(float(c), 0.0) * STANDARD_ATOMIC_MASSES.get(el, 50.0)
        for el, c in concentrations.items()
    }
    total = sum(weighted.values())
    if total <= 0.0:
        return {el: 0.0 for el in concentrations}
    return {el: w / total for el, w in weighted.items()}


#: Physically meaningful floors for a LIBS plasma solution. Unbounded BFGS (the
#: only method jax.scipy.optimize.minimize implements) can drive T/n_e to a
#: tiny-but-positive value (~1e-6 K, ~1e-38 cm^-3) that is non-physical yet
#: passes a bare ``> 0`` test. A LIBS plasma is several thousand K with
#: n_e well above 1e10 cm^-3.
_T_FLOOR_K = 1000.0
_NE_FLOOR_CM3 = 1e10


def _physical_solution(temperature_K, electron_density_cm3) -> bool:
    """True when T and n_e are finite and above the LIBS-plasma floors.

    A collapsed optimizer result (T ~ 0, n_e ~ 0, or NaN) must be scored as a
    failure rather than a misleadingly finite composition RMSE.
    """
    import math

    try:
        T = float(temperature_K)
        ne = float(electron_density_cm3)
    except (TypeError, ValueError):
        return False
    return math.isfinite(T) and math.isfinite(ne) and T >= _T_FLOOR_K and ne >= _NE_FLOOR_CM3


def _to_cflibs_result(
    *,
    concentrations_mass: dict,
    temperature_K: float,
    electron_density_cm3: float,
    converged: bool,
    failed: bool,
    iterations: int = 0,
    extra_quality: Optional[dict] = None,
):
    """Adapt a full-spectrum solver result to the :class:`CFLIBSResult` contract.

    ``_score_spectrum`` reads a fixed surface off the returned object:
    ``concentrations`` (scored as mass fractions), ``temperature_K``,
    ``electron_density_cm3``, ``converged``, and ``quality_metrics`` (whose
    ``failed`` key gates the all-FN failure branch). We synthesize exactly that
    surface so the joint/bayesian branches plug into the same scoring path as
    the peak-based solvers.
    """
    from cflibs.inversion.solve.iterative import CFLIBSResult

    quality_metrics = {"failed": 1.0 if failed else 0.0}
    if extra_quality:
        quality_metrics.update(extra_quality)
    return CFLIBSResult(
        temperature_K=float(temperature_K),
        temperature_uncertainty_K=0.0,
        electron_density_cm3=float(electron_density_cm3),
        concentrations=dict(concentrations_mass),
        concentration_uncertainties={el: 0.0 for el in concentrations_mass},
        iterations=int(iterations),
        converged=bool(converged),
        quality_metrics=quality_metrics,
    )


def _failed_full_spectrum_result(elements, reason: str):
    """All-FN result mirroring the scoreboard's failure-policy parity branch."""
    logger.warning("Full-spectrum solver produced no usable composition: %s", reason)
    return _to_cflibs_result(
        concentrations_mass={el: 0.0 for el in elements},
        temperature_K=0.0,
        electron_density_cm3=0.0,
        converged=False,
        failed=True,
    )


def _run_peak_based_solver(
    pipeline: AnalysisPipelineConfig,
    observations,
    atomic_db,
    stark_diagnostics,
    uncertainty_mode: str,
):
    """Iterative / closed-form solvers: consume the identified observations."""
    if pipeline.solver == "iterative":
        from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver

        solver = IterativeCFLIBSSolver(
            atomic_db=atomic_db,
            max_iterations=pipeline.max_iterations,
            t_tolerance_k=pipeline.t_tolerance_k,
            ne_tolerance_frac=pipeline.ne_tolerance_frac,
            pressure_pa=pipeline.pressure_pa,
            apply_self_absorption=pipeline.apply_self_absorption,
            min_boltzmann_r2=pipeline.min_boltzmann_r2,
            boltzmann_weight_cap=pipeline.boltzmann_weight_cap,
            saha_boltzmann_graph=pipeline.saha_boltzmann_graph,
            use_odr=pipeline.use_odr,
            odr_x_uncertainty=pipeline.odr_x_uncertainty,
            apply_ipd=pipeline.apply_ipd,
            two_region=pipeline.two_region,
            aki_uncertainty_weighting=pipeline.aki_uncertainty_weighting,
            degeneracy_dominance_threshold=pipeline.degeneracy_dominance_threshold,
            degeneracy_min_elements=pipeline.degeneracy_min_elements,
            assess_quality=pipeline.assess_quality,
        )
        closure_kwargs = _finalize_closure_kwargs(pipeline, observations)
        return _solve_analyze_result(
            solver,
            observations,
            pipeline.closure_mode,
            closure_kwargs,
            uncertainty_mode,
            stark_diagnostics=stark_diagnostics,
        )

    # closed_form: ClosedFormILRSolver builds its closure at construction time
    # and ``solve()`` takes NO closure_mode/stark_diagnostics/**closure_kwargs,
    # so we call it directly (bypassing _solve_analyze_result).
    from cflibs.inversion.solve.closed_form import ClosedFormConfig, ClosedFormILRSolver

    ov = dict(pipeline.solver_overrides)
    cf_closure = ov.get("cf_closure_mode", pipeline.closure_mode)
    if cf_closure not in _CLOSED_FORM_CLOSURES:
        cf_closure = "standard"

    config_kwargs: dict = {
        "closure_mode": cf_closure,
        "saha_passes": int(ov.get("saha_passes", 2)),
        "partition_refine": bool(ov.get("partition_refine", True)),
        "ne_mode": str(ov.get("ne_mode", "pressure")),
        "pressure_pa": float(ov.get("cf_pressure_pa", pipeline.pressure_pa)),
        "apply_ipd": bool(ov.get("cf_apply_ipd", pipeline.apply_ipd)),
    }
    if cf_closure == "oxide":
        oxide_stoich = _finalize_closure_kwargs(pipeline, observations).get("oxide_stoichiometry")
        if oxide_stoich is not None:
            config_kwargs["oxide_stoichiometry"] = oxide_stoich

    solver = ClosedFormILRSolver(atomic_db, config=ClosedFormConfig(**config_kwargs))
    return solver.solve(
        observations,
        initial_T_K=float(ov.get("initial_T_K", 10000.0)),
        initial_ne_cm3=float(ov.get("initial_ne_cm3", 1e17)),
    )


def _run_full_spectrum_solver(
    wavelength,
    intensity,
    atomic_db,
    pipeline: AnalysisPipelineConfig,
    *,
    warm_start,
    diagnostics: dict,
):
    """Run the SVD-conditioned, chunked full-spectrum fit on the iterative warm start.

    The iterative ``warm_start`` (a :class:`CFLIBSResult`) supplies the initial
    ``T``, ``n_e`` and composition.  The full-spectrum solver
    (:func:`cflibs.inversion.solve.full_spectrum.solve_full_spectrum`) fits the
    measured spectrum in a low-dimensional SVD basis using the
    memory-efficient chunked forward kernel.  Both the warm-start and the
    converged-fit accounting are written into ``diagnostics['full_spectrum']``
    so callers can honestly report whether a *real* converged fit was reached
    and whether it improved on the iterative warm start.

    Returns a :class:`CFLIBSResult`.  When the full-spectrum optimiser does not
    produce a real converged optimum, the warm start is returned UNCHANGED (the
    honest "fell back" outcome).
    """
    from cflibs.inversion.solve.full_spectrum import solve_full_spectrum

    warm_concentrations = {
        el: float(c) for el, c in warm_start.concentrations.items() if float(c) > 0.0
    }
    fit_elements = sorted(warm_concentrations) or list(pipeline.elements)

    try:
        fs = solve_full_spectrum(
            wavelength,
            intensity,
            fit_elements,
            str(atomic_db.db_path),
            warm_start_T_K=float(warm_start.temperature_K),
            warm_start_ne_cm3=float(warm_start.electron_density_cm3),
            warm_start_concentrations=warm_concentrations,
            resolving_power=pipeline.resolving_power,
            method=pipeline.solver,
        )
    except Exception as exc:  # noqa: BLE001 — never crash the pipeline on the fit
        logger.warning(
            "Full-spectrum solver (%s) raised %r; keeping the iterative warm start.",
            pipeline.solver,
            exc,
        )
        diagnostics["full_spectrum"] = {
            "solver": pipeline.solver,
            "error": f"{type(exc).__name__}: {exc}",
            "converged_fit": False,
            "adopted_fit": False,
        }
        return warm_start

    diagnostics["full_spectrum"] = {
        "solver": pipeline.solver,
        "converged_fit": bool(fs.converged),
        "adopted_fit": bool(fs.adopted_fit),
        "initial_objective": fs.initial_objective,
        "final_objective": fs.final_objective,
        "iterations": fs.iterations,
        "gradient_norm": fs.gradient_norm,
        "warm_start_T_K": fs.warm_start_temperature_K,
        "warm_start_ne_cm3": fs.warm_start_electron_density_cm3,
        "fit_T_K": fs.fit_temperature_K,
        "fit_ne_cm3": fs.fit_electron_density_cm3,
        "warm_start_concentrations": dict(fs.warm_start_concentrations),
        "fit_concentrations": dict(fs.fit_concentrations),
        **{f"diag_{k}": v for k, v in fs.diagnostics.items()},
    }
    logger.info(
        "Full-spectrum %s: converged_fit=%s adopted=%s obj %.4g -> %.4g "
        "(%d iters, |grad|=%.2g), T %.0f->%.0f K",
        pipeline.solver,
        fs.converged,
        fs.adopted_fit,
        fs.initial_objective,
        fs.final_objective,
        fs.iterations,
        fs.gradient_norm,
        fs.warm_start_temperature_K,
        fs.fit_temperature_K,
    )

    if not fs.adopted_fit:
        # Honest fall-back: the optimiser did not beat the warm start. Return
        # the iterative result unchanged so the reported composition is the
        # one that was actually solved for.
        return warm_start

    # Adopt the converged fit. Re-wrap into the CFLIBSResult contract so the
    # rest of the pipeline (scoreboard, CLI trust report) is unchanged.
    quality_metrics = dict(getattr(warm_start, "quality_metrics", {}) or {})
    quality_metrics["full_spectrum_converged"] = 1.0
    quality_metrics["full_spectrum_adopted"] = 1.0
    return warm_start.__class__(
        temperature_K=float(fs.temperature_K),
        temperature_uncertainty_K=float(getattr(warm_start, "temperature_uncertainty_K", 0.0)),
        electron_density_cm3=float(fs.electron_density_cm3),
        concentrations=dict(fs.concentrations),
        concentration_uncertainties=dict(
            getattr(warm_start, "concentration_uncertainties", {}) or {}
        ),
        iterations=int(fs.iterations),
        converged=bool(fs.converged),
        quality_metrics=quality_metrics,
        overall_reliable=getattr(warm_start, "overall_reliable", False),
    )


def _dispatch_solver(
    pipeline: AnalysisPipelineConfig,
    observations,
    atomic_db,
    wavelength,
    intensity,
    stark_diagnostics,
    uncertainty_mode: str,
):
    """Select and run the configured solver, returning a :class:`CFLIBSResult`.

    The single solver-selection fork: peak-based solvers (iterative,
    closed_form) consume the already-built ``observations``; full-spectrum
    solvers (joint, bayesian) consume the raw ``wavelength``/``intensity`` via a
    JAX forward model. ``coarse_to_fine`` is reserved (needs a manifold).
    """
    solver = pipeline.solver
    if solver in ("iterative", "closed_form"):
        return _run_peak_based_solver(
            pipeline, observations, atomic_db, stark_diagnostics, uncertainty_mode
        )
    if solver in ("joint", "bayesian"):
        from dataclasses import replace as _dc_replace

        # Warm-start the full-spectrum fit from a quick iterative solve on the
        # same observations (the converged solver refines it; falls back to it).
        warm = _run_peak_based_solver(
            _dc_replace(pipeline, solver="iterative"),
            observations,
            atomic_db,
            stark_diagnostics,
            uncertainty_mode,
        )
        return _run_full_spectrum_solver(
            wavelength, intensity, atomic_db, pipeline, warm_start=warm, diagnostics={}
        )
    if solver == "coarse_to_fine":
        raise NotImplementedError(
            "coarse_to_fine solver needs a prebuilt manifold (manifold backlog); "
            "not yet wired into the unified dispatch."
        )
    raise ValueError(
        f"Unknown solver {solver!r}; choose from "
        "{iterative, closed_form, joint, bayesian, coarse_to_fine}."
    )


def run_pipeline(
    wavelength,
    intensity,
    atomic_db,
    pipeline: AnalysisPipelineConfig,
    uncertainty_mode: str = "none",
):
    """Run the shared detection -> selection -> iterative-solve pipeline.

    The single execution path behind ``analyze``, ``invert``, ``batch`` and
    the benchmark harnesses. Returns ``(result, diagnostics)`` where
    ``diagnostics`` carries the resolved preset, the requested-but-dropped
    element map for the trust report, per-element observation counts
    (``observation_counts``) for benchmark scoring, and per-stage wall-clock
    timings (``stage_timings``: calibration / detection+ID / stark n_e /
    solve, in seconds) for the goal-metric scoreboard (bead A1).
    """
    # Function-local imports preserve the module's light import-time contract:
    # Solver imports are lazy (inside ``_dispatch_solver``) to avoid a circular
    # import (see module docstring); ``time`` is kept local for stage timings.
    import time

    _t_start = time.perf_counter()

    # Spectral-response correction FIRST (audit 02-F5): divide the measured
    # spectrum by the relative detection efficiency E(lambda) before any
    # observation building, so integrated line intensities, their shot-noise
    # uncertainties and the Boltzmann ordinates are all computed from
    # response-corrected data. Identity (no-op) when no curve is configured —
    # ChemCam CCS spectra arrive response-corrected upstream and must NOT be
    # corrected twice. Path resolution (config-relative etc.) happens at the
    # CLI layer; here the path is used as given.
    if pipeline.response_curve:
        from cflibs.inversion.preprocess.response_correction import SpectralResponseCorrection

        correction = SpectralResponseCorrection.from_file(pipeline.response_curve)
        intensity = correction.apply(wavelength, intensity)
        logger.info(
            "Applied spectral-response correction from %s (coverage %.1f-%.1f nm).",
            pipeline.response_curve,
            *correction.coverage_nm,
        )

    _t_detect0 = time.perf_counter()
    observations, diagnostics = detect_and_select_lines(
        wavelength,
        intensity,
        atomic_db,
        pipeline.elements,
        min_relative_intensity=pipeline.min_relative_intensity,
        top_k_per_element=pipeline.top_k_per_element,
        resolving_power=pipeline.resolving_power,
        wavelength_tolerance_nm=pipeline.wavelength_tolerance_nm,
        min_peak_height=pipeline.min_peak_height,
        peak_width_nm=pipeline.peak_width_nm,
        apply_self_absorption=pipeline.apply_self_absorption,
        exclude_resonance=pipeline.exclude_resonance,
        min_snr=pipeline.min_snr,
        min_energy_spread_ev=pipeline.min_energy_spread_ev,
        min_lines_per_element=pipeline.min_lines_per_element,
        isolation_wavelength_nm=pipeline.isolation_wavelength_nm,
        max_lines_per_element=pipeline.max_lines_per_element,
        wavelength_calibration=pipeline.wavelength_calibration,
        shift_coherence_veto=pipeline.shift_coherence_veto,
        residual_shift_scan_nm=pipeline.residual_shift_scan_nm,
        global_shift_scan_nm=pipeline.global_shift_scan_nm,
        affine_coverage_gate=pipeline.affine_coverage_gate,
        line_residual_gate=pipeline.line_residual_gate,
        calib_pool_cache=pipeline.calib_pool_cache,
        hough_calib_seed=pipeline.hough_calib_seed,
        # ``False`` (the default) -> ``None`` so the ``CFLIBS_RANSAC_EARLY_EXIT``
        # env var is still consulted inside the calibrator (config True forces on;
        # config False defers to the env var, matching the prototype's behaviour).
        ransac_early_exit=(pipeline.ransac_early_exit or None),
        grade_aware_selection=pipeline.grade_aware_selection,
        target_sigma_t=pipeline.target_sigma_t,
        plasma_temperature_K=pipeline.plasma_temperature_K,
        reliability_ranked_selection=pipeline.reliability_ranked_selection,
        detection_overrides=pipeline.detection_overrides,
        return_diagnostics=True,
    )
    _detect_s = time.perf_counter() - _t_detect0
    diagnostics["preset"] = pipeline.preset
    diagnostics["saha_boltzmann_graph"] = pipeline.saha_boltzmann_graph
    diagnostics["closure_mode"] = pipeline.closure_mode
    diagnostics["response_curve"] = pipeline.response_curve

    # Per-element observation counts (which elements survived detection +
    # selection, and with how many lines). Benchmark harnesses score these;
    # the CLI trust report ignores them.
    observation_counts: dict = {}
    for obs in observations:
        element = getattr(obs, "element", None)
        if element is not None:
            observation_counts[element] = observation_counts.get(element, 0) + 1
    diagnostics["observation_counts"] = observation_counts
    diagnostics["n_observations"] = len(observations)

    if len(observations) == 0:
        raise ValueError("No usable spectral lines detected for inversion.")

    # Stark-broadening n_e diagnostic (bead pxex; audit 02-F2): measure n_e
    # from the widths of observed literature-grade lines so it enters the Saha
    # terms as a measurement, not the 1-atm pressure-balance assumption. When
    # no line qualifies, stark_diagnostics stays None and the solver keeps the
    # existing (warned) fallback path — behaviour unchanged.
    stark_diagnostics = None
    _t_stark0 = time.perf_counter()
    if pipeline.stark_ne:
        from cflibs.inversion.physics.stark_ne import measure_stark_ne

        try:
            stark_result = measure_stark_ne(
                wavelength,
                intensity,
                observations,
                atomic_db,
                resolving_power=pipeline.resolving_power,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Stark n_e diagnostic failed ({exc!r}); using solver fallback n_e.")
            stark_result = None
        if stark_result is not None:
            diagnostics["stark_ne"] = {
                "n_lines": stark_result.n_lines,
                "ne_cm3": stark_result.ne_median_cm3,
                "ne_scatter_cm3": stark_result.ne_scatter_cm3,
                "instrument_fwhm_source": stark_result.instrument_fwhm_source,
                "lines": [
                    {
                        "element": m.element,
                        "ionization_stage": m.ionization_stage,
                        "wavelength_nm": m.wavelength_nm,
                        "lorentz_fwhm_nm": m.lorentz_fwhm_nm,
                        "ne_cm3": m.ne_cm3,
                    }
                    for m in stark_result.measurements
                ],
                "rejected": dict(stark_result.rejected),
            }
            if stark_result.usable:
                stark_diagnostics = stark_result.diagnostics
    _stark_s = time.perf_counter() - _t_stark0

    _t_solve0 = time.perf_counter()
    result = _dispatch_solver(
        pipeline,
        observations,
        atomic_db,
        wavelength,
        intensity,
        stark_diagnostics,
        uncertainty_mode,
    )
    _solve_s = time.perf_counter() - _t_solve0

    # Per-stage wall-clock timings (bead A1 scoreboard): runtime is a goal
    # metric, so the production pipeline reports where each second went.
    # ``calibration_s`` is measured inside ``detect_and_select_lines`` and is
    # a subset of the detection call; ``detection_id_s`` is the remainder
    # (peak detection, line matching, selection gates).
    _calibration_s = float(diagnostics.pop("calibration_s", 0.0))
    diagnostics["stage_timings"] = {
        "calibration_s": _calibration_s,
        "detection_id_s": max(_detect_s - _calibration_s, 0.0),
        "stark_ne_s": _stark_s,
        "solve_s": _solve_s,
        "total_s": time.perf_counter() - _t_start,
    }

    # Solver-stage drops: requested elements that survived detection and
    # selection but ended the solve with no mass attributed.
    dropped = diagnostics["dropped_elements"]
    for el in pipeline.elements:
        if el not in dropped and result.concentrations.get(el, 0.0) <= 0.0:
            dropped[el] = "solve"

    # Number->mass fractions (DED Gap 4): the peak-based solvers emit number/mole
    # fractions; the full-spectrum (joint/bayesian) path already returns mass.
    # Expose a consistent mass-fraction view so consumers compare wt% like-for-
    # like. For a metal-alloy known set there is no oxygen, so the conversion is
    # clean (unlike the geological O-excluded case).
    if not result.mass_fractions:
        if pipeline.solver in ("joint", "bayesian"):
            result.mass_fractions = dict(result.concentrations)
        else:
            result.mass_fractions = _number_to_mass_fractions(result.concentrations)
    return result, diagnostics
