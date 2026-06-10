"""Knob-space round-trip: suggestion -> config_overrides -> valid pipeline config."""

import inspect

import pytest

import knob_space


class _RandomTrial:
    """Duck-typed trial drawing uniformly (no optuna needed)."""

    def __init__(self, rng):
        self.rng = rng

    def suggest_categorical(self, name, choices):
        return choices[int(self.rng.integers(0, len(choices)))]

    def suggest_int(self, name, low, high):
        return int(self.rng.integers(low, high + 1))

    def suggest_float(self, name, low, high, log=False):
        import numpy as np

        if log:
            return float(np.exp(self.rng.uniform(np.log(low), np.log(high))))
        return float(self.rng.uniform(low, high))


def _pipeline_field_names():
    from cflibs.inversion.pipeline import AnalysisPipelineConfig

    return set(AnalysisPipelineConfig.__dataclass_fields__)


def _detection_param_names():
    from cflibs.inversion.identify.line_detection import detect_line_observations

    return set(inspect.signature(detect_line_observations).parameters)


@pytest.mark.unit
def test_knob_count_matches_design():
    # ~43 knobs incl. mode selectors; design doc says ~45 total knobs.
    assert 40 <= len(knob_space.SPACE) <= 50
    # Frozen-by-design axes must NOT be searchable.
    params = {k.param for k in knob_space.SPACE}
    assert "wavelength_calibration" not in params
    assert "presence_eps_massfrac" not in params
    assert "resolving_power" not in params


@pytest.mark.unit
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_suggestion_roundtrip_builds_valid_config(seed):
    import numpy as np

    from cflibs.inversion.pipeline import build_pipeline_config

    trial = _RandomTrial(np.random.default_rng(seed))
    params = knob_space.suggest_params(trial)
    overrides = knob_space.params_to_overrides(params)

    detection = overrides["detection_overrides"]
    pipeline_keys = set(overrides) - {"detection_overrides"}
    assert pipeline_keys <= _pipeline_field_names()
    assert set(detection) <= _detection_param_names()
    # The global comb scan is a first-class pipeline field, never a magic
    # detection key (altitude#2).
    assert "shift_scan_nm" not in detection
    assert "global_shift_scan_nm" in overrides
    assert isinstance(detection["kdet_weight_clip"], tuple)
    lo, hi = detection["kdet_weight_clip"]
    assert 0.05 <= lo <= 1.0 < hi <= 10.0

    # Overrides must resolve cleanly through the production builder's top tier.
    pipeline = build_pipeline_config(["Fe", "Si"], overrides=overrides)
    assert pipeline.detection_overrides == detection
    assert pipeline.global_shift_scan_nm == pytest.approx(overrides["global_shift_scan_nm"])
    for key in pipeline_keys:
        assert getattr(pipeline, key) == overrides[key], key
    if overrides["closure_mode"] is not None:
        assert pipeline.closure_mode in (
            "standard",
            "matrix",
            "oxide",
            "ilr",
            "pwlr",
            "dirichlet_residual",
        )


@pytest.mark.unit
def test_unknown_override_key_fails_fast():
    """A typo'd knob must never silently evaluate the production default
    (same strength as the deleted ``apply_config_overrides`` guard)."""
    from cflibs.inversion.pipeline import build_pipeline_config

    with pytest.raises(ValueError, match="no knob"):
        build_pipeline_config(["Fe"], overrides={"saha_boltzman_graph": True})
    # And the validation/normalization tiers apply to overrides too.
    with pytest.raises(ValueError, match="Valid modes"):
        build_pipeline_config(["Fe"], overrides={"closure_mode": "softmax"})


@pytest.mark.unit
def test_conditional_knobs():
    import numpy as np

    # Find one draw with adaptive tolerance + disabled rel-int floor.
    for seed in range(200):
        trial = _RandomTrial(np.random.default_rng(seed))
        params = knob_space.suggest_params(trial)
        if (
            params["wavelength_tolerance_mode"] == "adaptive"
            and not params["min_relative_intensity_enabled"]
        ):
            break
    else:  # pragma: no cover
        pytest.fail("no adaptive draw in 200 seeds")
    assert "wavelength_tolerance_nm" not in params
    assert "min_relative_intensity" not in params
    overrides = knob_space.params_to_overrides(params)
    assert overrides["wavelength_tolerance_nm"] is None
    assert overrides["min_relative_intensity"] is None


@pytest.mark.unit
def test_baseline_params_reproduce_production_defaults():
    """The derivation contract (simp#8/#9): knob defaults are DERIVED from
    AnalysisPipelineConfig / detect_line_observations, so the baseline
    candidate must resolve to the untouched production config. This pins the
    derivation machinery (factory routing, mode-selector defaults, condition
    gating), not hand-copied numbers."""
    from cflibs.inversion.pipeline import build_pipeline_config

    overrides = knob_space.params_to_overrides(knob_space.baseline_params())
    reference = build_pipeline_config(["Fe"])
    candidate = build_pipeline_config(["Fe"], overrides=overrides)
    # The baseline candidate must equal the untouched production config on
    # every searched pipeline field. exclude_resonance is special: the
    # production config carries None, which detect_and_select_lines resolves
    # to False — the knob space encodes the resolved value directly (the one
    # deliberately non-derived default).
    for key in set(overrides) - {"detection_overrides", "exclude_resonance"}:
        assert getattr(candidate, key) == getattr(reference, key), key
    assert candidate.exclude_resonance is False
    assert reference.exclude_resonance in (None, False)
    # Detection overrides must equal the detect_line_observations defaults.
    from cflibs.inversion.identify.line_detection import detect_line_observations

    sig = inspect.signature(detect_line_observations)
    for key, value in overrides["detection_overrides"].items():
        assert value == sig.parameters[key].default, key
    # The wedge axis stays out of the space (eff#1 pathology, c1-knobs-v2).
    assert "use_deconvolution" not in {k.param for k in knob_space.SPACE}
    assert "use_deconvolution" not in overrides["detection_overrides"]


@pytest.mark.unit
def test_optuna_fixedtrial_roundtrip():
    optuna = pytest.importorskip("optuna")

    params = knob_space.baseline_params()
    trial = optuna.trial.FixedTrial(params)
    suggested = knob_space.suggest_params(trial)
    assert suggested == params
    assert knob_space.params_to_overrides(suggested) == knob_space.params_to_overrides(params)


@pytest.mark.unit
def test_looser_gates_is_a_distinct_valid_candidate():
    base = knob_space.baseline_params()
    loose = knob_space.looser_gates_params()
    assert loose != base
    overrides = knob_space.params_to_overrides(loose)
    assert overrides["min_snr"] < base["min_snr"]
    assert overrides["detection_overrides"]["comb_min_matches"] == 2
