"""
Tests for the shared analyze/invert/batch pipeline defaults (bead l4a8).

Pins three contracts introduced by the pipeline-defaults overhaul
(docs/audit/2026-06-09-overhaul/04-pipeline-defaults.md):

1. **Preset resolution** — ``--preset`` (default ``geological``) resolves to
   the measured-best configuration on real ChemCam BHVO-2
   (``saha_boltzmann_graph=True`` + ``closure_mode='oxide'``, RMSE
   10.29 -> 4.03 wt%), with explicit flags / YAML keys overriding the preset.
2. **analyze == batch pipeline parity** — both commands feed the SAME
   resolved :class:`AnalysisPipelineConfig` into :func:`_run_pipeline`, so
   ``batch`` can never re-grow the pre-fix raw-detection wiring whose drift
   caused the Na=98% blowup.
3. **YAML unknown-key rejection** — typo'd ``analysis.*`` keys are a hard
   error instead of silently reverting the run to defaults.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cflibs.cli import main as cli_main
from cflibs.cli.main import (
    ANALYSIS_PRESETS,
    DEFAULT_ANALYSIS_PRESET,
    AnalysisPipelineConfig,
    _build_pipeline_config,
)
from cflibs.core.config import VALID_ANALYSIS_KEYS, validate_analysis_config

# =============================================================================
# 1. Preset resolution
# =============================================================================


class TestPresetResolution:
    def test_default_preset_is_geological_best_validated(self):
        """No flags -> the measured-best BHVO-2 configuration, exactly."""
        cfg = _build_pipeline_config(["Si", "Fe"])
        assert cfg.preset == "geological"
        assert DEFAULT_ANALYSIS_PRESET == "geological"
        assert cfg.saha_boltzmann_graph is True
        assert cfg.closure_mode == "oxide"
        # The rest of the validated-best bundle (audit acceptance criteria).
        assert cfg.boltzmann_weight_cap == pytest.approx(5.0)
        assert cfg.wavelength_calibration is True
        assert cfg.top_k_per_element == 60
        assert cfg.min_relative_intensity is None
        assert cfg.exclude_resonance is None  # helper resolves None -> keep
        assert cfg.apply_self_absorption is False
        # n_e is MEASURED from Stark widths by default (bead pxex / audit
        # 02-F2); the pressure-balance fallback only fires when no
        # literature-grade line qualifies.
        assert cfg.stark_ne is True

    def test_metallic_preset(self):
        cfg = _build_pipeline_config(["Fe", "Cr"], preset="metallic")
        assert cfg.saha_boltzmann_graph is True
        assert cfg.closure_mode == "standard"
        assert cfg.stark_ne is True

    def test_raw_preset_is_legacy_defaults(self):
        cfg = _build_pipeline_config(["Fe"], preset="raw")
        assert cfg.saha_boltzmann_graph is False
        assert cfg.closure_mode == "standard"
        assert cfg.stark_ne is False

    def test_stark_ne_flag_and_yaml_resolution(self):
        # Explicit flag beats the preset default...
        cfg = _build_pipeline_config(["Fe"], preset="geological", stark_ne=False)
        assert cfg.stark_ne is False
        # ...the YAML key beats the preset...
        cfg = _build_pipeline_config(["Fe"], analysis_cfg={"stark_ne": False})
        assert cfg.stark_ne is False
        # ...and the flag beats the YAML key.
        cfg = _build_pipeline_config(["Fe"], analysis_cfg={"stark_ne": False}, stark_ne=True)
        assert cfg.stark_ne is True

    def test_explicit_flags_override_preset(self):
        cfg = _build_pipeline_config(
            ["Fe"], preset="geological", saha_boltzmann_graph=False, closure_mode="ilr"
        )
        assert cfg.saha_boltzmann_graph is False
        assert cfg.closure_mode == "ilr"

    def test_yaml_keys_override_preset_and_flags_override_yaml(self):
        # YAML key beats the (YAML-chosen) preset...
        cfg = _build_pipeline_config(
            ["Fe"], analysis_cfg={"preset": "raw", "saha_boltzmann_graph": True}
        )
        assert cfg.preset == "raw"
        assert cfg.saha_boltzmann_graph is True
        assert cfg.closure_mode == "standard"
        # ...and an explicit CLI flag beats the YAML key.
        cfg = _build_pipeline_config(
            ["Fe"],
            analysis_cfg={"saha_boltzmann_graph": True},
            saha_boltzmann_graph=False,
        )
        assert cfg.saha_boltzmann_graph is False

    def test_unknown_preset_rejected(self):
        with pytest.raises(ValueError, match="Valid presets"):
            _build_pipeline_config(["Fe"], preset="metallurgical")

    def test_unknown_closure_mode_rejected(self):
        with pytest.raises(ValueError, match="Valid modes"):
            _build_pipeline_config(["Fe"], closure_mode="softmax")

    def test_preset_registry_contents(self):
        assert set(ANALYSIS_PRESETS) == {"geological", "metallic", "raw"}
        assert ANALYSIS_PRESETS["geological"] == {
            "saha_boltzmann_graph": True,
            "closure_mode": "oxide",
            "stark_ne": True,
        }

    def test_analyze_parser_exposes_preset_flag(self):
        """``analyze --preset metallic`` parses through the real CLI parser."""
        captured = {}

        def _capture(args):
            captured.update(vars(args))

        argv = ["cflibs", "analyze", "x.csv", "--elements", "Fe", "--preset", "metallic"]
        with patch.object(sys, "argv", argv):
            with patch("cflibs.cli.main.analyze_cmd", _capture):
                cli_main.main()
        assert captured["preset"] == "metallic"
        # The accuracy knobs default to None ("not given") so the preset wins.
        assert captured["saha_boltzmann_graph"] is None
        assert captured["closure_mode"] is None

    @pytest.mark.parametrize("command", ["analyze", "invert", "batch"])
    def test_all_three_commands_expose_preset(self, command, capsys):
        """--preset/--closure-mode/--saha-boltzmann-graph exist everywhere."""
        argv = ["cflibs", command, "--help"]
        with patch.object(sys, "argv", argv):
            with pytest.raises(SystemExit):
                cli_main.main()
        help_text = capsys.readouterr().out
        assert "--preset" in help_text
        assert "--closure-mode" in help_text
        assert "--saha-boltzmann-graph" in help_text


# =============================================================================
# 2. analyze == batch pipeline parity
# =============================================================================


class _StopPipeline(RuntimeError):
    """Sentinel raised by the patched ``_run_pipeline`` after capture."""


def _capture_pipeline_from(cmd_func, args, monkeypatch, tmp_path):
    """Run ``cmd_func(args)`` with the heavy stages stubbed; return the
    :class:`AnalysisPipelineConfig` the command handed to ``_run_pipeline``."""
    captured = {}

    def _fake_run_pipeline(wavelength, intensity, atomic_db, pipeline, uncertainty_mode="none"):
        captured["pipeline"] = pipeline
        raise _StopPipeline()

    fake_db = tmp_path / "fake.db"
    fake_db.touch()
    monkeypatch.setattr(cli_main, "_run_pipeline", _fake_run_pipeline)
    monkeypatch.setattr(cli_main, "_resolve_db_path", lambda *a, **k: fake_db)
    monkeypatch.setattr(
        "cflibs.io.spectrum.load_spectrum",
        lambda *_a, **_k: (np.linspace(300.0, 400.0, 10), np.ones(10)),
    )
    with patch("cflibs.atomic.database.AtomicDatabase", MagicMock()):
        try:
            cmd_func(args)
        except _StopPipeline:
            pass
    assert "pipeline" in captured, f"{cmd_func.__name__} never reached _run_pipeline"
    return captured["pipeline"]


class TestBatchAnalyzeParity:
    """batch must resolve the IDENTICAL pipeline config as analyze."""

    @staticmethod
    def _make_args(tmp_path, **overrides):
        class Args:
            spectrum = "spectrum.csv"
            directory = str(tmp_path / "spectra")
            elements = "Si,Fe,Mg"
            db_path = None
            output = None
            output_format = "table"
            uncertainty = "none"
            preset = None
            closure_mode = None
            saha_boltzmann_graph = None
            apply_self_absorption = None
            min_relative_intensity = None
            resolving_power = None

        for key, value in overrides.items():
            setattr(Args, key, value)
        return Args()

    @pytest.mark.parametrize(
        "overrides",
        [
            {},  # bare defaults -> geological preset
            {"preset": "metallic"},
            {"preset": "raw", "saha_boltzmann_graph": True},
            {"closure_mode": "standard", "resolving_power": 5000.0},
        ],
    )
    def test_batch_and_analyze_build_identical_pipeline(self, overrides, monkeypatch, tmp_path):
        spectra_dir = tmp_path / "spectra"
        spectra_dir.mkdir()
        (spectra_dir / "a.csv").write_text("wavelength_nm,counts\n300.0,1\n")

        analyze_pipeline = _capture_pipeline_from(
            cli_main.analyze_cmd, self._make_args(tmp_path, **overrides), monkeypatch, tmp_path
        )
        batch_pipeline = _capture_pipeline_from(
            cli_main.batch_cmd, self._make_args(tmp_path, **overrides), monkeypatch, tmp_path
        )

        assert isinstance(analyze_pipeline, AnalysisPipelineConfig)
        assert isinstance(batch_pipeline, AnalysisPipelineConfig)
        # Dataclass equality covers every detection, selection and solver knob.
        assert analyze_pipeline == batch_pipeline

    def test_default_pipeline_is_measured_best(self, monkeypatch, tmp_path):
        """A bare ``analyze`` resolves to the BHVO-2 measured-best config."""
        pipeline = _capture_pipeline_from(
            cli_main.analyze_cmd, self._make_args(tmp_path), monkeypatch, tmp_path
        )
        assert pipeline.preset == "geological"
        assert pipeline.saha_boltzmann_graph is True
        assert pipeline.closure_mode == "oxide"
        assert pipeline.boltzmann_weight_cap == pytest.approx(5.0)
        assert pipeline.wavelength_calibration is True


# =============================================================================
# 3. YAML analysis.* unknown-key rejection
# =============================================================================


class TestAnalysisConfigValidation:
    def test_valid_keys_accepted(self):
        config = {
            "analysis": {
                "elements": ["Fe"],
                "preset": "geological",
                "saha_boltzmann_graph": True,
                "closure_mode": "oxide",
                "boltzmann_weight_cap": 5.0,
            }
        }
        assert validate_analysis_config(config) is True

    def test_missing_analysis_section_accepted(self):
        assert validate_analysis_config({}) is True

    def test_typo_key_rejected_listing_valid_keys(self):
        """The audit's canonical typo silently reverted runs to defaults."""
        config = {"analysis": {"saha_boltzman_graph": True}}
        with pytest.raises(ValueError) as excinfo:
            validate_analysis_config(config)
        message = str(excinfo.value)
        assert "saha_boltzman_graph" in message
        assert "saha_boltzmann_graph" in message  # valid-keys listing

    def test_non_mapping_analysis_rejected(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            validate_analysis_config({"analysis": ["not", "a", "dict"]})

    def test_valid_keys_cover_builder_inputs(self):
        """Every YAML key the pipeline builder reads must be declared valid."""
        builder_keys = {
            "preset",
            "min_relative_intensity",
            "top_k_per_element",
            "resolving_power",
            "wavelength_tolerance_nm",
            "min_peak_height",
            "peak_width_nm",
            "apply_self_absorption",
            "exclude_resonance",
            "min_snr",
            "min_energy_spread_ev",
            "min_lines_per_element",
            "isolation_wavelength_nm",
            "max_lines_per_element",
            "wavelength_calibration",
            "max_iterations",
            "t_tolerance_k",
            "ne_tolerance_frac",
            "pressure_pa",
            "pressure",
            "self_absorption_column_density_cm3",
            "self_absorption_plasma_length_cm",
            "boltzmann_weight_cap",
            "min_boltzmann_r2",
            "saha_boltzmann_graph",
            "closure_mode",
            "closure_kwargs",
            "matrix_element",
            "oxide_elements",
            "stark_ne",
        }
        assert builder_keys <= VALID_ANALYSIS_KEYS

    def test_invert_cmd_rejects_typo_config(self, tmp_path):
        """End-to-end: ``invert --config`` with a typo'd key hard-errors
        before any spectrum or database work."""
        config_path = tmp_path / "bad.yaml"
        config_path.write_text("analysis:\n  elements: [Fe]\n  saha_boltzman_graph: true\n")

        class Args:
            spectrum = "spectrum.csv"
            config = str(config_path)
            elements = None
            output = None

        with pytest.raises(ValueError, match="Unknown analysis config key"):
            cli_main.invert_cmd(Args())


# =============================================================================
# Shipped example config must resolve to the validated defaults
# =============================================================================


def test_example_inversion_config_is_valid_and_safe():
    """The shipped example config must validate and must NOT re-enable the
    legacy major-deleting detection settings (exclude_resonance=true,
    min_relative_intensity=100.0)."""
    from pathlib import Path

    import yaml

    example = Path(__file__).resolve().parents[2] / "examples" / "inversion_config_example.yaml"
    config = yaml.safe_load(example.read_text())
    assert validate_analysis_config(config) is True
    analysis = config["analysis"]
    assert analysis.get("exclude_resonance") is False
    assert analysis.get("min_relative_intensity") is None
    assert analysis.get("preset", DEFAULT_ANALYSIS_PRESET) == "geological"
