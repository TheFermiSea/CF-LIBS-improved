"""
CLI / YAML plumbing tests for the spectral-response correction hook (gzwd).

The hook must be reachable from all three shared-pipeline entry points
(``analyze``, ``invert``, ``batch``) with the standard precedence
(CLI flag > YAML ``analysis.response_curve``), default to identity
(``None`` -> the exact same intensity array reaches detection), and apply
the division when configured.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import cflibs.cli.main as cli_main
from cflibs.cli.main import AnalysisPipelineConfig, _build_pipeline_config
from cflibs.core.config import VALID_ANALYSIS_KEYS, validate_analysis_config

CURVE_CSV = "wavelength_nm,relative_efficiency\n250.0,0.5\n550.0,1.0\n"


# ---------------------------------------------------------------------------
# Flag exposure and config resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("command", ["analyze", "invert", "batch"])
def test_all_three_commands_expose_response_curve(command, capsys):
    argv = ["cflibs", command, "--help"]
    with patch.object(sys, "argv", argv):
        with pytest.raises(SystemExit):
            cli_main.main()
    assert "--response-curve" in capsys.readouterr().out


def test_analyze_parser_accepts_response_curve_flag():
    captured = {}

    def _capture(args):
        captured.update(vars(args))

    argv = [
        "cflibs",
        "analyze",
        "x.csv",
        "--elements",
        "Fe",
        "--response-curve",
        "lamp.csv",
    ]
    with patch.object(sys, "argv", argv):
        with patch("cflibs.cli.main.analyze_cmd", _capture):
            cli_main.main()
    assert captured["response_curve"] == "lamp.csv"


def test_default_is_none_identity():
    pipeline = _build_pipeline_config(["Fe"])
    assert pipeline.response_curve is None


def test_yaml_key_resolves():
    pipeline = _build_pipeline_config(["Fe"], analysis_cfg={"response_curve": "lamp.csv"})
    assert pipeline.response_curve == "lamp.csv"


def test_cli_flag_overrides_yaml():
    pipeline = _build_pipeline_config(
        ["Fe"],
        analysis_cfg={"response_curve": "yaml.csv"},
        response_curve="flag.csv",
    )
    assert pipeline.response_curve == "flag.csv"


def test_validate_analysis_config_accepts_response_curve():
    assert "response_curve" in VALID_ANALYSIS_KEYS
    assert validate_analysis_config({"analysis": {"response_curve": "lamp.csv"}})


# ---------------------------------------------------------------------------
# Pipeline behaviour: identity regression + applied division
# ---------------------------------------------------------------------------


class _StopPipeline(RuntimeError):
    """Sentinel raised by the patched detection stage after capture."""


def _run_pipeline_capturing_intensity(pipeline, wavelength, intensity, monkeypatch):
    """Run ``_run_pipeline`` with detection stubbed; return the intensity
    array that reached ``_detect_and_select_lines``."""
    captured = {}

    def _fake_detect(wl, inten, *args, **kwargs):
        captured["wavelength"] = wl
        captured["intensity"] = inten
        raise _StopPipeline()

    monkeypatch.setattr(cli_main, "_detect_and_select_lines", _fake_detect)
    with pytest.raises(_StopPipeline):
        cli_main._run_pipeline(wavelength, intensity, MagicMock(), pipeline)
    return captured["intensity"]


def test_run_pipeline_identity_is_bit_identical(monkeypatch):
    """Regression pin: with response_curve=None the SAME intensity object
    (no copy, no arithmetic) reaches detection — default behaviour is
    bit-identical to the pre-hook pipeline."""
    pipeline = _build_pipeline_config(["Fe"])
    wavelength = np.linspace(300.0, 500.0, 50)
    intensity = np.random.default_rng(7).uniform(0.0, 1.0, 50)
    seen = _run_pipeline_capturing_intensity(pipeline, wavelength, intensity, monkeypatch)
    assert seen is intensity


def test_run_pipeline_applies_division(monkeypatch, tmp_path):
    curve_path = tmp_path / "lamp.csv"
    curve_path.write_text(CURVE_CSV)
    pipeline = _build_pipeline_config(["Fe"], response_curve=str(curve_path))
    wavelength = np.linspace(300.0, 500.0, 50)
    intensity = np.ones(50)
    seen = _run_pipeline_capturing_intensity(pipeline, wavelength, intensity, monkeypatch)
    # E(lambda) interpolates the 0.5 -> 1.0 ramp; corrected = 1 / E.
    expected_eff = np.interp(wavelength, [250.0, 550.0], [0.5, 1.0])
    np.testing.assert_allclose(seen, 1.0 / expected_eff, rtol=1e-12)


def test_run_pipeline_uncovered_curve_hard_errors(monkeypatch, tmp_path):
    from cflibs.inversion.preprocess.response_correction import ResponseCurveCoverageError

    curve_path = tmp_path / "narrow.csv"
    curve_path.write_text("400.0,0.9\n450.0,1.0\n")
    pipeline = _build_pipeline_config(["Fe"], response_curve=str(curve_path))
    wavelength = np.linspace(300.0, 500.0, 50)
    monkeypatch.setattr(
        cli_main, "_detect_and_select_lines", lambda *a, **k: pytest.fail("must not be reached")
    )
    with pytest.raises(ResponseCurveCoverageError, match="does not cover"):
        cli_main._run_pipeline(wavelength, np.ones(50), MagicMock(), pipeline)


def test_run_pipeline_records_response_curve_in_diagnostics(monkeypatch, tmp_path):
    curve_path = tmp_path / "lamp.csv"
    curve_path.write_text(CURVE_CSV)
    pipeline = _build_pipeline_config(["Fe"], response_curve=str(curve_path))

    def _fake_detect(wl, inten, *args, **kwargs):
        return [], {"dropped_elements": {}}

    monkeypatch.setattr(cli_main, "_detect_and_select_lines", _fake_detect)
    with pytest.raises(ValueError, match="No usable spectral lines"):
        cli_main._run_pipeline(np.linspace(300.0, 500.0, 5), np.ones(5), MagicMock(), pipeline)


# ---------------------------------------------------------------------------
# invert: YAML path resolved relative to the config file
# ---------------------------------------------------------------------------


def test_invert_resolves_yaml_response_curve_relative_to_config(monkeypatch, tmp_path):
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    (config_dir / "lamp.csv").write_text(CURVE_CSV)
    config_path = config_dir / "inversion.yaml"
    config_path.write_text("analysis:\n  elements: [Fe]\n  response_curve: lamp.csv\n")

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

    class Args:
        spectrum = "spectrum.csv"
        config = str(config_path)
        elements = None
        output = None

    with patch("cflibs.atomic.database.AtomicDatabase", MagicMock()):
        try:
            cli_main.invert_cmd(Args())
        except _StopPipeline:
            pass

    pipeline = captured["pipeline"]
    assert isinstance(pipeline, AnalysisPipelineConfig)
    assert pipeline.response_curve == str(config_dir / "lamp.csv")
