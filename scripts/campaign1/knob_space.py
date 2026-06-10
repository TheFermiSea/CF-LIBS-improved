"""Campaign 1 search-space definition (Optuna TPE over the inversion knob space).

Source of truth: docs/audit/2026-06-10-goalfirst/optimization-program-design.md
Section 3.1 (Tables A and B). Every knob carries its name, type, search range
and the *current production default* (geological preset / dataclass defaults
on this branch). Excluded by design (frozen, NOT searchable):

- ``wavelength_calibration`` — frozen True ("turning it off is a known
  catastrophic axis", design 3.1-A);
- ``PRESENCE_EPS_MASSFRAC`` (5e-3), the confounder set (Ag/Sn/W/Bi/Th) and the
  candidate-set policy — scoring constants live in the scoreboard, out of
  bounds (design 2.4);
- ``resolving_power`` — a per-dataset truth hint, not a tunable.

This module is intentionally **optuna-free**: ``suggest_params`` only calls
``trial.suggest_*`` methods (duck-typed), so the mapping is unit-testable
without the optimizer installed. The physics-only constraint applies to
``cflibs/`` only; this file lives in ``scripts/campaign1/`` and never ships.

A candidate is fully described by a flat Optuna params dict;
:func:`params_to_overrides` turns it into the single ``config_overrides``
dict consumed by the ``overrides`` tier of
``cflibs.inversion.pipeline.build_pipeline_config`` (detection-layer knobs
nest under the ``detection_overrides`` field added to
``AnalysisPipelineConfig`` for this campaign).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

KNOB_SPACE_VERSION = "c1-knobs-v1"

#: Repository root (scripts/campaign1/ -> repo).
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Knob:
    """One searchable knob.

    ``target`` routes the value: ``pipeline`` -> an ``AnalysisPipelineConfig``
    field override, ``detection`` -> a ``detect_line_observations`` kwarg
    (via ``detection_overrides``), ``mode`` -> a selector param consumed by
    :func:`params_to_overrides` post-processing (no direct field).
    """

    param: str
    target: str  # "pipeline" | "detection" | "mode"
    field: str
    kind: str  # "float" | "int" | "cat"
    default: Any
    low: Optional[float] = None
    high: Optional[float] = None
    log: bool = False
    choices: Optional[tuple] = None
    condition: Optional[tuple] = None  # (param_name, required_value)
    note: str = ""


# Order matters: condition params precede the knobs they gate.
SPACE: tuple[Knob, ...] = (
    # --- A. AnalysisPipelineConfig (design 3.1-A) -------------------------
    Knob(
        "min_relative_intensity_enabled",
        "mode",
        "",
        "cat",
        False,
        choices=(False, True),
        note="{None} branch of min_relative_intensity",
    ),
    Knob(
        "min_relative_intensity",
        "pipeline",
        "min_relative_intensity",
        "float",
        None,
        1e-4,
        0.05,
        log=True,
        condition=("min_relative_intensity_enabled", True),
    ),
    Knob("top_k_per_element", "pipeline", "top_k_per_element", "int", 60, 15, 200),
    Knob(
        "wavelength_tolerance_mode",
        "mode",
        "",
        "cat",
        "fixed",
        choices=("fixed", "adaptive"),
        note="adaptive => None (R-derived tolerance when resolving power known)",
    ),
    Knob(
        "wavelength_tolerance_nm",
        "pipeline",
        "wavelength_tolerance_nm",
        "float",
        0.1,
        0.02,
        0.3,
        condition=("wavelength_tolerance_mode", "fixed"),
    ),
    Knob("min_peak_height", "pipeline", "min_peak_height", "float", 0.01, 0.001, 0.05, log=True),
    Knob(
        "peak_width_mode",
        "mode",
        "",
        "cat",
        "fixed",
        choices=("fixed", "adaptive"),
        note="adaptive => None (R-derived width when resolving power known)",
    ),
    Knob(
        "peak_width_nm",
        "pipeline",
        "peak_width_nm",
        "float",
        0.2,
        0.05,
        0.5,
        condition=("peak_width_mode", "fixed"),
    ),
    Knob(
        "apply_self_absorption",
        "pipeline",
        "apply_self_absorption",
        "cat",
        "off",
        choices=("off", "observable"),
    ),
    Knob("exclude_resonance", "pipeline", "exclude_resonance", "cat", False, choices=(False, True)),
    Knob("min_snr", "pipeline", "min_snr", "float", 10.0, 2.0, 20.0),
    Knob("min_energy_spread_ev", "pipeline", "min_energy_spread_ev", "float", 2.0, 0.5, 4.0),
    Knob("min_lines_per_element", "pipeline", "min_lines_per_element", "int", 3, 1, 5),
    Knob("isolation_wavelength_nm", "pipeline", "isolation_wavelength_nm", "float", 0.1, 0.02, 0.3),
    Knob("max_lines_per_element", "pipeline", "max_lines_per_element", "int", 20, 5, 60),
    # wavelength_calibration: FROZEN True (catastrophic axis) — not searchable.
    Knob(
        "shift_coherence_veto",
        "pipeline",
        "shift_coherence_veto",
        "cat",
        True,
        choices=(False, True),
    ),
    Knob(
        "residual_shift_scan_nm",
        "pipeline",
        "residual_shift_scan_nm",
        "float",
        0.0,
        0.0,
        0.1,
        note="expect optimum at 0 (ye6t: the 0.05 legacy rode its window edge)",
    ),
    Knob(
        "affine_coverage_gate",
        "pipeline",
        "affine_coverage_gate",
        "cat",
        True,
        choices=(False, True),
    ),
    Knob(
        "line_residual_gate", "pipeline", "line_residual_gate", "cat", True, choices=(False, True)
    ),
    Knob("max_iterations", "pipeline", "max_iterations", "int", 20, 5, 50),
    Knob("t_tolerance_k", "pipeline", "t_tolerance_k", "float", 100.0, 10.0, 500.0, log=True),
    Knob("ne_tolerance_frac", "pipeline", "ne_tolerance_frac", "float", 0.1, 0.01, 0.5, log=True),
    Knob("boltzmann_weight_cap", "pipeline", "boltzmann_weight_cap", "float", 5.0, 1.0, 20.0),
    Knob("min_boltzmann_r2", "pipeline", "min_boltzmann_r2", "float", 0.3, 0.0, 0.8),
    Knob(
        "saha_boltzmann_graph",
        "pipeline",
        "saha_boltzmann_graph",
        "cat",
        True,
        choices=(False, True),
    ),
    Knob(
        "closure_mode",
        "pipeline",
        "closure_mode",
        "cat",
        "oxide",
        # 'matrix' excluded: it requires a global matrix_element, which does not
        # exist for per-spectrum candidate sets — every draw died with the
        # failure death penalty in the smoke study (~1/6 of startup budget).
        choices=("standard", "oxide", "ilr", "pwlr", "dirichlet_residual"),
        note="per-preset (geological run tunes the geological closure), not per-dataset",
    ),
    Knob("stark_ne", "pipeline", "stark_ne", "cat", True, choices=(False, True)),
    # --- B. detect_line_observations gates (design 3.1-B) -----------------
    Knob(
        "ground_state_threshold_ev",
        "detection",
        "ground_state_threshold_ev",
        "float",
        0.1,
        0.05,
        0.5,
    ),
    # Optuna param name "shift_scan_nm" is FROZEN for the live campaign1-phaseA
    # journal; it now routes to the first-class pipeline field
    # ``global_shift_scan_nm`` (formerly the magic
    # detection_overrides["shift_scan_nm"] key) with identical effective
    # behavior: the global comb scan used when calibration fails/is skipped.
    Knob(
        "shift_scan_nm",
        "pipeline",
        "global_shift_scan_nm",
        "float",
        0.5,
        0.1,
        1.0,
        note="global scan when calibration fails/skipped",
    ),
    Knob(
        "comb_max_lines_per_element",
        "detection",
        "comb_max_lines_per_element",
        "int",
        30,
        10,
        100,
    ),
    Knob("comb_min_matches", "detection", "comb_min_matches", "int", 3, 2, 6),
    Knob(
        "comb_min_precision", "detection", "comb_min_precision", "float", 0.02, 0.005, 0.2, log=True
    ),
    Knob("comb_min_recall", "detection", "comb_min_recall", "float", 0.1, 0.02, 0.5, log=True),
    Knob(
        "comb_max_missing_fraction",
        "detection",
        "comb_max_missing_fraction",
        "float",
        0.85,
        0.5,
        0.95,
    ),
    Knob(
        "comb_fallback_to_nearest",
        "detection",
        "comb_fallback_to_nearest",
        "cat",
        True,
        choices=(False, True),
    ),
    Knob("comb_fallback_max_elements", "detection", "comb_fallback_max_elements", "int", 5, 1, 10),
    Knob("kdet_enabled", "detection", "kdet_enabled", "cat", True, choices=(False, True)),
    Knob("kdet_min_score", "detection", "kdet_min_score", "float", 0.05, 0.005, 0.3, log=True),
    Knob("kdet_min_candidates", "detection", "kdet_min_candidates", "int", 2, 1, 5),
    Knob("kdet_rarity_power", "detection", "kdet_rarity_power", "float", 0.5, 0.0, 2.0),
    Knob("kdet_weight_clip_lo", "mode", "", "float", 0.25, 0.05, 1.0),
    Knob("kdet_weight_clip_hi", "mode", "", "float", 4.0, 1.0, 10.0),
    Knob("coherence_min_lines", "detection", "coherence_min_lines", "int", 2, 2, 5),
    Knob("coherence_min_fraction", "detection", "coherence_min_fraction", "float", 0.5, 0.2, 0.9),
    Knob(
        "residual_gate_min_kept_lines",
        "detection",
        "residual_gate_min_kept_lines",
        "int",
        3,
        1,
        6,
    ),
    Knob("poisson_floor_scale", "detection", "poisson_floor_scale", "float", 1.0, 0.3, 3.0),
    Knob(
        "use_deconvolution", "detection", "use_deconvolution", "cat", False, choices=(False, True)
    ),
)


def _is_active(knob: Knob, params: Mapping[str, Any]) -> bool:
    if knob.condition is None:
        return True
    name, required = knob.condition
    return params.get(name) == required


def suggest_params(trial: Any) -> dict[str, Any]:
    """Draw one candidate from the space (duck-typed Optuna trial)."""
    params: dict[str, Any] = {}
    for knob in SPACE:
        if not _is_active(knob, params):
            continue
        if knob.kind == "cat":
            params[knob.param] = trial.suggest_categorical(knob.param, list(knob.choices))
        elif knob.kind == "int":
            params[knob.param] = trial.suggest_int(knob.param, int(knob.low), int(knob.high))
        elif knob.kind == "float":
            params[knob.param] = trial.suggest_float(knob.param, knob.low, knob.high, log=knob.log)
        else:  # pragma: no cover - space definition error
            raise ValueError(f"Unknown knob kind {knob.kind!r} for {knob.param}")
    return params


def params_to_overrides(params: Mapping[str, Any]) -> dict[str, Any]:
    """Map a flat params dict to one ``config_overrides`` dict.

    The result is consumed by the ``overrides`` tier of
    ``cflibs.inversion.pipeline.build_pipeline_config``: pipeline knobs are
    top-level ``AnalysisPipelineConfig`` field overrides; detection knobs nest
    under ``detection_overrides``.
    """
    pipeline: dict[str, Any] = {}
    detection: dict[str, Any] = {}
    for knob in SPACE:
        if knob.target == "mode" or not _is_active(knob, params):
            continue
        if knob.param not in params:
            raise KeyError(f"params missing active knob {knob.param!r}")
        value = params[knob.param]
        if knob.target == "pipeline":
            pipeline[knob.field] = value
        elif knob.target == "detection":
            detection[knob.field] = value
    # Conditional / composite post-processing.
    if not params.get("min_relative_intensity_enabled", False):
        pipeline["min_relative_intensity"] = None
    if params.get("wavelength_tolerance_mode") == "adaptive":
        pipeline["wavelength_tolerance_nm"] = None
    if params.get("peak_width_mode") == "adaptive":
        pipeline["peak_width_nm"] = None
    detection["kdet_weight_clip"] = (
        float(params["kdet_weight_clip_lo"]),
        float(params["kdet_weight_clip_hi"]),
    )
    pipeline["detection_overrides"] = detection
    return pipeline


def baseline_params() -> dict[str, Any]:
    """The production-default candidate (geological preset), as a flat params dict."""
    params: dict[str, Any] = {
        "min_relative_intensity_enabled": False,
        "wavelength_tolerance_mode": "fixed",
        "peak_width_mode": "fixed",
    }
    for knob in SPACE:
        if knob.target == "mode" and knob.param in params:
            continue
        if not _is_active(knob, params):
            continue
        params[knob.param] = knob.default
    return params


def looser_gates_params() -> dict[str, Any]:
    """Hand-picked seed trial encoding the failure hypothesis (design 3.1).

    The 'No usable spectral lines' failures point at detection gates being too
    strict for low-SNR / pure-element spectra: loosen
    min_snr / min_peak_height / comb_min_matches / min_lines_per_element.
    """
    params = baseline_params()
    params.update(
        {
            "min_snr": 3.0,
            "min_peak_height": 0.002,
            "comb_min_matches": 2,
            "min_lines_per_element": 1,
            "residual_gate_min_kept_lines": 1,
            "kdet_min_score": 0.01,
        }
    )
    return params


def seed_trial_params() -> list[dict[str, Any]]:
    """All hand-picked trials to enqueue at study creation."""
    return [baseline_params(), looser_gates_params()]


# ---------------------------------------------------------------------------
# FROZEN_MANIFEST: the per-study reproducibility record
# ---------------------------------------------------------------------------

FROZEN_MANIFEST_FILENAME = "frozen_manifest.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_describe(repo_root: Path) -> dict[str, Any]:
    def _run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo_root, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        sha = _run("rev-parse", "HEAD")
        dirty = bool(_run("status", "--porcelain"))
        branch = _run("rev-parse", "--abbrev-ref", "HEAD")
    except Exception as exc:  # pragma: no cover - git absent on cluster stage
        return {"git_sha": None, "git_dirty": None, "git_branch": None, "git_error": repr(exc)}
    return {"git_sha": sha, "git_dirty": dirty, "git_branch": branch}


def write_frozen_manifest(
    study_dir: Path | str,
    *,
    splits_manifest: Mapping[str, Any],
    splits_manifest_path: Path | str,
    db_path: Path | str,
    seed: int,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write ``frozen_manifest.json`` next to the study.

    Captures everything needed to reproduce a trial: the exact dataset split
    id lists, the atomic-DB sha256, the sampling seed, the git SHA and the
    knob-space version/definition (design 6.3).
    """
    study_dir = Path(study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(db_path)
    manifest: dict[str, Any] = {
        "campaign": "campaign1",
        "knob_space_version": KNOB_SPACE_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        **_git_describe(REPO_ROOT),
        "db_path": str(db_path),
        "db_sha256": _sha256_file(db_path),
        "splits_manifest_path": str(splits_manifest_path),
        "splits_manifest_sha256": _sha256_file(Path(splits_manifest_path)),
        "splits": dict(splits_manifest),
        "knobs": [asdict(knob) for knob in SPACE],
    }
    if extra:
        manifest["extra"] = dict(extra)
    out = study_dir / FROZEN_MANIFEST_FILENAME
    out.write_text(json.dumps(manifest, indent=2, default=str))
    return out
