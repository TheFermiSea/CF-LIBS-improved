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
import inspect
import json
import subprocess
from dataclasses import MISSING, asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

KNOB_SPACE_VERSION = "c1-knobs-v2"  # v2: use_deconvolution dropped, defaults derived

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


# ---------------------------------------------------------------------------
# Production defaults, DERIVED from the shipped code (simp#8/#9)
# ---------------------------------------------------------------------------
# The knob table no longer hand-copies production defaults: the
# AnalysisPipelineConfig dataclass and the detect_line_observations signature
# ARE the source, so a shipped-default change cannot silently diverge from
# the baseline candidate. Search RANGES stay hand-written — they are search
# policy (design 3.1), not production fact.


def _pipeline_default(field_name: str) -> Any:
    from cflibs.inversion.pipeline import AnalysisPipelineConfig

    default = AnalysisPipelineConfig.__dataclass_fields__[field_name].default
    if default is MISSING:  # pragma: no cover - space definition error
        raise ValueError(f"AnalysisPipelineConfig.{field_name} has no plain default")
    return default


def _detection_default(param_name: str) -> Any:
    from cflibs.inversion.identify.line_detection import detect_line_observations

    default = inspect.signature(detect_line_observations).parameters[param_name].default
    if default is inspect.Parameter.empty:  # pragma: no cover - space definition error
        raise ValueError(f"detect_line_observations has no default for {param_name!r}")
    return default


def _pknob(param: str, kind: str, low=None, high=None, *, field=None, default=MISSING, **kw):
    """Pipeline knob; default derived from ``AnalysisPipelineConfig`` unless
    explicitly overridden (only ``exclude_resonance`` needs that). ``field``
    defaults to ``param`` — pass it only when the Optuna param name and the
    config field name differ (the frozen ``shift_scan_nm`` journal name)."""
    field = field if field is not None else param
    if default is MISSING:
        default = _pipeline_default(field)
    return Knob(param, "pipeline", field, kind, default, low, high, **kw)


def _dknob(param: str, kind: str, low=None, high=None, **kw):
    """Detection knob (param == kwarg name); default derived from the
    ``detect_line_observations`` signature."""
    return Knob(param, "detection", param, kind, _detection_default(param), low, high, **kw)


def _mknob(param: str, kind: str, default: Any, low=None, high=None, **kw):
    """Mode/selector knob consumed by :func:`params_to_overrides` (no field)."""
    return Knob(param, "mode", "", kind, default, low, high, **kw)


# Order matters: condition params precede the knobs they gate.
SPACE: tuple[Knob, ...] = (
    # --- A. AnalysisPipelineConfig (design 3.1-A) -------------------------
    _mknob(
        "min_relative_intensity_enabled",
        "cat",
        _pipeline_default("min_relative_intensity") is not None,
        choices=(False, True),
        note="{None} branch of min_relative_intensity",
    ),
    _pknob(
        "min_relative_intensity",
        "float",
        1e-4,
        0.05,
        log=True,
        condition=("min_relative_intensity_enabled", True),
    ),
    _pknob("top_k_per_element", "int", 15, 200),
    _mknob(
        "wavelength_tolerance_mode",
        "cat",
        "fixed" if _pipeline_default("wavelength_tolerance_nm") is not None else "adaptive",
        choices=("fixed", "adaptive"),
        note="adaptive => None (R-derived tolerance when resolving power known)",
    ),
    _pknob(
        "wavelength_tolerance_nm",
        "float",
        0.02,
        0.3,
        condition=("wavelength_tolerance_mode", "fixed"),
    ),
    _pknob("min_peak_height", "float", 0.001, 0.05, log=True),
    _mknob(
        "peak_width_mode",
        "cat",
        "fixed" if _pipeline_default("peak_width_nm") is not None else "adaptive",
        choices=("fixed", "adaptive"),
        note="adaptive => None (R-derived width when resolving power known)",
    ),
    _pknob("peak_width_nm", "float", 0.05, 0.5, condition=("peak_width_mode", "fixed")),
    _pknob("apply_self_absorption", "cat", choices=("off", "observable")),
    # exclude_resonance is the one non-derived default: production carries
    # None, which detect_and_select_lines RESOLVES to False — the knob space
    # encodes the resolved value (pinned by the baseline-params test).
    _pknob("exclude_resonance", "cat", default=False, choices=(False, True)),
    _pknob("min_snr", "float", 2.0, 20.0),
    _pknob("min_energy_spread_ev", "float", 0.5, 4.0),
    _pknob("min_lines_per_element", "int", 1, 5),
    _pknob("isolation_wavelength_nm", "float", 0.02, 0.3),
    _pknob("max_lines_per_element", "int", 5, 60),
    # wavelength_calibration: FROZEN True (catastrophic axis) — not searchable.
    _pknob("shift_coherence_veto", "cat", choices=(False, True)),
    _pknob(
        "residual_shift_scan_nm",
        "float",
        0.0,
        0.1,
        note="expect optimum at 0 (ye6t: the 0.05 legacy rode its window edge)",
    ),
    _pknob("affine_coverage_gate", "cat", choices=(False, True)),
    _pknob("line_residual_gate", "cat", choices=(False, True)),
    _pknob("max_iterations", "int", 5, 50),
    _pknob("t_tolerance_k", "float", 10.0, 500.0, log=True),
    _pknob("ne_tolerance_frac", "float", 0.01, 0.5, log=True),
    _pknob("boltzmann_weight_cap", "float", 1.0, 20.0),
    _pknob("min_boltzmann_r2", "float", 0.0, 0.8),
    _pknob("saha_boltzmann_graph", "cat", choices=(False, True)),
    _pknob(
        "closure_mode",
        "cat",
        # 'matrix' excluded: it requires a global matrix_element, which does not
        # exist for per-spectrum candidate sets — every draw died with the
        # failure death penalty in the smoke study (~1/6 of startup budget).
        choices=("standard", "oxide", "ilr", "pwlr", "dirichlet_residual"),
        note="per-preset (geological run tunes the geological closure), not per-dataset",
    ),
    _pknob("stark_ne", "cat", choices=(False, True)),
    # --- B. detect_line_observations gates (design 3.1-B) -----------------
    _dknob("ground_state_threshold_ev", "float", 0.05, 0.5),
    # Optuna param name "shift_scan_nm" is FROZEN for the live campaign1-phaseA
    # journal; it routes to the first-class pipeline field
    # ``global_shift_scan_nm`` (formerly the magic
    # detection_overrides["shift_scan_nm"] key) with identical effective
    # behavior: the global comb scan used when calibration fails/is skipped.
    _pknob(
        "shift_scan_nm",
        "float",
        0.1,
        1.0,
        field="global_shift_scan_nm",
        note="global scan when calibration fails/skipped",
    ),
    _dknob("comb_max_lines_per_element", "int", 10, 100),
    _dknob("comb_min_matches", "int", 2, 6),
    _dknob("comb_min_precision", "float", 0.005, 0.2, log=True),
    _dknob("comb_min_recall", "float", 0.02, 0.5, log=True),
    _dknob("comb_max_missing_fraction", "float", 0.5, 0.95),
    _dknob("comb_fallback_to_nearest", "cat", choices=(False, True)),
    _dknob("comb_fallback_max_elements", "int", 1, 10),
    _dknob("kdet_enabled", "cat", choices=(False, True)),
    _dknob("kdet_min_score", "float", 0.005, 0.3, log=True),
    _dknob("kdet_min_candidates", "int", 1, 5),
    _dknob("kdet_rarity_power", "float", 0.0, 2.0),
    _mknob("kdet_weight_clip_lo", "float", _detection_default("kdet_weight_clip")[0], 0.05, 1.0),
    _mknob("kdet_weight_clip_hi", "float", _detection_default("kdet_weight_clip")[1], 1.0, 10.0),
    _dknob("coherence_min_lines", "int", 2, 5),
    _dknob("coherence_min_fraction", "float", 0.2, 0.9),
    _dknob("residual_gate_min_kept_lines", "int", 1, 6),
    _dknob("poisson_floor_scale", "float", 0.3, 3.0),
    # use_deconvolution: REMOVED from the space (like the matrix-closure
    # exclusion). A True draw wedged a pool child inside JAX
    # backend_compile_and_load for 13+ minutes (GIL-released C/XLA that no
    # SIGALRM can interrupt) — the SLURM task died, ~32 core-hours went
    # unledgered, and TPE was blind to the cause. The detection default
    # (False) is what every candidate now gets. Optuna tolerates the space
    # change mid-study: old journal trials keep their recorded param, new
    # suggestions simply never draw it.
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
    """The production-default candidate (geological preset), as a flat params dict.

    A pure fold over ``SPACE``: every knob contributes its (derived)
    production default; condition-gated knobs drop out when their mode
    selector's default deactivates them (selectors precede their gated knobs
    in ``SPACE``).
    """
    params: dict[str, Any] = {}
    for knob in SPACE:
        if _is_active(knob, params):
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
    # git absent (OSError) or not a repo (CalledProcessError): the staged
    # cluster copy has no .git. Anything else should surface, not be eaten.
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover
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
