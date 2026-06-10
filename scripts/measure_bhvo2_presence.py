#!/usr/bin/env python3
"""Falsifiable element-presence measurement on real BHVO-2 (beads 29p1, vj82).

Runs the SHARED production pipeline (``cflibs.inversion.pipeline`` —
``build_pipeline_config`` -> ``run_pipeline``, the exact code path behind
``cflibs analyze``) on a real BHVO-2 spectrum and scores the result against
the USGS certified composition. Reports the five-criteria gate from the
29p1 redesign:

  (a) Al recovered          -- cert Al mass-fraction 0.0714, currently ~0
  (c) no Ag/Sn/W/Bi FPs     -- confounders that must stay absent
  (d) RMSE down, attributed -- not a Mn-deflation artifact
  (e) Na not blown up; Mn   -- Na fix preserved, Mn over-attribution

This is a measurement tool, not a fix. Run on dev for the baseline, then
after each change for the delta.

Harness bridge (bead vj82): the script no longer hand-builds its pipeline.
Every knob resolves through the same builder as the CLI, with flag defaults
equal to the CLI/preset defaults (geological preset: oxide closure + pooled
Saha-Boltzmann graph; wavelength calibration ON; shift-coherence veto ON).
The knobs where the legacy harness and the CLI historically diverged are
explicit flags so one-factor ablations are possible:

  --wavelength-calibration / --no-wavelength-calibration
  --shift-coherence-veto   / --no-shift-coherence-veto
  --confounders            / --no-confounders   (request Ag/Sn/W/Bi/Th too)

NOTE (default change vs the pre-vj82 script): ``--closure-mode`` and
``--saha-boltzmann-graph`` now default to the CLI preset resolution
(geological => oxide + SB graph) instead of the legacy ``standard`` + off.
The maintainer gate invocation ``--closure-mode oxide --saha-boltzmann-graph``
is unaffected (it resolves identically).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# --- Harness provenance guard (bead vj82) ----------------------------------
# ``python scripts/measure_bhvo2_presence.py`` puts ``scripts/`` (NOT the
# checkout root) first on sys.path, so ``cflibs`` silently resolves to
# whatever editable install is registered in the venv — historically the main
# checkout rather than the worktree under test. That made two "identical"
# harness runs measure different code. Prepend this script's own checkout
# root so the measurement always runs the code it sits next to, and record
# the resolved ``cflibs`` path in the output.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cflibs  # noqa: E402
from cflibs.atomic.database import AtomicDatabase  # noqa: E402
from cflibs.benchmark.reference_compositions import BHVO2_BASALT_USGS  # noqa: E402
from cflibs.inversion.pipeline import (  # noqa: E402
    ANALYSIS_PRESETS,
    CLOSURE_MODES,
    AnalysisPipelineConfig,
    build_pipeline_config,
    run_pipeline,
)
from cflibs.io.spectrum import load_spectrum  # noqa: E402

# Known FP confounders: neutral-resonance lines in the BHVO-2 band, same
# thermal E_k band as the real majors (29p1 Root cause B).
CONFOUNDERS = ["Ag", "Sn", "W", "Bi", "Th"]
EPS_PRESENT = 5e-3  # mass-fraction above which an element counts as "called present"


def _resolve_db() -> Path:
    for cand in (
        _REPO_ROOT / "ASD_da/libs_production.db",
        Path("ASD_da/libs_production.db"),
        Path("/home/brian/code/CF-LIBS-improved/ASD_da/libs_production.db"),
    ):
        if cand.exists():
            return cand
    raise FileNotFoundError("libs_production.db not found")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse harness flags. Defaults = the CLI/preset defaults (``None`` =
    'not given', resolved by :func:`build_pipeline_config` exactly as for a
    flagless ``cflibs analyze``)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spectrum", default="data/bhvo2_usgs/chemcam_bhvo2_loc1_spectrum.csv")
    ap.add_argument(
        "--preset",
        default=None,
        choices=sorted(ANALYSIS_PRESETS),
        help="Analysis preset (default: the CLI default, 'geological').",
    )
    ap.add_argument(
        "--min-relative-intensity", type=lambda s: None if s == "none" else float(s), default=None
    )
    ap.add_argument("--exclude-resonance", choices=["auto", "true", "false"], default="auto")
    ap.add_argument(
        "--apply-self-absorption",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Curve-of-growth self-absorption correction (default: CLI default, off).",
    )
    ap.add_argument(
        "--closure-mode",
        default=None,
        choices=list(CLOSURE_MODES),
        help="Closure equation (default: preset resolution; geological => oxide).",
    )
    ap.add_argument(
        "--saha-boltzmann-graph",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pooled Saha-Boltzmann graph intercepts (default: preset resolution).",
    )
    ap.add_argument(
        "--wavelength-calibration",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Segmented RANSAC wavelength calibration before detection "
        "(default: CLI default, on).",
    )
    ap.add_argument(
        "--shift-coherence-veto",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reject elements whose matched lines disagree on the residual "
        "shift (default: CLI default, on).",
    )
    ap.add_argument(
        "--confounders",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also request the Ag/Sn/W/Bi/Th confounders (legacy harness "
        "behavior, default on). --no-confounders requests only the 10 "
        "certified elements, matching a plain CLI invocation.",
    )
    ap.add_argument(
        "--stark-ne",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Measure n_e from the Stark widths of observed literature-grade lines "
            "(mirrors the production geological-preset default; --no-stark-ne for "
            "the legacy 1-atm pressure-balance fallback)."
        ),
    )
    ap.add_argument("--label", default="baseline")
    return ap.parse_args(argv)


def build_pipeline_from_args(args: argparse.Namespace) -> AnalysisPipelineConfig:
    """Resolve the harness pipeline through the SHARED CLI builder.

    This function is the bead-vj82 contract: the script may not construct
    detection/selection/solver objects itself. Tests assert the returned
    dataclass equals a CLI-built one for the same knobs.
    """
    cert_elems = list(BHVO2_BASALT_USGS)
    elements = list(cert_elems)
    if args.confounders:
        elements += [e for e in CONFOUNDERS if e not in cert_elems]

    excl = {"auto": None, "true": True, "false": False}[args.exclude_resonance]
    return build_pipeline_config(
        elements,
        preset=args.preset,
        closure_mode=args.closure_mode,
        saha_boltzmann_graph=args.saha_boltzmann_graph,
        apply_self_absorption=args.apply_self_absorption,
        min_relative_intensity=args.min_relative_intensity,
        exclude_resonance=excl,
        wavelength_calibration=args.wavelength_calibration,
        shift_coherence_veto=args.shift_coherence_veto,
        stark_ne=args.stark_ne,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    spec_path = Path(args.spectrum)
    if not spec_path.exists():
        spec_path = _REPO_ROOT / args.spectrum

    cert = dict(BHVO2_BASALT_USGS)
    cert_elems = list(cert.keys())

    pipeline = build_pipeline_from_args(args)

    wl, inten = load_spectrum(str(spec_path))
    db = AtomicDatabase(_resolve_db())

    result, diagnostics = run_pipeline(wl, inten, db, pipeline)

    # Stark n_e info for the JSON artifact (measured inside run_pipeline —
    # the bead-vj82 contract forbids this script from running pipeline
    # stages itself).
    stark_diag = diagnostics.get("stark_ne")
    if stark_diag is not None:
        stark_info: dict = {
            "enabled": True,
            "n_lines": stark_diag["n_lines"],
            "ne_cm3": stark_diag["ne_cm3"],
            "ne_scatter_cm3": stark_diag["ne_scatter_cm3"],
            "instrument_fwhm_source": stark_diag["instrument_fwhm_source"],
            "lines": [
                f"{m['element']} {'I' if m['ionization_stage'] == 1 else 'II'} "
                f"{m['wavelength_nm']:.2f} -> {m['ne_cm3']:.2e}"
                for m in stark_diag["lines"]
            ],
            "rejected": dict(stark_diag["rejected"]),
        }
    else:
        stark_info = {"enabled": bool(pipeline.stark_ne), "n_lines": 0}

    pred = dict(result.concentrations)
    obs_counts: dict[str, int] = dict(diagnostics["observation_counts"])
    n_obs = int(diagnostics["n_observations"])

    # --- scoring ---
    rmse = math.sqrt(sum((pred.get(e, 0.0) - cert[e]) ** 2 for e in cert_elems) / len(cert_elems))
    dropped = [e for e in cert_elems if pred.get(e, 0.0) < EPS_PRESENT]
    fps = {e: pred.get(e, 0.0) for e in CONFOUNDERS if pred.get(e, 0.0) >= EPS_PRESENT}

    cflibs_path = str(Path(cflibs.__file__).resolve().parent)
    print(f"\n{'='*72}\n  BHVO-2 PRESENCE MEASUREMENT  [{args.label}]")
    print(f"  cflibs={cflibs_path}")
    print(
        f"  spectrum={spec_path.name}  min_rel_int={pipeline.min_relative_intensity}  "
        f"exclude_resonance={pipeline.exclude_resonance}  SA={pipeline.apply_self_absorption}"
    )
    print(
        f"  preset={pipeline.preset}  closure={pipeline.closure_mode}  "
        f"saha_boltzmann_graph={pipeline.saha_boltzmann_graph}"
    )
    print(
        f"  wavelength_calibration={pipeline.wavelength_calibration}  "
        f"shift_coherence_veto={pipeline.shift_coherence_veto}  "
        f"confounders={bool(args.confounders)}"
    )
    print(
        f"  T={result.temperature_K:.0f}K  ne={result.electron_density_cm3:.2e}  "
        f"converged={result.converged}  n_obs={n_obs}"
    )
    ne_source = (
        "stark"
        if result.quality_metrics.get("ne_from_stark")
        else "pressure_balance_fallback (ASSUMED)"
    )
    print(
        f"  n_e source={ne_source}  stark_lines={stark_info.get('n_lines', 0)}  "
        f"scatter={stark_info.get('ne_scatter_cm3', 0.0) or 0.0:.2e}"
    )
    for line_desc in stark_info.get("lines", []) or []:
        print(f"    stark line: {line_desc}")
    print("=" * 72)
    print(f"  {'el':<4}{'cert(wt%)':>11}{'pred(wt%)':>11}{'err':>9}{'n_obs':>7}")
    for e in cert_elems:
        c, p = cert[e] * 100, pred.get(e, 0.0) * 100
        print(f"  {e:<4}{c:>11.3f}{p:>11.3f}{p-c:>9.3f}{obs_counts.get(e,0):>7}")
    print(f"\n  Al observation lines surviving detection: {obs_counts.get('Al',0)}")
    print(f"  RMSE (cert-10)        : {rmse*100:.3f} wt%")
    print(f"  Dropped majors (<{EPS_PRESENT}): {dropped}")
    print(f"  FP confounders present : { {k: round(v*100,3) for k,v in fps.items()} or 'NONE'}")
    # confounder obs counts even if below presence eps
    conf_obs = {e: obs_counts.get(e, 0) for e in CONFOUNDERS if obs_counts.get(e, 0)}
    print(f"  Confounder obs lines   : {conf_obs or 'NONE'}")

    summary = {
        "label": args.label,
        "spectrum": spec_path.name,
        "cflibs_path": cflibs_path,
        "rmse_wt%": rmse * 100,
        "knobs": {
            "preset": pipeline.preset,
            "closure_mode": pipeline.closure_mode,
            "saha_boltzmann_graph": pipeline.saha_boltzmann_graph,
            "wavelength_calibration": pipeline.wavelength_calibration,
            "shift_coherence_veto": pipeline.shift_coherence_veto,
            "apply_self_absorption": pipeline.apply_self_absorption,
            "exclude_resonance": pipeline.exclude_resonance,
            "min_relative_intensity": pipeline.min_relative_intensity,
            "confounders": bool(args.confounders),
            "elements": list(pipeline.elements),
        },
        "T_K": result.temperature_K,
        "ne": result.electron_density_cm3,
        "converged": result.converged,
        "pred": {e: pred.get(e, 0.0) for e in pipeline.elements},
        "ne_source": (
            "stark" if result.quality_metrics.get("ne_from_stark") else "pressure_balance"
        ),
        "stark_ne": stark_info,
        "obs_counts": obs_counts,
        "dropped": dropped,
        "fps": fps,
        "al_obs": obs_counts.get("Al", 0),
    }
    # Repo-local, user-owned output dir (not a world-writable shared /tmp path).
    out_dir = _REPO_ROOT / "output" / "bhvo2_measure"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.label}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
