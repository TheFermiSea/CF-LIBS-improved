#!/usr/bin/env python
"""
Grid-search ALIAS parameters on labeled real datasets.

This script calibrates false-positive/false-negative behavior by sweeping
key ALIAS thresholds and ranking configurations by global F1.
"""

import argparse
import csv
import itertools
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Keep consistent with existing scripts.
os.environ["JAX_PLATFORMS"] = "cpu"

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cflibs.atomic.database import AtomicDatabase  # noqa: E402
from cflibs.inversion.alias_identifier import ALIASIdentifier  # noqa: E402

from scripts.validate_real_data import (  # noqa: E402
    DATASETS,
    estimate_resolving_power,
    load_hdf5,
    load_hdf5_multishot,
    load_netcdf,
    load_scipp,
    load_scipp_depth_scan,
    select_representative_spectrum,
)


LoaderFn = Callable[[str], Tuple[np.ndarray, np.ndarray, Dict]]


@dataclass
class DatasetCase:
    name: str
    path: Path
    elements: List[str]
    expected: Set[str]
    resolving_power: float
    wavelength: np.ndarray = field(repr=False)
    spectrum: np.ndarray = field(repr=False)


@dataclass
class SweepResult:
    intensity_threshold_factor: float
    detection_threshold: float
    chance_window_scale: float
    min_relative_intensity: float
    max_lines_per_element: int
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    fpr: float
    exact_matches: int
    datasets_evaluated: int


def _parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _safe_ratio(n: float, d: float) -> float:
    return n / d if d > 0 else 0.0


def _build_loader_map() -> Dict[str, LoaderFn]:
    return {
        "netcdf": load_netcdf,
        "hdf5": load_hdf5,
        "hdf5_multishot": load_hdf5_multishot,
        "scipp": load_scipp,
        "scipp_depth_scan": load_scipp_depth_scan,
    }


def _select_cases(
    data_dir: Path,
    selected_datasets: Optional[Sequence[str]],
) -> List[DatasetCase]:
    loaders = _build_loader_map()
    selected = set(selected_datasets) if selected_datasets else None
    cases: List[DatasetCase] = []

    for ds in DATASETS:
        if ds.get("expected") is None:
            continue
        if selected is not None and ds["name"] not in selected:
            continue

        loader_name = ds["loader"]
        loader = loaders.get(loader_name)
        if loader is None:
            continue

        path = data_dir / ds["path"]
        if not path.exists():
            continue

        wavelength, data, _ = loader(str(path))
        spectrum = select_representative_spectrum(data, ds["name"])
        if "resolving_power" in ds:
            rp = float(ds["resolving_power"])
        else:
            rp = float(estimate_resolving_power(wavelength, spectrum))

        cases.append(
            DatasetCase(
                name=ds["name"],
                path=path,
                elements=list(ds["elements"]),
                expected=set(ds["expected"]),
                resolving_power=rp,
                wavelength=wavelength,
                spectrum=spectrum,
            )
        )

    return cases


def _score_one(
    db: AtomicDatabase,
    case: DatasetCase,
    intensity_threshold_factor: float,
    detection_threshold: float,
    chance_window_scale: float,
    min_relative_intensity: float,
    max_lines_per_element: int,
) -> Tuple[int, int, int, int, bool]:
    identifier = ALIASIdentifier(
        atomic_db=db,
        elements=case.elements,
        resolving_power=case.resolving_power,
        intensity_threshold_factor=float(intensity_threshold_factor),
        detection_threshold=float(detection_threshold),
        chance_window_scale=float(chance_window_scale),
        min_relative_intensity=float(min_relative_intensity),
        max_lines_per_element=int(max_lines_per_element),
    )
    result = identifier.identify(case.wavelength, case.spectrum)
    detected = {e.element for e in result.detected_elements}
    searched = set(case.elements)

    tp = len(detected & case.expected)
    fp = len(detected - case.expected)
    fn = len(case.expected - detected)
    tn = len((searched - case.expected) - detected)
    exact = detected == case.expected
    return tp, fp, fn, tn, exact


def run_sweep(
    db: AtomicDatabase,
    cases: List[DatasetCase],
    intensity_threshold_factors: Iterable[float],
    detection_thresholds: Iterable[float],
    chance_window_scales: Iterable[float],
    min_relative_intensities: Iterable[float],
    max_lines_per_element: Iterable[int],
    max_combinations: Optional[int] = None,
) -> List[SweepResult]:
    combo_iter = itertools.product(
        intensity_threshold_factors,
        detection_thresholds,
        chance_window_scales,
        min_relative_intensities,
        max_lines_per_element,
    )
    if max_combinations is not None and max_combinations > 0:
        combos = list(itertools.islice(combo_iter, max_combinations))
    else:
        combos = list(combo_iter)

    results: List[SweepResult] = []
    for idx, (itf, dt, cws, mri, mle) in enumerate(combos, start=1):
        tp = fp = fn = tn = 0
        exact_matches = 0
        evaluated = 0

        for case in cases:
            try:
                ctp, cfp, cfn, ctn, exact = _score_one(
                    db=db,
                    case=case,
                    intensity_threshold_factor=float(itf),
                    detection_threshold=float(dt),
                    chance_window_scale=float(cws),
                    min_relative_intensity=float(mri),
                    max_lines_per_element=int(mle),
                )
            except Exception as exc:
                logger.warning(
                    "Scoring failed for dataset %s with config %s: %s",
                    case.name,
                    f"itf={itf} dt={dt} cws={cws} mri={mri} mle={mle}",
                    exc,
                )
                logger.debug(
                    "Traceback for dataset %s config itf=%s dt=%s cws=%s mri=%s mle=%s",
                    case.name,
                    itf,
                    dt,
                    cws,
                    mri,
                    mle,
                    exc_info=True,
                )
                continue
            tp += ctp
            fp += cfp
            fn += cfn
            tn += ctn
            exact_matches += int(exact)
            evaluated += 1

        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        f1 = _safe_ratio(2 * precision * recall, precision + recall)
        fpr = _safe_ratio(fp, fp + tn)

        results.append(
            SweepResult(
                intensity_threshold_factor=float(itf),
                detection_threshold=float(dt),
                chance_window_scale=float(cws),
                min_relative_intensity=float(mri),
                max_lines_per_element=int(mle),
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn,
                precision=precision,
                recall=recall,
                f1=f1,
                fpr=fpr,
                exact_matches=exact_matches,
                datasets_evaluated=evaluated,
            )
        )
        print(
            f"[{idx:>3}/{len(combos)}] "
            f"itf={itf:.2f} dt={dt:.3f} cws={cws:.2f} mri={mri:.0f} mle={mle} "
            f"-> F1={f1:.3f} P={precision:.3f} R={recall:.3f} FPR={fpr:.3f}"
        )

    results.sort(key=lambda r: (-r.f1, -r.precision, -r.recall, r.fpr, -r.exact_matches))
    return results


def _write_outputs(results: List[SweepResult], output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "alias_calibration_results.json"
    csv_path = output_dir / "alias_calibration_results.csv"

    payload = [asdict(r) for r in results]
    json_path.write_text(json.dumps(payload, indent=2))

    if payload:
        headers = list(payload[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows([row[h] for h in headers] for row in payload)
    else:
        csv_path.write_text("")

    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate ALIAS thresholds using labeled datasets."
    )
    parser.add_argument("--db-path", type=str, default="ASD_da/libs_production.db")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="output/calibration")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--max-combinations", type=int, default=0)

    parser.add_argument("--intensity-threshold-factors", type=str, default="3.5,4.0,4.5")
    parser.add_argument("--detection-thresholds", type=str, default="0.03,0.04,0.05")
    parser.add_argument("--chance-window-scales", type=str, default="0.3,0.4,0.5")
    parser.add_argument("--min-relative-intensities", type=str, default="50,100,200")
    parser.add_argument("--max-lines-per-element", type=str, default="30,50")

    args = parser.parse_args()

    db_path = Path(args.db_path)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    if not db_path.exists():
        raise FileNotFoundError(f"Atomic DB not found: {db_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    db = AtomicDatabase(str(db_path))
    cases = _select_cases(data_dir=data_dir, selected_datasets=args.datasets)
    if not cases:
        raise RuntimeError("No labeled datasets available for calibration.")

    print(f"Loaded {len(cases)} labeled datasets:")
    for c in cases:
        print(f"  - {c.name:<20} RP={c.resolving_power:>6.0f}  expected={sorted(c.expected)}")

    results = run_sweep(
        db=db,
        cases=cases,
        intensity_threshold_factors=_parse_float_list(args.intensity_threshold_factors),
        detection_thresholds=_parse_float_list(args.detection_thresholds),
        chance_window_scales=_parse_float_list(args.chance_window_scales),
        min_relative_intensities=_parse_float_list(args.min_relative_intensities),
        max_lines_per_element=_parse_int_list(args.max_lines_per_element),
        max_combinations=args.max_combinations if args.max_combinations > 0 else None,
    )

    if not results:
        raise RuntimeError("Sweep produced no valid results.")

    top_k = min(max(args.top_k, 1), len(results))
    print("\nTop configurations:")
    for i, r in enumerate(results[:top_k], start=1):
        print(
            f"{i:>2}. F1={r.f1:.3f} P={r.precision:.3f} R={r.recall:.3f} FPR={r.fpr:.3f} "
            f"exact={r.exact_matches}/{r.datasets_evaluated} "
            f"| itf={r.intensity_threshold_factor:.2f} dt={r.detection_threshold:.3f} "
            f"cws={r.chance_window_scale:.2f} mri={r.min_relative_intensity:.0f} "
            f"mle={r.max_lines_per_element}"
        )

    best = results[0]
    print("\nRecommended defaults:")
    print(
        f"  intensity_threshold_factor={best.intensity_threshold_factor:.2f}, "
        f"detection_threshold={best.detection_threshold:.3f}, "
        f"chance_window_scale={best.chance_window_scale:.2f}, "
        f"min_relative_intensity={best.min_relative_intensity:.0f}, "
        f"max_lines_per_element={best.max_lines_per_element}"
    )

    json_path, csv_path = _write_outputs(results, output_dir)
    print(f"\nSaved results:\n  - {json_path}\n  - {csv_path}")


if __name__ == "__main__":
    main()
