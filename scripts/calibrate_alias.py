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
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

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


LoaderFn = Callable[[str], Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]


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

        try:
            wavelength, data, _ = loader(str(path))
            spectrum = select_representative_spectrum(data, ds["name"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping dataset %s due to load error: %s", ds["name"], exc)
            logger.debug("Traceback while loading dataset %s", ds["name"], exc_info=True)
            continue
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


# ---------------------------------------------------------------------------
# Aalto LIBS benchmark data (Pörö et al.)
# ---------------------------------------------------------------------------
# Major metallic/semi-metallic elements per mineral from standard formulae.
# Excludes O, H, C, N, S, F, Cl, B (weak/ambiguous LIBS emission in air).
MINERAL_COMPOSITIONS: Dict[str, List[str]] = {
    "adularia": ["K", "Al", "Si"],
    "almandine": ["Fe", "Al", "Si"],
    "apatite": ["Ca"],
    "augite": ["Ca", "Mg", "Fe", "Si"],
    "biotite": ["K", "Mg", "Fe", "Al", "Si"],
    "chalcopyrite": ["Cu", "Fe"],
    "cordierite": ["Mg", "Fe", "Al", "Si"],
    "corundum": ["Al"],
    "diopside": ["Ca", "Mg", "Si"],
    "fluorite": ["Ca"],
    "galena": ["Pb"],
    "garnet": ["Fe", "Al", "Si", "Ca"],
    "gypsum": ["Ca"],
    "hematite": ["Fe"],
    "hornblende": ["Ca", "Mg", "Fe", "Al", "Si"],
    "hypersthene": ["Mg", "Fe", "Si"],
    "kaolinite": ["Al", "Si"],
    "kyanite": ["Al", "Si"],
    "lepidolite": ["K", "Li", "Al", "Si"],
    "magnesite": ["Mg"],
    "magnetite": ["Fe"],
    "microcline": ["K", "Al", "Si"],
    "molybdenite": ["Mo"],
    "muscovite": ["K", "Al", "Si"],
    "olivine": ["Mg", "Fe", "Si"],
    "orthoclase": ["K", "Al", "Si"],
    "pentlandite": ["Fe", "Ni"],
    "phlogopite": ["K", "Mg", "Al", "Si"],
    "plagioclase": ["Na", "Ca", "Al", "Si"],
    "pyrite": ["Fe"],
    "pyrrhotite": ["Fe"],
    "quartz": ["Si"],
    "scapolite": ["Na", "Ca", "Al", "Si"],
    "serpentine": ["Mg", "Si"],
    "siderite": ["Fe"],
    "sphene": ["Ca", "Ti", "Si"],
    "spodumene": ["Li", "Al", "Si"],
    "staurolite": ["Fe", "Al", "Si"],
    "talc": ["Mg", "Si"],
    "topaz": ["Al", "Si"],
    "tourmaline": ["Na", "Fe", "Al", "Si"],
    "tremolite": ["Ca", "Mg", "Si"],
    "wollastonite": ["Ca", "Si"],
    "zircon": ["Zr", "Si"],
}

# Broad search list for Aalto benchmarks — all elements the DB likely covers.
AALTO_SEARCH_ELEMENTS = [
    "Fe",
    "Ca",
    "Mg",
    "Si",
    "Al",
    "Ti",
    "Na",
    "K",
    "Mn",
    "Cr",
    "Ni",
    "Cu",
    "Co",
    "V",
    "Li",
    "Sr",
    "Ba",
    "Zn",
    "Pb",
    "Mo",
    "Zr",
    "Sn",
]


def _load_aalto_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a 2-column (wavelength, spectrum) Aalto CSV file."""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def _select_aalto_cases(data_dir: Path) -> List[DatasetCase]:
    """Build benchmark cases from Aalto LIBS element and mineral spectra."""
    aalto_dir = data_dir / "aalto_libs"
    if not aalto_dir.exists():
        return []

    cases: List[DatasetCase] = []

    # Pure element spectra — ground truth is the element itself
    elem_dir = aalto_dir / "elements"
    if elem_dir.exists():
        for csv_path in sorted(elem_dir.glob("*_spectrum.csv")):
            element = csv_path.stem.replace("_spectrum", "")
            if element not in AALTO_SEARCH_ELEMENTS:
                continue
            try:
                wl, sp = _load_aalto_csv(str(csv_path))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping Aalto element %s: %s", element, exc)
                continue
            cases.append(
                DatasetCase(
                    name=f"aalto_elem_{element}",
                    path=csv_path,
                    elements=AALTO_SEARCH_ELEMENTS,
                    expected={element},
                    resolving_power=600.0,
                    wavelength=wl,
                    spectrum=sp,
                )
            )

    # Mineral spectra — ground truth from MINERAL_COMPOSITIONS
    mineral_dir = aalto_dir / "minerals"
    if mineral_dir.exists():
        for csv_path in sorted(mineral_dir.glob("*_spectrum.csv")):
            # Extract mineral name: e.g. "augite4_spectrum.csv" → "augite"
            stem = csv_path.stem.replace("_spectrum", "")
            # Strip trailing digits and sample IDs (e.g. "augite4", "adulariaE11")
            mineral = stem.rstrip("0123456789")
            if mineral.endswith("E"):
                mineral = mineral[:-1]
            mineral = mineral.lower()
            if mineral not in MINERAL_COMPOSITIONS:
                logger.warning("Unknown mineral %s from %s, skipping", mineral, csv_path.name)
                continue
            expected = set(MINERAL_COMPOSITIONS[mineral])
            # Only include expected elements that are in our search list
            expected &= set(AALTO_SEARCH_ELEMENTS)
            if not expected:
                continue
            try:
                wl, sp = _load_aalto_csv(str(csv_path))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping Aalto mineral %s: %s", csv_path.name, exc)
                continue
            cases.append(
                DatasetCase(
                    name=f"aalto_{stem}",
                    path=csv_path,
                    elements=AALTO_SEARCH_ELEMENTS,
                    expected=expected,
                    resolving_power=600.0,
                    wavelength=wl,
                    spectrum=sp,
                )
            )

    return cases


def _score_one(
    db: AtomicDatabase,
    case: DatasetCase,
    intensity_threshold_factor: float,
    detection_threshold: float,
    chance_window_scale: float,
    max_lines_per_element: int,
) -> Tuple[int, int, int, int, bool]:
    identifier = ALIASIdentifier(
        atomic_db=db,
        elements=case.elements,
        resolving_power=case.resolving_power,
        intensity_threshold_factor=float(intensity_threshold_factor),
        detection_threshold=float(detection_threshold),
        chance_window_scale=float(chance_window_scale),
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
    max_lines_per_element: Iterable[int],
    max_combinations: Optional[int] = None,
) -> List[SweepResult]:
    combo_iter = itertools.product(
        intensity_threshold_factors,
        detection_thresholds,
        chance_window_scales,
        max_lines_per_element,
    )
    if max_combinations is not None and max_combinations > 0:
        combos = list(itertools.islice(combo_iter, max_combinations))
    else:
        combos = list(combo_iter)

    results: List[SweepResult] = []
    for idx, (itf, dt, cws, mle) in enumerate(combos, start=1):
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
                    max_lines_per_element=int(mle),
                )
            except Exception as exc:
                logger.warning(
                    "Scoring failed for dataset %s with config %s: %s",
                    case.name,
                    f"itf={itf} dt={dt} cws={cws} mle={mle}",
                    exc,
                )
                logger.debug(
                    "Traceback for dataset %s config itf=%s dt=%s cws=%s mle=%s",
                    case.name,
                    itf,
                    dt,
                    cws,
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
            f"itf={itf:.2f} dt={dt:.3f} cws={cws:.2f} mle={mle} "
            f"-> F1={f1:.3f} P={precision:.3f} R={recall:.3f} FPR={fpr:.3f}"
        )

    results.sort(
        key=lambda r: (
            r.datasets_evaluated <= 0,
            -r.f1,
            -r.precision,
            -r.recall,
            r.fpr,
            -r.exact_matches,
        )
    )
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
    parser.add_argument(
        "--include-aalto",
        action="store_true",
        help="Include Aalto LIBS element and mineral spectra in the benchmark.",
    )
    parser.add_argument(
        "--aalto-only",
        action="store_true",
        help="Run ONLY Aalto LIBS spectra (skip lab datasets).",
    )

    parser.add_argument("--intensity-threshold-factors", type=str, default="3.0,3.5,4.0")
    parser.add_argument("--detection-thresholds", type=str, default="0.02,0.03,0.05")
    parser.add_argument("--chance-window-scales", type=str, default="0.3,0.4,0.5")
    parser.add_argument("--max-lines-per-element", type=str, default="20,30,50")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    db_path = Path(args.db_path)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    if not db_path.exists():
        raise FileNotFoundError(f"Atomic DB not found: {db_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    cases: List[DatasetCase] = []
    if not args.aalto_only:
        cases.extend(_select_cases(data_dir=data_dir, selected_datasets=args.datasets))
    if args.include_aalto or args.aalto_only:
        aalto_cases = _select_aalto_cases(data_dir=data_dir)
        cases.extend(aalto_cases)
        logger.info("Added %d Aalto benchmark cases", len(aalto_cases))
    if not cases:
        raise RuntimeError("No labeled datasets available for calibration.")

    print(f"Loaded {len(cases)} labeled datasets:")
    for c in cases:
        print(f"  - {c.name:<20} RP={c.resolving_power:>6.0f}  expected={sorted(c.expected)}")

    with AtomicDatabase(str(db_path)) as db:
        results = run_sweep(
            db=db,
            cases=cases,
            intensity_threshold_factors=_parse_float_list(args.intensity_threshold_factors),
            detection_thresholds=_parse_float_list(args.detection_thresholds),
            chance_window_scales=_parse_float_list(args.chance_window_scales),
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
            f"cws={r.chance_window_scale:.2f} mle={r.max_lines_per_element}"
        )

    best = results[0]
    print("\nRecommended defaults:")
    print(
        f"  intensity_threshold_factor={best.intensity_threshold_factor:.2f}, "
        f"detection_threshold={best.detection_threshold:.3f}, "
        f"chance_window_scale={best.chance_window_scale:.2f}, "
        f"max_lines_per_element={best.max_lines_per_element}"
    )

    json_path, csv_path = _write_outputs(results, output_dir)
    print(f"\nSaved results:\n  - {json_path}\n  - {csv_path}")


if __name__ == "__main__":
    main()
