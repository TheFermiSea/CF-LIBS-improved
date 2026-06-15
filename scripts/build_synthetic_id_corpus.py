#!/usr/bin/env python
"""
Build deterministic synthetic corpus for CF-LIBS spectral-ID debugging.
"""

from __future__ import annotations

import argparse
import json
import os

# Keep JAX on CPU for stable local/CI execution.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from cflibs.benchmark.synthetic_corpus import (
    PerturbationAxes,
    build_synthetic_id_corpus,
    default_recipes,
    diagnostic_recipes,
)


def _parse_csv_str_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _parse_csv_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _parse_range(name: str, value: str) -> list[float]:
    values = _parse_csv_float_list(value)
    if len(values) != 2:
        raise argparse.ArgumentTypeError(
            f"{name} requires exactly two comma-separated values: min,max"
        )
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=str, default="ASD_da/libs_production.db")
    parser.add_argument("--output-dir", type=str, default="output/synthetic_corpus")
    parser.add_argument("--dataset-name", type=str, default="ak3_1_3_corpus_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--elements",
        type=str,
        default="Fe,Ni,Cr,Mn,Cu,Ti,Si,Al,V,Mg,Co",
        help="Comma-separated candidate elements",
    )
    parser.add_argument("--lambda-min", type=float, default=224.6)
    parser.add_argument("--lambda-max", type=float, default=265.3)
    parser.add_argument("--pixels", type=int, default=2560)
    parser.add_argument("--temperature-range-eV", type=str, default="0.8,1.8")
    parser.add_argument("--log-ne-range", type=str, default="16.3,18.0")
    parser.add_argument(
        "--recipe-set",
        type=str,
        default="default",
        choices=["default", "diagnostic"],
        help=(
            "Recipe set to build. 'default' = legacy 4-recipe ak3.1.3 set "
            "(only Fe/Ni/Cr/Mn). 'diagnostic' = balanced 23-recipe w3 set "
            "(11 pure controls + Fe/Ni binary + 11 alloy matrices) where every "
            "panel element appears in >=4 recipes."
        ),
    )
    parser.add_argument(
        "--version",
        type=str,
        default="ak3.1.3",
        help="Version string stamped on the BenchmarkDataset (e.g. ak3.2.0 for the diagnostic set).",
    )
    parser.add_argument(
        "--max-spectra",
        type=int,
        default=0,
        help=(
            "If > 0, cap the corpus to the first N recipe/perturbation spectra "
            "(smoke/dry-run only). 0 = build the full corpus."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the resolved build plan (window, panel, recipe-set, recipe names, "
            "perturbation count, version) as JSON and exit without running the forward model."
        ),
    )

    # Controlled axes (full factorial)
    parser.add_argument("--snr-db", type=str, default="20,30,40")
    parser.add_argument("--continuum-level", type=str, default="0.00,0.03")
    parser.add_argument("--resolving-power", type=str, default="700,1000")
    parser.add_argument("--shift-nm", type=str, default="-1.0,0.0,1.0")
    parser.add_argument("--warp-quadratic-nm", type=str, default="0.0,0.15")

    args = parser.parse_args()
    try:
        temperature_range_eV = _parse_range("--temperature-range-eV", args.temperature_range_eV)
        log_ne_range = _parse_range("--log-ne-range", args.log_ne_range)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    candidate_elements = _parse_csv_str_list(args.elements)
    axes = PerturbationAxes(
        snr_db=_parse_csv_float_list(args.snr_db),
        continuum_level=_parse_csv_float_list(args.continuum_level),
        resolving_power=_parse_csv_float_list(args.resolving_power),
        shift_nm=_parse_csv_float_list(args.shift_nm),
        warp_quadratic_nm=_parse_csv_float_list(args.warp_quadratic_nm),
    )

    if args.recipe_set == "diagnostic":
        recipes = diagnostic_recipes(candidate_elements)
    else:
        recipes = default_recipes(candidate_elements)
    if not recipes:
        parser.error(f"No valid recipes for set '{args.recipe_set}' given elements {args.elements}")

    n_perturb = (
        len(axes.snr_db)
        * len(axes.continuum_level)
        * len(axes.resolving_power)
        * len(axes.shift_nm)
        * len(axes.warp_quadratic_nm)
    )
    full_n = len(recipes) * n_perturb
    planned_n = min(full_n, args.max_spectra) if args.max_spectra > 0 else full_n

    if args.dry_run:
        plan = {
            "dry_run": True,
            "dataset_name": args.dataset_name,
            "version": args.version,
            "recipe_set": args.recipe_set,
            "n_recipes": len(recipes),
            "recipe_names": [r.name for r in recipes],
            "candidate_elements": candidate_elements,
            "requested_window_nm": [args.lambda_min, args.lambda_max],
            "pixels": args.pixels,
            "n_perturbation_combinations": n_perturb,
            "n_spectra_full": full_n,
            "n_spectra_planned": planned_n,
            "max_spectra": args.max_spectra,
            "resolving_power": _parse_csv_float_list(args.resolving_power),
            "output_dir": f"{args.output_dir}/{args.dataset_name}",
        }
        print(json.dumps(plan, indent=2))
        return

    summary = build_synthetic_id_corpus(
        db_path=args.db_path,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        seed=args.seed,
        candidate_elements=candidate_elements,
        wavelength_min_nm=args.lambda_min,
        wavelength_max_nm=args.lambda_max,
        pixels=args.pixels,
        temperature_range_eV=temperature_range_eV,
        log_ne_range=log_ne_range,
        recipes=recipes,
        axes=axes,
        version=args.version,
        max_spectra=args.max_spectra,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
