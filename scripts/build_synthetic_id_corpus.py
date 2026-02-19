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

from cflibs.benchmark.synthetic_corpus import PerturbationAxes, build_synthetic_id_corpus


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

    summary = build_synthetic_id_corpus(
        db_path=args.db_path,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        seed=args.seed,
        candidate_elements=_parse_csv_str_list(args.elements),
        wavelength_min_nm=args.lambda_min,
        wavelength_max_nm=args.lambda_max,
        pixels=args.pixels,
        temperature_range_eV=temperature_range_eV,
        log_ne_range=log_ne_range,
        axes=PerturbationAxes(
            snr_db=_parse_csv_float_list(args.snr_db),
            continuum_level=_parse_csv_float_list(args.continuum_level),
            resolving_power=_parse_csv_float_list(args.resolving_power),
            shift_nm=_parse_csv_float_list(args.shift_nm),
            warp_quadratic_nm=_parse_csv_float_list(args.warp_quadratic_nm),
        ),
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
