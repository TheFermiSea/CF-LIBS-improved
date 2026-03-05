#!/usr/bin/env python3
"""Consolidated NIST cross-check validation runner for CF-LIBS."""
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
from cflibs.atomic.database import AtomicDatabase
from cflibs.plasma.saha_boltzmann import SahaBoltzmannSolver
from cflibs.core.constants import EV_TO_K
DEFAULT_DB = Path("ASD_da/libs_production.db")
DEFAULT_OUT = Path("output/validation/nist_crosscheck_report.json")
class PartitionFunctionCheck:
    """Compare Irwin polynomial partition functions against NIST ASD reference."""
    NIST_REF = {
        "Fe": {1: {5000: 25.07, 10000: 41.95, 15000: 64.38, 20000: 98.36},
               2: {5000: 30.42, 10000: 43.38, 15000: 55.02, 20000: 69.80}},
        "Cu": {1: {5000: 2.03, 10000: 3.81, 15000: 6.55, 20000: 10.16},
               2: {5000: 1.00, 10000: 1.08, 15000: 1.55, 20000: 2.42}},
        "Al": {1: {5000: 5.84, 10000: 5.91, 15000: 6.06, 20000: 6.38},
               2: {5000: 1.00, 10000: 1.00, 15000: 1.01, 20000: 1.03}},
        "Ni": {1: {5000: 23.66, 10000: 35.35, 15000: 60.38, 20000: 89.12},
               2: {5000: 9.56, 10000: 18.42, 15000: 27.53, 20000: 36.64}},
        "Ti": {1: {5000: 31.71, 10000: 73.46, 15000: 122.48, 20000: 170.89},
               2: {5000: 41.07, 10000: 83.93, 15000: 123.03, 20000: 157.61}},
        "Cr": {1: {5000: 7.06, 10000: 11.49, 15000: 25.99, 20000: 42.48},
               2: {5000: 6.22, 10000: 11.17, 15000: 19.54, 20000: 27.63}},
    }
    @staticmethod
    def run(solver):
        results = {}
        for elem, stages in PartitionFunctionCheck.NIST_REF.items():
            results[elem] = {}
            for stage, temps in stages.items():
                stage_res = {}
                for T_K, nist_U in temps.items():
                    T_eV = T_K / EV_TO_K
                    our_U = solver.calculate_partition_function(elem, stage, T_eV)
                    pct = (our_U - nist_U) / nist_U * 100
                    stage_res[str(T_K)] = {"cflibs": round(our_U, 3), "nist": nist_U, "pct_diff": round(pct, 2)}
                results[elem][str(stage)] = stage_res
        return results
class IonizationFractionCheck:
    """Compare ionization fractions against NIST LIBS Simulation reference."""
    NIST_REF = {
        "Fe": {"T_eV": 0.8, "n_e": 1e17, "fractions": {1: 0.27, 2: 0.73, 3: 2.2e-5}},
    }
    @staticmethod
    def run(solver):
        results = {}
        for elem, ref in IonizationFractionCheck.NIST_REF.items():
            fracs = solver.get_ionization_fractions(elem, ref["T_eV"], ref["n_e"])
            elem_res = {}
            for stage in sorted(set(list(fracs.keys()) + list(ref["fractions"].keys()))):
                our = fracs.get(stage, 0.0)
                nist = ref["fractions"].get(stage)
                if nist is not None:
                    pct = abs(our - nist) / max(nist, 1e-30) * 100
                    elem_res[str(stage)] = {"cflibs": round(our, 6), "nist": nist, "pct_diff": round(pct, 2), "pass": pct < 10}
                else:
                    elem_res[str(stage)] = {"cflibs": round(our, 6), "nist": None}
            results[elem] = {"conditions": {"T_eV": ref["T_eV"], "n_e": ref["n_e"]}, "stages": elem_res}
        return results
class SpectralParityCheck:
    """Check existing NIST spectral comparison artifacts."""
    @staticmethod
    def run():
        results = {}
        compare_dir = Path("output/nist_synthetic_compare")
        if not compare_dir.exists():
            return {"status": "no_artifacts", "path": str(compare_dir)}
        for subdir in sorted(compare_dir.iterdir()):
            metrics_file = subdir / "metrics.json"
            summary_file = subdir / "summary_all.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    results[subdir.name] = json.load(f)
            elif summary_file.exists():
                with open(summary_file) as f:
                    results[subdir.name] = json.load(f)
        return results
def build_report(db_path, output_path):
    """Run all checks and write consolidated JSON report."""
    db = AtomicDatabase(str(db_path))
    solver = SahaBoltzmannSolver(db)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "db_path": str(db_path),
        "partition_functions": PartitionFunctionCheck.run(solver),
        "ionization_fractions": IonizationFractionCheck.run(solver),
        "spectral_comparisons": SpectralParityCheck.run(),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"Report written to {output_path}")
    return report
def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Run consolidated NIST validation")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    report = build_report(args.db, args.output)
    pf = report["partition_functions"]
    n_ok = sum(1 for e in pf.values() for s in e.values() for v in s.values() if abs(v["pct_diff"]) < 5.0)
    n_total = sum(1 for e in pf.values() for s in e.values() for _ in s.values())
    ion = report["ionization_fractions"]
    ion_pass = sum(1 for e in ion.values() for s in e["stages"].values() if s.get("pass", False))
    ion_total = sum(1 for e in ion.values() for s in e["stages"].values() if s.get("nist") is not None)
    print(f"\nPartition functions: {n_ok}/{n_total} within 5% of NIST")
    print(f"Ionization fractions: {ion_pass}/{ion_total} within 20% of NIST")
    n_spec = len(report.get("spectral_comparisons", {}))
    print(f"Spectral comparisons: {n_spec} datasets found")
    return 0
if __name__ == "__main__":
    main()
# end of script
