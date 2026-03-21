"""Analyze calibration stress from synthetic benchmark outputs."""
import csv
import json
from pathlib import Path
class ShiftAnalysis:
    """Analyze identifier performance vs global wavelength shift."""
    @staticmethod
    def from_csv(path):
        rows = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows
    @staticmethod
    def analyze(bench_dir):
        shift_csv = bench_dir / "group_metrics_shift_nm.csv"
        if not shift_csv.exists():
            return {"status": "no_data", "path": str(shift_csv)}
        rows = ShiftAnalysis.from_csv(shift_csv)
        by_algo = {}
        for row in rows:
            algo = row["algorithm"]
            shift = float(row["group_value"])
            f1 = float(row["f1"])
            prec = float(row["precision"])
            rec = float(row["recall"])
            fpr = float(row["fpr"])
            if algo not in by_algo:
                by_algo[algo] = []
            by_algo[algo].append({"shift_nm": shift, "f1": round(f1, 4), "precision": round(prec, 4), "recall": round(rec, 4), "fpr": round(fpr, 4)})
        recommendations = {}
        for algo, points in by_algo.items():
            baseline = next((p for p in points if p["shift_nm"] == 0.0), None)
            if baseline:
                degradation = {p["shift_nm"]: round(baseline["f1"] - p["f1"], 4) for p in points}
                max_deg = max(abs(v) for v in degradation.values())
                recommendations[algo] = {"baseline_f1": baseline["f1"], "degradation_by_shift": degradation, "max_degradation": round(max_deg, 4), "robust": max_deg < 0.15}
        return {"by_algorithm": by_algo, "recommendations": recommendations}
def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Calibration stress analysis")
    parser.add_argument("--bench-dir", type=Path, default=Path("output/synthetic_benchmark/postmerge_synth_v1_auto_24"))
    parser.add_argument("--output", type=Path, default=Path("output/validation/calibration_stress_report.json"))
    args = parser.parse_args()
    shift_results = ShiftAnalysis.analyze(args.bench_dir)
    warp_csv = args.bench_dir / "group_metrics_warp_quadratic_nm.csv"
    warp_results = {"status": "single_value_only"} if warp_csv.exists() else {"status": "no_data"}
    report = {"benchmark_dir": str(args.bench_dir), "shift_analysis": shift_results, "warp_analysis": warp_results, "calibration_recommendations": _build_recommendations(shift_results)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"Report: {args.output}")
    recs = report.get("calibration_recommendations", {})
    for algo, rec in recs.items():
        print(f"  {algo}: baseline F1={rec.get('baseline_f1', 'N/A')}, max_degradation={rec.get('max_degradation', 'N/A')}, robust={rec.get('robust', False)}")
    return 0
def _build_recommendations(shift_results):
    recs = shift_results.get("recommendations", {})
    summary = {}
    for algo, data in recs.items():
        if data["robust"]:
            summary[algo] = {"strategy": "global_shift_sufficient", "note": f"Max F1 degradation {data['max_degradation']:.2%} under +/-1nm shift"}
        else:
            summary[algo] = {"strategy": "needs_careful_calibration", "note": f"Max F1 degradation {data['max_degradation']:.2%} under +/-1nm shift — consider higher-order correction"}
    return summary
if __name__ == "__main__":
    main()
