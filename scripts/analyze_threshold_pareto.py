"""Pareto analysis of identifier thresholds from benchmark data."""
import csv
import json
from pathlib import Path
class ParetoAnalyzer:
    """Analyze current operating points and generate threshold recommendations."""
    CURRENT_THRESHOLDS = {
        "ALIAS": {"intensity_threshold_factor": 3.0, "detection_threshold": 0.03},
        "Comb": {"min_correlation": 0.10, "tooth_activation_threshold": 0.5},
        "Correlation": {"min_confidence": 0.03},
    }
    @staticmethod
    def load_aggregate(bench_dir):
        path = Path(bench_dir) / "aggregate_metrics.csv"
        rows = []
        with open(path) as f:
            for row in csv.DictReader(f):
                rows.append(row)
        return rows
    @staticmethod
    def load_per_element(bench_dir):
        path = Path(bench_dir) / "per_element_metrics.csv"
        rows = []
        with open(path) as f:
            for row in csv.DictReader(f):
                rows.append(row)
        return rows
    @staticmethod
    def analyze_fp_sources(per_element):
        fp_sources = {}
        for row in per_element:
            algo = row["algorithm"]
            elem = row["element"]
            fp = int(row["fp"])
            if fp > 0:
                if algo not in fp_sources:
                    fp_sources[algo] = {}
                fp_sources[algo][elem] = fp
        return fp_sources
    @staticmethod
    def generate_report(bench_dir, output_path):
        bench_dir = Path(bench_dir)
        agg = ParetoAnalyzer.load_aggregate(bench_dir)
        per_elem = ParetoAnalyzer.load_per_element(bench_dir)
        fp_sources = ParetoAnalyzer.analyze_fp_sources(per_elem)
        current_ops = {}
        for row in agg:
            algo = row["algorithm"]
            current_ops[algo] = {
                "f1": round(float(row["f1"]), 4),
                "precision": round(float(row["precision"]), 4),
                "recall": round(float(row["recall"]), 4),
                "fpr": round(float(row["fpr"]), 4),
            }
        recommendations = {}
        for algo in ParetoAnalyzer.CURRENT_THRESHOLDS:
            ops = current_ops.get(algo, {})
            fps = fp_sources.get(algo, {})
            top_fp = sorted(fps.items(), key=lambda x: -x[1])[:3]
            if ops.get("fpr", 0) > 0.15:
                rec = {"action": "tighten_threshold", "reason": f"FPR={ops.get("fpr",0):.1%} exceeds 15% gate", "top_fp_elements": dict(top_fp)}
            else:
                rec = {"action": "keep_current", "reason": f"FPR={ops.get("fpr",0):.1%} acceptable", "top_fp_elements": dict(top_fp)}
            recommendations[algo] = rec
        report = {
            "version": "v1.0",
            "benchmark_dir": str(bench_dir),
            "current_thresholds": ParetoAnalyzer.CURRENT_THRESHOLDS,
            "current_operating_points": current_ops,
            "false_positive_sources": fp_sources,
            "recommendations": recommendations,
            "note": "Full Pareto sweep requires parameterized benchmark re-run with threshold grid. This report documents the current single operating point and FP analysis.",
        }
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))
        print(f"Report: {output_path}")
        for algo, rec in recommendations.items():
            print(f"  {algo}: {rec["action"]} ({rec["reason"]})")
        return report
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pareto threshold analysis")
    parser.add_argument("--bench-dir", type=Path, default=Path("output/synthetic_benchmark/postmerge_synth_v1_auto_24"))
    parser.add_argument("--output", type=Path, default=Path("output/validation/threshold_pareto_report.json"))
    args = parser.parse_args()
    ParetoAnalyzer.generate_report(args.bench_dir, args.output)
if __name__ == "__main__":
    main()
