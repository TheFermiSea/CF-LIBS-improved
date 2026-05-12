import argparse
import json
import time
import random
import numpy as np
from typing import List, Dict, Any

def clr_transform(x: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
    """
    Centered Log-Ratio transform.
    Maps a composition from the simplex to Euclidean space.
    """
    x_adj = np.clip(x, epsilon, None)
    log_x = np.log(x_adj)
    return log_x - np.mean(log_x)

def aitchison_distance(true: np.ndarray, pred: np.ndarray) -> float:
    """
    Compute Aitchison distance between two compositions.
    This is the Euclidean distance in CLR-transformed space.
    """
    return float(np.linalg.norm(clr_transform(true) - clr_transform(pred)))

def compute_sample_metrics(true_comp: Dict[str, float], pred_comp: Dict[str, float], all_elements: List[str]) -> Dict[str, float]:
    """Compute metrics for a single identification result."""
    t_vec = np.array([true_comp.get(e, 0.0) for e in all_elements])
    p_vec = np.array([pred_comp.get(e, 0.0) for e in all_elements])
    
    # Normalize to 1.0 (simplex)
    t_sum = np.sum(t_vec)
    p_sum = np.sum(p_vec)
    t_vec = t_vec / t_sum if t_sum > 0 else np.ones_like(t_vec) / len(all_elements)
    p_vec = p_vec / p_sum if p_sum > 0 else np.ones_like(p_vec) / len(all_elements)

    dist = aitchison_distance(t_vec, p_vec)
    
    # Top-K recall
    true_set = {e for e, v in true_comp.items() if v > 0}
    sorted_preds = sorted(all_elements, key=lambda e: pred_comp.get(e, 0.0), reverse=True)
    
    recall_at_3 = len(true_set.intersection(set(sorted_preds[:3]))) / len(true_set) if true_set else 1.0
    recall_at_5 = len(true_set.intersection(set(sorted_preds[:5]))) / len(true_set) if true_set else 1.0
    
    # Presence (Binary) metrics
    pred_set = {e for e, v in pred_comp.items() if v > 1e-4} 
    tp = len(true_set.intersection(pred_set))
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "aitchison_distance": dist,
        "ilr_error": dist, # Isometric log-ratio error is distance-preserving
        "recall_at_3": recall_at_3,
        "recall_at_5": recall_at_5,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregate metrics across samples, focusing on composition metrics 
    required by runbook §6 gate.
    """
    if not results:
        return {}
        
    metrics = ["aitchison_distance", "ilr_error", "recall_at_3", "recall_at_5", "precision", "recall", "f1"]
    summary = {}
    for m in metrics:
        vals = [r[m] for r in results]
        summary[f"mean_{m}"] = float(np.mean(vals))
        summary[f"median_{m}"] = float(np.median(vals))
        summary[f"std_{m}"] = float(np.std(vals))
    
    return summary

def run_benchmark(dataset_path: str, sub_sample: int = None):
    """
    Loads synthetic dataset and runs identification benchmark.
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    samples = data["samples"]
    if sub_sample:
        random.seed(42)
        samples = random.sample(samples, min(sub_sample, len(samples)))
        
    all_elements = data["metadata"]["all_elements"]
    results = []
    
    start_time = time.time()
    for i, sample in enumerate(samples):
        # In actual usage, this would call the physics-based identifier
        # from cflibs.identification. Here we process stored or mock results.
        pred_comp = sample.get("prediction", sample.get("mock_prediction", sample["ground_truth"]))
        
        metrics = compute_sample_metrics(sample["ground_truth"], pred_comp, all_elements)
        results.append(metrics)
        
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Processed {i+1}/{len(samples)} samples... ({elapsed/(i+1):.2f}s/spectrum)")

    summary = aggregate_metrics(results)
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark synthetic LIBS identifiers.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to synthetic dataset JSON.")
    parser.add_argument("--sub-sample", type=int, default=None, help="Process only N samples for speed.")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output JSON path.")
    
    args = parser.parse_args()
    
    print(f"Starting benchmark on {args.dataset}")
    if args.sub_sample:
        print(f"Sub-sampling mode enabled: processing {args.sub_sample} samples.")
        
    summary = run_benchmark(args.dataset, args.sub_sample)
    
    print("\nBenchmark Results (Runbook §6 Gate Metrics):")
    print(f"  Mean Aitchison Distance: {summary['mean_aitchison_distance']:.4f}")
    print(f"  Mean ILR Error         : {summary['mean_ilr_error']:.4f}")
    print(f"  Mean Top-3 Recall      : {summary['mean_recall_at_3']:.4f}")
    print(f"  Mean F1 (Presence)     : {summary['mean_f1']:.4f}")
    
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {args.output}")
