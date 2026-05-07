import argparse
import json
import sys
from typing import Dict

def evaluate_gate(candidate_results: Dict, adversarial_results: Dict, baseline_results: Dict) -> bool:
    """
    Evaluates whether a PR should be blocked based on adversarial overlap.
    Returns True if the PR passes, False if it is blocked.
    """
    baseline_da = baseline_results.get("aggregate_da", 1.0)
    candidate_da = candidate_results.get("aggregate_da", 1.0)
    adversarial_da = adversarial_results.get("aggregate_da", 1.0)
    
    candidate_gain = baseline_da - candidate_da
    adversarial_gain = baseline_da - adversarial_da
    
    print(f"--- Benchmark Gate Evaluation ---")
    print(f"Baseline dA:    {baseline_da:.4f}")
    print(f"Candidate dA:   {candidate_da:.4f} (Gain: {candidate_gain:.4f})")
    print(f"Adversarial dA: {adversarial_da:.4f} (Gain: {adversarial_gain:.4f})")
    
    if candidate_gain <= 0:
        print("REJECTED: Candidate failed to show improvement over baseline.")
        return False

    # Acceptance Criteria: if adversarial twin's gains overlap candidate's by > 50%, block merge
    if candidate_gain > 0:
        overlap_ratio = adversarial_gain / candidate_gain
        print(f"Adversarial Overlap Ratio: {overlap_ratio:.2%}")
        
        if overlap_ratio > 0.50:
            print(f"REJECTED: Adversarial overlap > 50% ({overlap_ratio:.2%}).")
            print("Possible Goodharting/gaming detected.")
            return False
            
    print("PASSED: Candidate improvement exceeds adversarial gaming potential.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, help="Path to candidate benchmark results JSON")
    parser.add_argument("--adversarial", required=True, help="Path to adversarial benchmark results JSON")
    parser.add_argument("--baseline", required=True, help="Path to baseline benchmark results JSON")
    
    args = parser.parse_args()
    
    try:
        with open(args.candidate) as f:
            cand = json.load(f)
        with open(args.adversarial) as f:
            adv = json.load(f)
        with open(args.baseline) as f:
            base = json.load(f)
    except Exception as e:
        print(f"Error loading benchmark results: {e}")
        sys.exit(1)
        
    if not evaluate_gate(cand, adv, base):
        sys.exit(1)
