#!/usr/bin/env python
"""
Parameter sweep for comb algorithm to fix FN-bound issue.

Sweeps min_correlation and tooth_activation_threshold to find optimal
combination that maximizes recall without regressing precision > 0.02.
"""

import sys
import numpy as np
from pathlib import Path

from cflibs.atomic.database import AtomicDatabase
from cflibs.inversion.identify.comb import CombIdentifier
from scripts.calibrate_alias import (
    AALTO_SEARCH_ELEMENTS,
    _select_aalto_cases,
    _safe_ratio,
)

def score_result(result, case):
    """Score one identification result against ground truth."""
    detected = {e.element for e in result.detected_elements}
    searched = set(case.elements)
    tp = len(detected & case.expected)
    fp = len(detected - case.expected)
    fn = len(case.expected - detected)
    tn = len((searched - case.expected) - detected)
    return tp, fp, fn, tn

def main():
    db_path = Path("ASD_da/libs_production.db")
    data_dir = Path("data")
    
    if not db_path.exists():
        print(f"Atomic DB not found: {db_path}")
        return
    if not data_dir.exists():
        print(f"Data dir not found: {data_dir}")
        return
    
    # Load benchmark cases
    cases = _select_aalto_cases(data_dir)
    if not cases:
        print("No Aalto benchmark spectra found")
        return
    
    print(f"Loaded {len(cases)} Aalto benchmark spectra")
    
    # Parameter sweep
    min_correlations = [0.05, 0.08, 0.10, 0.12]
    tooth_activation_thresholds = [0.3, 0.4, 0.5]
    
    # Track baseline
    baseline_tp = baseline_fp = baseline_fn = baseline_tn = 0
    
    with AtomicDatabase(str(db_path)) as db:
        # Run baseline with default parameters
        baseline_identifier = CombIdentifier(
            atomic_db=db,
            elements=AALTO_SEARCH_ELEMENTS,
            min_correlation=0.12,  # default
            tooth_activation_threshold=0.5,  # default
        )
        
        for case in cases:
            try:
                result = baseline_identifier.identify(case.wavelength, case.spectrum)
                tp, fp, fn, tn = score_result(result, case)
                baseline_tp += tp
                baseline_fp += fp
                baseline_fn += fn
                baseline_tn += tn
            except Exception as exc:
                print(f"Baseline failed on {case.name}: {exc}")
        
        baseline_precision = _safe_ratio(baseline_tp, baseline_tp + baseline_fp)
        baseline_recall = _safe_ratio(baseline_tp, baseline_tp + baseline_fn)
        baseline_f1 = _safe_ratio(2 * baseline_precision * baseline_recall, baseline_precision + baseline_recall)
        
        print(f"\nBaseline (min_correlation=0.12, tooth_activation=0.5):")
        print(f"  TP={baseline_tp}, FP={baseline_fp}, FN={baseline_fn}, TN={baseline_tn}")
        print(f"  Precision={baseline_precision:.4f}, Recall={baseline_recall:.4f}, F1={baseline_f1:.4f}")
        
        # Now sweep parameters
        print(f"\n{'='*100}")
        print(f"PARAMETER SWEEP RESULTS")
        print(f"{'='*100}")
        print(f"{'min_corr':<10} {'tooth_act':<12} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} {'Prec':>7} {'Recall':>7} {'F1':>7} {'ΔPrec':>7}")
        print(f"{'-'*100}")
        
        best_config = None
        best_recall = -1
        best_f1 = -1
        
        for min_corr in min_correlations:
            for tooth_act in tooth_activation_thresholds:
                tp = fp = fn = tn = 0
                
                identifier = CombIdentifier(
                    atomic_db=db,
                    elements=AALTO_SEARCH_ELEMENTS,
                    min_correlation=min_corr,
                    tooth_activation_threshold=tooth_act,
                )
                
                for case in cases:
                    try:
                        result = identifier.identify(case.wavelength, case.spectrum)
                        ctp, cfp, cfn, ctn = score_result(result, case)
                        tp += ctp
                        fp += cfp
                        fn += cfn
                        tn += ctn
                    except Exception as exc:
                        print(f"  Failed on {case.name}: {exc}")
                        break
                
                precision = _safe_ratio(tp, tp + fp)
                recall = _safe_ratio(tp, tp + fn)
                f1 = _safe_ratio(2 * precision * recall, precision + recall)
                delta_precision = precision - baseline_precision
                
                # Check if precision regressed too much
                if delta_precision > 0.02:
                    status = "PRECISION REGRESSED"
                else:
                    status = "OK"
                
                print(f"{min_corr:<10.2f} {tooth_act:<12.2f} {tp:>5} {fp:>5} {fn:>5} {tn:>5} "
                      f"{precision:>7.4f} {recall:>7.4f} {f1:>7.4f} {delta_precision:>+7.4f} {status}")
                
                # Track best configuration (maximize recall, then F1)
                if delta_precision <= 0.02 and recall > best_recall:
                    best_recall = recall
                    best_f1 = f1
                    best_config = (min_corr, tooth_act, tp, fp, fn, tn, precision, recall, f1)
        
        print(f"\n{'='*100}")
        if best_config:
            min_corr, tooth_act, tp, fp, fn, tn, precision, recall, f1 = best_config
            print(f"BEST CONFIGURATION:")
            print(f"  min_correlation={min_corr}, tooth_activation_threshold={tooth_act}")
            print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
            print(f"  Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        else:
            print("No configuration found that satisfies constraints")

if __name__ == "__main__":
    main()
