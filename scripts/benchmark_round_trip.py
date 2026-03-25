import json
import time
import os
from pathlib import Path
from cflibs.atomic.database import AtomicDatabase
from cflibs.validation.round_trip import RoundTripValidator

def main():
    print("Starting Phase 1: Round-Trip Validation Benchmark")
    
    # Initialize DB and Validator
    db_path = "ASD_da/libs_production.db"
    if not os.path.exists(db_path):
        print(f"WARNING: Database not found at {db_path}. Using mock or skipping if it fails.")
        
    db = AtomicDatabase(db_path)
    
    # We set tolerances slightly looser for the validator object itself so we can capture
    # "failed" states manually and compute our own metrics for the report.
    validator = RoundTripValidator(
        atomic_db=db,
        temperature_tolerance=0.15,
        density_tolerance=0.30,
        concentration_tolerance=0.15
    )

    # Parameter Grid
    temperatures = [6000.0, 8000.0, 10000.0, 12000.0]
    densities = [1e16, 3e16, 1e17, 3e17]
    compositions = [
        {"Fe": 1.0},
        {"Fe": 0.7, "Cu": 0.3},
        {"Fe": 0.4, "Si": 0.3, "Al": 0.2, "Ca": 0.1}
    ]
    noise_states = [False, True]

    results_data = []
    
    total_runs = len(temperatures) * len(densities) * len(compositions) * len(noise_states)
    print(f"Running {total_runs} validation tests...")
    
    # Console table header
    print(f"{'T (K)':<6} | {'n_e (cm-3)':<9} | {'Comp':<15} | {'Noise':<5} | {'ΔT/T %':<8} | {'Δn_e/n_e %':<10} | {'Max ΔC pp':<10} | {'Conv?':<5}")
    print("-" * 85)

    success_noiseless = 0
    success_noisy = 0
    total_noiseless = 0
    total_noisy = 0

    seed_base = 42

    for noise in noise_states:
        for comp_idx, comp in enumerate(compositions):
            comp_name = f"Mix {comp_idx+1}" if len(comp) > 1 else "Pure Fe"
            for T in temperatures:
                for ne in densities:
                    seed = seed_base + int(T) + int(ne/1e15) + comp_idx
                    
                    start_time = time.perf_counter()
                    
                    result = validator.validate(
                        temperature_K=T,
                        electron_density_cm3=ne,
                        concentrations=comp,
                        seed=seed,
                        n_lines_per_element=10,
                        add_noise=noise
                    )
                    
                    latency = time.perf_counter() - start_time
                    
                    # Compute max concentration error in percentage points
                    max_c_err_pp = 0.0
                    for el, true_c in comp.items():
                        rec_c = result.recovered_concentrations.get(el, 0.0)
                        err_pp = abs(rec_c - true_c) * 100.0
                        if err_pp > max_c_err_pp:
                            max_c_err_pp = err_pp
                            
                    t_err_pct = result.temperature_error_frac * 100.0
                    ne_err_pct = result.electron_density_error_frac * 100.0
                    
                    # Determine success against targets
                    if not noise:
                        total_noiseless += 1
                        passed_target = (t_err_pct < 5.0) and (ne_err_pct < 20.0) and (max_c_err_pp < 5.0) and result.converged
                        if passed_target: success_noiseless += 1
                    else:
                        total_noisy += 1
                        passed_target = (t_err_pct < 10.0) and (ne_err_pct < 30.0) and (max_c_err_pp < 10.0) and result.converged
                        if passed_target: success_noisy += 1

                    # Log to console
                    noise_str = "Yes" if noise else "No"
                    conv_str = "Yes" if result.converged else "No"
                    
                    if not result.converged:
                        t_err_str = "FAIL"
                        ne_err_str = "FAIL"
                        c_err_str = "FAIL"
                    else:
                        t_err_str = f"{t_err_pct:.1f}%"
                        ne_err_str = f"{ne_err_pct:.1f}%"
                        c_err_str = f"{max_c_err_pp:.1f}pp"
                        
                    print(f"{T:<6.0f} | {ne:<9.1e} | {comp_name:<15} | {noise_str:<5} | {t_err_str:<8} | {ne_err_str:<10} | {c_err_str:<10} | {conv_str:<5}")

                    # Store for JSON
                    results_data.append({
                        "true_T": T,
                        "true_ne": ne,
                        "true_comp": comp,
                        "noise": noise,
                        "recovered_T": result.recovered_temperature_K,
                        "recovered_ne": result.recovered_electron_density,
                        "recovered_comp": result.recovered_concentrations,
                        "error_T_pct": t_err_pct,
                        "error_ne_pct": ne_err_pct,
                        "max_error_C_pp": max_c_err_pp,
                        "converged": result.converged,
                        "iterations": result.iterations,
                        "latency_s": latency,
                        "passed_strict_targets": passed_target
                    })

    print("-" * 85)
    print("Summary:")
    if total_noiseless > 0:
        print(f"Noiseless Success Rate: {success_noiseless}/{total_noiseless} ({success_noiseless/total_noiseless*100:.1f}%) [Target: >90%]")
    if total_noisy > 0:
        print(f"Noisy Success Rate:     {success_noisy}/{total_noisy} ({success_noisy/total_noisy*100:.1f}%) [Target: >75%]")

    # Identify failure modes
    failures = [r for r in results_data if not r["passed_strict_targets"]]
    if failures:
        print("\nFailure Modes Analysis:")
        # Group by noise
        for noise_state in [False, True]:
            state_failures = [f for f in failures if f["noise"] == noise_state]
            if not state_failures: continue
            print(f"  Noise={noise_state}: {len(state_failures)} failures")
            # See if failures correlate with low T or high ne
            low_t_fails = len([f for f in state_failures if f["true_T"] <= 8000])
            print(f"    - Failures at T <= 8000K: {low_t_fails}/{len(state_failures)}")
            mix_fails = len([f for f in state_failures if len(f["true_comp"]) > 2])
            print(f"    - Failures on Complex Mix: {mix_fails}/{len(state_failures)}")

    out_file = Path("output/benchmarks/round_trip_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({
            "summary": {
                "total_noiseless": total_noiseless,
                "success_noiseless": success_noiseless,
                "total_noisy": total_noisy,
                "success_noisy": success_noisy
            },
            "results": results_data
        }, f, indent=2)
        
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    main()
