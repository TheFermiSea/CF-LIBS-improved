import json
import time
import os
import numpy as np
from pathlib import Path
import warnings

# Suppress JAX warnings for cleaner benchmark output
warnings.filterwarnings("ignore")

from cflibs.atomic.database import AtomicDatabase
from cflibs.manifold.config import ManifoldConfig
from cflibs.manifold.generator import ManifoldGenerator
from cflibs.manifold.loader import ManifoldLoader
from cflibs.manifold.vector_index import VectorIndexConfig
from cflibs.inversion.hybrid import HybridInverter
from cflibs.inversion.solver import IterativeCFLIBSSolver
from cflibs.validation.round_trip import GoldenSpectrumGenerator, NoiseModel

def main():
    print("Starting Phases 2 & 3: Manifold Generation and Hybrid Inference Benchmark\n")
    
    db_path = "ASD_da/libs_production.db"
    output_dir = Path("output/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifold_path = output_dir / "benchmark_manifold.h5"
    
    # Phase 2: Manifold Generation
    print("--- Phase 2: Generating Manifold ---")
    elements = ["Fe", "Cu", "Si", "Al", "Ca", "Ti"]
    config = ManifoldConfig(
        elements=elements,
        temperature_range=(0.5, 2.0),
        density_range=(1e16, 1e18),
        temperature_steps=30,
        density_steps=15,
        concentration_steps=5, # Generates roughly 10k-20k spectra total for reasonable build times
        wavelength_range=(200.0, 800.0),
        pixels=4096,
        output_path=str(manifold_path),
        db_path=db_path,
        batch_size=1000
    )
    
    if not manifold_path.exists():
        generator = ManifoldGenerator(config)
        t0 = time.perf_counter()
        generator.generate_manifold()
        t_gen = time.perf_counter() - t0
        print(f"Manifold generated in {t_gen:.2f}s")
    else:
        print("Manifold already exists, skipping generation.")
        
    print("\n--- Building FAISS Index ---")
    loader = ManifoldLoader(str(manifold_path))
    t0 = time.perf_counter()
    loader.build_vector_index(n_components=30, index_config=VectorIndexConfig(index_type="flat"))
    t_idx = time.perf_counter() - t0
    print(f"Index built in {t_idx:.2f}s\n")
    
    # Phase 3: Hybrid Inference Benchmark
    print("--- Phase 3: Hybrid Inference Benchmark ---")
    if not os.path.exists(db_path):
        print(f"Database missing at {db_path}. Exiting benchmark.")
        return

    db = AtomicDatabase(db_path)
    golden_gen = GoldenSpectrumGenerator(db, wavelength_range=(200.0, 800.0))
    noise_model = NoiseModel(shot_noise=True, readout_noise=5.0)
    
    inverter = HybridInverter(loader, max_iterations=100)
    iterative_solver = IterativeCFLIBSSolver(db, max_iterations=20)
    
    n_tests = 50
    print(f"Running {n_tests} test cases (comparing strategies)...")
    print(f"{'Method':<15} | {'ΔT/T %':<8} | {'Δn_e/n_e %':<10} | {'Max ΔC pp':<10} | {'Latency (ms)':<12} | {'Conv?':<5}")
    print("-" * 75)
    
    metrics = {
        "Manifold-only": {"t_err": [], "ne_err": [], "c_err": [], "latency": [], "conv": []},
        "Hybrid": {"t_err": [], "ne_err": [], "c_err": [], "latency": [], "conv": []},
        "Gradient-only": {"t_err": [], "ne_err": [], "c_err": [], "latency": [], "conv": []},
        "Iterative": {"t_err": [], "ne_err": [], "c_err": [], "latency": [], "conv": []}
    }
    
    # Warmup JAX
    print("Warming up JAX JIT compilation (this takes a moment)...")
    dummy_spec = np.ones(config.pixels)
    inverter.invert(dummy_spec, use_manifold_init=True)
    inverter.invert(dummy_spec, use_manifold_init=False)
    print("JAX Warmup complete.\n")
    
    np.random.seed(42)
    
    for i in range(n_tests):
        T_true = np.random.uniform(0.6, 1.8) * 11604.5
        ne_true = 10**np.random.uniform(16.5, 17.5)
        
        c_raw = np.random.uniform(0, 1, len(elements))
        c_raw /= c_raw.sum()
        comp_true = {el: c for el, c in zip(elements, c_raw)}
        
        golden = golden_gen.generate(T_true, ne_true, comp_true, n_lines_per_element=10, seed=i)
        golden_noisy = noise_model.apply(golden)
        
        # Hybrid/Gradient solvers need full spectrum, build it via forward model
        T_eV = T_true / 11604.5
        c_arr = np.array([comp_true[el] for el in elements])
        spec_clean = np.array(inverter.forward_model(T_eV, ne_true, c_arr, loader.wavelength))
        spec_noisy = spec_clean + np.random.normal(0, 5.0, spec_clean.shape)
        spec_noisy = np.maximum(spec_noisy, 1.0)
        spec_noisy += np.random.normal(0, np.sqrt(spec_noisy))
        
        # 1. Manifold-only
        t0 = time.perf_counter()
        idx, sim, params = loader.find_nearest_spectrum(spec_noisy)
        t_man = (time.perf_counter() - t0) * 1000
        metrics["Manifold-only"]["t_err"].append(abs(params['T_eV']*11604.5 - T_true) / T_true * 100)
        metrics["Manifold-only"]["ne_err"].append(abs(params['n_e_cm3'] - ne_true) / ne_true * 100)
        metrics["Manifold-only"]["c_err"].append(max([abs(params.get(el, 0) - comp_true[el]) for el in elements]) * 100)
        metrics["Manifold-only"]["latency"].append(t_man)
        metrics["Manifold-only"]["conv"].append(True)
        
        # 2. Hybrid
        t0 = time.perf_counter()
        res_hybrid = inverter.invert(spec_noisy, use_manifold_init=True)
        t_hyb = (time.perf_counter() - t0) * 1000
        metrics["Hybrid"]["t_err"].append(abs(res_hybrid.temperature_K - T_true) / T_true * 100)
        metrics["Hybrid"]["ne_err"].append(abs(res_hybrid.electron_density_cm3 - ne_true) / ne_true * 100)
        metrics["Hybrid"]["c_err"].append(max([abs(res_hybrid.concentrations.get(el, 0) - comp_true[el]) for el in elements]) * 100)
        metrics["Hybrid"]["latency"].append(t_hyb)
        metrics["Hybrid"]["conv"].append(res_hybrid.converged)
        
        # 3. Gradient-only
        t0 = time.perf_counter()
        res_grad = inverter.invert(spec_noisy, use_manifold_init=False)
        t_grad = (time.perf_counter() - t0) * 1000
        metrics["Gradient-only"]["t_err"].append(abs(res_grad.temperature_K - T_true) / T_true * 100)
        metrics["Gradient-only"]["ne_err"].append(abs(res_grad.electron_density_cm3 - ne_true) / ne_true * 100)
        metrics["Gradient-only"]["c_err"].append(max([abs(res_grad.concentrations.get(el, 0) - comp_true[el]) for el in elements]) * 100)
        metrics["Gradient-only"]["latency"].append(t_grad)
        metrics["Gradient-only"]["conv"].append(res_grad.converged)
        
        # 4. Iterative
        t0 = time.perf_counter()
        try:
            res_iter = iterative_solver.solve(golden_noisy.line_observations)
            t_iter = (time.perf_counter() - t0) * 1000
            metrics["Iterative"]["t_err"].append(abs(res_iter.temperature_K - T_true) / T_true * 100)
            metrics["Iterative"]["ne_err"].append(abs(res_iter.electron_density_cm3 - ne_true) / ne_true * 100)
            metrics["Iterative"]["c_err"].append(max([abs(res_iter.concentrations.get(el, 0) - comp_true[el]) for el in elements]) * 100)
            metrics["Iterative"]["conv"].append(res_iter.converged)
        except Exception:
            t_iter = (time.perf_counter() - t0) * 1000
            metrics["Iterative"]["t_err"].append(100.0)
            metrics["Iterative"]["ne_err"].append(100.0)
            metrics["Iterative"]["c_err"].append(100.0)
            metrics["Iterative"]["conv"].append(False)
        metrics["Iterative"]["latency"].append(t_iter)

    results_out = {}
    for method, data in metrics.items():
        avg_t = np.mean(data["t_err"])
        avg_ne = np.mean(data["ne_err"])
        avg_c = np.mean(data["c_err"])
        avg_lat = np.median(data["latency"])
        pct_conv = np.mean(data["conv"]) * 100
        
        print(f"{method:<15} | {avg_t:<8.1f} | {avg_ne:<10.1f} | {avg_c:<10.1f} | {avg_lat:<12.2f} | {pct_conv:<5.0f}")
        
        results_out[method] = {
            "avg_t_err_pct": avg_t,
            "avg_ne_err_pct": avg_ne,
            "avg_c_err_pp": avg_c,
            "median_latency_ms": avg_lat,
            "converged_pct": pct_conv
        }
        
    out_file = output_dir / "hybrid_inference_results.json"
    with open(out_file, "w") as f:
        json.dump(results_out, f, indent=2)
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    main()
