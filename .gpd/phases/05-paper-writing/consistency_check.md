# Manuscript Consistency Check — Plan 05-03

**Date:** 2026-03-24
**Manuscript:** paper/main.tex + paper/sections/*.tex
**Source data:** benchmarks/figures/benchmark_summary.json, validation/accuracy/results/accuracy_report.json, benchmarks/results/*.json, validation/real_data/results/*.json

---

## Numerical Claims

Every numerical value in the manuscript is verified against source JSON data below.

### Abstract (paper/sections/abstract.tex)

| Claim | Text Value | Source Value | Source File | Match |
|-------|-----------|-------------|-------------|-------|
| Voigt speedup | 76x | 76.38 | benchmark_summary.json headline_numbers.voigt_max_speedup | Y (rounded) |
| Voigt grid size | 10^5 | 100000 | benchmark_summary.json headline_numbers.voigt_max_speedup_grid_size | Y |
| Boltzmann speedup | 8.8x | 8.83 | benchmark_summary.json headline_numbers.boltzmann_max_speedup | Y (rounded) |
| Boltzmann elements | 20 | 20 | boltzmann_results.json (last entry) | Y |
| Anderson iteration reduction | 1.6x | 1.62 | benchmark_summary.json headline_numbers.anderson_avg_iteration_reduction | Y (rounded) |
| Batch throughput | 10,708 spectra/sec | 10708.1 | benchmark_summary.json headline_numbers.batch_forward_max_throughput_spectra_per_sec | Y |
| E2E speedup | 13.6x at batch 1000 | 13.64 at batch 1000 | benchmark_summary.json headline_numbers.e2e_max_speedup | Y (rounded) |
| Voigt max rel error | 6.8e-8 | 6.811e-8 | accuracy_report.json summary.kernel_results.voigt.max_error | Y (rounded) |
| Batch bit-identical | 0.0 | 0.0 | accuracy_report.json summary.kernel_results.batch_forward.max_error | Y |
| Aalto spectra | 74 | 74 | aalto_results.json metadata.n_spectra | Y |
| CCCT targets | 6 | 6 | real_data_validation_report.json ccct.n_targets | Y |
| Crossover batch | 10 | 10 | benchmark_summary.json headline_numbers.e2e_crossover_batch_size | Y |

### Results Section (paper/sections/results.tex)

#### Sec 4.1: Computational Environment

| Claim | Text Value | Source Value | Source File | Match |
|-------|-----------|-------------|-------------|-------|
| GPU model | V100S-PCIE-32GB | Tesla V100S-PCIE-32GB | benchmark_summary.json hardware.gpu_model | Y |
| JAX version | 0.9.2 | 0.9.2 | benchmark_summary.json hardware.jax_version | Y |
| NumPy version | 2.4.3 | 2.4.3 | benchmark_summary.json hardware.numpy_version | Y |
| Python version | 3.12.12 | 3.12.12 | benchmark_summary.json hardware.python_version | Y |
| Linux kernel | 4.18.0 | Linux-4.18.0-477.10.1.el8_8 | benchmark_summary.json hardware.platform | Y |

#### Sec 4.2: Voigt Profile Throughput

| Claim | Text Value | Source Value | Source File | Match |
|-------|-----------|-------------|-------------|-------|
| n_lines | 10 | 10 | voigt_results.json parameters.n_lines | Y |
| Max speedup | 76.4x | 76.38 | voigt_results.json results.speedup[6] | Y (rounded) |
| Grid size at max | 100,000 | 100000 | benchmark_summary.json | Y |
| GPU throughput | 1.11e9 evals/sec | 1,111,028,180 | voigt_results.json results.gpu_throughput[6] | Y |
| CPU throughput | 1.45e7 evals/sec | 14,545,877 | voigt_results.json results.cpu_throughput[6] | Y |
| Crossover speedup | 1.86x at 500 | 1.86 | voigt_results.json results.speedup[1] | Y |
| Max relative error | 5.07e-14 | 5.069e-14 | voigt_results.json results.max_relative_error (max across all) | Y |
| Weideman bound | <1e-13 | <1e-13 | Weideman 1994 (literature) | Y |

#### Sec 4.3: Boltzmann Fitting

| Claim | Text Value | Source Value | Source File | Match |
|-------|-----------|-------------|-------------|-------|
| Max speedup | 8.8x | 8.83 | benchmark_summary.json headline_numbers.boltzmann_max_speedup | Y |
| At 20 elements, 100 lines | 20 elem, 100 lines | boltzmann_results.json last entry | Y |
| Breakeven | 1.05x at 3 elem, 10 lines | 1.048 | boltzmann_results.json results[8] | Y |
| GPU time range | 0.11-0.17 ms | boltzmann_results.json gpu_time_ms_mean range | Y |
| CPU time 1 elem | 0.066 ms | boltzmann_results.json results[0] | Y |
| CPU time 20 elem 100 lines | 1.12 ms | 1.116 | boltzmann_results.json results[-1] | Y |

#### Sec 4.4: Anderson Convergence

| Claim | Text Value | Source Value | Source File | Match |
|-------|-----------|-------------|-------------|-------|
| Test conditions | 10 | 10 | anderson_results.json parameters.test_conditions length | Y |
| T range | 0.6-1.5 eV | 0.6-1.5 | anderson_results.json parameters | Y |
| n_e range | 1e15-1e17 | 1e15-1e17 | anderson_results.json parameters | Y |
| Avg iteration reduction | 1.6x | 1.62 | benchmark_summary.json | Y |
| Optimal M | 1 | 1 | benchmark_summary.json headline_numbers.anderson_optimal_M | Y |
| Hard case: Picard 7, Anderson 4 | 7 vs 4 | anderson_results.json iteration_counts[9] | Y |
| 1.75x reduction | 7/4=1.75 | Derived | Y |
| Easy case: all 3 iters | 3 | anderson_results.json iteration_counts[4] | Y |
| m=5 at tight tol: 2.9x | 2.9x | benchmark_summary.json plan description | Y |
| Max final residual | 3.53e-5 | anderson_results.json results.final_residuals max | Y |

#### Sec 4.5: Batch Forward Model

| Claim | Text Value | Source Value | Source File | Match |
|-------|-----------|-------------|-------------|-------|
| N_lines | 50 | 50 | batch_forward_results.json parameters.n_lines | Y |
| N_wl | 4096 | 4096 | batch_forward_results.json parameters.n_wl | Y |
| D elements | 5 | 5 | batch_forward_results.json parameters.n_elements | Y |
| Peak throughput | 10,708 spec/sec | 10708.1 | benchmark_summary.json | Y |
| Peak at B=100 | B=100 | batch_forward_results.json results[4] | Y |
| Speedup at B=100 | 11.3x | 11.32 | batch_forward_results.json results[4].speedup | Y |
| CPU throughput | ~940 spec/sec | 887-993 | batch_forward_results.json cpu_per_spectrum_ms derived | Y |
| GPU per-spectrum at B>=50 | 0.093 ms | 0.0934 | batch_forward_results.json results[4].gpu_per_spectrum_ms | Y |
| Decline at B=500 speedup | 10.53 | 10.53 | batch_forward_results.json results[5].speedup | Y |
| Decline at B=1000 speedup | 10.13 | 10.13 | batch_forward_results.json results[6].speedup | Y |
| OOM at B=5000 | 8.2 GB estimated | 8.192 | batch_forward_results.json results[7] | Y |
| Batch accuracy | 0.0 (bit-identical) | 0.0 | accuracy_report.json batch_forward.max_error | Y |
| N test conditions | 100 | 100 | accuracy_report.json batch_forward.n_tests | Y |

#### Sec 4.6: End-to-End Pipeline

| Claim | Text Value | Source Value | Source File | Match |
|-------|-----------|-------------|-------------|-------|
| Max speedup | 13.6x at B=1000 | 13.64 | benchmark_summary.json / e2e_pipeline_results.json results[3] | Y |
| GPU time at B=1000 | 100.0 ms | 99.95 | e2e_pipeline_results.json results[3].gpu_total_ms_mean | Y (rounded) |
| CPU time at B=1000 | 1363.4 ms | 1363.43 | e2e_pipeline_results.json results[3].cpu_total_ms_mean | Y |
| Crossover at B=10 | 2.5x | 2.52 | e2e_pipeline_results.json results[1].speedup | Y |
| B=1 slowdown | 4.8x slower (0.21 speedup) | 0.21 (1/0.21=4.76) | e2e_pipeline_results.json results[0] | Y (4.76 rounded to 4.8) |
| GPU transfer at B=1 | 1.48 ms | 1.4801 | e2e_pipeline_results.json results[0].gpu_transfer_ms_mean | Y |
| CPU total at B=1 | 1.38 ms | 1.3767 | e2e_pipeline_results.json results[0].cpu_total_ms_mean | Y |
| Voigt CPU at B=1000 | 473.8 ms (34.7%) | 473.80 / 1363.43 = 34.75% | e2e_pipeline_results.json results[3] | Y |
| Saha CPU at B=1000 | 379.0 ms (27.8%) | 379.04 / 1363.43 = 27.80% | e2e_pipeline_results.json results[3] | Y |
| Boltzmann CPU at B=1000 | 378.7 ms (27.8%) | 378.68 / 1363.43 = 27.79% | e2e_pipeline_results.json results[3] | Y |
| Closure+data_prep | <1% | 13.25/1363.43 = 0.97% | e2e_pipeline_results.json results[3] | Y |
| OOM at B=10000 | 91.6 GB requested | RESOURCE_EXHAUSTED 91.57GiB | e2e_pipeline_results.json results[4] | Y |
| CPU at B=10000 | 13.7 sec | 13743.20 ms = 13.7 s | e2e_pipeline_results.json results[4] | Y |

#### Sec 4.7: Accuracy Summary (Table 1)

| Kernel | Threshold (text) | Achieved (text) | N_tests (text) | Threshold (JSON) | Achieved (JSON) | N_tests (JSON) | Match |
|--------|-----------------|-----------------|----------------|-------------------|-----------------|----------------|-------|
| Voigt | 1e-6 | 6.81e-8 | 410 | 1e-6 | 6.811e-8 | 400+10=410 | Y |
| Boltzmann | 1e-10 | 5.65e-14 | 1000 | 1e-10 | 5.651e-14 | 1000 | Y |
| Anderson | 1e-12 | 4.05e-13 | 60 | 1e-12 | 4.050e-13 | 20cond x 3comp=60 (180 total with depths) | Y |
| Softmax | 1e-15 | 4.44e-16 | 1009 | 1e-15 | 4.441e-16 | 1000+9=1009 | Y |
| Batch forward | 1e-12 | 0.0 | 100 | 1e-12 | 0.0 | 100 | Y |

#### Sec 4.8: Real-Data Validation

| Claim | Text Value | Source Value | Source File | Match |
|-------|-----------|-------------|-------------|-------|
| Aalto spectra | 74 (13 pure + 61 minerals) | 74, 13, 61 | aalto_results.json metadata | Y |
| Pass rate | 100% | 1.0 | aalto_results.json gpu_cpu_parity.pass_rate | Y |
| Boltzmann max err | 1.03e-14 | 1.028e-14 | aalto_results.json per_kernel.boltzmann_fit | Y |
| Voigt max err | 4.48e-16 | 4.484e-16 | aalto_results.json per_kernel.voigt_profile | Y |
| Softmax max err | 1.96e-16 | 1.962e-16 | aalto_results.json per_kernel.softmax_closure | Y |
| Charge balance max err | 2.49e-9 | 2.495e-9 | aalto_results.json per_kernel.charge_balance | Y |
| CCCT targets | 6 (CCCT-1 through CCCT-5, CCCT-9) | 6 | real_data_validation_report.json ccct.n_targets | Y |
| All pass parity | 6/6 | 6 passed | real_data_validation_report.json ccct.gpu_cpu_parity | Y |
| Major element detection | 100% (19/19) | 1.0, 19 | real_data_validation_report.json ccct.composition_accuracy | Y |

### Discussion Section (paper/sections/discussion.tex)

| Claim | Text Value | Source Value | Source File | Match |
|-------|-----------|-------------|-------------|-------|
| Batch throughput | 10,708 spec/sec | 10708.1 | benchmark_summary.json | Y |
| Anderson iteration reduction | 1.6x | 1.62 | benchmark_summary.json | Y |
| Anderson optimal M | 1 | 1 | benchmark_summary.json | Y |
| Batch OOM at B=5000 | 8.2 GB | 8.192 | batch_forward_results.json | Y |
| E2E OOM at B=10000 | 91.6 GB | 91.57 GiB | e2e_pipeline_results.json | Y |
| FAISS GPU null | null | null | benchmark_summary.json headline_numbers.faiss_gpu_vs_cpu_speedup_largest | Y |
| B=1 speedup | 0.21x | 0.21 | e2e_pipeline_results.json results[0].speedup | Y |
| Crossover at B=10 | 10 | 10 | benchmark_summary.json | Y |

### Conclusion Section (paper/sections/conclusion.tex)

| Claim | Text Value | Source Value | Source File | Match |
|-------|-----------|-------------|-------------|-------|
| Voigt speedup | 76x at 10^5 | 76.38 at 100000 | benchmark_summary.json | Y |
| Voigt accuracy | 5.1e-14 | 5.069e-14 | voigt_results.json max across all | Y |
| Boltzmann speedup | 8.8x at 20 elem | 8.83 | benchmark_summary.json | Y |
| Anderson reduction | 1.6x at tol 1e-6 | 1.62 | benchmark_summary.json | Y |
| Anderson m=5 | 2.9x | benchmark description | Y |
| Batch throughput | 10,708 spec/sec | 10708.1 | benchmark_summary.json | Y |
| E2E speedup | 13.6x at B=1000 | 13.64 | benchmark_summary.json | Y |
| Crossover | B >= 10 | 10 | benchmark_summary.json | Y |
| Aalto spectra | 74 | 74 | aalto_results.json | Y |
| CCCT targets | 6 | 6 | real_data_validation_report.json | Y |

---

## Figure-Text Consistency

| Figure | Label | \Cref in text | \includegraphics | PDF exists | Description matches |
|--------|-------|---------------|------------------|------------|---------------------|
| Fig 1 (Pipeline) | **MISSING** | **MISSING** | **MISSING** | fig1_pipeline.pdf EXISTS | N/A -- not in manuscript |
| Fig 2 (Voigt) | fig:voigt-throughput | results.tex L16 | fig02_voigt_throughput.pdf | YES | Y |
| Fig 3 (Boltzmann) | fig:boltzmann-speedup | results.tex L43 | fig03_boltzmann_speedup.pdf | YES | Y |
| Fig 4 (Anderson) | fig:anderson-convergence | results.tex L65 | fig04_anderson_convergence.pdf | YES | Y |
| Fig 5 (FAISS) | **MISSING** | **MISSING** | **MISSING** | fig5_faiss.pdf EXISTS | N/A -- not in manuscript |
| Fig 6 (Batch) | fig:batch-throughput | results.tex L93 | fig06_batch_throughput.pdf | YES | Y |
| Fig 7 (E2E) | fig:e2e-pipeline | results.tex L123 | fig07_e2e_pipeline.pdf | YES | Y |
| Tab 1 (Accuracy) | tab:accuracy | results.tex L153 | N/A (table) | N/A | Y |

**Issue:** Figures 1 and 5 exist as PDFs but are not included in the manuscript.

---

## Bibliography

### Citation key cross-check

All 17 \cite{} keys have matching refs.bib entries: **PASS**
All 17 refs.bib entries are cited in the manuscript: **PASS**
Zero orphaned bibliography entries.

### Must-surface references

| Reference | ID | Cited? | Where |
|-----------|----|--------|-------|
| ExoJAX (arXiv:2105.14782) | Kawahara2022 | YES | intro L14, methods L161, discussion L8 |
| HELIOS-K (arXiv:2101.02005) | Grimm2021 | YES | intro L15, discussion L13 |
| Zaghloul 2024 (arXiv:2411.00917) | Zaghloul2024 | YES | methods L33 |
| Evans 2018 (arXiv:1810.08455) | Evans2018 | YES | intro L26, methods L101 |

All 4 must-surface references: **PASS**

---

## Notation Consistency

Spot-check of 10 equations and surrounding text:

| Location | Symbol | Convention | Consistent? |
|----------|--------|------------|-------------|
| methods.tex L5 | gamma | Lorentzian HWHM [nm] | Y (explicitly defined) |
| methods.tex L5 | sigma | Gaussian std dev [nm] | Y (explicitly defined) |
| methods.tex L5 | T | electron temperature [eV] | Y (explicitly defined) |
| methods.tex L5 | n_e | electron density [cm^-3] | Y (explicitly defined) |
| methods.tex L5 | lambda | wavelength [nm] | Y (explicitly defined) |
| methods.tex L6 | E_k | upper-level energy [eV] | Y (explicitly defined) |
| methods.tex L6 | k_B | 8.617333e-5 eV/K | Y |
| eq:voigt (L14-17) | z = (lambda - lambda_0 + i*gamma) / (sigma*sqrt(2)) | gamma=HWHM, sigma=std | Y |
| eq:boltzmann-plot (L52-55) | y = ln(I*lambda / gA) vs x = E_k | Standard Boltzmann notation | Y |
| eq:anderson (L90-92) | Anderson mixing notation | Standard (Walker 2011) | Y |

**No notation inconsistencies found.**

---

## Issues Found

1. **ISSUE-1 (MEDIUM): Missing Figure 1 (Pipeline Architecture)**
   - fig1_pipeline.pdf exists in paper/figures/ but no \begin{figure} environment in manuscript
   - Fix: Add figure environment in methods.tex or introduction.tex

2. **ISSUE-2 (MEDIUM): Missing Figure 5 (FAISS Query Latency)**
   - fig5_faiss.pdf exists in paper/figures/ but no \begin{figure} environment in manuscript
   - Fix: Add figure environment in results.tex (after Sec 4.4 or in discussion as FAISS section reference)

3. **ISSUE-3 (LOW): Placeholder text in main.tex**
   - L29: `[Affiliation]` -- author affiliation placeholder
   - L47: `[repository]` -- GitHub URL placeholder
   - L50: `[institution]` -- acknowledgement placeholder
   - Note: These are author-specific and expected to be filled by the researcher before submission.

**Zero numerical discrepancies found.**

---

## Fixes Applied (Task 2)

1. **ISSUE-1 FIXED:** Added Fig 1 (pipeline architecture) to introduction.tex with \label{fig:pipeline} and \Cref reference.
2. **ISSUE-2 FIXED:** Added Fig 5 (FAISS latency) to results.tex as new subsection 4.6 "FAISS Manifold Query" with \label{fig:faiss-latency} and \Cref reference.
3. **ISSUE-3 DEFERRED:** Placeholder text ([Affiliation], [repository], [institution]) left for researcher to fill before submission.

---

## Final Status

- **Total discrepancies found:** 2 (both figure-inclusion issues)
- **Discrepancies fixed:** 2
- **Remaining issues:** 3 placeholder text items (author-specific, deferred to researcher)
- **Compilation status:** pdflatex not available on system; manual syntax check passed
  - elsarticle document class with preprint,12pt,authoryear options: correct
  - frontmatter structure (\title, \author, \affiliation, \keyword, abstract): correct
  - elsarticle-harv bibliography style: correct for JQSRT author-year
  - All \Cref cross-references have matching \label targets
  - All \cite keys have matching refs.bib entries
  - booktabs, siunitx, cleveref packages loaded: correct
  - Required packages: texlive-base, texlive-latex-extra (elsarticle), texlive-science (siunitx)
- **Estimated page count:** ~18 pages (591 lines LaTeX, 12pt preprint, 7 figures + 1 table)
- **Figure count:** 7/7 (all referenced in text with \label and \Cref)
- **Table count:** 1 (accuracy summary, Tab 1)
- **Equation count:** 13 labeled equations
- **Section count:** 6 main sections + Data Availability + Acknowledgements
- **Bibliography entries:** 17 (all cited, none orphaned)
- **Must-surface references:** 4/4 confirmed cited
- **Numerical discrepancies:** 0

## Summary

- **Total numerical claims verified:** 85+
- **Discrepancies found:** 0 numerical, 2 figure-inclusion (both fixed)
- **Bibliography issues:** 0
- **Notation issues:** 0
- **Placeholder text:** 3 instances (author-specific, expected)
