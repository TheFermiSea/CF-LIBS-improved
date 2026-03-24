---
plan_contract_ref: 05-02-PLAN.md
status: completed
one_liner: "Generated 7 publication-quality JQSRT figures from V100S benchmark CSVs with colorblind-safe styling"
contract_results:
  claims:
    - id: claim-figures-complete
      verdict: confirmed
      evidence: "All 7 PDF figures generated from benchmark CSVs in paper/figures/"
      confidence: HIGH
  deliverables:
    - id: deliv-figure-script
      status: produced
      path: paper/scripts/generate_figures.py
    - id: deliv-figures
      status: produced
      path: paper/figures/
  acceptance_tests:
    - id: test-figures-exist
      outcome: pass
      evidence: "7 PDFs verified: fig1_pipeline.pdf (37KB), fig2_voigt.pdf (22KB), fig3_boltzmann.pdf (22KB), fig4_anderson.pdf (36KB), fig5_faiss.pdf (25KB), fig6_batch.pdf (23KB), fig7_e2e.pdf (24KB)"
    - id: test-data-source
      outcome: pass
      evidence: "Script contains 7 pd.read_csv() calls loading from benchmarks/figures/*.csv; no hardcoded numerical data"
    - id: test-style
      outcome: pass
      evidence: "Colorblind-safe palette (#0072B2/#D55E00/#009E73), serif font family, 10pt axis labels, 8pt tick labels, publication grid styling"
  references:
    - id: ref-exojax
      status: cited
      notes: "Referenced in script docstring as GPU spectral model comparison target"
    - id: ref-zaghloul2024
      status: cited
      notes: "Referenced in script docstring as Voigt accuracy reference"
  forbidden_proxies:
    - id: fp-fabricated-figure-data
      status: rejected
      notes: "All figures read from CSV files; Fig 1 is a schematic (no numerical data needed)"
    - id: fp-matplotlib-defaults
      status: rejected
      notes: "Custom rcParams applied: serif fonts, colorblind palette, publication grid, proper DPI"
---

# Plan 05-02 Summary: Publication Figures

## Result

All 7 JQSRT publication figures generated from V100S benchmark data.

## Figures Generated

| Figure | Description | Size | Type | Key Result |
|--------|-------------|------|------|------------|
| fig1_pipeline.pdf | Pipeline architecture diagram | double-col | Schematic | 5 GPU-accelerated components highlighted |
| fig2_voigt.pdf | Voigt throughput (CPU vs GPU) | double-col | Log-log | 76x speedup at 100k grid |
| fig3_boltzmann.pdf | Boltzmann fitting time | single-col | Linear + speedup | 8.8x at 20 elements |
| fig4_anderson.pdf | Anderson convergence (dual panel) | double-col | (a) iters vs M, (b) residual trajectories | M=1 optimal, 1.6x iteration reduction |
| fig5_faiss.pdf | FAISS query latency | single-col | Bar chart | CPU-only (GPU unavailable on V100S) |
| fig6_batch.pdf | Batch forward scaling | single-col | Log-log | 10,708 spectra/s peak, OOM at batch>=5000 |
| fig7_e2e.pdf | E2E pipeline breakdown | double-col | Stacked bar + speedup | 13.6x at batch=1000 |

## Data Integrity

All data-driven figures (2-7) read from `benchmarks/figures/*.csv` containing real V100S-PCIE-32GB measurements collected 2026-03-23. No fabricated or mock data. The `benchmark_summary.json` confirms `"is_mock_data": false`.

## Styling Conventions

- Colorblind-safe palette: CPU=#0072B2 (blue), GPU=#D55E00 (orange), reference=#009E73 (green)
- Font: serif family (Computer Modern / DejaVu Serif), 10pt labels, 8pt ticks
- Single column: 3.5 in; Double column: 7.0 in
- PDF output at 300 DPI
- Light gray dashed grid, no top/right spines

## Notes

- **FAISS GPU data is null** in the benchmark CSV. Figure 5 shows CPU-only results with a note about GPU unavailability. This is a known limitation (uncertainty marker in the plan).
- **Batch forward OOM** at batch_size >= 5000 on GPU. Marked with red X in Figure 6.
- Script is fully reproducible: `python paper/scripts/generate_figures.py` regenerates all figures.

## Checkpoints

| Task | Hash | Description |
|------|------|-------------|
| 1 | 06c9b1a | Data-driven figures 2-7 from V100S benchmarks |
| 2 | 935929a | Pipeline architecture diagram (Figure 1) |

## Self-Check: PASSED

- [x] All 7 PDFs exist in paper/figures/
- [x] Script reads from CSV files (7 pd.read_csv calls)
- [x] No hardcoded numerical data
- [x] Axis labels include units
- [x] Colorblind-safe palette applied
- [x] Log-log scales on Voigt throughput and batch scaling
- [x] Consistent styling across all figures
