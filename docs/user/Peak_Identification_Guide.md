# Peak Identification and Line Matching Guide

Peak identification is the bridge between a measured LIBS spectrum and the
thermodynamic CF-LIBS inversion. The inversion solver does not operate on raw
pixels. It operates on `LineObservation` objects: integrated line intensities
with element labels, ion stages, wavelengths, upper-level energies, statistical
weights, and transition probabilities.

Use this guide when you need to decide which elements are present, tune peak
finding for real spectra, or understand why the quantitative inversion accepted
or rejected a line. For the equations used after line observations are built,
see [Physics: Equations](../physics/Equations.md) and
[Physics: Inversion Algorithm](../physics/Inversion_Algorithm.md). For the
module map, see [Codebase Architecture](../reference/Codebase_Architecture.md).

## Where Identification Fits

The real-data pipeline is:

```
Measured spectrum
  -> baseline and noise estimation
  -> peak detection
  -> candidate element/database matching
  -> wavelength-shift and interference checks
  -> line scoring and line selection
  -> LineObservation objects
  -> iterative CF-LIBS inversion
```

The scientific reason for this separation is simple: a wrong line label is not a
small preprocessing error. It changes the Boltzmann plot, the Saha correction,
and the closure equation, so it can produce a plausible but wrong temperature,
electron density, and composition.

## 1. Start With a Physically Plausible Candidate Set

When using `cflibs analyze` or `cflibs invert`, `--elements` is prior knowledge,
not a periodic-table sweep. Include all major and plausible minor elements from
the sample history, alloy grade, substrate, environment, or target matrix.
Avoid searching every element unless you are explicitly benchmarking an
identifier, because line density in the NIST database makes chance coincidences
common in dense LIBS spectra.

Good candidate sets are usually based on:

- Known sample class, such as Fe-Cr-Ni-Mn-Si for steel.
- Substrate or matrix knowledge, such as Si-Al-Ca-Mg-Fe-Na-K for minerals.
- Ambient or ablation contaminants, such as H, C, N, and O, interpreted with
  care because they can come from air, water, binders, or surface films.
- Instrument wavelength range. Do not include elements whose useful lines are
  outside the acquired band.

If the sample is genuinely unknown, run an element identifier first, inspect its
candidate list and diagnostics, then rerun the quantitative inversion with a
short, defensible element list.

## 2. Baseline and Noise Handling

Raw LIBS spectra combine discrete emission lines, bremsstrahlung/recombination
continuum, detector noise, and occasional cosmic-ray artifacts. CF-LIBS keeps
baseline and noise handling in `cflibs.inversion.preprocess.preprocessing`.

The canonical implementation is:

1. Estimate a baseline with a median filter, SNIP, or asymmetric least squares.
2. Estimate noise from baseline residuals using iterative sigma-clipped MAD.
3. Detect peaks above noise-scaled height and prominence thresholds.
4. Reject unrealistically narrow peaks with a minimum-FWHM filter.
5. Optionally require second-derivative confirmation.

Relevant entry points:

```python
from cflibs.inversion.preprocessing import (
    BaselineMethod,
    detect_peaks_auto,
    estimate_baseline,
    estimate_noise,
)

peaks, baseline, noise = detect_peaks_auto(
    wavelength_nm,
    intensity,
    resolving_power=5000.0,
    threshold_factor=4.0,
    prominence_factor=1.5,
    baseline_method=BaselineMethod.MEDIAN,
)
```

The literature supports this multi-stage view rather than raw local-maximum
matching. Chen et al. (2014) used continuous wavelet transform peak detection
for LIBS spectra because scale-space ridge persistence suppresses baseline and
noise artifacts. Lin et al. (2025) proposed adaptive-window symmetric zero-area
conversion for closely spaced noisy LIBS peaks. Shin et al. (2020) showed that
noise reduction and in-situ wavelength calibration are prerequisites for robust
automated spectral-line annotation in harsh plasma-diagnostic environments.

## 3. Wavelength Tolerance, Resolution, and Shift

Database wavelengths are not enough by themselves. Real spectra include pixel
calibration drift, spectrometer resolving power, Stark shifts, unresolved blends,
and finite integration windows.

Use these controls first:

| Parameter | Meaning | Typical action |
|-----------|---------|----------------|
| `resolving_power` | Spectrometer `R = lambda / delta_lambda` | Set for echelle instruments and any wavelength-dependent resolution. |
| `wavelength_tolerance_nm` | Match window between detected peak and database line | Tighten when false positives dominate; loosen if known lines miss by a stable offset. |
| `shift_scan_nm` | Global wavelength-shift search range | Use when all matches appear displaced in the same direction. |
| `shift_step_nm` | Step size for shift scan | Keep smaller than the matching tolerance. |
| `peak_width_nm` | Integration half-window scale for line area | Match the observed FWHM and instrument resolution. |

`detect_line_observations` uses a comb-style shift scan before building line
observations. This is important because Gajarska et al. (2024) found that small
instrumental shifts and broadened lines should be accommodated before deciding
that a line is absent. Noël et al. (2025) likewise require wavelength-calibrated
spectra for ALIAS and note that large Stark shifts remain a limitation of fixed
geometric matching.

## 4. Identification Algorithms in CF-LIBS

CF-LIBS exposes several identification strategies. They answer related but
different questions.

### Classic Line Observations

`cflibs.inversion.line_detection.detect_line_observations` is the path used by
`cflibs analyze`, `cflibs invert`, and `cflibs batch`. It converts a candidate
element list into `LineObservation` objects for the Boltzmann/Saha inversion.

The implementation:

- Loads transitions for the requested elements and wavelength range.
- Finds experimental peaks.
- Optionally filters candidates with a `kdet` rarity-weighted prefilter.
- Scans global wavelength shifts with comb precision/recall scoring.
- Matches accepted element transitions to peaks.
- Integrates line area, optionally using Voigt deconvolution for blends.
- Marks resonance lines so `LineSelector` can exclude self-absorbed lines.

This path is conservative: its job is not to discover every possible trace
element, but to produce a defensible line set for quantitative CF-LIBS.

### ALIAS

`ALIASIdentifier` implements the Automated Line Identification for Atomic
Spectroscopy workflow. Noël et al. (2025) describe ALIAS as a sequence of peak
detection, simplified LTE emissivity simulation, line fusion, matching,
thresholding, similarity scoring, and probability assessment.

The main scores are:

- `k_sim`: cosine similarity between theoretical emissivities and measured peak
  intensities.
- `k_rate`: emissivity-weighted detection rate for expected lines.
- `k_shift`: wavelength agreement between observed and database lines.
- `k_det`: combined detection coefficient.
- `CL`: confidence level that also accounts for signal-to-noise and abundance
  priors.

Use ALIAS when you want an interpretable candidate element list with explicit
score components. Treat it as a candidate generator, then validate line quality
before quantitative inversion.

### Comb Templates

`CombIdentifier` follows the comb/template-matching approach of Gajarska et al.
(2024). Each element is represented as a fingerprint of triangular template
teeth at expected line positions. The implementation estimates a baseline,
sets an adaptive threshold, shifts/widths templates, and correlates each
fingerprint with the measured spectrum.

Comb methods are useful when several lines from the same element should move or
broaden together. They are less reliable when line-specific Stark shifts or
severe blends break the common-shift assumption.

### Correlation Identifier

`CorrelationIdentifier` modernizes the model-spectrum correlation idea used by
Labutin et al. (2013): generate physically plausible spectra over a grid of
`T` and `n_e`, then compare model and measured spectra. This can identify
overlapping line groups that simple peak matching treats ambiguously.

Use classic correlation mode for transparent, small candidate sets. Use vector
mode only when a validated manifold/vector library is available.

### Full-Spectrum NNLS

`SpectralNNLSIdentifier` decomposes the observed spectrum as a non-negative
linear combination of single-element basis spectra at an estimated `(T, n_e)`.
This is a full-spectrum prefilter rather than a line-by-line identifier. It is
especially useful for narrowing a large candidate set before ALIAS or classic
line observation construction.

## 5. From Candidate Elements to Quantitative Lines

After candidate identification, CF-LIBS still applies line selection before the
inversion. `LineSelector` filters for:

- Sufficient signal-to-noise.
- Enough lines per element.
- Enough upper-energy spread for a stable Boltzmann slope.
- Spectral isolation from neighboring lines.
- Optional resonance-line exclusion.

This is where qualitative identification becomes quantitative spectroscopy. A
candidate element with one strong resonance line may be real, but it may still
be unsafe for CF-LIBS composition fitting because resonance lines are often
self-absorbed. John and Anoop (2023) and related self-absorption studies show
that self-absorption depresses line intensity, distorts line width, and biases
Boltzmann temperature and concentration estimates. When in doubt, exclude
resonance lines or use the documented self-absorption corrections in
[Physics: Equations](../physics/Equations.md).

## 6. Practical Parameter Tuning

| Symptom | Likely cause | First checks |
|---------|--------------|--------------|
| No peaks detected | Threshold too high, baseline too high, spectrum not in expected columns | Lower `min_peak_height` or `threshold_factor`; inspect baseline. |
| Hundreds of peaks detected | Noise threshold too low or continuum not removed | Raise `threshold_factor`; use SNIP/ALS baseline; require FWHM filter. |
| Known lines miss by same offset | Wavelength calibration drift | Increase `shift_scan_nm`; inspect `applied_shift_nm`; recalibrate spectrum. |
| Many plausible but wrong elements | Search set too broad or tolerance too loose | Reduce `--elements`; tighten `wavelength_tolerance_nm`; raise `min_relative_intensity`. |
| Transition metals over-identified | Dense Fe/Ti/Ni/Cr/Mo line fields | Require multiple matched lines, adequate recall, and Boltzmann consistency. |
| Strong element absent | Saturated/self-absorbed line, missing wavelength band, or Stark shift | Inspect resonance lines, FWHM, and line profile; try non-resonance lines. |
| Inversion accepts too few lines | Line selection is stricter than identification | Lower `min_snr` only if noise supports it; check `min_energy_spread_ev`. |

## 7. Diagnostics to Report

For publication-quality or shared analyses, report more than the final
composition:

- Candidate element list and how it was chosen.
- Baseline method and peak thresholds.
- Wavelength tolerance, resolving power, and any fitted global shift.
- Number of detected peaks, matched peaks, and unmatched peaks.
- Lines retained and rejected by `LineSelector`.
- Whether resonance lines were excluded or corrected.
- Boltzmann plot `R^2`, residuals, and energy spread per element.
- McWhirter LTE ratio and any self-absorption warnings.

These diagnostics make the difference between a reproducible CF-LIBS analysis
and an uninspectable element list.

## 8. Minimal Python Workflow

```python
from cflibs.atomic.database import AtomicDatabase
from cflibs.io.spectrum import load_spectrum
from cflibs.inversion.line_detection import detect_line_observations
from cflibs.inversion.line_selection import LineSelector
from cflibs.inversion.solver import IterativeCFLIBSSolver

wl, intensity = load_spectrum("my_spectrum.csv")

with AtomicDatabase("ASD_da/libs_production.db") as db:
    detection = detect_line_observations(
        wavelength=wl,
        intensity=intensity,
        atomic_db=db,
        elements=["Fe", "Cr", "Ni", "Mn"],
        resolving_power=5000.0,
        wavelength_tolerance_nm=0.1,
        min_peak_height=0.01,
        min_relative_intensity=100.0,
    )

    selector = LineSelector(
        min_snr=10.0,
        min_energy_spread_ev=2.0,
        min_lines_per_element=3,
        exclude_resonance=True,
    )
    selected = selector.select(
        detection.observations,
        resonance_lines=detection.resonance_lines,
    )

    solver = IterativeCFLIBSSolver(atomic_db=db)
    result = solver.solve(selected.selected_lines)

print(detection.warnings)
print(result.concentrations)
```

## References

1. Chen, P., Tian, D., Qiao, S., and Yang, G. (2014). "An automatic peak detection method for LIBS spectrum based on continuous wavelet transform."
2. Noël, C., Neoricic, L., Alvarez-Llamas, C., Cugerone, A., Fabre, C., Duponchel, L., and Motto-Ros, V. (2025). "Automated line identification for atomic spectroscopy (ALIAS): Application to LIBS imaging data processing." *Spectrochimica Acta Part B*, 231, 107255. DOI: 10.1016/j.sab.2025.107255.
3. Gajarska, Z., Faruzelova, A., Kepes, E., Prochazka, D., Porizka, P., Kaiser, J., Lohninger, H., and Limbeck, A. (2024). "Automated detection of element-specific features in LIBS spectra." *Journal of Analytical Atomic Spectrometry*, 39, 3151-3161. DOI: 10.1039/d4ja00247d.
4. Labutin, T. A., Zaytsev, S. M., and Popov, A. M. (2013). "Automatic identification of emission lines in laser-induced plasma by correlation of model and experimental spectra." *Analytical Chemistry*, 85, 1985-1990. DOI: 10.1021/ac303270q.
5. Tobin, M., and Nations, M. (2022). "Whose line is it anyway? A self-training spectral line identification code for plasma physics experiments." *Review of Scientific Instruments*. DOI: 10.1063/5.0107578.
6. Shin, J., et al. (2020). "Automatic impurity spectral line identification algorithm with noise reduction for fusion plasmas." *Fusion Engineering and Design*. DOI: 10.1016/j.fusengdes.2020.111459.
7. Lin, et al. (2025). "Research on laser-induced breakdown spectroscopy peak detection algorithm based on adaptive window width symmetric zero-area conversion." DOI: 10.1080/22297928.2025.2556488.
8. Matsumura, T., Takahashi, T., Nagata, K., Ando, Y., Yada, A., Thornton, B., and Kuwatani, T. (2024). "High-Throughput Calibration-Free Laser-Induced Breakdown Spectroscopy." *ACS Earth and Space Chemistry*, 8, 1259-1271.
9. John, L. M., and Anoop, K. K. (2023). "A numerical procedure for understanding the self-absorption effects in laser induced breakdown spectroscopy." *RSC Advances*, 13, 29613-29624. DOI: 10.1039/D3RA06226K.
