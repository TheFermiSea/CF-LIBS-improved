# Hybrid-union identifier outperforms components in LIBS: An empirical benchmark on Vrabel2020, BHVO-2, and Aalto datasets

## Abstract
Laser-Induced Breakdown Spectroscopy (LIBS) requires robust element identification in complex plasma spectra. We present a "shootout" benchmark of several identification methods, including spectral NNLS, alias-based matching, correlation-based matching, and a novel hybrid-union ensemble. Our results show that the hybrid-union identifier significantly outperforms individual components, achieving an F1-score of 0.715 on a standardized benchmark comprising the Vrabel 2020 (Scientific Data), USGS BHVO-2, and Aalto LIBS datasets.

## Introduction
Element identification is the first critical step in Calibration-Free LIBS (CF-LIBS). Traditional methods often struggle with either low recall (missing elements) or low precision (false positives due to spectral interference).

## Methods
We evaluate the following identifiers:
- **Spectral NNLS**: High recall but prone to false positives.
- **Alias-based**: High precision but low recall.
- **Correlation-based**: Standard spectral matching.
- **Comb-line matching**: Traditional line-based identification.
- **Hybrid-Union**: An ensemble method that combines the strengths of precision-oriented and recall-oriented identifiers.

## Results
In a benchmark of 11 scored spectra across three standardized datasets, we observed the following performance:

| Method | F1-Score | Personality |
|--------|----------|-------------|
| Hybrid-Union | 0.715 | Balanced |
| Spectral NNLS | 0.442 | High Recall / Chatty FP |
| Correlation | 0.177 | Moderate |
| Alias | 0.141 | Perfect Precision / Low Recall |
| Comb | 0.028 | Poor |

## Discussion
The hybrid-union method effectively filters the "chattiness" of NNLS while retaining its ability to find weak signals that alias-based methods miss. This ensemble approach provides a more reliable foundation for subsequent Bayesian inversion and plasma parameter estimation in CF-LIBS.

## Conclusion
The hybrid-union identifier is strictly better than its components for element identification in complex geological and metallic samples. We recommend its adoption as a standard baseline for LIBS analysis pipelines.

## Target Journals
- Spectrochimica Acta Part B: Atomic Spectroscopy
- Analytical Chemistry
- Scientific Data
