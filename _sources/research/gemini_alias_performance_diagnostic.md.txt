# Technical Diagnostic Report: Investigating Performance Discrepancies of the ALIAS Algorithm on Geological LIBS Datasets

The evolution of Laser-Induced Breakdown Spectroscopy (LIBS) from a fundamental analytical laboratory technique to a high-throughput, automated imaging modality has necessitated a radical shift in spectral data processing methodologies. Traditional manual peak-picking and qualitative interpretation by expert spectroscopists are no longer viable in an era where scanners like the Aalto University LASO-LIBS system generate megapixel-scale datasets at acquisition rates reaching 1.3 kHz. The Automated Line Identification for Atomic Spectroscopy (ALIAS) algorithm, proposed by Noel et al. (2025), represents a seminal attempt to bridge this gap by applying principles of the Vector Space Model (VSM) from computational linguistics to atomic emission spectra. However, the implementation of this algorithm on the Aalto University validated mineral and elemental libraries has yielded a significant performance deficit, with precision falling to 12% and recall to 28%, in stark contrast to the >95% success rates reported in controlled environments. This report provides an exhaustive diagnostic analysis of the fundamental physical, mathematical, and methodological misalignments responsible for this discrepancy.

## Theoretical Foundations of the ALIAS Framework and the Vector Space Model

The ALIAS algorithm operates by abstracting a LIBS spectrum into a high-dimensional vector, where each dimension represents a specific wavelength or spectral bin. This approach is rooted in the "bag-of-words" philosophy of information retrieval, where the frequency and uniqueness of "terms" (spectral peaks) determine the relevance of a "document" (an element in the database) to a "query" (the measured sample spectrum). This paradigm shift moves away from threshold-based peak searching toward a holistic geometric comparison in a multi-dimensional Hilbert space.

### The Peak Weighting Mechanism: Intensity and Specificity

The core of the ALIAS logic is its dual-factor weighting scheme, which calculates the significance of each spectral line based on two primary variables: local intensity and global specificity. In the context of LIBS, intensity is proportional to the probability of an atomic transition and the population of the upper energy level, while specificity is intended to act as an inverse document frequency (IDF) metric, penalizing lines that are common across multiple elements in the reference database.

The mathematical weight $w$ for a peak at wavelength $\lambda$ in the ALIAS implementation is defined as:

$$w(\lambda) = I(\lambda) \times Spec(\lambda)$$

where $I(\lambda)$ is the normalized peak intensity and $Spec(\lambda)$ is the specificity. The specificity is typically derived from the inverse of the count of entries in the reference database (e.g., Kurucz or NIST) that fall within a defined wavelength tolerance $\epsilon$. This can be formulated as:

$$Spec(\lambda) = \left( \sum_{E \in DB} \mathbb{1}_{|\lambda - \lambda_E| \leq \epsilon} \right)^{-1}$$

In this equation, $\mathbb{1}$ is the indicator function, $DB$ is the reference database, and $\lambda_E$ are the wavelengths associated with various elements $E$ in the database. While this model is effective for identifying "signature" lines in relatively simple matrices, it encounters profound challenges in the complex geochemical environments characteristic of the Aalto University spectral libraries.

### Vector Proximity and Similarity Metrics

Once weights are assigned, the algorithm represents both the element $E$ and the sample $S$ as vectors $V_E$ and $V_S$. The likelihood of an element's presence is then determined by the proximity between these vectors. Standard implementations of ALIAS utilize Cosine Similarity or Euclidean distance to rank elements. Cosine Similarity is defined as:

$$\cos(\theta) = \frac{V_E \cdot V_S}{\|V_E\| \|V_S\|} = \frac{\sum_{i=1}^{n} w_{E,i} w_{S,i}}{\sqrt{\sum_{i=1}^{n} w_{E,i}^2} \sqrt{\sum_{i=1}^{n} w_{S,i}^2}}$$

This metric measures the angular divergence between vectors, making it theoretically robust to variations in absolute intensity caused by fluctuations in laser-sample coupling or pulse energy. However, this robustness is predicated on the assumption that the "direction" of the vector accurately represents the elemental identity, an assumption that falters when matrix effects or plasma temperature variations alter the relative line intensities.

## Characteristics of the Aalto University LASO-LIBS Experimental Setup

To understand why ALIAS fails on the Aalto University datasets, it is necessary to examine the physical parameters of the data acquisition process. The Aalto Department of Civil Engineering has developed the LASO-LIBS scanner, a large-area high-resolution instrument designed primarily for geological applications, such as mine wall imaging and drill core analysis.

### Instrumental Configuration and Environmental Control

The Aalto research group employs a Q-switched Nd:YAG laser operating at 1064 nm with a 5 ns pulse duration. A defining feature of their methodology is the use of a 10 mbar Argon background pressure. This controlled atmosphere is designed to enhance the signal-to-noise ratio and improve the separation of critical lines, such as the Balmer-alpha peaks of Hydrogen and Deuterium, which are vital for fusion-relevant material studies and isotope quantification.

The spectral data is typically captured using high-resolution spectrometers. For example, some Aalto studies report using a grating with 1350 grooves/mm, providing a spectral resolution of approximately 0.1 nm. This high resolution is necessary for resolving the complex "forest" of lines in transition metals but also makes the algorithm highly sensitive to wavelength jitter and database inaccuracies.

### Validation and Data Diversity

The Aalto University LIBS libraries are not merely collections of raw spectra; they are validated using secondary analytical techniques. For geological samples, mineral identifications and elemental concentrations are confirmed via X-ray diffraction (XRD), X-ray fluorescence (XRF), and inductively coupled plasma atomic emission spectroscopy (ICP-AES).

| Parameter | Aalto University LASO-LIBS Setup | Standard ALIAS Baseline (Assumed) |
|-----------|----------------------------------|-----------------------------------|
| Laser Wavelength | 1064 nm (Nd:YAG) | Variable (often 266 nm or 532 nm) |
| Pulse Duration | 5 ns | 10 ns |
| Atmosphere | 10 mbar Argon | Ambient Air or Vacuum |
| Spectral Resolution | ~0.1 nm | ~0.5 nm |
| Target Matrix | Heterogeneous Minerals/Ores | Homogeneous Alloys/Liquids |
| Data Throughput | Megapixel Imaging (25 $\mu$m) | Bulk Spot Analysis |

The high spatial resolution of 25 micrometers allows the LASO-LIBS system to resolve single mineral grains. This granularity means that the spectra processed by the ALIAS algorithm are often from nearly pure minerals but are set within a complex geological context, such as the Kittila gold mine samples, which include arsenopyrite, pyrite, and various silicate matrices.

## Diagnostic Analysis of the 12% Precision and 28% Recall

The catastrophic failure of the ALIAS implementation -- characterized by low precision (high false positives) and low recall (high false negatives) -- indicates a fundamental disconnect between the algorithm's statistical assumptions and the physical reality of the LIBS plasma.

### Precision Deficit: The Specificity Paradox in Complex Matrices

The 12% precision observed suggests that the algorithm is frequently identifying elements that are not present. This is primarily a result of the specificity weighting logic. In a geological sample containing high concentrations of Iron (Fe), Manganese (Mn), or Chromium (Cr), the spectrum is saturated with thousands of emission lines.

The ALIAS algorithm's specificity weight penalizes peaks located in crowded spectral regions. In an Iron-rich matrix, almost every viable wavelength for identification coincides with an Iron line. Consequently, the algorithm devalues the most reliable, high-intensity lines of trace elements if they reside in these "crowded" neighborhoods. This forces the matching logic to rely on weak, obscure lines that happen to be "unique" in the database. In a real-world spectrum, these weak lines are often indistinguishable from noise spikes, baseline fluctuations, or minor artifacts, leading to a high rate of false-positive identifications of rare elements.

Furthermore, the "database disagreement" issue exacerbates this. If the reference Kurucz database contains incorrect oscillator strengths or wavelength positions for certain transitions, the weighted vectors for these elements will be distorted. When the algorithm encounters a noise peak that superficially matches an incorrectly weighted database entry, it assigns a high similarity score, further eroding precision.

### Recall Deficit: Physical Decoupling and Matrix Effects

The 28% recall indicates that nearly three-quarters of the true elements are being missed. This is a consequence of the ALIAS algorithm's neglect of the thermodynamic laws governing plasma emission. The relative intensities of lines from a single element are strictly dictated by the Saha-Boltzmann distribution.

The population $N_j$ of an ionization state $j$ is related to the partition function $Z$ and the plasma temperature $T$ according to the Saha equation:

$$\frac{N_{j+1} N_e}{N_j} = \frac{(2 \pi m_e k T)^{3/2}}{h^3} \frac{2 Z_{j+1}(T)}{Z_j(T)} e^{-\frac{\chi_j}{k T}}$$

where $N_e$ is the electron density and $\chi_j$ is the ionization energy. The ALIAS VSM treats each line weight as a largely independent dimension. If the experimental plasma temperature in the LASO-LIBS setup (influenced by the Argon pressure and mineral hardness) deviates from the temperature assumed when building the reference vector, the "direction" of the sample vector will rotate away from the reference vector. Because ALIAS does not dynamically update its reference vectors based on calculated plasma parameters ($T_e, n_e$), it fails to recognize elements when their relative line intensities shift.

In geological samples, matrix effects -- variations in the ablation process due to physical properties like grain size and porosity -- can alter the total intensity and the continuum baseline. If the ALIAS implementation does not include robust baseline subtraction, the VSM similarity will be dominated by the shape of the Bremsstrahlung continuum rather than the discrete atomic peaks, leading to low recall for trace elements whose peaks are "buried" in the background.

## Comparative Analysis of Spectral Identification Algorithms

Aalto University researchers have explored various spectral matching algorithms, providing a benchmark against which the ALIAS failure can be measured. The software utilized in some Aalto projects includes five primary algorithms: Euclidean Distance, Spectral Angle Mapper (SAM), Linear Correlation, Multidimensional Normal Distribution (MND), and Spectral Information Divergence (SID).

### Spectral Angle Mapper (SAM) vs. ALIAS

SAM is closely related to the Cosine Similarity used in ALIAS but is often implemented with more rigid thresholding in the hyperspectral imaging community. SAM treats the spectrum as a vector and calculates the angle between it and a reference:

$$\alpha = \arccos\left(\frac{\sum s_i r_i}{\sqrt{\sum s_i^2} \sqrt{\sum r_i^2}}\right)$$

where $s$ and $r$ are the sample and reference spectra. Aalto's implementation of SAM and SID often uses several library spectra for each material type to express spectral variability. This contrasts with the ALIAS approach of using a single reference vector derived from a database. By not accounting for the variability in line ratios caused by the 10 mbar Argon environment, the ALIAS implementation lacks the flexibility inherent in the Aalto researchers' preferred methods.

### The Role of Multidimensional Normal Distribution (MND)

The MND algorithm used at Aalto attempts to model the statistical distribution of spectral features rather than treating them as static vectors. This is particularly relevant for geological LIBS, where shot-to-shot variability is high due to mineral heterogeneity. ALIAS's reliance on a deterministic weighting scheme based on a static database (Kurucz) makes it fragile in the face of the stochastic nature of laser-induced plasmas on raw rock surfaces.

| Algorithm | Mechanism | Strength | Weakness in LIBS |
|-----------|-----------|----------|------------------|
| ALIAS | Weighted VSM (TF-IDF style) | High throughput, automated | Sensitive to DB errors and matrix effects |
| SAM | Angular divergence | Insensitive to absolute intensity | Ignores thermodynamic line coupling |
| SID | Information theory (divergence) | Captures stochastic variations | Computationally intensive for megapixel data |
| MND | Statistical distribution | Robust to shot-to-shot noise | Requires large training/reference sets |
| CF-LIBS | Physical modeling ($T_e, n_e$) | Quantitative, no standards needed | Extremely sensitive to spectral accuracy |

## Methodological Pitfalls in Dataset Selection

An often-overlooked source of error in algorithm implementation is the selection of the validation dataset. The Aalto University spectral repository is diverse, and misinterpreting the nature of the "validated spectra" can lead to catastrophic results.

### Reflectance vs. Emission Spectroscopy

One prominent Aalto dataset, described as the "largest spectral measurement campaign of boreal tree species," contains reflectance and transmittance spectra (350-2500 nm) for leaves, needles, and bark. This dataset is intended for remote sensing applications and is fundamentally different from LIBS emission spectra. Reflectance spectra consist of broad, overlapping molecular features and pigments, whereas LIBS spectra consist of narrow, discrete atomic lines.

If the ALIAS implementation was inadvertently tested against the Aalto vegetation reflectance library rather than the LASO-LIBS mineral library, the 12% precision and 28% recall would be entirely expected. The ALIAS specificity weighting is designed for line-rich atomic spectra; applying it to the continuous, smooth curves of vegetation reflectance would result in the algorithm "searching" for non-existent peaks in the noise. However, even assuming the correct LIBS mineral dataset was used, the complexity of the "Aalto mineral library" (which includes thousands of ore and rock material samples validated by XRD and XRF) presents a much higher bar than the synthetic or alloy-based datasets used in the original ALIAS publication.

### Instrumental Broadening and Epsilon Sensitivity

The Aalto LASO-LIBS system's 0.1 nm resolution and the 5 ns pulse duration produce specific line profiles. The ALIAS algorithm's wavelength tolerance parameter $\epsilon$ is critical. If $\epsilon$ is set based on lower-resolution systems or theoretical database widths, it will fail to capture the peaks in the high-resolution Aalto data.

In LIBS, the line width (FWHM) is often dominated by Stark broadening, which is a function of the electron density $n_e$. In the Aalto 10 mbar Argon environment, the Stark broadening of the H-alpha line can reach 0.4-0.5 nm. If the ALIAS $\epsilon$ is significantly smaller than the physical line width (e.g., $\epsilon = 0.05$ nm), the algorithm will only consider the central "pixel" of the peak, ignoring the majority of the signal and making it highly vulnerable to spectrometer calibration drift. This leads to a massive loss in recall, as peaks are effectively "filtered out" by the narrow matching window.

## Thermodynamic and Geochemical Factors in Geological LIBS

Geological samples introduce variables that are rarely present in the materials used to validate ALIAS in the original Noel et al. (2025) paper. These include mineral hardness, porosity, and the presence of rare earth elements (REEs) with extremely dense emission patterns.

### Mineral Hardness and Ablation Efficiency

Research at Aalto has demonstrated that the mechanical properties of rocks, such as Uniaxial Compressive Strength (UCS) and Leeb Equotip Hardness, are correlated with the acoustic emissions generated during the LIBS process. Harder minerals like quartz yield different plasma conditions than softer minerals like calcite or gypsum. These mechanical variations lead to fluctuations in the "plasma temperature" and "ablation rate," which in turn change the relative intensities of the lines.

Because ALIAS is a purely statistical algorithm, it lacks a mechanism to normalize for these physical matrix effects. In contrast, the Calibration-Free LIBS (CF-LIBS) approach, which is also used by Aalto-affiliated researchers, explicitly calculates $T_e$ and $n_e$ to normalize the spectra. The 28% recall of ALIAS likely reflects its inability to handle the variability in line intensities caused by these mechanical-physical interactions.

### The Problem of Light Elements

One of the stated advantages of LIBS is its sensitivity to light elements like Lithium (Li) and Carbon (C), where X-ray fluorescence (XRF) is often inapplicable. However, these light elements often have only a few, very intense lines (e.g., Li I at 670.7 nm). In the ALIAS weighting scheme, if these lines are in a spectral region with many other database entries (such as a region with atmospheric or matrix overlaps), their "specificity" will be reduced, causing the algorithm to de-prioritize the very lines that are most diagnostic of the element's presence.

| Element Type | ALIAS Specificity | Detection Challenge in Aalto Data |
|-------------|-------------------|-----------------------------------|
| Major Metals (Fe, Mn) | Low (Many lines) | Over-representation/False positives |
| Light Elements (Li, B, Be) | High (Few lines) | Under-weighted if overlapping matrix lines |
| Trace Elements (Au, Ag) | Moderate | Signals buried in matrix "forest" |
| Rare Earths (REEs) | Very Low | Complex patterns confuse VSM distance |

## Root Causes of Performance Degradation

Synthesizing the data from the Aalto University repositories and the ALIAS algorithm's constraints, four fundamental errors in implementation can be identified.

### 1. Static Reference Vectors vs. Dynamic Plasma Conditions

The ALIAS algorithm's use of a static database (Kurucz) to generate reference vectors is its most significant flaw when applied to geological data. The Aalto setup's 10 mbar Argon atmosphere and 5 ns pulse produce specific excitation conditions. If the reference vectors were built using local thermodynamic equilibrium (LTE) assumptions that do not match the experimental plasma temperature, the VSM direction will be fundamentally incorrect. This mismatch is the primary driver of the 28% recall, as the algorithm fails to find a high-similarity match even when the element is present.

### 2. The "Bag-of-Lines" Fallacy

By treating spectral lines through a VSM/TF-IDF lens, ALIAS ignores the physical "coupling" of lines. In text, the word "Laser" and the word "Induced" can appear independently. In LIBS, the 288.1 nm Si line cannot exist without the other Si lines being present in ratios dictated by the Boltzmann distribution. ALIAS devalues "common" lines to emphasize "unique" ones, but in LIBS, the "common" lines are often the most physically reliable indicators of an element's presence. Devaluing them in favor of "rare" database lines (which might be noise) is what collapses precision to 12%.

### 3. Wavelength Neighborhood ($\epsilon$) Misalignment

The Aalto spectrometers' high resolution (0.1 nm) combined with the Stark broadening in the plasma (0.4 nm) requires a sophisticated approach to the wavelength tolerance $\epsilon$. A single, static $\epsilon$ value is insufficient. If $\epsilon$ is too small, it misses broadened peaks. If $\epsilon$ is too large, it blends adjacent lines from different elements, further devaluing the specificity weights and increasing the confusion between transition metals.

### 4. Inadequate Pre-processing for Geological Matrices

Geological spectra are characterized by high baseline noise and complex continua. Unlike the relatively "clean" spectra used in many ML/AI validations (such as the radiomics or genomic studies mentioned in related research), LIBS spectra require physical pre-processing. Without aggressive baseline subtraction and continuum normalization, the ALIAS VSM similarity is more a measure of the "spectral background" than of the elemental content.

## Proposed Optimization and Remediation Strategies

To align the ALIAS performance on the Aalto dataset with the published >95% benchmark, the following technical adjustments are necessary.

### Transition to Calibration-Free Informed Weighting

Instead of a purely statistical specificity weight, the algorithm should incorporate a Calibration-Free (CF-LIBS) pre-processing step. This involves estimating the plasma temperature $T_e$ and electron density $n_e$ from the spectrum using the H-alpha line or the multi-elemental Saha-Boltzmann (MESB) plots. The reference vectors $V_E$ should then be dynamically generated for that specific $T_e$ and $n_e$, ensuring the line ratios in the database match those in the experimental plasma.

### Implementation of an Adaptive $\epsilon$ Window

The wavelength neighborhood $\epsilon$ used for specificity weighting and vector construction should not be a constant. It should be an adaptive function of the wavelength and the calculated electron density:

$$\epsilon(\lambda) = \text{FWHM}_{inst} + \omega_{Stark}(\lambda, n_e)$$

where $\text{FWHM}_{inst}$ is the instrumental broadening and $\omega_{Stark}$ is the Stark broadening coefficient. This would ensure that the VSM captures the entire integrated intensity of the peak, rather than just a single, jitter-sensitive pixel.

### Incorporating Matrix-Matched Sub-Libraries

The Aalto University "mineral library" should be used to create specific "sub-spaces" within the VSM. Rather than comparing a sample to every element in the Kurucz database (which contains over 100 elements), the algorithm should first use the 25-micrometer resolution data to classify the matrix (e.g., "silicate," "sulfide," or "carbonate") using a preliminary MND or SAM classifier. Once the matrix is identified, the ALIAS weighting can be applied using a reduced, matrix-specific reference set. This reduces the "specificity penalty" caused by irrelevant elements in the global database and will significantly boost precision.

### Leveraging Auxiliary Data for Spectral Gating

Aalto's inclusion of acoustic emissions and LiDAR 3D point clouds provides a unique opportunity for "spectral gating". Spectra acquired from highly porous or fractured areas of the drill core (identifiable via LiDAR and RQD assessments) will have different plasma dynamics than those from solid, intact rock. By using these auxiliary data streams to select specialized ALIAS weighting parameters for different rock textures, the recall and precision deficits can be mitigated.

## Concluding Synthesis

The investigation into the 12% precision and 28% recall of the ALIAS algorithm on Aalto University validated LIBS spectra points to a confluence of theoretical and experimental factors. The algorithm, as originally proposed, is a powerful tool for high-throughput data processing, but its reliance on statistical "specificity" weighting is fundamentally at odds with the physical behavior of transition-metal-rich geological plasmas. In these environments, the most "common" lines are also the most "reliable," and devaluing them leads to the misinterpretation of noise as rare elemental signals.

To achieve expert-level performance, the ALIAS framework must be evolved from a "blind" information retrieval model into a physically-informed chemometric system. This requires the integration of the Aalto-specific environmental parameters -- specifically the 10 mbar Argon atmosphere and high-resolution spectrometer profiles -- into a dynamic vector-space construction process. By replacing static database weights with thermodynamic line-coupling logic and adaptive wavelength windows, the ALIAS algorithm can be successfully adapted for the rigors of geological and mining applications, fulfilling its potential as a transformative tool for rapid elemental identification.
