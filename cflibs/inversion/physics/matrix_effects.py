"""
Matrix effect correction framework for CF-LIBS.

Matrix effects arise from differences in laser-matter interaction across different
sample types. This module provides:

1. **MatrixType** - Classification of sample matrices
2. **MatrixEffectCorrector** - Applies empirical corrections based on matrix type
3. **InternalStandardizer** - Normalizes concentrations using internal standard ratios
4. **CorrectionFactorDB** - Database of empirical correction factors

Physics Background
------------------
In LIBS, matrix effects manifest as:
- Variations in ablation efficiency (mass removed per pulse)
- Changes in plasma temperature and electron density
- Modified emission line intensities independent of concentration

Standard CF-LIBS assumes these effects cancel in the closure equation, but this
assumption breaks down for:
- Samples with very different thermal/optical properties
- Organic matrices (lower ablation temperatures)
- Highly reflective metallic samples
- Geological samples with complex mineralogy

References
----------
- Hahn & Omenetto (2010): LIBS: Part I. Fundamentals and Diagnostics
- Cremers & Radziemski (2013): Handbook of LIBS, Chapter 6
- Tognoni et al. (2010): Signal and noise in LIBS - Review
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Optional, Tuple, Set
import json
from pathlib import Path

from cflibs.core.logging_config import get_logger
from cflibs.inversion.solver import CFLIBSResult

logger = get_logger("inversion.matrix_effects")


class MatrixType(Enum):
    """
    Classification of sample matrix types for correction selection.

    Each matrix type has characteristic laser-matter interaction properties
    that affect LIBS measurements systematically.

    Attributes
    ----------
    METALLIC : auto
        Metallic alloys (Fe, Al, Cu, Ti alloys). High reflectivity, high
        ablation threshold, efficient plasma coupling.
    OXIDE : auto
        Metal oxides and ceramics. Lower thermal conductivity, different
        ablation dynamics than pure metals.
    ORGANIC : auto
        Biological tissues, polymers, wood. Low ablation threshold, lower
        plasma temperatures, molecular emission bands.
    GEOLOGICAL : auto
        Rocks, minerals, soils. Heterogeneous composition, variable grain
        sizes, often analyzed as pressed pellets.
    GLASS : auto
        Silicate glasses, slag. Amorphous structure, relatively homogeneous.
    LIQUID : auto
        Aqueous solutions, oils. Analyzed via dried residue or jet.
    UNKNOWN : auto
        Unclassified matrix. No correction applied.
    """

    METALLIC = auto()
    OXIDE = auto()
    ORGANIC = auto()
    GEOLOGICAL = auto()
    GLASS = auto()
    LIQUID = auto()
    UNKNOWN = auto()


@dataclass
class MatrixClassificationResult:
    """
    Result of matrix type classification.

    Attributes
    ----------
    matrix_type : MatrixType
        Classified matrix type
    confidence : float
        Confidence score (0-1) for the classification
    evidence : Dict[str, float]
        Element fractions or other evidence used for classification
    notes : str
        Human-readable explanation of classification
    """

    matrix_type: MatrixType
    confidence: float
    evidence: Dict[str, float]
    notes: str


@dataclass
class CorrectionFactor:
    """
    Empirical correction factor for a specific element in a matrix type.

    The correction is applied as:
        C_corrected = C_measured * multiplicative + additive

    Attributes
    ----------
    element : str
        Element symbol this correction applies to
    matrix_type : MatrixType
        Matrix type this correction was derived for
    multiplicative : float
        Multiplicative correction factor (default 1.0 = no correction)
    additive : float
        Additive correction factor (default 0.0 = no correction)
    uncertainty : float
        Estimated uncertainty in the correction (1-sigma, fractional)
    source : str
        Reference or calibration dataset source
    valid_range : Tuple[float, float]
        Concentration range (min, max) where correction is valid
    """

    element: str
    matrix_type: MatrixType
    multiplicative: float = 1.0
    additive: float = 0.0
    uncertainty: float = 0.1
    source: str = "default"
    valid_range: Tuple[float, float] = (0.0, 1.0)

    def apply(self, concentration: float) -> Tuple[float, float]:
        """
        Apply correction to a concentration value.

        Parameters
        ----------
        concentration : float
            Measured concentration (mass fraction)

        Returns
        -------
        Tuple[float, float]
            (corrected_concentration, correction_uncertainty)
        """
        corrected = concentration * self.multiplicative + self.additive
        # Propagate uncertainty
        uncert = abs(corrected) * self.uncertainty
        return corrected, uncert


class CorrectionFactorDB:
    """
    Database of empirical correction factors for matrix effects.

    Stores and retrieves correction factors organized by matrix type and element.
    Can be populated from JSON files or programmatically.

    Examples
    --------
    >>> db = CorrectionFactorDB()
    >>> db.add_factor(CorrectionFactor("Fe", MatrixType.METALLIC, 1.05))
    >>> factor = db.get_factor("Fe", MatrixType.METALLIC)
    >>> corrected, uncert = factor.apply(0.50)
    """

    # Default correction factors: matrix_type -> {element: (multiplicative, uncertainty)}
    _DEFAULT_FACTORS: Dict[MatrixType, Dict[str, Tuple[float, float]]] = {
        MatrixType.METALLIC: {
            "Fe": (1.0, 0.05),
            "Cr": (1.02, 0.08),
            "Ni": (0.98, 0.06),
            "Mn": (1.05, 0.10),
            "Si": (0.95, 0.12),
            "C": (0.85, 0.15),
            "Al": (1.0, 0.06),
            "Cu": (0.98, 0.05),
            "Ti": (1.03, 0.07),
        },
        MatrixType.OXIDE: {
            "Si": (1.10, 0.12),
            "Al": (1.08, 0.10),
            "Fe": (1.15, 0.12),
            "Ca": (1.05, 0.08),
            "Mg": (1.12, 0.10),
            "Na": (0.90, 0.15),
            "K": (0.88, 0.15),
        },
        MatrixType.GEOLOGICAL: {
            "Si": (1.08, 0.10),
            "Al": (1.05, 0.10),
            "Fe": (1.12, 0.12),
            "Ca": (1.02, 0.08),
            "Mg": (1.08, 0.10),
            "Ti": (1.10, 0.12),
            "Mn": (1.15, 0.15),
        },
        MatrixType.ORGANIC: {
            "C": (0.70, 0.20),
            "H": (0.80, 0.25),
            "N": (0.85, 0.20),
            "O": (0.90, 0.15),
            "Ca": (1.20, 0.15),
            "K": (0.85, 0.18),
            "Na": (0.82, 0.18),
            "Mg": (1.15, 0.15),
            "P": (1.10, 0.15),
            "S": (1.05, 0.12),
        },
    }

    def __init__(self) -> None:
        """Initialize correction factor database with defaults."""
        self._factors: Dict[MatrixType, Dict[str, CorrectionFactor]] = {mt: {} for mt in MatrixType}
        self._populate_defaults()

    def _populate_defaults(self) -> None:
        """Populate database with default correction factors from literature."""
        for matrix_type, elements in self._DEFAULT_FACTORS.items():
            for el, (mult, uncert) in elements.items():
                self._factors[matrix_type][el] = CorrectionFactor(
                    element=el,
                    matrix_type=matrix_type,
                    multiplicative=mult,
                    uncertainty=uncert,
                    source="default_literature",
                )

    def add_factor(self, factor: CorrectionFactor) -> None:
        """
        Add or update a correction factor.

        Parameters
        ----------
        factor : CorrectionFactor
            Correction factor to add
        """
        self._factors[factor.matrix_type][factor.element] = factor
        logger.debug(
            f"Added correction factor for {factor.element} in "
            f"{factor.matrix_type.name}: mult={factor.multiplicative:.3f}"
        )

    def get_factor(self, element: str, matrix_type: MatrixType) -> Optional[CorrectionFactor]:
        """
        Retrieve correction factor for an element in a matrix type.

        Parameters
        ----------
        element : str
            Element symbol
        matrix_type : MatrixType
            Matrix type

        Returns
        -------
        CorrectionFactor or None
            Correction factor if found, None otherwise
        """
        return self._factors[matrix_type].get(element)

    def get_factors_for_matrix(self, matrix_type: MatrixType) -> Dict[str, CorrectionFactor]:
        """
        Get all correction factors for a matrix type.

        Parameters
        ----------
        matrix_type : MatrixType
            Matrix type

        Returns
        -------
        Dict[str, CorrectionFactor]
            Dictionary mapping element symbols to correction factors
        """
        return self._factors[matrix_type].copy()

    def save_to_json(self, filepath: Path) -> None:
        """
        Save correction factors to JSON file.

        Parameters
        ----------
        filepath : Path
            Output file path
        """
        data = {}
        for matrix_type, elements in self._factors.items():
            data[matrix_type.name] = {
                el: {
                    "multiplicative": cf.multiplicative,
                    "additive": cf.additive,
                    "uncertainty": cf.uncertainty,
                    "source": cf.source,
                    "valid_range": list(cf.valid_range),
                }
                for el, cf in elements.items()
            }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved correction factors to {filepath}")

    def load_from_json(self, filepath: Path) -> None:
        """
        Load correction factors from JSON file.

        Parameters
        ----------
        filepath : Path
            Input file path
        """
        with open(filepath) as f:
            data = json.load(f)

        for matrix_name, elements in data.items():
            try:
                matrix_type = MatrixType[matrix_name]
            except KeyError:
                logger.warning(f"Unknown matrix type '{matrix_name}' in file, skipping")
                continue

            for el, cf_data in elements.items():
                factor = CorrectionFactor(
                    element=el,
                    matrix_type=matrix_type,
                    multiplicative=cf_data.get("multiplicative", 1.0),
                    additive=cf_data.get("additive", 0.0),
                    uncertainty=cf_data.get("uncertainty", 0.1),
                    source=cf_data.get("source", str(filepath)),
                    valid_range=tuple(cf_data.get("valid_range", [0.0, 1.0])),
                )
                self.add_factor(factor)

        logger.info(f"Loaded correction factors from {filepath}")

    @property
    def elements(self) -> Set[str]:
        """Get all elements with correction factors."""
        all_elements: Set[str] = set()
        for elements in self._factors.values():
            all_elements.update(elements.keys())
        return all_elements


@dataclass
class MatrixCorrectionResult:
    """
    Result of applying matrix effect corrections.

    Attributes
    ----------
    original_concentrations : Dict[str, float]
        Input concentrations before correction
    corrected_concentrations : Dict[str, float]
        Concentrations after matrix correction
    correction_uncertainties : Dict[str, float]
        Uncertainties introduced by corrections
    matrix_type : MatrixType
        Matrix type used for correction selection
    factors_applied : Dict[str, CorrectionFactor]
        Correction factors that were applied
    renormalized : bool
        Whether concentrations were renormalized to sum to 1
    """

    original_concentrations: Dict[str, float]
    corrected_concentrations: Dict[str, float]
    correction_uncertainties: Dict[str, float]
    matrix_type: MatrixType
    factors_applied: Dict[str, CorrectionFactor]
    renormalized: bool


def _dummy_cflibs_result(concentrations: Dict[str, float]) -> CFLIBSResult:
    """Create a minimal CFLIBSResult for compatibility."""
    return CFLIBSResult(
        temperature_K=10000.0,
        temperature_uncertainty_K=0.0,
        electron_density_cm3=1e17,
        concentrations=concentrations,
        concentration_uncertainties={},
        iterations=0,
        converged=True,
    )


class MatrixEffectCorrector:
    """
    Applies matrix-specific corrections to CF-LIBS concentration results.

    This class:
    1. Classifies the sample matrix based on composition
    2. Retrieves appropriate correction factors
    3. Applies corrections and optionally renormalizes

    Examples
    --------
    >>> corrector = MatrixEffectCorrector()
    >>> result = solver.solve(observations)
    >>> corrected = corrector.correct(result)
    >>> print(corrected.corrected_concentrations)

    Notes
    -----
    Matrix classification uses heuristics based on major element composition.
    For best results, specify the matrix type explicitly if known.
    """

    # Element sets for classification
    METALLIC_ELEMENTS = {"Fe", "Cr", "Ni", "Co", "Cu", "Zn", "Al", "Ti", "Mn", "Mo", "W", "V"}
    ORGANIC_ELEMENTS = {"C", "H", "N", "O", "S", "P"}
    GEOLOGICAL_MAJORS = {"Si", "Al", "Fe", "Ca", "Mg", "Na", "K", "Ti"}

    def __init__(
        self,
        correction_db: Optional[CorrectionFactorDB] = None,
        renormalize: bool = True,
    ) -> None:
        """
        Initialize the matrix effect corrector.

        Parameters
        ----------
        correction_db : CorrectionFactorDB, optional
            Database of correction factors. If None, uses default database.
        renormalize : bool
            Whether to renormalize concentrations to sum to 1 after correction
        """
        self.correction_db = correction_db or CorrectionFactorDB()
        self.renormalize = renormalize

    def classify(
        self,
        concentrations: Dict[str, float],
        known_type: Optional[MatrixType] = None,
    ) -> MatrixClassificationResult:
        """
        Classify the sample matrix based on composition.

        Parameters
        ----------
        concentrations : Dict[str, float]
            Element concentrations (mass fractions)
        known_type : MatrixType, optional
            If specified, skip classification and use this type

        Returns
        -------
        MatrixClassificationResult
            Classification result with confidence and evidence
        """
        if known_type is not None:
            return MatrixClassificationResult(
                matrix_type=known_type,
                confidence=1.0,
                evidence=concentrations.copy(),
                notes="Matrix type specified by user",
            )

        total = sum(concentrations.values())
        if total == 0:
            return MatrixClassificationResult(
                matrix_type=MatrixType.UNKNOWN,
                confidence=0.0,
                evidence={},
                notes="No concentration data",
            )

        # Normalize for analysis
        norm_conc = {el: c / total for el, c in concentrations.items()}

        # Calculate fraction in each category
        metallic_fraction = sum(norm_conc.get(el, 0) for el in self.METALLIC_ELEMENTS)
        organic_fraction = sum(norm_conc.get(el, 0) for el in self.ORGANIC_ELEMENTS)
        geological_fraction = sum(norm_conc.get(el, 0) for el in self.GEOLOGICAL_MAJORS)

        # Check for specific patterns
        o_fraction = norm_conc.get("O", 0)
        si_fraction = norm_conc.get("Si", 0)
        c_fraction = norm_conc.get("C", 0)
        fe_fraction = norm_conc.get("Fe", 0)

        evidence = {
            "metallic_fraction": metallic_fraction,
            "organic_fraction": organic_fraction,
            "geological_fraction": geological_fraction,
            "oxygen_fraction": o_fraction,
        }

        # Classification logic
        # High carbon + organic elements -> ORGANIC
        if c_fraction > 0.20 and organic_fraction > 0.50:
            return MatrixClassificationResult(
                matrix_type=MatrixType.ORGANIC,
                confidence=min(0.9, organic_fraction),
                evidence=evidence,
                notes=f"High carbon ({c_fraction:.1%}) and organic element content",
            )

        # High metallic elements, low oxygen -> METALLIC
        if metallic_fraction > 0.60 and o_fraction < 0.10:
            return MatrixClassificationResult(
                matrix_type=MatrixType.METALLIC,
                confidence=min(0.95, metallic_fraction),
                evidence=evidence,
                notes=f"High metallic content ({metallic_fraction:.1%}), low oxygen",
            )

        # Significant oxygen + silicon -> OXIDE or GEOLOGICAL or GLASS
        if o_fraction > 0.30 and si_fraction > 0.15:
            # Glass: very high Si + O, low other geological elements
            al_fraction = norm_conc.get("Al", 0)
            ca_fraction = norm_conc.get("Ca", 0)

            if si_fraction > 0.25 and (al_fraction + ca_fraction) < 0.10:
                return MatrixClassificationResult(
                    matrix_type=MatrixType.GLASS,
                    confidence=0.7,
                    evidence=evidence,
                    notes=f"High Si ({si_fraction:.1%}) + O, low Al+Ca suggests glass",
                )

            # Geological: multiple major elements present
            major_count = sum(1 for el in self.GEOLOGICAL_MAJORS if norm_conc.get(el, 0) > 0.02)
            if major_count >= 4:
                return MatrixClassificationResult(
                    matrix_type=MatrixType.GEOLOGICAL,
                    confidence=0.8,
                    evidence=evidence,
                    notes=f"Multiple geological majors present ({major_count} elements >2%)",
                )

            # Default to OXIDE
            return MatrixClassificationResult(
                matrix_type=MatrixType.OXIDE,
                confidence=0.7,
                evidence=evidence,
                notes=f"Significant oxygen ({o_fraction:.1%}) + Si content",
            )

        # High Fe + moderate O -> could be iron oxide or steel
        if fe_fraction > 0.50 and o_fraction > 0.10:
            return MatrixClassificationResult(
                matrix_type=MatrixType.OXIDE,
                confidence=0.7,
                evidence=evidence,
                notes=f"High Fe ({fe_fraction:.1%}) with oxygen suggests oxide",
            )

        # Default: highest fraction determines type
        if metallic_fraction >= organic_fraction and metallic_fraction >= geological_fraction:
            return MatrixClassificationResult(
                matrix_type=MatrixType.METALLIC,
                confidence=0.5,
                evidence=evidence,
                notes="Default classification based on metallic elements",
            )

        if geological_fraction > organic_fraction:
            return MatrixClassificationResult(
                matrix_type=MatrixType.GEOLOGICAL,
                confidence=0.5,
                evidence=evidence,
                notes="Default classification based on geological elements",
            )

        return MatrixClassificationResult(
            matrix_type=MatrixType.UNKNOWN,
            confidence=0.3,
            evidence=evidence,
            notes="Could not confidently classify matrix",
        )

    def correct(
        self,
        result: CFLIBSResult,
        matrix_type: Optional[MatrixType] = None,
    ) -> MatrixCorrectionResult:
        """
        Apply matrix effect corrections to CF-LIBS results.

        Parameters
        ----------
        result : CFLIBSResult
            CF-LIBS inversion result with concentrations
        matrix_type : MatrixType, optional
            Matrix type to use. If None, will be classified automatically.

        Returns
        -------
        MatrixCorrectionResult
            Corrected concentrations with metadata
        """
        concentrations = result.concentrations.copy()

        # Classify matrix if not specified
        classification = self.classify(concentrations, matrix_type)
        effective_type = classification.matrix_type

        logger.info(
            f"Matrix classification: {effective_type.name} "
            f"(confidence: {classification.confidence:.2f})"
        )

        if effective_type == MatrixType.UNKNOWN:
            logger.warning("Unknown matrix type - no corrections applied")
            return MatrixCorrectionResult(
                original_concentrations=concentrations,
                corrected_concentrations=concentrations.copy(),
                correction_uncertainties={el: 0.0 for el in concentrations},
                matrix_type=effective_type,
                factors_applied={},
                renormalized=False,
            )

        # Apply corrections
        corrected: Dict[str, float] = {}
        uncertainties: Dict[str, float] = {}
        factors_applied: Dict[str, CorrectionFactor] = {}

        for element, conc in concentrations.items():
            factor = self.correction_db.get_factor(element, effective_type)

            if factor is not None:
                corrected_val, uncert = factor.apply(conc)
                corrected[element] = corrected_val
                uncertainties[element] = uncert
                factors_applied[element] = factor
                logger.debug(
                    f"{element}: {conc:.4f} -> {corrected_val:.4f} "
                    f"(factor: {factor.multiplicative:.3f})"
                )
            else:
                # No correction available - pass through
                corrected[element] = conc
                uncertainties[element] = 0.0
                logger.debug(f"{element}: no correction factor, passing through")

        # Optionally renormalize
        renormalized = False
        if self.renormalize:
            total = sum(corrected.values())
            if total > 0 and abs(total - 1.0) > 1e-6:
                corrected = {el: c / total for el, c in corrected.items()}
                # Scale uncertainties accordingly
                uncertainties = {el: u / total for el, u in uncertainties.items()}
                renormalized = True
                logger.debug(f"Renormalized concentrations (total was {total:.4f})")

        return MatrixCorrectionResult(
            original_concentrations=concentrations,
            corrected_concentrations=corrected,
            correction_uncertainties=uncertainties,
            matrix_type=effective_type,
            factors_applied=factors_applied,
            renormalized=renormalized,
        )

    def correct_concentrations(
        self,
        concentrations: Dict[str, float],
        matrix_type: Optional[MatrixType] = None,
    ) -> MatrixCorrectionResult:
        """
        Apply corrections to raw concentration dictionary.

        Convenience method when you have concentrations but not a full CFLIBSResult.

        Parameters
        ----------
        concentrations : Dict[str, float]
            Element concentrations
        matrix_type : MatrixType, optional
            Matrix type. If None, will be classified.

        Returns
        -------
        MatrixCorrectionResult
        """
        # Create a minimal CFLIBSResult for compatibility
        dummy_result = _dummy_cflibs_result(concentrations)
        return self.correct(dummy_result, matrix_type)


@dataclass
class InternalStandardResult:
    """
    Result of internal standardization.

    Attributes
    ----------
    original_concentrations : Dict[str, float]
        Input concentrations
    standardized_concentrations : Dict[str, float]
        Concentrations normalized to internal standard
    internal_standard : str
        Element used as internal standard
    standard_concentration : float
        Assumed concentration of internal standard
    scale_factor : float
        Ratio of assumed to measured standard concentration
    """

    original_concentrations: Dict[str, float]
    standardized_concentrations: Dict[str, float]
    internal_standard: str
    standard_concentration: float
    scale_factor: float


class InternalStandardizer:
    """
    Normalizes CF-LIBS concentrations using an internal standard element.

    Internal standardization corrects for shot-to-shot variations in:
    - Laser energy fluctuations
    - Ablation efficiency variations
    - Plasma property changes

    The method requires knowing (or assuming) the concentration of one element,
    then scales all other concentrations relative to it.

    Examples
    --------
    >>> standardizer = InternalStandardizer("Fe", known_concentration=0.65)
    >>> result = standardizer.standardize(cflibs_result)
    >>> print(f"Scale factor: {result.scale_factor:.3f}")

    Notes
    -----
    The internal standard should be:
    - Present in significant quantity (>1%)
    - Have well-characterized spectral lines
    - Be relatively homogeneously distributed
    - Not subject to severe self-absorption
    """

    def __init__(
        self,
        standard_element: str,
        known_concentration: float,
    ) -> None:
        """
        Initialize internal standardizer.

        Parameters
        ----------
        standard_element : str
            Element to use as internal standard
        known_concentration : float
            Known (or assumed) concentration of standard element (mass fraction)
        """
        if known_concentration <= 0 or known_concentration > 1:
            raise ValueError(f"Known concentration must be in (0, 1], got {known_concentration}")

        self.standard_element = standard_element
        self.known_concentration = known_concentration

    def standardize(
        self,
        result: CFLIBSResult,
        propagate_uncertainty: bool = True,
    ) -> InternalStandardResult:
        """
        Apply internal standardization to CF-LIBS results.

        Parameters
        ----------
        result : CFLIBSResult
            CF-LIBS inversion result
        propagate_uncertainty : bool
            Whether to propagate uncertainty from standard (not yet implemented)

        Returns
        -------
        InternalStandardResult
            Standardized concentrations
        """
        concentrations = result.concentrations

        if self.standard_element not in concentrations:
            raise ValueError(
                f"Internal standard element '{self.standard_element}' "
                f"not found in concentrations: {list(concentrations.keys())}"
            )

        measured_standard = concentrations[self.standard_element]

        if measured_standard <= 0:
            raise ValueError(
                f"Measured concentration of standard '{self.standard_element}' "
                f"is zero or negative: {measured_standard}"
            )

        # Calculate scale factor
        scale_factor = self.known_concentration / measured_standard

        # Apply scaling
        standardized = {el: conc * scale_factor for el, conc in concentrations.items()}

        # The standard element should now have the known concentration
        standardized[self.standard_element] = self.known_concentration

        logger.info(
            f"Internal standardization: {self.standard_element} "
            f"({measured_standard:.4f} -> {self.known_concentration:.4f}), "
            f"scale factor = {scale_factor:.3f}"
        )

        return InternalStandardResult(
            original_concentrations=concentrations.copy(),
            standardized_concentrations=standardized,
            internal_standard=self.standard_element,
            standard_concentration=self.known_concentration,
            scale_factor=scale_factor,
        )

    def standardize_concentrations(
        self,
        concentrations: Dict[str, float],
    ) -> InternalStandardResult:
        """
        Apply internal standardization to raw concentrations.

        Convenience method when you have concentrations but not a full CFLIBSResult.

        Parameters
        ----------
        concentrations : Dict[str, float]
            Element concentrations

        Returns
        -------
        InternalStandardResult
        """
        dummy_result = _dummy_cflibs_result(concentrations)
        return self.standardize(dummy_result)

    def compute_ratios(
        self,
        concentrations: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute concentration ratios relative to internal standard.

        Parameters
        ----------
        concentrations : Dict[str, float]
            Element concentrations

        Returns
        -------
        Dict[str, float]
            Ratios C_element / C_standard for each element
        """
        if self.standard_element not in concentrations:
            raise ValueError(f"Standard element '{self.standard_element}' not in concentrations")

        standard_conc = concentrations[self.standard_element]
        if standard_conc <= 0:
            raise ValueError("Standard concentration is zero or negative")

        return {el: conc / standard_conc for el, conc in concentrations.items()}


def combine_corrections(
    result: CFLIBSResult,
    internal_standard: Optional[str] = None,
    standard_concentration: Optional[float] = None,
    matrix_type: Optional[MatrixType] = None,
    correction_db: Optional[CorrectionFactorDB] = None,
) -> Dict[str, float]:
    """
    Convenience function to apply both internal standardization and matrix corrections.

    Order of operations:
    1. Internal standardization (if specified)
    2. Matrix effect correction

    Parameters
    ----------
    result : CFLIBSResult
        CF-LIBS inversion result
    internal_standard : str, optional
        Element to use as internal standard
    standard_concentration : float, optional
        Known concentration of internal standard
    matrix_type : MatrixType, optional
        Matrix type for corrections
    correction_db : CorrectionFactorDB, optional
        Custom correction factor database

    Returns
    -------
    Dict[str, float]
        Final corrected concentrations
    """
    concentrations = result.concentrations.copy()

    # Step 1: Internal standardization
    if internal_standard is not None and standard_concentration is not None:
        standardizer = InternalStandardizer(internal_standard, standard_concentration)
        std_result = standardizer.standardize(result)
        concentrations = std_result.standardized_concentrations

        # Create updated result for matrix correction
        result = CFLIBSResult(
            temperature_K=result.temperature_K,
            temperature_uncertainty_K=result.temperature_uncertainty_K,
            electron_density_cm3=result.electron_density_cm3,
            concentrations=concentrations,
            concentration_uncertainties=result.concentration_uncertainties,
            iterations=result.iterations,
            converged=result.converged,
            quality_metrics=result.quality_metrics,
            electron_density_uncertainty_cm3=result.electron_density_uncertainty_cm3,
            boltzmann_covariance=result.boltzmann_covariance,
        )

    # Step 2: Matrix effect correction
    corrector = MatrixEffectCorrector(
        correction_db=correction_db,
        renormalize=True,
    )
    matrix_result = corrector.correct(result, matrix_type)

    return matrix_result.corrected_concentrations
