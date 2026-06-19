"""
Matrix effect correction framework for CF-LIBS.

Matrix effects arise from differences in laser-matter interaction across different
sample types. This module provides:

1. **MatrixType** - Classification of sample matrices
2. **MatrixEffectCorrector** - Applies empirical corrections based on matrix type
3. **InternalStandardizer** - Normalizes concentrations using internal standard ratios
4. **CorrectionFactorDB** - Database of empirical correction factors
5. **OnePointCalibrator** - OPT-IN one-point calibration (OPC) of the Boltzmann
   plot using a single certified reference standard (Cavalcanti et al., 2013)

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
- Cavalcanti, Teixeira, Legnaioli, Lorenzetti, Pardini & Palleschi (2013):
  "One-point calibration for calibration-free laser-induced breakdown
  spectroscopy quantitative analysis", Spectrochim. Acta B 87, 51-56,
  doi:10.1016/j.sab.2013.05.016  (basis for :class:`OnePointCalibrator`)
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import ClassVar, Dict, List, Optional, Sequence, Tuple, Set, Union
import json
from pathlib import Path

import numpy as np

from cflibs.core.constants import KB_EV
from cflibs.core.logging_config import get_logger
from cflibs.inversion.common import LineObservation
from cflibs.inversion.solve.iterative import CFLIBSResult

logger = get_logger("inversion.matrix_effects")


class AblationRegime(Enum):
    """
    Laser ablation regime governing the empirical matrix-correction factor set.

    The default empirical correction factors in :class:`CorrectionFactorDB` were
    derived from *nanosecond*-pulse LIBS, where thermal/differential ablation
    drives strongly element-dependent, non-stoichiometric mass removal (e.g. the
    ~15% carbon loss baked into ``METALLIC.C = 0.85``). These factors are not
    physically appropriate for ultrashort-pulse regimes.

    Attributes
    ----------
    NS : auto
        Nanosecond-pulse LIBS (default). Uses the historical empirical
        ns-derived correction factors. Behavior is unchanged from prior
        versions of this module.
    PS : auto
        Picosecond / ultrashort-pulse LIBS. Ablation is much closer to
        stoichiometric (negligible thermal/differential ablation), so the
        generic default factors collapse to 1.0 for every element. Supply a
        calibrated :class:`CorrectionFactorDB` (or load one from JSON / add
        factors) when matrix-matched ps calibration data is available.

    Notes
    -----
    Picosecond ablation is *more* stoichiometric than nanosecond ablation, but
    it is not perfectly so for every matrix; the 1.0 ps defaults are an
    explicit "no generic correction known" placeholder rather than a claim of
    perfect stoichiometry. See Hahn & Omenetto (2010), Cremers & Radziemski
    (2013) for the ns empirical basis and the regime dependence of ablation.

    References
    ----------
    - Hahn & Omenetto (2010): LIBS Part I. Fundamentals and Diagnostics.
    - Cremers & Radziemski (2013): Handbook of LIBS, Chapter 6.
    - Russo et al. (2002): Femtosecond vs nanosecond laser ablation —
      reduced fractionation / more stoichiometric removal at ultrashort pulses.
    """

    NS = auto()
    PS = auto()

    @classmethod
    def coerce(cls, value: "Union[AblationRegime, str]") -> "AblationRegime":
        """
        Coerce a string or :class:`AblationRegime` into an :class:`AblationRegime`.

        Accepts the enum directly or a case-insensitive string ('ns', 'ps').

        Parameters
        ----------
        value : AblationRegime or str
            Regime specifier.

        Returns
        -------
        AblationRegime

        Raises
        ------
        ValueError
            If a string value does not name a known regime.
        """
        if isinstance(value, cls):
            return value
        try:
            return cls[str(value).strip().upper()]
        except KeyError as exc:
            valid = ", ".join(repr(r.name.lower()) for r in cls)
            raise ValueError(f"Unknown ablation_regime {value!r}; expected one of {valid}") from exc


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

    Notes
    -----
    The built-in ``_DEFAULT_FACTORS`` were derived from *nanosecond* LIBS and
    encode element-dependent, non-stoichiometric ablation (see
    :class:`AblationRegime`). For ``ablation_regime=AblationRegime.PS`` the
    generic defaults collapse to 1.0 (no correction) because ultrashort-pulse
    ablation is approximately stoichiometric; provide calibrated factors when
    available.
    """

    # Process-level record of which (uncalibrated) default regimes have already
    # emitted the generic-defaults warning, so it fires at most once per regime.
    _warned_default_regimes: ClassVar[Set[AblationRegime]] = set()

    # Default correction factors: matrix_type -> {element: (multiplicative, uncertainty)}
    # NANOSECOND-regime empirical factors (the historical default).
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

    def __init__(
        self,
        ablation_regime: Union[AblationRegime, str] = AblationRegime.NS,
    ) -> None:
        """
        Initialize correction factor database with regime-appropriate defaults.

        Parameters
        ----------
        ablation_regime : AblationRegime or str, optional
            Laser ablation regime selecting the default factor set. Defaults to
            :attr:`AblationRegime.NS` (nanosecond), which reproduces the
            historical empirical factors exactly (no behavior change). Use
            ``AblationRegime.PS`` (or ``"ps"``) for picosecond / ultrashort-pulse
            data, which uses stoichiometric (1.0) generic defaults; override with
            calibrated factors via :meth:`add_factor` / :meth:`load_from_json`
            when matrix-matched ps calibration is available.
        """
        self.ablation_regime = AblationRegime.coerce(ablation_regime)
        self._factors: Dict[MatrixType, Dict[str, CorrectionFactor]] = {mt: {} for mt in MatrixType}
        self._populate_defaults()

    def _default_factors_for_regime(self) -> Dict[MatrixType, Dict[str, Tuple[float, float]]]:
        """
        Return the default ``{matrix_type: {element: (mult, uncert)}}`` factor
        table for the active ablation regime.

        For NS this is the historical ns-empirical table. For PS the factor set
        is *stoichiometric*: every element covered by the ns defaults maps to a
        multiplicative factor of exactly 1.0 (no generic correction). We mirror
        the ns element coverage rather than inventing ps-specific numbers, since
        ps ablation is approximately stoichiometric and no validated generic ps
        correction factors exist in the cited literature.
        """
        if self.ablation_regime is AblationRegime.NS:
            return self._DEFAULT_FACTORS
        # PS: stoichiometric generic defaults (1.0). Carry the ns uncertainty so
        # the placeholder still propagates a finite, regime-honest uncertainty.
        return {
            matrix_type: {el: (1.0, uncert) for el, (_mult, uncert) in elements.items()}
            for matrix_type, elements in self._DEFAULT_FACTORS.items()
        }

    def _populate_defaults(self) -> None:
        """Populate database with default correction factors for the active regime."""
        self._maybe_warn_uncalibrated_defaults()
        source = f"default_{self.ablation_regime.name.lower()}_generic"
        for matrix_type, elements in self._default_factors_for_regime().items():
            for el, (mult, uncert) in elements.items():
                self._factors[matrix_type][el] = CorrectionFactor(
                    element=el,
                    matrix_type=matrix_type,
                    multiplicative=mult,
                    uncertainty=uncert,
                    source=source,
                )

    def _maybe_warn_uncalibrated_defaults(self) -> None:
        """Emit a one-time warning (per regime) that generic, uncalibrated defaults are in use."""
        regime = self.ablation_regime
        if regime in CorrectionFactorDB._warned_default_regimes:
            return
        CorrectionFactorDB._warned_default_regimes.add(regime)
        if regime is AblationRegime.PS:
            detail = (
                "ps-LIBS ablation is approximately stoichiometric; generic defaults "
                "are 1.0 (no correction). Override with calibrated factors when available."
            )
        else:
            detail = (
                "these are nanosecond-pulse empirical factors and may bias non-ns data. "
                "Pass ablation_regime='ps' for ultrashort-pulse data, or supply calibrated factors."
            )
        logger.warning(
            "CorrectionFactorDB using uncalibrated generic %s-regime matrix-correction "
            "defaults: %s",
            regime.name.lower(),
            detail,
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
        ablation_regime: Union[AblationRegime, str] = AblationRegime.NS,
    ) -> None:
        """
        Initialize the matrix effect corrector.

        Parameters
        ----------
        correction_db : CorrectionFactorDB, optional
            Database of correction factors. If None, a default database is built
            for ``ablation_regime``.
        renormalize : bool
            Whether to renormalize concentrations to sum to 1 after correction
        ablation_regime : AblationRegime or str, optional
            Laser ablation regime for the default correction database. Defaults
            to :attr:`AblationRegime.NS` (nanosecond), preserving prior behavior.
            Ignored when an explicit ``correction_db`` is supplied (the database
            carries its own regime); a mismatch logs an informational note.
        """
        self.ablation_regime = AblationRegime.coerce(ablation_regime)
        if correction_db is None:
            self.correction_db = CorrectionFactorDB(ablation_regime=self.ablation_regime)
        else:
            self.correction_db = correction_db
            if correction_db.ablation_regime is not self.ablation_regime:
                logger.info(
                    "MatrixEffectCorrector ablation_regime=%s but supplied correction_db "
                    "was built for %s; using the database's factors as given.",
                    self.ablation_regime.name.lower(),
                    correction_db.ablation_regime.name.lower(),
                )
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
            return self._classify_oxygen_silicon(norm_conc, evidence, o_fraction, si_fraction)

        # High Fe + moderate O -> could be iron oxide or steel
        if fe_fraction > 0.50 and o_fraction > 0.10:
            return MatrixClassificationResult(
                matrix_type=MatrixType.OXIDE,
                confidence=0.7,
                evidence=evidence,
                notes=f"High Fe ({fe_fraction:.1%}) with oxygen suggests oxide",
            )

        # Default: highest fraction determines type
        return self._classify_default(
            evidence, metallic_fraction, organic_fraction, geological_fraction
        )

    def _classify_oxygen_silicon(
        self,
        norm_conc: Dict[str, float],
        evidence: Dict[str, float],
        o_fraction: float,
        si_fraction: float,
    ) -> MatrixClassificationResult:
        """Classify the significant oxygen + silicon branch (glass/geological/oxide)."""
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

    def _classify_default(
        self,
        evidence: Dict[str, float],
        metallic_fraction: float,
        organic_fraction: float,
        geological_fraction: float,
    ) -> MatrixClassificationResult:
        """Default tie-break classification based on the dominant category fraction."""
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
            Reserved for future uncertainty propagation from the standard.
            Currently accepted but ignored: the method body does not read this
            flag and no uncertainty is propagated regardless of its value.

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
    ablation_regime: Union[AblationRegime, str] = AblationRegime.NS,
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
    ablation_regime : AblationRegime or str, optional
        Laser ablation regime for the default correction database. Defaults to
        :attr:`AblationRegime.NS` (nanosecond), preserving prior behavior. Use
        ``"ps"`` for ultrashort-pulse data (stoichiometric generic defaults).

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
        ablation_regime=ablation_regime,
    )
    matrix_result = corrector.correct(result, matrix_type)

    return matrix_result.corrected_concentrations


# ============================================================================
# One-Point Calibration (OPC) -- Cavalcanti et al., SAB 2013
# ============================================================================
#
# Physics
# -------
# Standardless CF-LIBS reads the composition of each species ``s`` from the
# intercept of its Boltzmann plot. For an optically thin LTE plasma the
# integrated intensity of line ``i`` of species ``s`` obeys (Cremers &
# Radziemski 2013, Ch. 6; Tognoni et al. 2010, Eq. 1):
#
#     I_i = F * (g_i A_i / lambda_i) * (n_s / U_s(T)) * exp(-E_i / (kB T))   (1)
#
# where ``F`` is a single experimental factor (collection efficiency, ablated
# mass, plasma volume) common to *all* lines of *all* species in one spectrum.
# Taking the logarithm of the standard Boltzmann-plot ordinate
# ``y_i = ln( I_i * lambda_i / (g_i A_i) )`` gives the familiar linear form
#
#     y_i = -E_i / (kB T) + ln( F * n_s / U_s(T) ) = -E_i/(kB T) + q_s ,     (2)
#
# i.e. each species contributes a line of common slope ``-1/(kB T)`` and an
# intercept ``q_s`` from which ``n_s`` -- hence the composition -- is recovered.
#
# In practice the *measured* ordinate is biased by a wavelength-dependent term
# that Eq. (1) ignores: the spectrometer/detector spectral response
# ``R(lambda)``, errors in the tabulated transition probabilities ``A_i``, and
# residual (uncorrected) self-absorption. Collect all of these into one
# multiplicative intensity bias ``B(lambda)`` so that the recorded intensity is
# ``I_meas,i = B(lambda_i) * I_i``. This corrupts the ordinate additively:
#
#     y_meas,i = y_i + ln B(lambda_i) .                                       (3)
#
# Cavalcanti et al. (2013) one-point calibration: measure the SAME bias from a
# single certified reference standard. With the standard's certified number
# fractions ``n_s^cert`` and its measured plasma temperature ``T``, the
# bias-free ordinate predicted by Eq. (2) is known up to the global, species-
# *independent* offset ``ln F`` (which cancels in the closure Sum C_s = 1).
# The per-line log-correction is therefore
#
#     delta(lambda_i) = ln F_corr(lambda_i)
#                     = y_pred,i - y_meas,i                                   (4)
#         with  y_pred,i = -E_i/(kB T) + ln( n_s^cert / U_s(T) ) .
#
# ``delta(lambda) = -ln B(lambda) + ln F`` recovers the *negative* bias plus an
# irrelevant global constant. Applying the correction to an unknown sample that
# shares the matrix bias multiplies its line intensity by
# ``F_corr(lambda) = exp(delta(lambda))`` (equivalently adds ``delta`` to its
# Boltzmann ordinate), removing ``B`` and forcing the standard's CF result onto
# its certified composition by construction (residual ~ 0).
#
# DEFAULT = no OPC. Nothing above runs unless the caller explicitly fits an
# :class:`OnePointCalibrator` on a standard and applies it -- this is strictly
# opt-in and changes no existing CF-LIBS code path.


@dataclass
class OnePointCalibration:
    """
    Fitted one-point-calibration (OPC) correction, per Cavalcanti et al. (2013).

    Holds the wavelength-dependent log-correction ``delta(lambda)`` derived from a
    single certified standard. Apply it to the Boltzmann-plot points
    (:class:`~cflibs.inversion.common.LineObservation`) of an unknown sample with
    :meth:`apply` (or :meth:`apply_intensity`).

    Attributes
    ----------
    wavelengths_nm : np.ndarray
        Sorted line wavelengths (nm) of the certified standard used as the
        interpolation knots for ``delta(lambda)``.
    log_correction : np.ndarray
        Per-knot additive Boltzmann-ordinate correction ``delta`` of Eq. (4),
        i.e. ``ln F_corr``. The equivalent multiplicative intensity factor is
        ``exp(log_correction)``.
    temperature_K : float
        Plasma excitation temperature (K) of the standard at which ``delta`` was
        fitted. OPC is exact only when the unknown shares this ``T`` (same
        ``E_k/(kB T)`` mapping); a large mismatch degrades the correction.
    reference_label : str
        Free-text label identifying the certified standard (provenance only).
    extrapolate : bool
        If True, ``delta`` is held flat (clamped to the nearest knot) outside the
        fitted wavelength range; if False, out-of-range lines receive
        ``delta = 0`` (no correction) and are logged. Default False.

    Notes
    -----
    See module-level "One-Point Calibration (OPC)" section for the derivation.
    Cavalcanti et al., Spectrochim. Acta B 87 (2013) 51-56,
    doi:10.1016/j.sab.2013.05.016.
    """

    wavelengths_nm: np.ndarray
    log_correction: np.ndarray
    temperature_K: float
    reference_label: str = "standard"
    extrapolate: bool = False

    def correction_at(self, wavelength_nm: float) -> float:
        """
        Return the additive Boltzmann-ordinate correction ``delta(lambda)`` at one
        wavelength via linear interpolation over the fitted knots.

        Parameters
        ----------
        wavelength_nm : float
            Line wavelength (nm).

        Returns
        -------
        float
            ``delta(lambda)`` (= ``ln F_corr``). Zero outside the fitted range when
            ``extrapolate`` is False.
        """
        wl = float(wavelength_nm)
        lo, hi = float(self.wavelengths_nm[0]), float(self.wavelengths_nm[-1])
        if (wl < lo or wl > hi) and not self.extrapolate:
            return 0.0
        # np.interp clamps to the endpoints outside [lo, hi], which is exactly the
        # "flat extrapolation" behaviour we want when extrapolate is True.
        return float(np.interp(wl, self.wavelengths_nm, self.log_correction))

    def factor_at(self, wavelength_nm: float) -> float:
        """Multiplicative intensity factor ``F_corr(lambda) = exp(delta(lambda))``."""
        return float(np.exp(self.correction_at(wavelength_nm)))

    def apply_intensity(self, wavelength_nm: float, intensity: float) -> float:
        """
        Apply the OPC factor to a single raw line intensity.

        ``I_corrected = F_corr(lambda) * I_measured`` -- the multiplicative
        rescaling of Eq. (3)/(4) that cancels the matrix/response bias.
        """
        return self.factor_at(wavelength_nm) * float(intensity)

    def apply(self, observations: Sequence[LineObservation]) -> List[LineObservation]:
        """
        Apply OPC to a list of Boltzmann-plot points of an unknown sample.

        Returns new :class:`LineObservation` objects with intensity (and its
        uncertainty) multiplied by ``F_corr(lambda)``. Because the Boltzmann
        ordinate is ``ln( I lambda / (g A) )``, this is equivalent to *adding*
        ``delta(lambda)`` to each ordinate -- the OPC rescaling of the
        Boltzmann-plot points requested by Cavalcanti et al. (2013).

        Parameters
        ----------
        observations : Sequence[LineObservation]
            Boltzmann-plot points for the unknown sample (must share the
            standard's matrix bias for the correction to be physical).

        Returns
        -------
        list[LineObservation]
            New observations with OPC-corrected intensities. Input is unmodified.
        """
        corrected: List[LineObservation] = []
        for obs in observations:
            f = self.factor_at(obs.wavelength_nm)
            corrected.append(
                LineObservation(
                    wavelength_nm=obs.wavelength_nm,
                    intensity=obs.intensity * f,
                    intensity_uncertainty=obs.intensity_uncertainty * f,
                    element=obs.element,
                    ionization_stage=obs.ionization_stage,
                    E_k_ev=obs.E_k_ev,
                    g_k=obs.g_k,
                    A_ki=obs.A_ki,
                    aki_uncertainty=obs.aki_uncertainty,
                )
            )
        return corrected


class OnePointCalibrator:
    """
    OPT-IN one-point calibration (OPC) of the CF-LIBS Boltzmann plot.

    Implements Cavalcanti et al. (2013): a single certified reference standard is
    analysed first; the deviation of its measured Boltzmann-plot points from the
    points predicted by its *certified* composition defines a wavelength-dependent
    correction ``F_corr(lambda)`` that absorbs the combined spectral-response /
    transition-probability / residual-self-absorption bias. The same
    ``F_corr(lambda)`` is then applied (multiplicatively to line intensity, i.e.
    additively to the Boltzmann ordinate) to unknown samples sharing that bias.

    This class is **strictly opt-in**: constructing/using it changes no default
    CF-LIBS path. The standardless pipeline behaves exactly as before unless a
    caller fits a calibrator on a standard and applies it.

    Examples
    --------
    >>> calibrator = OnePointCalibrator()
    >>> opc = calibrator.fit(
    ...     standard_observations,            # measured Boltzmann-plot points
    ...     certified_number_fractions={"Fe": 0.70, "Cu": 0.30},
    ...     partition_functions={"Fe": {1: 30.0}, "Cu": {1: 8.0}},
    ...     temperature_K=10000.0,
    ... )
    >>> corrected = opc.apply(unknown_observations)   # rescaled Boltzmann points

    Notes
    -----
    - The correction is referenced *per species' neutral plane* in the same way
      CF-LIBS reads composition: lines are grouped by (element, ionization stage)
      and the ordinate predicted from the species' certified number fraction.
    - The global, species-independent factor ``ln F`` of Eq. (2) is not
      identifiable from one standard and is irrelevant to relative composition
      (it cancels in closure). It is fixed here by the intensity-weighted mean of
      the raw residuals so the typical correction magnitude stays small and the
      multiplicative factors stay near unity; choosing a different global offset
      only multiplies every unknown intensity by a constant, leaving the recovered
      *relative* composition unchanged.

    References
    ----------
    Cavalcanti, Teixeira, Legnaioli, Lorenzetti, Pardini & Palleschi,
    Spectrochim. Acta B 87 (2013) 51-56, doi:10.1016/j.sab.2013.05.016.
    """

    def __init__(self, extrapolate: bool = False) -> None:
        """
        Parameters
        ----------
        extrapolate : bool
            Whether the fitted correction is held flat outside the standard's
            wavelength range when applied to an unknown (True), or set to zero
            there (False, default -- conservative: never correct where the
            standard provides no information).
        """
        self.extrapolate = extrapolate

    @staticmethod
    def _partition_value(
        partition_functions: Dict[str, Dict[int, float]],
        element: str,
        ionization_stage: int,
    ) -> float:
        """Look up U_s(T) for (element, stage); default 1.0 with a warning."""
        per_element = partition_functions.get(element, {})
        u = per_element.get(ionization_stage)
        if u is None or u <= 0:
            logger.warning(
                "OPC: no positive partition function for %s stage %d; using U=1.0",
                element,
                ionization_stage,
            )
            return 1.0
        return float(u)

    def fit(
        self,
        standard_observations: Sequence[LineObservation],
        certified_number_fractions: Dict[str, float],
        partition_functions: Dict[str, Dict[int, float]],
        temperature_K: float,
        reference_label: str = "standard",
    ) -> OnePointCalibration:
        """
        Fit the OPC correction ``delta(lambda)`` from one certified standard.

        For every standard line ``i`` of species ``s = (element, stage)`` the
        bias-free Boltzmann ordinate predicted by the certified composition is

            ``y_pred,i = -E_i/(kB T) + ln( n_s^cert / U_s(T) )``     (Eq. 4)

        and the per-line log-correction is ``delta_i = y_pred,i - y_meas,i`` with
        ``y_meas,i = ln( I_i lambda_i / (g_i A_i) )``. An intensity-weighted mean
        of ``delta_i`` is removed as the (irrelevant) global offset ``ln F``, and
        the residual is stored on the sorted wavelength grid for interpolation.

        Parameters
        ----------
        standard_observations : Sequence[LineObservation]
            Measured Boltzmann-plot points of the certified standard.
        certified_number_fractions : Dict[str, float]
            Certified composition of the standard as *number* fractions per
            element (need not sum exactly to 1; only relative values matter).
            Must be positive for every element appearing in the observations.
        partition_functions : Dict[str, Dict[int, float]]
            ``U_s(T)`` per element and ionization stage, evaluated at
            ``temperature_K`` (e.g. ``{"Fe": {1: 30.0, 2: 45.0}}``).
        temperature_K : float
            Plasma excitation temperature (K) of the standard.
        reference_label : str
            Provenance label for the standard.

        Returns
        -------
        OnePointCalibration
            Fitted correction ready to apply to unknown samples.

        Raises
        ------
        ValueError
            If no usable lines remain, or a certified number fraction needed by
            the observations is missing/non-positive, or ``temperature_K`` is not
            positive.
        """
        if temperature_K <= 0:
            raise ValueError(f"temperature_K must be positive, got {temperature_K}")

        kT = KB_EV * float(temperature_K)  # eV
        wls: List[float] = []
        deltas: List[float] = []
        weights: List[float] = []

        for obs in standard_observations:
            if obs.intensity <= 0 or obs.g_k <= 0 or obs.A_ki <= 0:
                logger.debug(
                    "OPC: skipping non-physical standard line at %.3f nm " "(I=%.3g, g=%s, A=%.3g)",
                    obs.wavelength_nm,
                    obs.intensity,
                    obs.g_k,
                    obs.A_ki,
                )
                continue
            n_s = certified_number_fractions.get(obs.element)
            if n_s is None or n_s <= 0:
                raise ValueError(
                    f"OPC: certified number fraction for element '{obs.element}' "
                    f"is missing or non-positive ({n_s}); required to fit the "
                    "standard's Boltzmann ordinate."
                )
            u_s = self._partition_value(partition_functions, obs.element, obs.ionization_stage)
            # Predicted bias-free ordinate (Eq. 4), modulo global ln F.
            y_pred = -obs.E_k_ev / kT + np.log(n_s / u_s)
            y_meas = obs.y_value  # ln( I lambda / (g A) )
            delta = float(y_pred - y_meas)
            wls.append(float(obs.wavelength_nm))
            deltas.append(delta)
            # Weight by line intensity: brighter (higher-SNR) lines pin the global
            # offset more strongly. Strictly positive by the guard above.
            weights.append(float(obs.intensity))

        if not wls:
            raise ValueError("OPC: no usable standard lines to fit the correction.")

        wl_arr = np.asarray(wls, dtype=float)
        delta_arr = np.asarray(deltas, dtype=float)
        w_arr = np.asarray(weights, dtype=float)

        # Remove the species-independent global offset ln F (Eq. 2). It does not
        # affect relative composition (cancels in closure); subtracting the
        # intensity-weighted mean keeps the multiplicative factors near unity.
        global_offset = float(np.average(delta_arr, weights=w_arr))
        delta_arr = delta_arr - global_offset

        # Collapse duplicate wavelengths (intensity-weighted) and sort for interp.
        order = np.argsort(wl_arr)
        wl_sorted = wl_arr[order]
        delta_sorted = delta_arr[order]
        w_sorted = w_arr[order]
        uniq_wl, inverse = np.unique(wl_sorted, return_inverse=True)
        if uniq_wl.size != wl_sorted.size:
            num = np.zeros(uniq_wl.size)
            den = np.zeros(uniq_wl.size)
            np.add.at(num, inverse, delta_sorted * w_sorted)
            np.add.at(den, inverse, w_sorted)
            uniq_delta = num / den
        else:
            uniq_delta = delta_sorted

        logger.info(
            "OPC fit on '%s': %d lines, T=%.0f K, |delta| max=%.3f "
            "(global ln F offset=%.3f removed)",
            reference_label,
            uniq_wl.size,
            temperature_K,
            float(np.max(np.abs(uniq_delta))) if uniq_delta.size else 0.0,
            global_offset,
        )

        return OnePointCalibration(
            wavelengths_nm=uniq_wl,
            log_correction=uniq_delta,
            temperature_K=float(temperature_K),
            reference_label=reference_label,
            extrapolate=self.extrapolate,
        )
