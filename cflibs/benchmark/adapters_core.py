"""Core scoreboard dataset adapters (bead A1).

Adapters for the truth-bearing datasets that work TODAY, built to the
contract in :mod:`cflibs.benchmark.scoreboard_registry`: zero-argument
generators yielding ``(spectrum_id, wavelength_nm, intensity, SpectrumTruth)``,
lazy (no I/O at import), deterministic, skip-with-log when data files are
absent.

Datasets
--------
``bhvo2_chemcam``
    4 real ChemCam (LANL Mars-analog testbed) spectra of USGS BHVO-2 basalt
    (``data/bhvo2_usgs/chemcam_bhvo2_loc{1..4}_spectrum.csv``). Truth: the
    USGS/GeoReM certified oxide composition converted to ELEMENT wt%
    (:data:`cflibs.benchmark.reference_compositions.BHVO2_BASALT_USGS`).
``aalto``
    74 real spectra from the Aalto University LIBS spectral library
    (https://users.aalto.fi/~lainei1/pages/elements/): 13 pure-element
    standards + 61 mineral samples. Presence-only truth — pure-element
    labels and mineral-formula stoichiometry respectively; quantitative
    element-wt% conversion is ambiguous for both (metallic standards vs the
    geological oxide-closure basis; minerals have solid-solution formulas),
    so composition scoring is intentionally skipped.
``nist_srm_612`` / ``nist_steel``
    Placeholders: the data directories contain no spectra (READMEs document
    the public-data gap — see ``data/nist_srm_612/README.md`` and bead
    CF-LIBS-improved-9jvd). The adapters skip-with-log so the board records
    the gap honestly instead of hiding it.
``synthetic_fixedforward``
    The regenerated 288-spectrum synthetic ID corpus
    (``w2_fixedforward_v1/corpus.json``) produced by OUR OWN fixed forward
    model. Tagged ``synthetic``: useful for *relative* comparisons between
    pipeline versions, never a headline accuracy number (the forward model
    that generated it shares physics with the inversion under test).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Iterator, Optional

import numpy as np

from cflibs.core.logging_config import get_logger
from cflibs.benchmark.scoreboard_registry import (
    AdapterYield,
    SpectrumTruth,
    register_dataset,
)

logger = get_logger("benchmark.adapters_core")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "data"

#: Trace-element presence cutoff (wt%): certified elements below this are
#: excluded from ``elements_present`` (documented in each adapter's notes).
PRESENCE_CUTOFF_WT = 0.01

# ---------------------------------------------------------------------------
# bhvo2_chemcam — real ChemCam spectra of USGS BHVO-2 basalt
# ---------------------------------------------------------------------------

_BHVO2_SPECTRA = (
    "chemcam_bhvo2_loc1_spectrum.csv",
    "chemcam_bhvo2_loc2_spectrum.csv",
    "chemcam_bhvo2_loc3_spectrum.csv",
    "chemcam_bhvo2_loc4_spectrum.csv",
)


def bhvo2_chemcam_adapter() -> Iterator[AdapterYield]:
    """Yield the 4 ChemCam BHVO-2 location spectra with USGS certified truth."""
    from cflibs.benchmark.reference_compositions import BHVO2_BASALT_USGS
    from cflibs.io.spectrum import load_spectrum

    data_dir = _DATA_DIR / "bhvo2_usgs"
    composition_wt = {el: 100.0 * frac for el, frac in BHVO2_BASALT_USGS.items()}
    elements_present = frozenset(
        el for el, wt in composition_wt.items() if wt >= PRESENCE_CUTOFF_WT
    )
    notes = (
        "ChemCam (LANL testbed) LIBS of USGS BHVO-2 Hawaiian basalt; "
        "truth = USGS/GeoReM certified oxide composition converted to element wt% "
        "(cflibs.benchmark.reference_compositions.BHVO2_BASALT_USGS; oxygen excluded, "
        "oxide-bound). All 10 certified elements are above the "
        f"{PRESENCE_CUTOFF_WT} wt% presence cutoff. CCS spectra arrive "
        "response-corrected upstream; resolving power not certified (None)."
    )
    for fname in _BHVO2_SPECTRA:
        path = data_dir / fname
        if not path.exists():
            logger.warning("bhvo2_chemcam: %s missing — skipping this spectrum.", path)
            continue
        wavelength, intensity = load_spectrum(str(path))
        truth = SpectrumTruth(
            elements_present=elements_present,
            composition_wt=dict(composition_wt),
            resolving_power=None,
            notes=notes,
        )
        yield fname.replace("_spectrum.csv", ""), wavelength, intensity, truth


# ---------------------------------------------------------------------------
# aalto — Aalto University LIBS spectral library (elements + minerals)
# ---------------------------------------------------------------------------

#: Mineral formula -> constituent elements (idealized stoichiometry; same
#: mapping as scripts/run_aalto_benchmark.py, kept as data here because
#: shipped code cannot import from scripts/).
_AALTO_MINERAL_ELEMENTS: Dict[str, frozenset[str]] = {
    "adularia": frozenset({"K", "Al", "Si", "O"}),  # KAlSi3O8
    "aegerine": frozenset({"Na", "Fe", "Si", "O"}),  # NaFeSi2O6
    "almandine": frozenset({"Fe", "Al", "Si", "O"}),  # Fe3Al2(SiO4)3
    "apatite": frozenset({"Ca", "P", "O"}),  # Ca5(PO4)3(OH,F,Cl)
    "augite": frozenset({"Ca", "Mg", "Fe", "Al", "Si", "O"}),  # (Ca,Na)(Mg,Fe,Al)(Si,Al)2O6
    "beryl": frozenset({"Be", "Al", "Si", "O"}),  # Be3Al2(SiO3)6
    "biotite": frozenset({"K", "Mg", "Fe", "Al", "Si", "O"}),  # K(Mg,Fe)3(AlSi3O10)(OH)2
    "chalcopyrite": frozenset({"Cu", "Fe", "S"}),  # CuFeS2
    "cinnabar": frozenset({"Hg", "S"}),  # HgS
    "cordierite": frozenset({"Mg", "Fe", "Al", "Si", "O"}),  # (Mg,Fe)2Al4Si5O18
    "corundum": frozenset({"Al", "O"}),  # Al2O3
    "diopside": frozenset({"Ca", "Mg", "Si", "O"}),  # CaMgSi2O6
    "fluorite": frozenset({"Ca"}),  # CaF2 (F not LIBS-detectable)
    "galena": frozenset({"Pb", "S"}),  # PbS
    "garnet": frozenset({"Ca", "Mg", "Fe", "Al", "Si", "O"}),  # (Ca,Mg,Fe,Mn)3(Al,..)2(SiO4)3
    "gypsum": frozenset({"Ca", "S", "O"}),  # CaSO4.2H2O
    "hematite": frozenset({"Fe", "O"}),  # Fe2O3
    "hornblende": frozenset({"Ca", "Mg", "Fe", "Al", "Si", "O"}),  # Ca2(Mg,Fe,Al)5(Al,Si)8O22
    "hypersthene": frozenset({"Fe", "Mg", "Si", "O"}),  # (Fe,Mg)2Si2O6
    "kaolinite": frozenset({"Al", "Si", "O"}),  # Al2Si2O5(OH)4
    "kyanite": frozenset({"Al", "Si", "O"}),  # Al2SiO5
    "lepidolite": frozenset({"K", "Li", "Al", "Si", "O"}),  # KLi2Al(Si4O10)(F,OH)2
    "magnesite": frozenset({"Mg", "C", "O"}),  # MgCO3
    "magnetite": frozenset({"Fe", "O"}),  # Fe3O4
    "microcline": frozenset({"K", "Al", "Si", "O"}),  # KAlSi3O8
    "molybdenite": frozenset({"Mo", "S"}),  # MoS2
    "muscovite": frozenset({"K", "Al", "Si", "O"}),  # KAl2(AlSi3O10)(OH)2
    "olivine": frozenset({"Mg", "Fe", "Si", "O"}),  # (Mg,Fe)2SiO4
    "orthoclase": frozenset({"K", "Al", "Si", "O"}),  # KAlSi3O8
    "pentlandite": frozenset({"Fe", "Ni", "S"}),  # (Fe,Ni)9S8
    "phlogopite": frozenset({"K", "Mg", "Al", "Si", "O"}),  # KMg3(AlSi3O10)(OH)2
    "plagioclase": frozenset({"Na", "Ca", "Si", "Al", "O"}),  # (Na,Ca)(Si,Al)4O8
    "pyrite": frozenset({"Fe", "S"}),  # FeS2
    "pyrrhotite": frozenset({"Fe", "S"}),  # Fe(1-x)S
    "quartz": frozenset({"Si", "O"}),  # SiO2
    "scapolite": frozenset({"Na", "Ca", "Al", "Si", "O"}),  # (Na,Ca)4(Al,Si)12O24(Cl,CO3,SO4)
    "serpentine": frozenset({"Mg", "Si", "O"}),  # Mg3Si2O5(OH)4
    "siderite": frozenset({"Fe", "C", "O"}),  # FeCO3
    "sphalerite": frozenset({"Zn", "S"}),  # ZnS
    "sphene": frozenset({"Ca", "Ti", "Si", "O"}),  # CaTiSiO5 (titanite)
    "spodumene": frozenset({"Li", "Al", "Si", "O"}),  # LiAlSi2O6
    "staurolite": frozenset({"Fe", "Al", "Si", "O"}),  # Fe2+Al9Si4O23(OH)
    "talc": frozenset({"Mg", "Si", "O"}),  # Mg3Si4O10(OH)2
    "topaz": frozenset({"Al", "Si", "O"}),  # Al2SiO4(F,OH)2
    "tourmaline": frozenset({"Na", "Mg", "Fe", "Al", "Si", "B", "O"}),  # Na(..)3Al6(BO3)3Si6O18
    "tremolite": frozenset({"Ca", "Mg", "Fe", "Si", "O"}),  # Ca2(Mg,Fe)5Si8O22(OH)2
    "wollastonite": frozenset({"Ca", "Si", "O"}),  # CaSiO3
    "zircon": frozenset({"Zr", "Si", "O"}),  # ZrSiO4
    "mntantalite": frozenset({"Mn", "Fe", "Ta", "O"}),  # (Mn,Fe)Ta2O6
}

#: Elements LIBS can realistically detect in this library's band; O, H, F and
#: Cl are excluded from mineral truth (high excitation energies / VUV lines).
_LIBS_HARD_ELEMENTS = frozenset({"O", "H", "F", "Cl"})

_AALTO_SAMPLE_RE = re.compile(r"^([A-Za-z]+?)(?:E?\d+)?$")


def _aalto_mineral_name(stem: str) -> Optional[str]:
    """``'adulariaE11_spectrum'`` -> ``'adularia'`` (None when unparseable)."""
    match = _AALTO_SAMPLE_RE.match(stem.replace("_spectrum", ""))
    return match.group(1).lower() if match else None


def aalto_adapter() -> Iterator[AdapterYield]:
    """Yield Aalto pure-element and mineral spectra with presence-only truth."""
    base = _DATA_DIR / "aalto_libs"
    elements_dir = base / "elements"
    minerals_dir = base / "minerals"
    if not elements_dir.is_dir() and not minerals_dir.is_dir():
        logger.warning(
            "aalto: %s has no elements/ or minerals/ directory — skipping dataset.", base
        )
        return

    def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
        # Aalto CSVs use the header ``wavelength,spectrum``.
        data = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=float)
        return data[:, 0], data[:, 1]

    if elements_dir.is_dir():
        for path in sorted(elements_dir.glob("*_spectrum.csv")):
            element = path.stem.replace("_spectrum", "")
            wavelength, intensity = _load(path)
            truth = SpectrumTruth(
                elements_present=frozenset({element}),
                composition_wt=None,
                resolving_power=None,
                notes=(
                    f"Aalto LIBS library pure-element standard ({element}); "
                    "presence-only truth. Nominal composition is ~100 wt% metal but "
                    "element-wt comparison is skipped: the production geological "
                    "preset closes on an oxide basis, which is ill-defined for a "
                    "metallic standard. https://users.aalto.fi/~lainei1/pages/elements/"
                ),
            )
            yield f"element_{element}", wavelength, intensity, truth

    if minerals_dir.is_dir():
        for path in sorted(minerals_dir.glob("*_spectrum.csv")):
            sample = path.stem.replace("_spectrum", "")
            mineral = _aalto_mineral_name(sample)
            formula_elements = _AALTO_MINERAL_ELEMENTS.get(mineral or "")
            if formula_elements is None:
                logger.warning(
                    "aalto: no mineral-formula truth for %s (parsed %r) — skipping.",
                    path.name,
                    mineral,
                )
                continue
            detectable = frozenset(formula_elements - _LIBS_HARD_ELEMENTS)
            wavelength, intensity = _load(path)
            truth = SpectrumTruth(
                elements_present=detectable,
                composition_wt=None,
                resolving_power=None,
                notes=(
                    f"Aalto LIBS library mineral sample {sample} ({mineral}); "
                    "presence-only truth from idealized mineral-formula stoichiometry "
                    "(solid-solution minerals make quantitative wt% ambiguous). "
                    "O/H/F/Cl excluded (not LIBS-detectable in this band). "
                    "https://users.aalto.fi/~lainei1/pages/elements/"
                ),
            )
            yield f"mineral_{sample}", wavelength, intensity, truth


# ---------------------------------------------------------------------------
# nist_srm_612 / nist_steel — placeholders, no public spectra ingested
# ---------------------------------------------------------------------------


def nist_srm_612_adapter() -> Iterator[AdapterYield]:
    """Skip-with-log: data/nist_srm_612 holds a README documenting the gap."""
    data_dir = _DATA_DIR / "nist_srm_612"
    spectra = sorted(data_dir.glob("*_spectrum.csv")) if data_dir.is_dir() else []
    if not spectra:
        logger.warning(
            "nist_srm_612: no spectra in %s (README documents the public-data gap, "
            "bead CF-LIBS-improved-9jvd) — dataset skipped.",
            data_dir,
        )
        return
    # Spectra appeared after the gap closes: wire them to the certified truth.
    from cflibs.benchmark.reference_compositions import NIST_SRM_612_GLASS
    from cflibs.io.spectrum import load_spectrum

    composition_wt = {el: 100.0 * frac for el, frac in NIST_SRM_612_GLASS.items()}
    elements_present = frozenset(
        el for el, wt in composition_wt.items() if wt >= PRESENCE_CUTOFF_WT
    )
    for path in spectra:
        wavelength, intensity = load_spectrum(str(path))
        truth = SpectrumTruth(
            elements_present=elements_present,
            composition_wt=dict(composition_wt),
            resolving_power=None,
            notes=(
                "NIST SRM 612 trace-element glass; truth = certified majors "
                "(Pearce et al. 1997) converted to element wt%; ~38 ppm trace "
                f"dopants are far below the {PRESENCE_CUTOFF_WT} wt% cutoff and excluded."
            ),
        )
        yield path.stem.replace("_spectrum", ""), wavelength, intensity, truth


def nist_steel_adapter() -> Iterator[AdapterYield]:
    """Skip-with-log: data/nist_steel contains no spectra (placeholder only)."""
    data_dir = _DATA_DIR / "nist_steel"
    spectra = sorted(data_dir.glob("*_spectrum.csv")) if data_dir.is_dir() else []
    if not spectra:
        logger.warning(
            "nist_steel: no spectra in %s (placeholder; no validated public NIST "
            "steel-SRM LIBS spectra ingested) — dataset skipped.",
            data_dir,
        )
        return
    logger.warning(
        "nist_steel: %d spectrum file(s) found in %s but no certified-truth "
        "mapping is wired for this layout — dataset skipped (extend "
        "nist_steel_adapter with the SRM cert mapping).",
        len(spectra),
        data_dir,
    )
    yield from ()  # empty generator: dataset yields nothing when skipped


# ---------------------------------------------------------------------------
# synthetic_fixedforward — our own fixed-forward-model synthetic corpus
# ---------------------------------------------------------------------------

#: Environment override for the corpus path (useful on machines where the
#: corpus was regenerated elsewhere).
SYNTH_CORPUS_ENV = "CFLIBS_SCOREBOARD_SYNTH_CORPUS"

_SYNTH_CORPUS_CANDIDATES = (
    _REPO_ROOT / "output/synthetic_corpus_w2/w2_fixedforward_v1/corpus.json",
    Path(
        "/home/brian/code/CF-LIBS-improved/.worktrees/w1-integration/"
        "output/synthetic_corpus_w2/w2_fixedforward_v1/corpus.json"
    ),
)


def _resolve_synth_corpus() -> Optional[Path]:
    override = os.environ.get(SYNTH_CORPUS_ENV)
    candidates = (Path(override),) if override else _SYNTH_CORPUS_CANDIDATES
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def synthetic_fixedforward_adapter() -> Iterator[AdapterYield]:
    """Yield the w2_fixedforward_v1 synthetic corpus (288 spectra, known truth)."""
    corpus_path = _resolve_synth_corpus()
    if corpus_path is None:
        logger.warning(
            "synthetic_fixedforward: corpus.json not found (tried $%s and %s) — "
            "dataset skipped. Regenerate with scripts/build_synthetic_id_corpus.py.",
            SYNTH_CORPUS_ENV,
            [str(p) for p in _SYNTH_CORPUS_CANDIDATES],
        )
        return
    corpus = json.loads(corpus_path.read_text())
    notes_base = (
        f"SYNTHETIC corpus {corpus.get('name', '?')} (version {corpus.get('version', '?')}) "
        "generated by OUR OWN fixed forward model — valid for RELATIVE comparisons "
        "between pipeline versions only, never a headline accuracy number "
        "(the generator shares physics with the inversion under test). "
        f"Truth = generation recipe mass fractions; elements below {PRESENCE_CUTOFF_WT} wt% "
        "excluded from elements_present. Source: "
    ) + str(corpus_path)
    for spec in corpus["spectra"]:
        composition_wt = {el: 100.0 * float(frac) for el, frac in spec["true_composition"].items()}
        elements_present = frozenset(
            el for el, wt in composition_wt.items() if wt >= PRESENCE_CUTOFF_WT
        )
        rp_estimate = spec.get("rp_estimate")
        truth = SpectrumTruth(
            elements_present=elements_present,
            composition_wt=composition_wt,
            resolving_power=float(rp_estimate) if rp_estimate else None,
            notes=notes_base,
        )
        yield (
            str(spec["spectrum_id"]),
            np.asarray(spec["wavelength_nm"], dtype=float),
            np.asarray(spec["intensity"], dtype=float),
            truth,
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_core_adapters(*, replace: bool = True) -> None:
    """Register the core datasets. Idempotent (``replace=True`` by default).

    Tier assignments mirror the campaign split design (design 2.1, single
    source of truth — ``scripts/campaign1/splits.py`` derives its name sets
    from these registrations): ``bhvo2_chemcam`` is HOLDOUT (the adoption-gate
    headline number must never leak into tuning loops or default boards).
    """
    register_dataset(
        "bhvo2_chemcam",
        bhvo2_chemcam_adapter,
        tags=("real", "geological"),
        tier="holdout",
        replace=replace,
    )
    register_dataset("aalto", aalto_adapter, tags=("real", "minerals"), replace=replace)
    register_dataset(
        "nist_srm_612", nist_srm_612_adapter, tags=("real", "placeholder"), replace=replace
    )
    register_dataset(
        "nist_steel", nist_steel_adapter, tags=("real", "placeholder"), replace=replace
    )
    register_dataset(
        "synthetic_fixedforward",
        synthetic_fixedforward_adapter,
        tags=("synthetic",),
        replace=replace,
    )
