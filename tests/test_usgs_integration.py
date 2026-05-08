import numpy as np
import pandas as pd
from cflibs.benchmark.datasets.usgs import USGSDataset


def test_usgs_dataset_compositions():
    """Verify that USGS compositions are correctly hardcoded and converted to elements."""
    ds = USGSDataset()
    samples = ds.available_samples()
    assert "BHVO-2" in samples
    assert "AGV-2" in samples
    assert "BCR-2" in samples
    assert "G-2" in samples

    # Check BHVO-2 Si content (~23.18 wt% element)
    # SiO2 = 49.60. Si factor = 0.46744. 49.60 * 0.46744 = 23.185...
    comp = ds.get_composition("BHVO-2")
    assert "Si" in comp.element_wt_percent
    assert np.isclose(comp.element_wt_percent["Si"], 23.185, atol=0.01)

    # Check Fe2O3T -> Fe conversion for BHVO-2
    # Fe2O3T = 12.39. Fe factor = 0.69944. 12.39 * 0.69944 = 8.666...
    assert np.isclose(comp.element_wt_percent["Fe"], 8.666, atol=0.01)


def test_usgs_spectrum_loading(tmp_path):
    """Verify that USGSDataset can locate and load spectrum files."""
    # Create a dummy spectrum file
    usgs_dir = tmp_path / "usgs_geostandards"
    usgs_dir.mkdir()

    # BHVO-2.csv naming convention
    df = pd.DataFrame(
        {"wavelength": np.linspace(200, 900, 1000), "intensity": np.random.rand(1000)}
    )
    df.to_csv(usgs_dir / "BHVO-2.csv", index=False)

    ds = USGSDataset(data_dir=usgs_dir)
    spec = ds.get_spectrum("BHVO-2")

    assert spec is not None
    assert spec.spectrum_id == "usgs_bhvo-2"
    assert len(spec.wavelength_nm) == 1000
    assert "Si" in spec.true_composition

    # Test alternative naming (lowercase)
    df.to_csv(usgs_dir / "agv_2.csv", index=False)
    spec_agv = ds.get_spectrum("AGV-2")
    assert spec_agv is not None
    assert spec_agv.spectrum_id == "usgs_agv-2"


def test_usgs_integration_logic(tmp_path):
    """Test the logic used in scripts/run_accuracy_ablation.py for USGS integration."""
    usgs_dir = tmp_path / "usgs"
    usgs_dir.mkdir()

    # Create dummy files for 3 standards
    for sid in ["BHVO-2", "AGV-2", "BCR-2"]:
        df = pd.DataFrame({"wavelength": np.linspace(300, 500, 100), "intensity": np.ones(100)})
        df.to_csv(usgs_dir / f"{sid}.csv", index=False)

    from cflibs.benchmark.datasets.usgs import USGSDataset

    ds = USGSDataset(data_dir=usgs_dir)

    # Minimal list of detectable elements for testing
    LIBS_DETECTABLE = {"Si", "Ti", "Al", "Fe", "Mn", "Mg", "Ca", "Na", "K", "P"}

    spectra = []
    usgs_count = 0
    for sid in ds.available_samples():
        benchmark_spec = ds.get_spectrum(sid)
        if benchmark_spec is None:
            continue

        true_comp = benchmark_spec.true_composition
        det_elements = {k: v for k, v in true_comp.items() if k in LIBS_DETECTABLE}
        det_total = sum(det_elements.values())

        if det_total > 0:
            stoich = {k: v / det_total for k, v in det_elements.items()}
            spectra.append(
                {
                    "mineral": sid,
                    "stoichiometric": stoich,
                }
            )
            usgs_count += 1

    assert usgs_count == 3
    assert len(spectra) == 3
    for s in spectra:
        # Stoichiometric values should sum to 1.0 after renormalization
        assert np.isclose(sum(s["stoichiometric"].values()), 1.0)
