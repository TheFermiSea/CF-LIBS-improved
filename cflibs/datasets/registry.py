"""Registry for LIBS datasets and their acquisition status."""

DATASET_REGISTRY = {
    "nist_srm_610": {
        "loader": "cflibs.datasets.nist.load_srm_610",
        "type": "reference_glass",
        "status": "active",
    },
    "nist_srm_612": {
        "loader": None,  # Loader wired in PR #116, but spectra missing
        "type": "reference_glass",
        "status": "pending_spectra",
        "acquisition_plan": {
            "gap": "No publicly-downloadable validated raw LIBS spectra exist in open archives (PDS, USGS, NIST, Zenodo).",
            "actions": [
                "Email Roger Wiens / Sam Clegg / Sylvestre Maurice (IRAP) for SuperCam internal pre-flight spectra.",
                "Contact LIBS-ENFL group (U Malaga, Laserna group) regarding published SRM 612 LIBS work data release.",
                "Check ORDaR (ordar.obspm.fr) for French lab deposits."
            ],
            "directive": "Synthesis cannot be the sole data source; real measured spectra are required."
        }
    },
}

# Datasets excluded from accuracy gates due to missing raw data or pending validation
EXCLUDED_DATASETS = [
    "nist_srm_612",
]
