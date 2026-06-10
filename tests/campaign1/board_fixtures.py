"""Shared rigged-board builders for the campaign fitness tests."""


def make_row(
    name,
    *,
    tags=("real",),
    f1=0.7,
    precision=1.0,
    recall=0.6,
    fp=0,
    n_failed=0,
    rmse=None,
    basis="presence_only",
    wall_s=1.0,
    n_spectra=10,
):
    spectra = [
        {
            "spectrum_id": f"{name}/s{i}",
            "status": "ok",
            "wall_s": wall_s,
            "composition_basis": basis,
            "tp": ["Fe"],
            "fp": [],
            "fn": [],
        }
        for i in range(n_spectra)
    ]
    composition = {"rmse_wt_median": rmse} if rmse is not None else None
    return {
        "name": name,
        "tags": list(tags),
        "n_run": n_spectra,
        "n_failed": n_failed,
        "id_metrics": {
            "tp": 10,
            "fp": fp,
            "fn": 4,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "composition": composition,
        "spectra": spectra,
        "failures": {},
        "runtime": {"median_wall_s": wall_s},
    }
