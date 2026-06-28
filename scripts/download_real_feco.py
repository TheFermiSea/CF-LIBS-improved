"""Download the Fe-Co certified-composition LIBS benchmark (figshare 21984989).

A binary Fe->Co alloy ladder (11 certified samples, 10 wt% steps + trace Mn/Pb),
50 spectra/sample/system, several spectrometer systems. MIT licensed. Useful as a
cross-matrix (cobalt-bearing) held-out test for the known-matrix OPC mode, which is
otherwise validated only on PhdYoda steels.

Source: J. Vrabel et al., CEITEC. https://figshare.com/articles/dataset/21984989
License: MIT.

Mirrors scripts/download_real_steel.py. Writes to data/real_feco/ (gitignored).
Run from the worktree root:  PYTHONPATH=$PWD python scripts/download_real_feco.py
"""

import json
import os
import urllib.request

OUT_DIR = "data/real_feco"
os.makedirs(OUT_DIR, exist_ok=True)

# figshare file id -> local name. Skip discovery_catalina (176 MB) by default.
FILES = {
    "readme.txt": "39018473",
    "import_h5.py": "39018464",
    "sample_composition.xlsx": "39018476",
    "labtrace_avantes_7mJ.h5": "39018467",
    "labtrace_avantes_15mJ.h5": "39018470",
    "firefly_avantes_multi_20mJ.h5": "39018461",
    # "discovery_catalina_20mJ.h5": "39018458",  # 176 MB, uncomment if needed
}
BASE_URL = "https://ndownloader.figshare.com/files/{}"


def download() -> None:
    for name, fid in FILES.items():
        dst = os.path.join(OUT_DIR, name)
        if os.path.exists(dst):
            print(f"skip (exists): {dst}")
            continue
        url = BASE_URL.format(fid)
        print(f"downloading {name} <- {url}")
        urllib.request.urlretrieve(url, dst)  # noqa: S310 (trusted figshare host)
        print(f"  saved -> {dst} ({os.path.getsize(dst)} bytes)")


def summarize() -> None:
    import h5py  # local import: only needed for the optional inspection
    import pandas as pd

    comp = os.path.join(OUT_DIR, "sample_composition.xlsx")
    if os.path.exists(comp):
        # Real header sits on the 2nd row: Sample, Fe, Co, Mn, Pb.
        df = pd.read_excel(comp, header=1).dropna(how="all")
        print("\ncertified composition (wt%):")
        print(df.to_string(index=False))

    for name in FILES:
        if not name.endswith(".h5"):
            continue
        path = os.path.join(OUT_DIR, name)
        if not os.path.exists(path):
            continue
        with h5py.File(path, "r") as f:
            spec = f["spectra"]
            wl = f["wavelengths"][()] if "wavelengths" in f else None
            ep = json.loads(spec.attrs["experimental_parameters"])
            samples = list(spec.attrs["samples"])
            print(f"\n{name}: spectra={spec.shape} dtype={spec.dtype}")
            if wl is not None:
                print(f"  wavelength {float(min(wl)):.1f}-{float(max(wl)):.1f} nm, n_bins={len(wl)}")
            print(f"  n_samples={len(set(samples))} (50 shots each), params={ep}")


if __name__ == "__main__":
    download()
    try:
        summarize()
    except Exception as exc:  # noqa: BLE001
        print("summary skipped:", repr(exc)[:160])
