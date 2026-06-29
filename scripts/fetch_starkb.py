"""Fetch electron-impact Stark widths from STARK-B (stark-b.obspm.fr).

Reverse-engineered POST API (no VAMDC, no manual export):
  periodictable:datasets&element=<El>      -> ions, each /index.php/data/ion/<id>
  GET /index.php/data/ion/<id>             -> <select name=dataset> datasetId (+ session)
  querypage:query (queryType=wavelength, density=1e17, temperature=all,
                   wavelengthMin=0, wavelengthMax=100000, format=html)
                                           -> table: Wavelength(Å), T(K), N(cm-3), w(Å), d(Å)

Writes data/stark_b/raw/<El>_<ion>.csv in the ingest_stark_b.py format:
  transition (nm), T_e (K), n_e (cm-3), gamma_W (Å), gamma_d (Å)
"""
from __future__ import annotations

import html as _html
import os
import re
import sys
import time

import requests

BASE = "https://stark-b.obspm.fr"
OUT = "data/stark_b/raw"
ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII",
         9: "IX", 10: "X", 11: "XI", 12: "XII", 13: "XIII"}

# Full periodic table — STARK-B only has a subset; elements with no data are
# skipped after one cheap datasets probe. Pass args to override (a subset).
ELEMENTS = sys.argv[1:] or [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si",
    "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
    "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb",
    "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
    "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np",
    "Pu", "Am", "Cm", "Bk", "Cf", "Es",
]

S = requests.Session()
S.headers.update({"User-Agent": "cflibs-starkb/1.0"})


def _retry(fn, tries=4):
    for i in range(tries):
        try:
            return fn()
        except requests.exceptions.RequestException:
            if i == tries - 1:
                raise
            time.sleep(3 * (i + 1))
    return None


def post(action, **params):
    data = {"module": "stark_b", "action": action}
    data.update({k: v for k, v in params.items() if v is not None})
    return _retry(lambda: S.post(f"{BASE}/index.php", data=data, timeout=60).text)


def get_text(url):
    return _retry(lambda: S.get(url, timeout=60).text)


def cells(row_html):
    return [_html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
            for c in re.findall(r"<td[^>]*>(.*?)</td>", row_html, re.S | re.I)]


def parse_table(page):
    """Header-driven parse of the WIDTHS table -> (widths_header, list[dict]).

    The page has two tables with a 'Wavelength' header: the widths table
    (N, ..., Wavelength, C, T (K), A, *w, w (Å), *d, d (Å)) and a fit-coeffs
    table (..., a0, a1, a2, b0, b1, b2). Lock onto the widths one (it has a
    'T (K)' column) and stop collecting when any other header appears.
    """
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", page, re.S | re.I)
    cur = None
    widths_header = None
    out = []
    for r in rows:
        hdr = [_html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
               for c in re.findall(r"<th[^>]*>(.*?)</th>", r, re.S | re.I)]
        if hdr:
            lo = [h.lower() for h in hdr]
            is_widths = any("t (k)" in h for h in lo) and any("avelength" in h for h in lo)
            cur = hdr if is_widths else None
            if is_widths:
                widths_header = hdr
            continue
        if cur is None:
            continue
        cs = cells(r)
        if len(cs) < len(cur):
            continue
        out.append(dict(zip(cur, cs)))
    return widths_header, out


def colidx(header, *needles):
    for i, h in enumerate(header):
        hl = h.lower()
        if all(n in hl for n in needles):
            return h
    return None


def fetch_ion(ion_id, dataset):
    # The query is STATEFUL: the server needs the full densities->transitions->
    # temperatures sequence (same session) to set the wavelength query context,
    # else it returns "incorrect parameters". Gentle delays (small academic server).
    get_text(f"{BASE}/index.php/data/ion/{ion_id}")
    time.sleep(0.6)
    post("querypage:densities", elementId=ion_id, datasetId=dataset,
         targetFormElement="densities")
    time.sleep(0.6)
    post("querypage:transitions", elementId=ion_id, datasetId=dataset, density="1e17",
         targetFormElement="transitions")
    time.sleep(0.6)
    post("querypage:temperatures", elementId=ion_id, datasetId=dataset, density="1e17",
         queryType="wavelength", wavelengthMin="0", wavelengthMax="100000",
         targetFormElement="temperatures")
    time.sleep(0.6)
    page = post("querypage:query", elementId=ion_id, datasetId=dataset, density="1e17",
                temperature="all", queryType="wavelength", wavelengthMin="0",
                wavelengthMax="100000", format="html", targetFormElement="results")
    header, rows = parse_table(page)
    if not header:
        return []
    c_wl = colidx(header, "wavelength")
    c_t = colidx(header, "t (k)") or colidx(header, "t(k)") or colidx(header, "t ")
    c_n = colidx(header, "n (cm") or colidx(header, "n(cm") or colidx(header, "electron")
    c_w = colidx(header, "w (")
    c_d = colidx(header, "d (")
    recs = []
    for r in rows:
        try:
            wl_aa = float(r[c_wl])
            t_k = float(r[c_t])
            n_e = float(re.sub(r"[^0-9eE.+-]", "", r[c_n])) if c_n else 1e17
            w_aa = float(r[c_w]) if r.get(c_w) else None
            d_aa = float(r[c_d]) if (c_d and r.get(c_d)) else ""
        except (KeyError, ValueError, TypeError):
            continue
        if w_aa is None:
            continue
        recs.append((wl_aa * 0.1, t_k, n_e, w_aa, d_aa))  # nm, K, cm-3, Å, Å
    return recs


def main():
    os.makedirs(OUT, exist_ok=True)
    S.get(BASE, timeout=40)  # establish base session
    time.sleep(0.5)
    total = 0
    for el in ELEMENTS:
        ds = post("periodictable:datasets", element=el)
        ions = re.findall(r"/index\.php/data/ion/(\d+)\">\s*([A-Za-z]+)\s+([IVXL]+)", ds)
        if not ions:
            print(f"{el}: no ions in STARK-B", flush=True)
            time.sleep(0.4)
            continue
        for ion_id, _eln, roman in ions:
            path = f"{OUT}/{el}_{roman}.csv"
            if os.path.exists(path):  # resumable: skip already-downloaded
                print(f"  {el} {roman}: cached, skip", flush=True)
                continue
            try:
                page = get_text(f"{BASE}/index.php/data/ion/{ion_id}")
                opts = re.findall(r'<option[^>]*value="([^"]+)"', page)
                dataset = next((v for v in opts if v not in ("-1", "all", "")), None)
                if not dataset:
                    print(f"  {el} {roman}: no dataset", flush=True)
                    continue
                recs = fetch_ion(ion_id, dataset)
            except Exception as exc:  # noqa: BLE001 — keep going on a flaky request
                print(f"  {el} {roman}: ERROR {type(exc).__name__} {str(exc)[:60]}", flush=True)
                time.sleep(3)
                continue
            if not recs:
                print(f"  {el} {roman} (ion {ion_id}): 0 rows", flush=True)
                time.sleep(0.5)
                continue
            with open(path, "w") as fh:
                fh.write("transition,T_e,n_e,gamma_W,gamma_d\n")
                for wl_nm, t_k, n_e, w, d in recs:
                    fh.write(f"{wl_nm:.4f},{t_k:.0f},{n_e:.3e},{w},{d}\n")
            print(f"  {el} {roman}: {len(recs)} rows -> {path}", flush=True)
            total += len(recs)
            time.sleep(0.5)
    print(f"\nTOTAL: {total} Stark-width rows across {len(ELEMENTS)} elements -> {OUT}/")


if __name__ == "__main__":
    main()
