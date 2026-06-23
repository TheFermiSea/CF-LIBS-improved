#!/usr/bin/env python
"""Submit VALD3 'Extract Element' requests headlessly via Playwright.

Port of HELIOS-K's Selenium ``vald_request.py`` (Grimm, exoclime/HELIOS-K) to
Playwright + headless Chromium (no geckodriver / system Firefox; runs on a headless
server). Per species it: logs in (email), opens 'Extract Element', enters the
wavelength range + ``El ion`` token, selects Long format + FTP + HFS splitting, and
submits. The completed extraction is later pulled by ``vald_auto_download.py``.

UNITS: this does NOT set 'Unit selections' — it relies on the account's SAVED units.
The existing data/vald/ slices are AIR / Angstrom / eV, so wavelength bounds here are
ANGSTROM (1000-10000 A = 100-1000 nm). If your account is set otherwise, fix it once
on the VALD 'Unit selections' page (or run with --debug to inspect the live form).

SELECTOR RISK: VALD's form may have changed since the 2020 HELIOS-K script. Run once
with ``--debug`` on a single species first: it saves a screenshot + page HTML after
each step to --debug-dir so the selectors can be verified before a full sweep.

Usage:
    # de-risk first (one species, dump form state):
    python scripts/vald_auto_request.py --email you@example.com --species "Fe 1" \
        --debug --debug-dir /tmp/vald_debug
    # then a sweep (atomic neutrals+ions + the 7 minor molecules):
    python scripts/vald_auto_request.py --email you@example.com \
        --elements all --ions 1 2 3 --molecules --throttle 20
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

# Z-ordered element symbols (1..92, H..U) — matches HELIOS-K elt0 ordering.
ELEMENTS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
    "Fr Ra Ac Th Pa U"
).split()
MOLECULES = ["CN", "C2", "OH", "CH", "CO", "NH", "MgH"]  # TiO excluded (ExoMol bulk)


def build_species(args) -> list[str]:
    if args.species:
        return list(args.species)
    out: list[str] = []
    els = ELEMENTS if args.elements == ["all"] else args.elements
    for el in els:
        for ion in args.ions:
            out.append(f"{el} {ion}")
    if args.molecules:
        out.extend(f"{m} 1" for m in MOLECULES)
    return out


def _dump(page, dbg: Path, tag: str) -> None:
    dbg.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(dbg / f"{tag}.png"), full_page=True)
    (dbg / f"{tag}.html").write_text(page.content())
    print(f"  [debug] wrote {tag}.png + {tag}.html")


def submit_one(
    page, species: str, wl_min_a: float, wl_max_a: float, email: str, debug: bool, dbg: Path
) -> str | None:
    """Run the Extract-Element form for one species; return job-number text or None."""
    page.goto("http://vald.astro.uu.se", timeout=60000)
    # login (email-only)
    page.fill('input[name="user"]', email)
    page.click('input[type="submit"]')
    page.wait_for_load_state("networkidle")
    if debug:
        _dump(page, dbg, "1_after_login")
    # Extract Element
    page.click('input[value="Extract Element"]')
    page.wait_for_load_state("networkidle")
    if debug:
        _dump(page, dbg, "2_extract_element")
    # wavelength range (Angstrom; account units assumed air/Angstrom) + species
    page.fill('input[name="stwvl"]', str(wl_min_a))
    page.fill('input[name="endwvl"]', str(wl_max_a))
    page.fill('input[name="elmion"]', species)
    page.click('input[value="long"]')
    page.click('input[value="via ftp"]')
    page.click('input[value="HFS splitting"]')
    if debug:
        _dump(page, dbg, "3_filled")
    page.click('input[value="Submit request"]')
    page.wait_for_load_state("networkidle")
    if debug:
        _dump(page, dbg, "4_submitted")
    text = page.inner_text("body")
    m = re.search(r"\b(\d{5,7})\b", text)  # job number heuristic; verify in --debug
    return m.group(1) if m else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--email", required=True, help="VALD-registered email (login)")
    ap.add_argument("--species", nargs="+", help='explicit species, e.g. "Fe 1" "CN 1"')
    ap.add_argument("--elements", nargs="+", default=["all"], help="elements or 'all'")
    ap.add_argument("--ions", nargs="+", type=int, default=[1, 2, 3], help="ion stages")
    ap.add_argument("--molecules", action="store_true", help="add the 7 minor molecules")
    ap.add_argument("--wl-min-a", type=float, default=1000.0, help="min Angstrom (100 nm)")
    ap.add_argument("--wl-max-a", type=float, default=10000.0, help="max Angstrom (1000 nm)")
    ap.add_argument("--throttle", type=float, default=20.0, help="seconds between submits")
    ap.add_argument("--manifest", default="data/vald/submitted_manifest.tsv")
    ap.add_argument("--debug", action="store_true", help="one species, dump form state")
    ap.add_argument("--debug-dir", default="/tmp/vald_debug")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    species = build_species(args)
    if args.debug:
        species = species[:1]
    print(f"{len(species)} species to submit; throttle {args.throttle}s")
    dbg = Path(args.debug_dir)
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        with manifest.open("a") as mf:
            for i, sp in enumerate(species):
                try:
                    job = submit_one(
                        page, sp, args.wl_min_a, args.wl_max_a, args.email, args.debug, dbg
                    )
                    print(f"[{i+1}/{len(species)}] submitted {sp!r} -> job {job}")
                    mf.write(f"{sp}\t{job}\n")
                    mf.flush()
                except Exception as e:
                    print(f"[{i+1}/{len(species)}] FAILED {sp!r}: {type(e).__name__}: {e}")
                    if args.debug:
                        _dump(page, dbg, "error")
                    break
                if not args.debug and i < len(species) - 1:
                    time.sleep(args.throttle)
        browser.close()
    print(f"manifest -> {manifest}")


if __name__ == "__main__":
    main()
