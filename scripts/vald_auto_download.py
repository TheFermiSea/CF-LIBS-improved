#!/usr/bin/env python
"""Download VALD3 'Extract' results directly off the VALD FTP area by job number.

VALD serves every completed extraction at a PREDICTABLE public URL keyed on the
account username + job number (no email-link needed), discovered from the HELIOS-K
``vald_download.py`` tooling (Grimm, exoclime/HELIOS-K):

    http://vald.astro.uu.se/~vald/FTP/<jobname>.<jobnum>.gz       (line list)
    http://vald.astro.uu.se/~vald/FTP/<jobname>.<jobnum>.bib.gz   (bibliography)

``<jobname>`` is the VALD username (e.g. the existing data/vald/ files are
``BrianSquires.<jobnum>.gz`` -> jobname = ``BrianSquires``). This script fetches a
job-number RANGE, keeping files GZIPPED and named ``vald_<jobname>_<num>.linelist.gz``
so ``ingest_vald_atomic.py`` (glob ``data/vald/*.linelist.gz``) consumes them directly.
404s (job not ready / expired / wrong number) are skipped, not fatal.

Usage:
    python scripts/vald_auto_download.py --jobname BrianSquires \
        --jobstart 19830 --jobend 19883 --out data/vald
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

FTP = "http://vald.astro.uu.se/~vald/FTP"


def _curl(url: str, dest: Path) -> bool:
    """Fetch url -> dest with curl; return True on success, False on 404/error."""
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    r = subprocess.run(
        ["curl", "-fsS", "--max-time", "300", "--retry", "2", "-o", str(tmp), url],
        capture_output=True,
    )
    if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
        tmp.replace(dest)
        return True
    tmp.unlink(missing_ok=True)
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobname", required=True, help="VALD username (e.g. BrianSquires)")
    ap.add_argument("--jobstart", type=int, required=True, help="first job number")
    ap.add_argument("--jobend", type=int, required=True, help="last job number (inclusive)")
    ap.add_argument("--out", default="data/vald", help="output dir")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    got = missing = 0
    for num in range(args.jobstart, args.jobend + 1):
        ll = out / f"vald_{args.jobname}_{num}.linelist.gz"
        if ll.exists() and ll.stat().st_size > 0:
            print(f"SKIP {num} (already have)")
            got += 1
            continue
        # VALD names job files zero-padded to 6 digits (e.g. BrianSquires.019856.gz);
        # fall back to the unpadded form for >6-digit numbers / other accounts.
        jobstrs = [f"{num:06d}", str(num)]
        ok = False
        for js in jobstrs:
            if _curl(f"{FTP}/{args.jobname}.{js}.gz", ll):
                size_mb = ll.stat().st_size / 1e6
                _curl(f"{FTP}/{args.jobname}.{js}.bib.gz", out / f"vald_{args.jobname}_{num}.bib.gz")
                print(f"OK   {num} (as {js})  ({size_mb:.1f} MB) -> {ll.name}")
                got += 1
                ok = True
                break
        if not ok:
            print(f"MISS {num} (404 / not ready)")
            missing += 1
    print(f"\nDONE: {got} downloaded/present, {missing} missing in [{args.jobstart},{args.jobend}]")


if __name__ == "__main__":
    main()
