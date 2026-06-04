#!/usr/bin/env python3
"""Why does Al get 0 observation lines on BHVO-2? Direct detection probe."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

from cflibs.atomic.database import AtomicDatabase
from cflibs.io.spectrum import load_spectrum

spec = sys.argv[1] if len(sys.argv) > 1 else "data/bhvo2_usgs/chemcam_bhvo2_loc1_spectrum.csv"
sp = Path(spec)
if not sp.exists():
    sp = Path("/home/brian/code/CF-LIBS-improved") / spec

wl, inten = load_spectrum(str(sp))
db = AtomicDatabase(Path("/home/brian/code/CF-LIBS-improved/ASD_da/libs_production.db"))

print(f"spectrum: {sp.name}")
print(f"  wl range: {wl.min():.2f} - {wl.max():.2f} nm, n={len(wl)}, median dλ={np.median(np.diff(wl)):.4f} nm")
inorm = inten / inten.max()

# strong Al I resonance lines
al_lines = [308.215, 309.271, 394.401, 396.152]
trans = db.get_transitions("Al", wavelength_min=wl.min(), wavelength_max=wl.max())
print(f"\nAl transitions in DB within spectrum range: {len(trans)}")
strong = sorted(trans, key=lambda t: (t.A_ki or 0) * (t.g_k or 1), reverse=True)[:12]
print(f"  {'wl_nm':>9}{'stage':>6}{'A_ki':>11}{'E_k_eV':>8}{'g_k':>5}{'rel_int':>9}{'reson':>7}")
for t in strong:
    ri = t.relative_intensity if t.relative_intensity is not None else -1
    print(f"  {t.wavelength_nm:>9.3f}{t.ionization_stage:>6}{(t.A_ki or 0):>11.2e}"
          f"{t.E_k_ev:>8.2f}{t.g_k:>5}{ri:>9.1f}{str(t.is_resonance):>7}")

# detect peaks the way the pipeline roughly does
peaks_idx, _ = find_peaks(inorm, height=0.01, distance=2)
peak_wl = wl[peaks_idx]
print(f"\ndetected peaks (height>0.01): {len(peak_wl)}")

# For each strong Al line, find nearest detected peak + offset
print(f"\nNearest detected peak to each Al resonance line:")
print(f"  {'Al_line':>9}{'nearest_peak':>13}{'offset_nm':>11}{'peak_height':>12}")
for line in al_lines:
    if line < wl.min() or line > wl.max():
        print(f"  {line:>9.3f}   OUT OF SPECTRUM RANGE")
        continue
    d = np.abs(peak_wl - line)
    j = int(np.argmin(d))
    h = inorm[peaks_idx[j]]
    print(f"  {line:>9.3f}{peak_wl[j]:>13.3f}{peak_wl[j]-line:>+11.3f}{h:>12.4f}")

# Is there a systematic offset? cross-check with a couple of other known strong majors
print(f"\nSystematic-offset check via strong unblended major lines:")
ref = {"Mg II 279.55": 279.553, "Mg II 280.27": 280.270, "Ca II 393.37": 393.366,
       "Ca II 396.85": 396.847, "Si I 288.16": 288.158, "Na I 589.0": 588.995}
for name, line in ref.items():
    if line < wl.min() or line > wl.max():
        print(f"  {name:<16} OUT OF RANGE"); continue
    d = np.abs(peak_wl - line); j = int(np.argmin(d))
    print(f"  {name:<16} line={line:.3f} nearest_peak={peak_wl[j]:.3f} offset={peak_wl[j]-line:+.3f} h={inorm[peaks_idx[j]]:.3f}")
