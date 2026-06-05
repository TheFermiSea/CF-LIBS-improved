from pathlib import Path
import numpy as np
from cflibs.atomic.database import AtomicDatabase
from cflibs.io.spectrum import load_spectrum
from cflibs.inversion.identify import line_detection as LD
DB="/home/brian/code/CF-LIBS-improved/ASD_da/libs_production.db"
SPEC="/home/brian/code/CF-LIBS-improved/data/bhvo2_usgs/chemcam_bhvo2_loc1_spectrum.csv"
ELEMS=["Si","Ti","Al","Fe","Mn","Mg","Ca","Na","K","P","Ag","Sn","W","Bi"]
wl,inten=load_spectrum(SPEC); wl=wl-0.10; db=AtomicDatabase(Path(DB))
peaks=LD._find_peaks(wl,inten,0.01,0.2,use_jax_fallback=False)
pwl=np.array([p[1] for p in peaks]); tot=len(pwl)
wl_step=float(np.median(np.diff(wl)))
grid=LD._build_shift_grid(0.5,None,wl_step,0.1)
wl_range=max(float(pwl.max()-pwl.min()),1e-6)
for floor in (100.0, 1.0):
    trans=LD._load_transitions(db,ELEMS,wl.min(),wl.max(),floor)
    tbe={}
    for t in trans: tbe.setdefault(t.element,[]).append(t)
    dens={e:len(v)/wl_range for e,v in tbe.items()}
    med=float(np.median(list(dens.values())))
    print(f"\n=== floor={floor} median_density={med:.3f} (kdet_min_score=0.05) ===")
    print(f"  {'el':3}{'nlines':>7}{'best_cand':>10}{'frac':>7}{'rarity_w':>9}{'score':>7}{'PASS':>6}")
    for e in ["Al","Mg","Na","K","Fe","Ti","Ca","Si","Ag","Bi"]:
        v=tbe.get(e,[])
        if not v: 
            print(f"  {e:3}  (no transitions @ floor)"); continue
        twl=np.sort(np.array([t.wavelength_nm for t in v]))
        best=0
        for s in grid:
            m=LD._peaks_within_tolerance(pwl+s,twl,0.1); c=int(m.sum())
            best=max(best,c)
        frac=best/tot
        rw=(med/max(dens[e],1e-6))**0.5; rw=float(np.clip(rw,0.25,4.0))
        sc=frac*rw
        print(f"  {e:3}{len(v):>7}{best:>10}{frac:>7.3f}{rw:>9.3f}{sc:>7.3f}{('Y' if (best>=2 and sc>=0.05) else 'n'):>6}")
