from pathlib import Path
import numpy as np
from cflibs.atomic.database import AtomicDatabase
from cflibs.io.spectrum import load_spectrum
from cflibs.inversion.identify.line_detection import detect_line_observations
DB="/home/brian/code/CF-LIBS-improved/ASD_da/libs_production.db"
SPEC="/home/brian/code/CF-LIBS-improved/data/bhvo2_usgs/chemcam_bhvo2_loc1_spectrum.csv"
MAJORS=["Si","Ti","Al","Fe","Mn","Mg","Ca","Na","K","P"]; CONF=["Ag","Sn","W","Bi"]; ELEMS=MAJORS+CONF
wl,inten=load_spectrum(SPEC); wl=wl-0.10; db=AtomicDatabase(Path(DB))
def run(label,**kw):
    res=detect_line_observations(wl,inten,db,ELEMS,**kw)
    by={}
    for o in res.observations: by[o.element]=by.get(o.element,0)+1
    print(f"[{label}] majors={{ {', '.join(f'{e}:{by.get(e,0)}' for e in MAJORS)} }}")
    print(f"   conf={{ {', '.join(f'{e}:{by.get(e,0)}' for e in CONF)} }}  warns={res.warnings}")
run("floor=1 kdet=off comb prec=0.06 miss=0.6 recall=0.2 FALLBACK=OFF",
    min_relative_intensity=1.0, kdet_enabled=False, comb_min_precision=0.06,
    comb_max_missing_fraction=0.6, comb_min_recall=0.2, comb_fallback_to_nearest=False)
run("floor=1 kdet=ON comb tight FALLBACK=OFF",
    min_relative_intensity=1.0, comb_min_precision=0.06,
    comb_max_missing_fraction=0.6, comb_min_recall=0.2, comb_fallback_to_nearest=False)
