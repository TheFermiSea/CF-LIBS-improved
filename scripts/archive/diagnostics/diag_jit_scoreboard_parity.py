"""J12 sanity: jit vs reference scoreboard scorer agree on a few spectra.

Runs ``_score_spectrum`` with pipeline_impl='reference' and ='jit' on a handful
of synthetic_fixedforward spectra and compares the scored outputs (presence
TP/FP/FN, predicted wt%, T, n_e). Validates the --pipeline jit plumbing before
launching a full board.

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=$PWD python -u \
        scripts/diag_jit_scoreboard_parity.py [--n 3] [--dataset synthetic_fixedforward]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cflibs.atomic.database import AtomicDatabase
from cflibs.benchmark.scoreboard import _score_spectrum
from cflibs.benchmark.scoreboard_registry import ensure_default_datasets, iter_datasets

REPO = Path(__file__).resolve().parent.parent
DB = REPO / "ASD_da" / "libs_production.db"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--dataset", default="synthetic_fixedforward")
    args = ap.parse_args()

    ensure_default_datasets()
    db = AtomicDatabase(str(DB))
    entry = next(e for e in iter_datasets() if e.name == args.dataset)
    items = list(entry.adapter_factory())[: args.n]
    print(f"dataset={args.dataset}  comparing {len(items)} spectra (preset=geological)\n")

    n_agree_presence = 0
    for sid, wl, inten, truth in items:
        ref = _score_spectrum(db, sid, wl, inten, truth, pipeline_impl="reference")
        jit = _score_spectrum(db, sid, wl, inten, truth, pipeline_impl="jit")
        rt = (ref.get("tp"), ref.get("fp"), ref.get("fn"))
        jt = (jit.get("tp"), jit.get("fp"), jit.get("fn"))
        presence_match = rt == jt
        n_agree_presence += int(presence_match)
        rcomp = ref.get("predicted_wt", {})
        jcomp = jit.get("predicted_wt", {})
        keys = sorted(set(rcomp) | set(jcomp))
        max_dwt = max((abs(rcomp.get(k, 0.0) - jcomp.get(k, 0.0)) for k in keys), default=0.0)
        print(f"[{sid}] status ref={ref['status']} jit={jit['status']}")
        print(f"    presence match={presence_match}  ref(tp/fp/fn)={rt}  jit(tp/fp/fn)={jt}")
        print(f"    T  ref={ref.get('temperature_K')!s:>10} jit={jit.get('temperature_K')!s:>10}")
        print(f"    ne ref={ref.get('electron_density_cm3')!s:>12} "
              f"jit={jit.get('electron_density_cm3')!s:>12}")
        print(f"    max |Δwt%| (ref vs jit) = {max_dwt:.4f}")
        if ref.get("rmse_wt") is not None or jit.get("rmse_wt") is not None:
            print(f"    rmse_wt ref={ref.get('rmse_wt')} jit={jit.get('rmse_wt')}")
        print()

    print(f"presence agreement: {n_agree_presence}/{len(items)}")


if __name__ == "__main__":
    main()
