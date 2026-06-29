"""End-to-end validation driver for the ExoJAX-grade CF-LIBS reference forward
model + structured-Jacobian K=1 Gauss-Newton inversion.

Modes:
  --synth          Generate a spectrum from a KNOWN (T, n_e, composition) via the
                   reference forward, add realistic Poisson+Gaussian noise, then
                   invert and report recovery accuracy + sub-ms inversion latency.
  --real <npz>     Invert a real spectrum loaded from the real-spectra agent output
                   (/tmp/rtval/data/*.npz). Reports recovered composition vs the
                   stored ground-truth wt% (mapped onto the bundle's element set).

Shared atomic-line bundle: --bundle <npz> (build_atomic_bundle.py output).

Usage:
  python validate.py --synth  --bundle /tmp/rtval/atomic/bundle.npz
  python validate.py --real /tmp/rtval/data/chemcam_calib.npz --bundle ... --index 0
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import jax
import jax.numpy as jnp

from reference_forward import AtomicBundle, InstrumentConfig, make_reference_forward
from reference_inversion import make_structured_inversion, invert, _peak_norm

jax.config.update("jax_enable_x64", False)  # float32: GPU-fast, realistic for RT


# ---------------------------------------------------------------------------
def _device_info():
    return dict(
        device=str(jax.devices()[0]),
        backend=jax.default_backend(),
        jax_version=jax.__version__,
    )


def _build_forward(bundle_path, wl_grid, fwhm_nm=0.05, self_absorption=False, tau=0.0):
    bundle = AtomicBundle.from_npz(bundle_path)
    instr = InstrumentConfig(
        wl_grid_nm=np.asarray(wl_grid, dtype=np.float32),
        fwhm_nm=fwhm_nm,
        use_self_absorption=self_absorption,
        escape_tau_scale=tau,
    )
    fwd = make_reference_forward(bundle, instr)
    return bundle, instr, fwd


def comp_rmse(c_rec, c_true):
    return float(np.sqrt(np.mean((np.asarray(c_rec) - np.asarray(c_true)) ** 2)))


# ---------------------------------------------------------------------------
# SYNTH MODE
# ---------------------------------------------------------------------------
def run_synth(args):
    info = _device_info()
    # fixed instrument grid (instrument-space, nm)
    wl = np.linspace(args.wl_min, args.wl_max, args.channels).astype(np.float32)
    bundle, instr, fwd = _build_forward(
        args.bundle, wl, fwhm_nm=args.fwhm,
        self_absorption=args.self_absorption, tau=args.tau,
    )
    n_elem = bundle.n_elem
    rng = np.random.default_rng(args.seed)

    # ---- known ground truth ----
    T_true = float(args.T_true)
    logne_true = float(np.log10(args.ne_true))
    # a structured, non-uniform composition (dominant matrix element + minors)
    raw_true = rng.uniform(-1.5, 1.5, n_elem).astype(np.float32)
    e = np.exp(raw_true - raw_true.max())
    comp_true = (e / e.sum()).astype(np.float32)

    forward_comp = fwd["forward_comp"]
    clean = np.asarray(jax.jit(forward_comp)(
        jnp.float32(T_true), jnp.float32(10.0 ** logne_true), jnp.asarray(comp_true)
    ))
    clean = np.clip(clean, 0.0, None)

    # ---- realistic detector noise: Poisson (shot) + read noise ----
    peak = clean.max() + 1e-30
    photon_scale = args.peak_counts / peak
    counts = clean * photon_scale
    noisy = rng.poisson(np.clip(counts, 0, None)).astype(np.float32)
    noisy = noisy + rng.normal(0.0, args.read_noise, counts.shape).astype(np.float32)
    spectrum = np.clip(noisy / photon_scale, 0.0, None).astype(np.float32)

    # ---- warm-start (realistic Boltzmann-plot prior on T, ne; neutral comp) ----
    T0 = T_true * rng.uniform(0.9, 1.1)
    logne0 = logne_true + rng.uniform(-0.2, 0.2)
    theta0 = jnp.asarray(
        np.concatenate([[T0, logne0], np.zeros(n_elem)]).astype(np.float32)
    )

    runner = make_structured_inversion(fwd, k=args.k)
    res = invert(spectrum, fwd, theta0=theta0, k=args.k,
                 n_timing=args.n_timing, runner=runner)

    rmse = comp_rmse(res["composition"], comp_true)
    T_err = abs(res["T"] - T_true) / T_true
    ne_err = abs(res["ne"] - 10.0 ** logne_true) / (10.0 ** logne_true)

    out = dict(
        mode="synth",
        **info,
        elements=list(bundle.elements),
        n_elem=n_elem,
        n_lines=bundle.n_lines,
        channels=args.channels,
        wl_range_nm=[args.wl_min, args.wl_max],
        k=args.k,
        forward_engine="exojax.opacity.lpf (Voigt-Hjerting xsvector)",
        self_absorption=bool(args.self_absorption),
        truth=dict(
            T=T_true, ne=10.0 ** logne_true,
            composition={el: float(c) for el, c in zip(bundle.elements, comp_true)},
        ),
        recovered=dict(
            T=res["T"], ne=res["ne"],
            composition={el: float(c) for el, c in zip(bundle.elements, res["composition"])},
        ),
        accuracy=dict(comp_rmse=rmse, T_rel_err=T_err, ne_rel_err=ne_err),
        inversion_latency_us=res["latency_us"],
        sub_ms=res["latency_us"]["median"] < 1000.0,
    )
    _emit(out, args.out)
    return out


# ---------------------------------------------------------------------------
# REAL MODE
# ---------------------------------------------------------------------------
def run_real(args):
    info = _device_info()
    d = np.load(args.real, allow_pickle=True)
    wl_real = np.asarray(d["wavelength_nm"], dtype=np.float32)
    inten = np.asarray(d["intensity"], dtype=np.float32)
    ds_elements = [str(x) for x in d["elements"]]
    comp_wt = np.asarray(d["composition_wt"], dtype=np.float64)  # (N, E_ds) wt%
    idx = args.index
    spectrum_full = inten[idx]

    # restrict to the bundle's wavelength window (the reference grid for inversion)
    lo = float(max(args.wl_min, wl_real.min()))
    hi = float(min(args.wl_max, wl_real.max()))
    mask = (wl_real >= lo) & (wl_real <= hi)
    wl = wl_real[mask]
    spectrum = np.clip(spectrum_full[mask], 0.0, None).astype(np.float32)

    bundle, instr, fwd = _build_forward(
        args.bundle, wl, fwhm_nm=args.fwhm,
        self_absorption=args.self_absorption, tau=args.tau,
    )
    n_elem = bundle.n_elem

    runner = make_structured_inversion(fwd, k=args.k)
    res = invert(spectrum, fwd, T0=args.T0, logne0=np.log10(args.ne0),
                 k=args.k, n_timing=args.n_timing, runner=runner)

    # map ground-truth wt% onto the bundle's element order (only shared elements);
    # renormalize over shared elements for an apples-to-apples comparison.
    gt_map = {el: float(comp_wt[idx, j]) for j, el in enumerate(ds_elements)}
    shared = [el for el in bundle.elements if el in gt_map]
    gt_shared = np.array([gt_map[el] for el in shared], dtype=np.float64)
    rec_full = {el: float(c) for el, c in zip(bundle.elements, res["composition"])}
    rec_shared = np.array([rec_full[el] for el in shared], dtype=np.float64)
    gt_norm = gt_shared / (gt_shared.sum() + 1e-30)
    rec_norm = rec_shared / (rec_shared.sum() + 1e-30)
    rmse_shared = comp_rmse(rec_norm, gt_norm) if len(shared) else None

    out = dict(
        mode="real",
        **info,
        dataset=str(d["dataset"]) if "dataset" in d.files else args.real,
        spectrum_index=idx,
        spectrum_id=str(d["spectrum_ids"][idx]) if "spectrum_ids" in d.files else None,
        elements=list(bundle.elements),
        n_elem=n_elem,
        n_lines=bundle.n_lines,
        channels=int(mask.sum()),
        wl_range_nm=[lo, hi],
        k=args.k,
        forward_engine="exojax.opacity.lpf (Voigt-Hjerting xsvector)",
        recovered=dict(
            T=res["T"], ne=res["ne"],
            composition_numberfrac=rec_full,
        ),
        ground_truth_wt_shared={el: float(g) for el, g in zip(shared, gt_shared)},
        shared_elements=shared,
        comp_rmse_shared_renorm=rmse_shared,
        inversion_latency_us=res["latency_us"],
        sub_ms=res["latency_us"]["median"] < 1000.0,
        note="real-data comp comparison is number-fraction(rec) vs wt%(truth) over "
             "SHARED elements, both renormalized; not a calibrated CF-LIBS quant.",
    )
    _emit(out, args.out)
    return out


def _emit(out, path):
    txt = json.dumps(out, indent=2)
    print(txt)
    if path:
        with open(path, "w") as fh:
            fh.write(txt)
        print(f"\nWROTE {path}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--synth", action="store_true", help="synthetic round-trip")
    g.add_argument("--real", type=str, help="path to real-spectra .npz")

    ap.add_argument("--bundle", required=True, help="atomic bundle .npz")
    ap.add_argument("--k", type=int, default=1, help="GN steps (default 1)")
    ap.add_argument("--channels", type=int, default=4000)
    ap.add_argument("--wl-min", type=float, default=240.0)
    ap.add_argument("--wl-max", type=float, default=850.0)
    ap.add_argument("--fwhm", type=float, default=0.05, help="instrument FWHM nm")
    ap.add_argument("--self-absorption", action="store_true")
    ap.add_argument("--tau", type=float, default=0.0, help="escape-factor strength")
    ap.add_argument("--n-timing", type=int, default=120)
    ap.add_argument("--out", type=str, default=None, help="write JSON result here")

    # synth
    ap.add_argument("--T-true", type=float, default=9500.0)
    ap.add_argument("--ne-true", type=float, default=1.0e17)
    ap.add_argument("--peak-counts", type=float, default=5000.0)
    ap.add_argument("--read-noise", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=7)
    # real
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--T0", type=float, default=9000.0)
    ap.add_argument("--ne0", type=float, default=1.0e17)

    args = ap.parse_args()
    if args.synth:
        run_synth(args)
    else:
        run_real(args)


if __name__ == "__main__":
    main()
