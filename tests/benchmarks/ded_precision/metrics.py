"""Accuracy + precision metrics for the DED benchmark (DED-PLAN section 3).

ABSOLUTE trueness is primary (RMSEP + bias on the known elements), then ratio
accuracy, then precision/repeatability and Delta-sensitivity. All inputs are
wt% dicts on the constrained element set; renormalize references to that set
before scoring (basis-consistency rule).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from cflibs.inversion.physics.closure import log_ratios as _log_ratios


def renorm_to_set(comp_wt: Dict[str, float], elements: Sequence[str]) -> Dict[str, float]:
    """Renormalize a composition to the K-element set, summing to 100 wt%."""
    s = sum(float(comp_wt.get(e, 0.0)) for e in elements) or 1.0
    return {e: float(comp_wt.get(e, 0.0)) / s * 100.0 for e in elements}


def absolute_metrics(
    predicted: Sequence[Dict[str, float]],
    truth: Sequence[Dict[str, float]],
    elements: Sequence[str],
) -> Dict[str, object]:
    """Per-element RMSEP / bias / MAE / MaxAE (wt%) + joint RMSEP over a paired
    set of predicted/truth compositions."""
    assert len(predicted) == len(truth) and predicted, "paired, non-empty required"
    per: Dict[str, Dict[str, float]] = {}
    all_err: List[np.ndarray] = []
    for e in elements:
        d = np.array(
            [predicted[i].get(e, 0.0) - truth[i].get(e, 0.0) for i in range(len(truth))],
            dtype=float,
        )
        per[e] = {
            "rmsep": float(np.sqrt(np.mean(d**2))),
            "bias": float(np.mean(d)),
            "mae": float(np.mean(np.abs(d))),
            "maxae": float(np.max(np.abs(d))),
        }
        all_err.append(d)
    joint = float(np.sqrt(np.mean(np.concatenate(all_err) ** 2)))
    return {"per_element": per, "rmsep_joint": joint, "n": len(truth)}


def ratio_rmsep(
    predicted: Sequence[Dict[str, float]],
    truth: Sequence[Dict[str, float]],
    numerator: str,
    denominator: str,
    eps: float = 0.01,
) -> float:
    """RMSEP of the numerator/denominator wt% ratio (e.g. Al/Ti)."""
    pr = np.array(
        [p.get(numerator, 0.0) / max(p.get(denominator, 0.0), eps) for p in predicted],
        dtype=float,
    )
    tr = np.array(
        [t.get(numerator, 0.0) / max(t.get(denominator, 0.0), eps) for t in truth],
        dtype=float,
    )
    return float(np.sqrt(np.mean((pr - tr) ** 2)))


def log_ratio_metrics(
    predicted: Sequence[Dict[str, float]],
    truth: Sequence[Dict[str, float]],
    reference: str,
    numerators: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Aitchison log-ratio accuracy: error of ``ln(C_num / C_ref)`` (Issue 2).

    For each numerator element the per-realization error is
    ``ln(pred_num/pred_ref) - ln(truth_num/truth_ref)``. This is the matrix- and
    detected-set-invariant DED deliverable: the closure denominator cancels, so
    unlike absolute wt% this quantity does not slosh when one element's line set
    is perturbed (``MatrixEffects.lean`` ``recoveredComposition_ratio_matrix_invariant``).

    The predicted-minus-truth error is identical whether the input dicts are wt%
    or number/mole fractions -- the atomic-weight offset ``ln(AW_num/AW_ref)``
    cancels in the difference -- so this scores the same quantity as
    :meth:`CFLIBSResult.log_ratios` regardless of basis.

    Non-finite log-ratios (a zero/missing numerator, or a composition with a
    zero/absent reference) drop out of the aggregate as NaN.

    Returns
    -------
    dict
        ``{"reference": str, "n": int, "per_numerator": {el: {rmsep, bias,
        std, n}}}`` (log-ratio units, natural log).
    """
    assert len(predicted) == len(truth) and predicted, "paired, non-empty required"

    def _safe(comp: Dict[str, float]) -> Dict[str, float]:
        try:
            return _log_ratios(comp, reference)
        except (KeyError, ValueError):
            return {}

    pred_lr = [_safe(p) for p in predicted]
    truth_lr = [_safe(t) for t in truth]
    if numerators is None:
        keys: set = set()
        for d in truth_lr:
            keys.update(d)
        numerators = sorted(keys)

    per: Dict[str, Dict[str, float]] = {}
    for num in numerators:
        d = np.array(
            [
                pred_lr[i].get(num, float("nan")) - truth_lr[i].get(num, float("nan"))
                for i in range(len(truth))
            ],
            dtype=float,
        )
        d = d[np.isfinite(d)]
        per[num] = {
            "rmsep": float(np.sqrt(np.mean(d**2))) if d.size else float("nan"),
            "bias": float(np.mean(d)) if d.size else float("nan"),
            "std": float(np.std(d)) if d.size else float("nan"),
            "n": int(d.size),
        }
    return {"reference": reference, "per_numerator": per, "n": len(truth)}


def precision_std(
    predicted_replicates: Sequence[Dict[str, float]], elements: Sequence[str]
) -> Dict[str, float]:
    """Per-element std (wt%) over replicate predictions at FIXED composition."""
    return {e: float(np.std([p.get(e, 0.0) for p in predicted_replicates])) for e in elements}


def min_detectable_change(sigma_wt: float, n_window: int, k_sigma: float = 3.0) -> float:
    """3-sigma minimum detectable composition change at n-shot averaging."""
    return float(k_sigma * sigma_wt / np.sqrt(max(n_window, 1)))
