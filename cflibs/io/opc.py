"""JSON persistence for a known-matrix :class:`OPCCalibration`.

An OPC calibration is a tiny payload (one temperature, a handful of per-element
factors, and provenance), so it serializes cleanly to a small JSON file that is
reusable across a DED build / measurement campaign:

* :func:`save_opc_calibration` writes the calibration to disk.
* :func:`load_opc_calibration` reads it back into an :class:`OPCCalibration`.

The on-disk schema is versioned (``schema = "cflibs.opc.v1"``) so a future
field change can be detected rather than silently mis-read.
"""

from __future__ import annotations

import json
from pathlib import Path

from cflibs.inversion.physics.opc import OPCCalibration

__all__ = ["save_opc_calibration", "load_opc_calibration", "OPC_SCHEMA"]

#: On-disk schema tag for an OPC calibration JSON file.
OPC_SCHEMA = "cflibs.opc.v1"


def save_opc_calibration(calibration: OPCCalibration, path: "str | Path") -> Path:
    """Serialize an :class:`OPCCalibration` to a JSON file.

    Parameters
    ----------
    calibration : OPCCalibration
        The known-matrix calibration to persist.
    path : str or Path
        Destination JSON path (parent directories are created if needed).

    Returns
    -------
    Path
        The path written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": OPC_SCHEMA,
        "robust_T_K": float(calibration.robust_T_K),
        "F": {str(el): float(f) for el, f in calibration.F.items()},
        "selected_standards": [str(s) for s in calibration.selected_standards],
        "conditioning_rule": str(calibration.conditioning_rule),
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def load_opc_calibration(path: "str | Path") -> OPCCalibration:
    """Load an :class:`OPCCalibration` from a JSON file written by
    :func:`save_opc_calibration`.

    Parameters
    ----------
    path : str or Path
        Path to an OPC calibration JSON file.

    Returns
    -------
    OPCCalibration
        The deserialized calibration.

    Raises
    ------
    ValueError
        If the file is missing required keys or carries an unknown schema tag.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"OPC calibration file {path!s} is not a JSON object.")
    schema = data.get("schema")
    if schema != OPC_SCHEMA:
        raise ValueError(
            f"OPC calibration file {path!s} has schema {schema!r}; expected {OPC_SCHEMA!r}."
        )
    try:
        robust_T_K = float(data["robust_T_K"])
        F = {str(el): float(f) for el, f in dict(data["F"]).items()}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"OPC calibration file {path!s} is malformed: {exc}") from exc
    selected = [str(s) for s in data.get("selected_standards", [])]
    rule = str(data.get("conditioning_rule", ""))
    return OPCCalibration(
        robust_T_K=robust_T_K,
        F=F,
        selected_standards=selected,
        conditioning_rule=rule,
    )
