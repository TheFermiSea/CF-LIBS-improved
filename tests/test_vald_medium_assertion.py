"""VALD ingest must detect/assert the wavelength medium (air vs vacuum) [audit air-vacuum-vald].

VALD3 delivers air OR vacuum depending on extraction units; the ingest assumes air
for λ>=200nm. A vacuum-extracted file silently shifts visible lines ~+114 pm, so the
ingest now detects the WL_air/WL_vac header and fails loudly on vacuum/unknown.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from ingest_vald_atomic import _detect_vald_medium, parse_vald  # noqa: E402


def _write(tmp_path, name, header):
    p = tmp_path / name
    p.write_text(f"# VALD extract\nElm Ion {header} loggf E_low(eV)\n'Fe 1', 4045.8, -0.5, 1.4\n")
    return p


def test_detect_air(tmp_path):
    assert _detect_vald_medium(_write(tmp_path, "air.txt", "WL_air(A)")) == "air"


def test_detect_vacuum(tmp_path):
    assert _detect_vald_medium(_write(tmp_path, "vac.txt", "WL_vac(A)")) == "vacuum"


def test_missing_header_raises(tmp_path):
    p = tmp_path / "nohdr.txt"
    p.write_text("'Fe 1', 4045.8, -0.5, 1.4\n")
    with pytest.raises(ValueError, match="determine wavelength medium"):
        _detect_vald_medium(p)


def test_parse_vald_rejects_vacuum_file(tmp_path):
    p = _write(tmp_path, "vac.txt", "WL_vac(A)")
    with pytest.raises(ValueError, match="VACUUM units"):
        list(parse_vald(p, 200.0, 900.0))
