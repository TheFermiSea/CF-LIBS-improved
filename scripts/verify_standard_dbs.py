#!/usr/bin/env python3
"""Verify the shipped standard atomic databases against a checksum manifest.

HARD IMMUTABILITY MANDATE: NIST ASD, Kurucz, Stark-B and every pre-compiled
standard DB shipped in this repo are **immutable**. A past ASD corruption cost
months of rework, so any silent byte-change to a standard DB must be caught
loudly. This script recomputes the ``sha256`` + size of every DB recorded in
``ASD_da/STANDARD_DB_MANIFEST.json`` and diffs them against the manifest.

Derived/rebuildable data (``ASD_da/overlays/``) is intentionally **excluded** --
enrichments live in NEW overlay DBs, never in the standard sources.

Usage:
    # verify (CI / pre-commit): exits non-zero on ANY mismatch
    PYTHONPATH=. python3 scripts/verify_standard_dbs.py

    # regenerate the manifest after a *legitimate* standard-DB refresh
    PYTHONPATH=. python3 scripts/verify_standard_dbs.py --update

Physics-only: hashlib + json + stdlib (no ML libraries, no cflibs imports).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

# Repo root = parent of scripts/. Resolve absolutely so the worktree trap
# (scripts/ on sys.path[0]) cannot point us at another checkout.
REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "ASD_da" / "STANDARD_DB_MANIFEST.json"

#: Standard (immutable) DBs to track, keyed by repo-relative POSIX path, with a
#: human description. To add a newly shipped standard DB, add it here and run
#: ``--update``. NEVER add anything under ``ASD_da/overlays/`` (derived data).
STANDARD_DBS: dict[str, str] = {
    "ASD_da/libs_production.db": (
        "Primary CF-LIBS atomic database (SQLite): lines, energy_levels, "
        "species_physics, partition_functions. Compiled from the gold-standard "
        "NIST ASD dump. IMMUTABLE -- enrichments belong in ASD_da/overlays/."
    ),
}

_CHUNK = 1 << 20  # 1 MiB


def _sha256_and_size(path: Path) -> tuple[str, int]:
    """Return (hex sha256, size in bytes) of ``path`` reading raw bytes only.

    Reads bytes directly (never opens the file via sqlite), so no WAL frame or
    shared-memory index is ever touched and the checksum stays reproducible.
    """
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        while True:
            block = fh.read(_CHUNK)
            if not block:
                break
            size += len(block)
            h.update(block)
    return h.hexdigest(), size


def build_manifest() -> dict:
    """Recompute a fresh manifest for every tracked standard DB (must exist)."""
    databases: dict[str, dict] = {}
    for rel, description in sorted(STANDARD_DBS.items()):
        path = REPO_ROOT / rel
        if not path.exists():
            raise FileNotFoundError(f"Standard DB not found (cannot build manifest): {path}")
        sha256, size = _sha256_and_size(path)
        databases[rel] = {
            "sha256": sha256,
            "size_bytes": size,
            "date": date.today().isoformat(),
            "description": description,
        }
    return {
        "_comment": (
            "Checksum manifest for IMMUTABLE standard atomic databases. Verified by "
            "scripts/verify_standard_dbs.py. Overlays (ASD_da/overlays/) are derived "
            "and intentionally excluded. Regenerate with --update only after a "
            "legitimate, reviewed refresh of a standard DB."
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "databases": databases,
    }


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}. Run with --update to create it."
        )
    return json.loads(MANIFEST_PATH.read_text())


def verify() -> list[str]:
    """Recompute + diff every manifest entry. Returns a list of mismatch strings.

    An empty list means every standard DB is byte-identical to the manifest.
    """
    manifest = load_manifest()
    recorded = manifest.get("databases", {})
    problems: list[str] = []

    # 1. Every tracked standard DB must be present in the manifest.
    for rel in STANDARD_DBS:
        if rel not in recorded:
            problems.append(f"{rel}: tracked standard DB missing from manifest (run --update)")

    # 2. Every manifest entry must match the on-disk file.
    for rel, entry in recorded.items():
        path = REPO_ROOT / rel
        if not path.exists():
            problems.append(f"{rel}: file missing on disk ({path})")
            continue
        sha256, size = _sha256_and_size(path)
        if size != entry.get("size_bytes"):
            problems.append(
                f"{rel}: size mismatch (manifest={entry.get('size_bytes')} disk={size})"
            )
        if sha256 != entry.get("sha256"):
            problems.append(
                f"{rel}: sha256 MISMATCH -- standard DB was modified!\n"
                f"    manifest={entry.get('sha256')}\n    disk    ={sha256}"
            )
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--update",
        action="store_true",
        help="Regenerate the manifest from current on-disk DBs (legitimate refresh only).",
    )
    args = ap.parse_args(argv)

    if args.update:
        manifest = build_manifest()
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Wrote manifest: {MANIFEST_PATH}")
        for rel, entry in manifest["databases"].items():
            print(f"  {rel}  sha256={entry['sha256'][:16]}...  {entry['size_bytes']} bytes")
        return 0

    problems = verify()
    if problems:
        print("STANDARD DB VERIFICATION FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print(
            "\nStandard databases are IMMUTABLE. If this change is legitimate, review it "
            "and regenerate the manifest with:\n"
            "  PYTHONPATH=. python3 scripts/verify_standard_dbs.py --update",
            file=sys.stderr,
        )
        return 1

    print("OK: all standard databases match STANDARD_DB_MANIFEST.json")
    for rel in load_manifest().get("databases", {}):
        print(f"  verified {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
