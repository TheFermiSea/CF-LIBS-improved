#!/usr/bin/env python3
"""CI gate for the CF-LIBS ast-grep invariant rule pack.

Runs ``ast-grep scan --json`` over the repo and enforces a *ratchet*:

* **error**-severity rules are the GATE. They are clean on the current tree
  (see ``rules/README.md``), so this script exits 0 today. If a future commit
  introduces a match for any error-severity rule (e.g. an ``import torch`` or a
  ``jax.clear_caches()`` in tests), the gate exits non-zero.
* **warning / info / hint**-severity rules are ADVISORY. They have legitimate
  pre-existing debt (silent-fallback returns, module-level ``setdefault`` in
  tests, ``print`` in benchmark harnesses, ...). Their counts are printed but
  never fail the build, so CI does not break on day one.

Promote an advisory rule to a gate by (1) cleaning its current hits and (2)
flipping its ``severity`` to ``error`` in ``rules/<id>.yml``.

Usage::

    python scripts/ast_grep_gate.py            # gate: exit 1 on any error hit
    python scripts/ast_grep_gate.py --advisory # never exit non-zero (report only)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GATING_SEVERITIES = {"error"}


def _run_scan() -> list[dict]:
    binary = shutil.which("ast-grep") or shutil.which("sg")
    if binary is None:
        print(
            "SKIP: ast-grep not installed; cannot run the invariant gate.\n"
            "      Install with `npm i -g @ast-grep/cli` or from the GitHub "
            "release, then re-run.",
            file=sys.stderr,
        )
        # Absent tool -> cannot gate. Do not hard-fail so a dev without the
        # binary is not blocked; CI installs the binary so it gates there.
        raise SystemExit(0)

    proc = subprocess.run(
        [binary, "scan", "--json=stream"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode not in (0, 1):  # ast-grep uses 1 when findings exist
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"ast-grep scan failed (exit {proc.returncode})")

    findings: list[dict] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line:
            findings.append(json.loads(line))
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--advisory",
        action="store_true",
        help="report only; never exit non-zero (for a soft CI step)",
    )
    args = ap.parse_args()

    findings = _run_scan()

    by_rule: dict[str, list[dict]] = defaultdict(list)
    severity_of: dict[str, str] = {}
    for m in findings:
        rid = m["ruleId"]
        by_rule[rid].append(m)
        severity_of[rid] = m.get("severity", "warning")

    gating_hits: list[tuple[str, dict]] = []
    print("CF-LIBS ast-grep invariant gate")
    print("=" * 60)
    if not findings:
        print("No findings.")
    for rid in sorted(by_rule):
        sev = severity_of[rid]
        hits = by_rule[rid]
        tag = "GATE " if sev in GATING_SEVERITIES else "adv. "
        print(f"[{tag}] {sev:7} {rid}: {len(hits)} hit(s)")
        if sev in GATING_SEVERITIES:
            for h in hits:
                loc = f"{h['file']}:{h['range']['start']['line'] + 1}"
                print(f"          -> {loc}")
                gating_hits.append((rid, h))
    print("=" * 60)

    if gating_hits and not args.advisory:
        print(
            f"FAIL: {len(gating_hits)} error-severity (gating) finding(s). "
            "These invariants must stay clean; see rules/README.md.",
            file=sys.stderr,
        )
        return 1

    if gating_hits:
        print(f"(advisory mode) {len(gating_hits)} gating finding(s) ignored.")
    else:
        print("OK: all gating (error-severity) invariants are clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
