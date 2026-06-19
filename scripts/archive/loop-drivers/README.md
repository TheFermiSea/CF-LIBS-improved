# Archived host-specific autonomous-loop drivers

These shell drivers ran a one-time autonomous 24-hour benchmark loop on a
specific cluster host (vasp-03). They are operationally coupled to that host
and to a dated `output/loop-<DATE>/` layout, and are not part of any current
automated workflow. Retained here only as provenance for how that run was
orchestrated; nothing in the repo invokes them.

| Script | Role |
|--------|------|
| `loop_24h_driver.sh` | Outer 24h loop driver. |
| `loop_iteration.sh` | Single-iteration body invoked by the 24h driver. |
| `run_cell.sh` | Per-cell runner used inside an iteration. |

Moved from `scripts/` in the 2026-06 repo-cleanliness sweep
(docs/audit/2026-06-09-overhaul/05-repo-cleanliness.md).
