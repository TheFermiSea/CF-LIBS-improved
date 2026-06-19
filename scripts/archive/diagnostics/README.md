# Archived one-off diagnostic / investigation scripts

These are per-investigation diagnostic drivers that documented a specific,
now-closed investigation. They are retained here only as provenance for how a
finding was reached; none are part of any current automated workflow and
nothing in `cflibs/`, `tests/`, or `scripts/` imports them.

| Script | Investigation (closed) |
|--------|------------------------|
| `diag_segmented_calib_flip.py` | J9 R8 segmented-calibration model-flip — **fixed** (per-segment re-detect + unconditional dense-hull coverage tiebreak; now on-device, parity-confirmed). |
| `diag_seg2_bic_margin.py` | J9/segment-2 BIC-margin probe for the same R8 flip. |
| `diag_jit_scoreboard_parity.py` | jit-vs-reference scoreboard parity diagnostic. |
| `run_j12_board_compare.py` | J12/M3 board comparison driver (jit vs reference board). |
| `analyze_window_separability.py` | Fe-separability root-cause analysis tied to a specific corpus/audit. |
| `validate_kramida_2024.py` | One-shot PR #71 follow-up DB delta (Kramida 2024 levels). |

Moved from `scripts/` in the 2026-06 repo-cleanliness sweep
(docs/audit/2026-06-09-overhaul/05-repo-cleanliness.md).
