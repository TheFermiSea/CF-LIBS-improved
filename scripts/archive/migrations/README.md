# Archived one-time DB migration scripts

These scripts were one-shot builders/patchers for the committed atomic
database (`ASD_da/libs_production.db`). They have **already been applied** —
their effects are baked into the committed DB — and are retained here only as
provenance for how specific tables/columns were produced (Stark widths,
partition-function coefficients, species physics, missing
elements/ions/levels, broadening columns, ...).

Do not re-run them against the production DB unless you are deliberately
rebuilding it; several are not idempotent. For a from-scratch rebuild, the
canonical entry point is `datagen_v2.py` at the repo root.

Moved from `scripts/` in the 2026-06 repo-cleanliness sweep
(docs/audit/2026-06-09-overhaul/05-repo-cleanliness.md §3.4).
