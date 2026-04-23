"""CF-LIBS evolution driver (optimization-process tooling only).

Modules here support the LLM-driven algorithm search (perturbation generation,
fitness evaluation, blocklist enforcement). They are part of the OPTIMIZATION
PROCESS, not the shipped CF-LIBS algorithm.

HARD CONSTRAINT (see beads CF-LIBS-improved-3fy3): modules here must not
import any ML / neural-network library. The shipped algorithm is physics-only.
The blocklist scanner in :mod:`cflibs.evolution.evaluator` enforces this on
every candidate produced by the evolution loop.
"""
