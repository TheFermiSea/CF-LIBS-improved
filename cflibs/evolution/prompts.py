"""Physics-grounding string primitives for LLM-driven evolution prompts.

Shared building blocks used by the perturbation generator, the council
synthesiser, and the structural-mutation proposer. Each downstream task
composes its own full prompt out of these constants plus task-specific
framing — the canonical forbidden list lives exactly once on
``cflibs.evolution.evaluator.FORBIDDEN_PREFIXES``.
"""

from __future__ import annotations

from cflibs.evolution.evaluator import FORBIDDEN_PREFIXES

# Modules / primitives the evolved algorithm may reference freely.
ALLOWED_PRIMITIVES: tuple[str, ...] = (
    "numpy",
    "scipy (optimize.nnls, optimize.minimize, linalg, special)",
    "jax and jax.numpy (for autodiff / jit of physics expressions)",
    "math, dataclasses, typing, functools",
    "cflibs.plasma (Saha-Boltzmann, partition functions)",
    "cflibs.radiation (Voigt / Gaussian / Lorentzian profiles)",
    "cflibs.atomic (NIST line database lookups)",
    "cflibs.inversion.common (LineObservation, BoltzmannFitResult)",
)


PHYSICS_GROUNDING_PREAMBLE: str = """\
You are proposing a change to a physics-only CF-LIBS algorithm.

HARD CONSTRAINT — NON-NEGOTIABLE:
The shipped algorithm must be physics-based only. You may NOT introduce
neural networks, trained models, PLS/PCR, learned embeddings, learned
features, or anything that requires a training phase on labelled spectra.

FORBIDDEN modules (any import, attribute access, or dynamic import of
these will be rejected at AST scan, fitness = -inf):
{forbidden_lines}

ALLOWED primitives for the evolved algorithm:
{allowed_lines}

ML is permitted in the optimization process that produces your proposal
(you, the search loop, the council). It is NEVER permitted in the
candidate code you emit.

Candidates that violate this constraint are dropped before physics
evaluation — you do not get another attempt on a flagged candidate.
"""


def render_preamble() -> str:
    """Return the physics-grounding preamble with the current allow/deny lists substituted."""
    forbidden_lines = "\n".join(f"  - {name}" for name in FORBIDDEN_PREFIXES)
    allowed_lines = "\n".join(f"  - {name}" for name in ALLOWED_PRIMITIVES)
    return PHYSICS_GROUNDING_PREAMBLE.format(
        forbidden_lines=forbidden_lines,
        allowed_lines=allowed_lines,
    )
