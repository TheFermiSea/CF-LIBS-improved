"""Physics-grounding string primitives for LLM-driven evolution prompts.

Shared building blocks used by the perturbation generator, the council
synthesiser, and the structural-mutation proposer. Each downstream task
(``jrw``, ``1i1s``, ``ra0b``, ``hdat``) composes its own full prompt
out of these constants plus task-specific framing — keeping the canonical
allow/deny lists in exactly one place so they cannot drift.

The list contents are kept in lockstep with the evaluator's
:data:`cflibs.evolution.evaluator.FORBIDDEN_PREFIXES` tuple; the module
exposes :func:`ensure_consistency` for a CI-time check.
"""

from __future__ import annotations

from cflibs.evolution.evaluator import FORBIDDEN_PREFIXES

# Modules / primitives the evolved algorithm may reference freely. Listed
# in a form that reads naturally when embedded in a prompt.
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

# Modules / namespaces the evolved algorithm must never reference. Kept
# in sync with FORBIDDEN_PREFIXES so the prompt and the scanner agree
# byte-for-byte.
FORBIDDEN_LIBRARIES: tuple[str, ...] = FORBIDDEN_PREFIXES

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
    forbidden_lines = "\n".join(f"  - {name}" for name in FORBIDDEN_LIBRARIES)
    allowed_lines = "\n".join(f"  - {name}" for name in ALLOWED_PRIMITIVES)
    return PHYSICS_GROUNDING_PREAMBLE.format(
        forbidden_lines=forbidden_lines,
        allowed_lines=allowed_lines,
    )


def ensure_consistency() -> None:
    """Raise if the prompt's forbidden list drifts from the scanner's blocklist.

    Intended for CI / pytest: this catches the case where someone updates
    the scanner but forgets to update the prompt (or vice versa).
    """
    if tuple(FORBIDDEN_LIBRARIES) != tuple(FORBIDDEN_PREFIXES):
        raise RuntimeError(
            "cflibs.evolution.prompts.FORBIDDEN_LIBRARIES has drifted from "
            "cflibs.evolution.evaluator.FORBIDDEN_PREFIXES. Update both in "
            "the same commit so the LLM prompt and the scanner agree."
        )
