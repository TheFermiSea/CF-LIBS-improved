# Development

Internal documentation for the CF-LIBS project: deployment, the
LLM-driven evolution framework, and other dev-process notes. **None of
this is required reading for using CF-LIBS to analyze spectra** — for
that, see [../user/](../user/README.md).

| Document | Use it for |
|----------|-----------|
| [Evolution_Framework.md](Evolution_Framework.md) | LLM-driven hierarchical-ES algorithm optimization, physics-only blocklist scanner, ruff TID251 enforcement. Tooling only — not part of the shipped algorithm. |
| [CODEEVOLVE_WAVE2_PLAN.md](CODEEVOLVE_WAVE2_PLAN.md) | Wave-2 development plan for the evolution framework. |
| [Deployment.md](Deployment.md) | Local and cluster deployment (Apple Silicon, NVIDIA CUDA + MPI). |
| [REFERENCE_ANALYSIS_LIBSSA.md](REFERENCE_ANALYSIS_LIBSSA.md) | Comparison notes against the LIBSSA reference implementation. |

The repository-root files [`CLAUDE.md`](../../CLAUDE.md) and
[`AGENTS.md`](../../AGENTS.md) describe how Claude Code and other AI
agents operate in this repo. They are not user documentation and are
not the canonical source for physics — that is
[../physics/](../physics/README.md).
