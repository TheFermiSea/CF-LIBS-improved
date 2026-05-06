# CF-LIBS Documentation

This directory holds the user-facing scientific documentation, the
physics reference, the codebase reference, and the developer/AI tooling
notes for CF-LIBS.

`CLAUDE.md` and `AGENTS.md` at the repository root are AI-agent operating
manuals. They are **not** part of the user or scientific documentation
and they are **not** the canonical source for physics or equations. The
physics lives under [`physics/`](physics/).

## Layout

```
docs/
├── user/         User-facing scientific documentation
│                  (start here if you have a LIBS spectrum)
├── physics/      Equations, assumptions, inversion algorithm
├── reference/    API and codebase architecture
├── development/  AI/dev tooling: evolution framework, deployment, etc.
├── design/       Internal design documents
├── literature/   Literature reviews
├── planning/     Roadmaps, PRDs
├── reports/      Benchmark reports
├── research/     Research notes
├── validation/   Validation specs and reports
├── changelog/    Per-change notes
├── archive/      Superseded historical material
├── index.rst     Sphinx entry point
└── conf.py       Sphinx configuration
```

## Where to start

| You are … | Start here |
|-----------|-----------|
| A scientist with a measured LIBS spectrum | [user/Quick_Start_Real_Data.md](user/Quick_Start_Real_Data.md) |
| Identifying elements and matching peaks before inversion | [user/Peak_Identification_Guide.md](user/Peak_Identification_Guide.md) |
| Generating synthetic spectra for experimental design | [user/Quick_Start_Synthetic.md](user/Quick_Start_Synthetic.md) |
| Reading deeper user-level configuration and Python API | [user/User_Guide.md](user/User_Guide.md) |
| Looking for the equations the code evaluates | [physics/Equations.md](physics/Equations.md) |
| Asking whether your plasma satisfies CF-LIBS assumptions | [physics/Assumptions_And_Validity.md](physics/Assumptions_And_Validity.md) |
| Tracing the iterative solver step by step | [physics/Inversion_Algorithm.md](physics/Inversion_Algorithm.md) |
| Looking up an API call signature | [reference/API_Reference.md](reference/API_Reference.md) |
| Mapping the source tree to the physics | [reference/Codebase_Architecture.md](reference/Codebase_Architecture.md) |
| Building or extending the atomic database | [reference/Database_Generation.md](reference/Database_Generation.md) |
| Working on the LLM-driven evolution framework | [development/Evolution_Framework.md](development/Evolution_Framework.md) |
| Deploying CF-LIBS on a cluster | [development/Deployment.md](development/Deployment.md) |

## Building the Sphinx site

The HTML site is generated from the same Markdown files plus
`index.rst`:

```bash
pip install sphinx myst-parser sphinx_rtd_theme
sphinx-build docs docs/_build/html
```
