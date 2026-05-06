# Physics Reference

Detailed equations, algorithms, and physical assumptions of the CF-LIBS
forward model and iterative inversion solver.

This directory replaces previous pointers to `CLAUDE.md` for physics
questions. `CLAUDE.md` is an AI-agent operating manual and is not part
of the user/scientific documentation.

## Documents

- **[Equations](Equations.md)** — every equation CF-LIBS evaluates:
  Saha–Boltzmann, line emissivity, Voigt profile, Stark broadening,
  Boltzmann plot, closure modes, self-absorption, McWhirter criterion,
  Bayesian likelihood. Symbol table with units. Citations.

- **[Assumptions and Validity](Assumptions_And_Validity.md)** — every
  assumption baked into the algorithm (LTE, optical thinness, single-zone
  uniformity, …) with regime of validity, failure mode, diagnostic, and
  options when each fails. Includes an audit checklist.

- **[Inversion Algorithm](Inversion_Algorithm.md)** — step-by-step
  walkthrough of the iterative CF-LIBS solver including pseudocode,
  rationale for common-slope fitting and iteration, code-location map,
  and uncertainty propagation paths.

## When to Read Which

| You want to … | Read |
|--------------|------|
| Understand what a recovered `T` means | [Equations](Equations.md) |
| Verify your spectrum is in the LTE regime | [Assumptions and Validity](Assumptions_And_Validity.md) |
| Decide between `standard` / `matrix` / `oxide` closure | [Equations](Equations.md) |
| Check if self-absorption is biasing your fit | [Assumptions and Validity](Assumptions_And_Validity.md) |
| Trace the iterative solver step-by-step | [Inversion Algorithm](Inversion_Algorithm.md) |
| Cite the underlying physics in a paper | [Equations](Equations.md) |
