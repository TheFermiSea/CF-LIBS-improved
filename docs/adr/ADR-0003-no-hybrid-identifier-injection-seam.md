# ADR-0003: Do not inject voters into `HybridIdentifier`

**Status:** accepted (2026-05-27)

## Context

A 2026-05-27 architecture review proposed changing `HybridIdentifier`
(`cflibs/inversion/identify/hybrid.py`) to accept its two stages — an
NNLS screener and an ALIAS confirmer — as constructor parameters
instead of instantiating them inline inside `.identify()`. The pitch
was that an injection seam would (a) make `combine()` testable without
a real `AtomicDatabase` / `BasisLibrary`, and (b) prepare the ground for
alternative hybrid variants (e.g. "Comb + NNLS").

## Decision

**Do not introduce the injection seam.** Keep `HybridIdentifier`
instantiating its NNLS screener and ALIAS confirmer inline.

## Why

The friction signal — "you'd have to fake the atomic DB to reach
`combine()`" — turned out to describe a test that doesn't exist. Audit
on 2026-05-27:

| Question | Answer |
|---|---|
| Tests that import `HybridIdentifier` | 1 (`test_hybrid_consensus_2of3.py`) |
| Tests that **instantiate** `HybridIdentifier` | **0** |
| Tests that exercise `HybridIdentifier.identify()` | **0** |
| Tests that exercise `HybridIdentifier._combine()` | **0** |

The single import is a signature-pinning assertion that explicitly
chooses *not* to instantiate the class because of fixture cost. There
are zero tests of the combining logic — that is a test-coverage gap,
not a constraint that an injection seam would relieve.

An interface seam is justified when a real second adapter exists or
when faking the dependencies unlocks existing painful tests. Here,
neither condition holds:

- One adapter today (production: NNLS + ALIAS). No "Hybrid with Comb +
  NNLS" variant exists or is planned. **One adapter = hypothetical
  seam.**
- No tests of `combine()` exist that the seam would simplify. If a
  maintainer wants to test `combine()`, they can either accept the
  `AtomicDatabase` fixture cost (same cost as the seven other
  identifier tests in the suite) or write the fake themselves at the
  test site — neither requires changing `HybridIdentifier`'s
  constructor.

The deletion test on the seam itself: if the proposed
`(stage1, stage2)` constructor parameters were already there and we
deleted them in favour of inline construction, complexity would not
reappear at the call sites (production has two call sites, both pass
the same identifiers). The seam concentrates nothing.

## Consequences

- `HybridIdentifier`'s constructor surface stays as-is (NNLS + ALIAS
  parameters, no stage-object injection).
- If a second hybrid variant ever materialises (Comb + NNLS, ALIAS +
  Comb, etc.), revisit this ADR — two adapters would justify the seam
  on the leverage side.
- If `combine()` coverage becomes important enough to write tests for,
  the appropriate fix is to add those tests with real fixtures, not to
  invert the constructor surface.

This decision should not be re-surfaced by future architecture reviews
unless one of the two conditions above changes.
