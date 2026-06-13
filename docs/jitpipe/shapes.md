# jitpipe shape-bucketing + padding constants (ADR-0004 §5.2)

The jittable pipeline keeps the jit cache small by padding every per-spectrum
array up to a fixed **shape bucket**. The bucket id (plus the species/level pad
and broadening mode) is the only thing that keys the jit cache; all continuous
knobs flow as traced `PipelineParams` leaves and never recompile.

## Superset snapshot (host-built, `ASD_da/libs_production.db`)

Measured against the current production DB (see `tests/jitpipe/test_snapshot.py`):

| Block | Shape | dtype | Notes |
|---|---|---|---|
| lines | (28727,) per column | mixed (see below) | sorted by (element, sp_num, wavelength, id) |
| energy_levels | (175, 676) | f64 g/E + bool mask | uniform pad; Fe II has the 676 max |
| partition polys (canonical) | (175, 5) | f64 | `partition_spec_for` re-fit; concrete, no NaN |
| partition polys (stored) | (175, 5) | f64 | raw table; NaN sentinel when absent |
| canonical scalar fallbacks | (175, 2) | f64 | eager `[U_I, U_II]` |
| species_physics | (175, 2) | f64 | `[ip_ev, atomic_mass]` |
| doublet pairs | (P, 2) | i32 | shared-upper-level line pairs (+ rho, r_thin) |
| oxide stoichiometry | (175,) | f64 | O atoms per cation |

Per-line column dtypes: `line_element_index` i16, `line_sp_num` i8,
`line_species_index` i32, `line_is_resonance` bool, `line_stark_source_class`
u8, all physical quantities f64.

### Stark source class codes (`line_stark_source_class`)

| code | meaning |
|---|---|
| 0 | null / unknown |
| 1 | konjevic-λ²-scaled |
| 2 | interpolated |
| 3 | hydrogenic |
| 4 | `stark_b` |

## Line-count buckets (`StaticConfig.bucket_id`)

A spectrum's candidate line set is padded up to the smallest bucket that fits
it (`cflibs.jitpipe.host.bucket_for_n_lines`):

```
64, 128, 256, 512, 1024, 2048, 4096   (then next power of two)
```

## Per-spectrum host↔device seam

The ONLY per-spectrum host↔device hop is candidate-set assembly: element masks
over the superset snapshot + the gather that produces the per-bucket padded
arrays. The bridge methods `PipelineSnapshot.to_lax_snapshot` /
`to_atomic_snapshot` implement this gather against the two legacy consumers
(see `contracts.md` tier table).
