# cflibs/atomic — Adversarial Verification Report

Worktree: `/home/brian/code/CF-LIBS-improved/.worktrees/v4-m5`
Verifier: Adversarial re-read (ripgrep + Read)
Date: 2026-06-25

---

## Verdict Summary

| Finding | REAL | Confirmed Severity | Notes |
|---------|------|--------------------|-------|
| F1 — stark_w docstring 10^16 | **TRUE** | high | Confirmed: structures.py:61 says 10^16; stark.py REF_NE = 1e17 |
| F2 — log basis ambiguity | **TRUE** | medium | Confirmed: structures.py:127 uses `log` not `ln` |
| F3 — is_resonance threshold 0.01 eV | **TRUE** | medium (not high) | Confirmed present; impact partially mitigated by runtime fallback at 0.1 eV |
| F4 — iterrows in get_energy_levels/get_available_elements | **TRUE** | medium | Confirmed: database.py:518,521,842 |
| F5 — LRUCache None not cached | **TRUE** | medium | Confirmed: cache.py:149 `if cached_value is not None` |
| F6 — migration on every __init__ | **TRUE** | medium | Confirmed: database.py:68, no version stamp |
| F7 — pickle self in cache key | **TRUE** | low | Confirmed: cache.py:45; __getstate__ strips pool/conn but db_path still pickled every call |
| F8 — PartitionFunction missing g0 | **TRUE** | low | Confirmed: multiple direct callers of get_partition_coefficients; saha_boltzmann.py works around it but iterative.py/streaming.py do not |
| F9 — snapshot docstring wrong | **TRUE** | low | Confirmed: docstring says "non-NULL rel_int"; actual code includes NULL-rel_int lines |
| F10 — test gaps | **TRUE** | medium | Confirmed: no is_resonance backfill test, no stark_w convention parity test |

**Highest confirmed severity: HIGH (F1)**

---

## Per-Finding Verification

### F1 — `Transition.stark_w` docstring states 10^16 (CONFIRMED TRUE, HIGH)

`structures.py` line 61 reads:
```
Stark width at reference density (10^16 cm^-3)
```
`cflibs/radiation/stark.py` declares `REF_NE = 1.0e17` on line 33, with an extensive block comment
(lines 17–40) explicitly documenting that this is the project-wide single source of truth:
"The atomic database stores `lines.stark_w` as the electron-impact FWHM at n_e = 1e17 cm^-3,
T = 10000 K." The database.py `get_stark_parameters_with_source` docstring at line 803 also says
"n_e = 1e17 cm^-3". The discrepancy is unambiguous: the struct docstring is wrong by 10×. Any
new contributor implementing their own Stark scaling from the struct docstring would produce Stark
widths 10× too large, since `w_fwhm = stark_w_ref * (n_e / REF_NE)` is linear in the reference
density. The radiation module is internally consistent and correct; the damage is confined to
documentation misleading external readers of the struct. Severity confirmed HIGH.

### F2 — `PartitionFunction` log basis ambiguity (CONFIRMED TRUE, MEDIUM)

`structures.py` lines 126–128 use the ambiguous formula `log(U) = Σ a_n (log T)^n`. Confirmed
via `plasma/partition.py`: `PartitionFunctionSpec` explicitly states the natural-log basis, and
the `partition.py` pipeline uses `np.log` (natural log) throughout, consistent with the CLAUDE.md
`partition function polynomial: log U = Σ aₙ(log T)ⁿ` notation (which also uses the ambiguous
`log`). The `saha-boltzmann-lte.md` literature doc (P6) warns explicitly: "Historical confusion
between log_10 and natural log polynomial fits. cflibs partition.py uses natural log." The
structures.py docstring should say `ln U(T) = Σ a_n (ln T)^n` to eliminate the ambiguity.
Severity confirmed MEDIUM (documentation-only; the computation is correct).

### F3 — `is_resonance` backfill threshold 0.01 eV (CONFIRMED TRUE, downgraded to MEDIUM)

`database.py` lines 143–144 confirm the SQL:
```sql
UPDATE lines SET is_resonance = 1 WHERE ei_ev < 0.01
UPDATE lines SET is_resonance = 0 WHERE ei_ev >= 0.01 OR ei_ev IS NULL
```
This is physically imprecise: a resonance line should have E_i = 0 exactly (NIST ground-state
convention). The 0.01 eV threshold will incorrectly flag fine-structure ground-state multiplet
members and other very low-lying metastable levels of elements like Fe I (where several levels
exist below 0.01 eV). The impact on the Stark n_e diagnostic is that flagged lines are
*down-ranked* (not hard-excluded) by a factor of `RESONANCE_DOWNRANK` (stark.py:1031), which
is a `0.5×` penalty. However, one mitigating factor not noted in the census: the runtime fallback
in `cflibs/inversion/identify/line_detection.py:456` applies `E_i_ev < ground_state_threshold_ev`
where the default threshold is **0.1 eV** — *wider* than the DB backfill threshold. So lines
with 0.01 eV < E_i < 0.1 eV are correctly NOT flagged in the DB but ARE caught by the fallback
when the DB field is None. This partially compensates for the 0.01 eV threshold being too tight.
The real harm is that lines with 0 < E_i < 0.01 eV (true metastables) ARE falsely marked
`is_resonance=1` in the DB, causing them to be down-ranked when they may be valid Stark
diagnostics. Severity confirmed MEDIUM (not high — the down-rank is 0.5×, not an exclusion;
and the fallback uses a stricter resonance criterion only when the DB value is None/missing).

### F4 — iterrows in `get_energy_levels` and `get_available_elements` (CONFIRMED TRUE, MEDIUM)

`database.py:518,521` confirm `pd.read_sql_query` + `df.iterrows()` in `get_energy_levels`.
`database.py:842` confirms `pd.read_sql_query` (no iterrows, but still a pandas round-trip) in
`get_available_elements`. Line 413 contains the comment documenting the anti-pattern: "~100x
faster than pd.Series scalar access". The same anti-pattern remains in both methods. For
`get_energy_levels`, this affects every `snapshot(include_levels=True)` call (2× per element
for stage I and II) and the cold-cache partition function path. Severity confirmed MEDIUM.

### F5 — `LRUCache` silently fails to cache `None` returns (CONFIRMED TRUE, MEDIUM)

`core/cache.py:149` reads:
```python
if cached_value is not None:
    return cached_value
```
`cache.get` returns `None` for both a cache miss (line 64) and a stored `None` value. The
decorator at line 148 cannot distinguish. `get_ionization_potential` (decorated with
`@cached_ionization`) returns `None` for missing species (database.py:559). Every call for
a stage-III ion (commonly missing) re-queries the DB. For a 10-element inversion with 10 missing
stage-III entries, this means 10 repeated DB round-trips on every call. Severity confirmed MEDIUM.

### F6 — Migration runs on every `__init__` (CONFIRMED TRUE, MEDIUM)

`database.py:68` unconditionally calls `self._check_and_migrate_schema()`. The migration method
`_perform_migration` at lines 89–101 calls 7 sub-methods, each of which runs at minimum one
`PRAGMA table_info`, `COUNT(*)`, or `sqlite_master` query regardless of whether the schema has
already been migrated. There is no `cflibs_meta` version stamp or any fast-path. For cluster
workers constructing a fresh `AtomicDatabase` per task, this is 7+ SQLite round-trips of pure
overhead at startup. Severity confirmed MEDIUM.

### F7 — Cache key pickles the full `AtomicDatabase` instance (CONFIRMED TRUE, LOW)

`core/cache.py:45–46`:
```python
key_data = pickle.dumps((args, sorted(kwargs.items())))
return hashlib.md5(key_data).hexdigest()
```
When called as a method decorator, `args = (self, element, stage, ...)`, so the entire
`AtomicDatabase` instance is passed to `pickle.dumps`. `__getstate__` at database.py:1336
removes `_pool` and `conn`, but the remaining state (`db_path`, `_use_pool`) is still serialized
on every cache call, including the hot `get_transitions` path. For the common case where all
workers share the same DB path, this produces identical keys on every call — the key is correct
but the key-generation cost (pickle + md5) includes the DB instance overhead on every potential
cache miss check. Severity confirmed LOW (correct behavior but wasteful).

### F8 — `PartitionFunction` dataclass missing `g0` (CONFIRMED TRUE, LOW)

`structures.py:123–150` confirms the `PartitionFunction` dataclass has no `g0` field.
`plasma/partition.py:358` confirms `PartitionFunctionSpec` carries `g0: float`. Direct callers
of `get_partition_coefficients` that bypass `partition_spec_for` do not get the g0 floor:
- `cflibs/inversion/solve/iterative.py:359` — uses raw `pf.coefficients`, no g0 lookup
- `cflibs/inversion/runtime/streaming.py:681,1217` — same pattern
- `cflibs/inversion/physics/self_absorption_inputs.py:104` — same pattern
- `cflibs/plasma/anderson_solver.py:143–144` — same pattern

`saha_boltzmann.py:744–756` is the one caller that DOES correctly call `get_ground_state_g`
separately and pass it as `g0=g0`. The other callers lose the g0 floor guarantee (U can fall
below the ground-state degeneracy at low T). The partition function polynomial can return values
below `g0` at low T where the polynomial is extrapolating; without the floor, temperatures near
or below `t_min` produce physically incorrect partition functions. Severity confirmed LOW
(only affects edge cases at T < t_min, which is outside normal LIBS operating range of 5000–20000 K).

### F9 — `snapshot()` docstring says "non-NULL rel_int" (CONFIRMED TRUE, LOW)

`database.py:869–870` reads:
```
min_relative_intensity : float, optional
    Minimum relative intensity threshold; default 0.0 keeps all
    lines that have a non-NULL ``rel_int`` entry.
```
The actual code at line 1252–1254 passes `None` to `get_transitions` when `min_relative_intensity
<= 0`, which produces a SQL query with NO `rel_int` WHERE clause, including lines where
`rel_int IS NULL`. The docstring description is incorrect: 0.0 keeps ALL lines regardless of
whether they have rel_int data. Severity confirmed LOW (documentation only; code behavior is
correct and sensible).

### F10 — Test gaps (CONFIRMED TRUE, MEDIUM)

Verified by examining `tests/test_atomic.py`: the file covers basic round-trips but has no test
for `is_resonance` backfill correctness, no assertion that the `stark_w` convention matches
`radiation.stark.REF_NE`, no cache-None regression guard, and no `partition_spec_for` round-trip
test. Severity confirmed MEDIUM.

---

## Additional Findings Not in Census

### NEW-1 — `estimate_stark_parameter` hardcoded fallback calibrated for 1e16 convention (LOW — Physics)

**Location:** `cflibs/radiation/stark.py:308–310`

```python
if ionization_potential_ev is None or upper_energy_ev >= ionization_potential_ev:
    # Default ~0.005 nm at 1e16 for typical lines
    return 0.005
```

The function is documented (line 292–304) to return "FWHM at REF_NE = 1e17 cm^-3, matching the
stored `lines.stark_w` convention." However, the hardcoded fallback constant 0.005 nm is
commented as "at 1e16", meaning it was calibrated against the old (now-corrected) wrong
convention. At the current 1e17 convention, this fallback should return ~0.0005 nm (10× smaller)
for the same physical width, or conversely the returned 0.005 nm overstates the estimated
Stark width 10× vs the stored convention. This inconsistency is a direct residue of the
convention confusion documented in the stark.py module-level comment (lines 23–32: "Historically
the runtime treated the column as HWHM at 1e16, which over-broadened every Stark line by a
factor of 20"). The correction fixed the consumption path and the REF_NE constant but left the
fallback literal. Impact: the semi-empirical fallback is a last resort used only when no DB
Stark data exists; lines using it will have modestly overstated Stark widths. Severity: LOW
(estimation path only; DB-backed lines are not affected).

### NEW-2 — `_migrate_lines_columns` runs unconditionally even after `is_resonance` is populated (LOW — Performance)

**Location:** `cflibs/atomic/database.py:104–125`

The `_migrate_lines_columns` migration step at line 106 always issues `PRAGMA table_info(lines)`
and fetches all column metadata. If all columns already exist (post-migration), the loop body
(lines 124–125) is never entered, but the PRAGMA round-trip still occurs on every `__init__`.
This is a facet of F6, but it specifically means the `is_resonance` backfill at lines 143–144
only runs once (on the migration that adds the column), which is correct — but the PRAGMA check
still runs every time. No separate finding required; this is absorbed by F6's fix (version stamp
would skip all of `_perform_migration`).

---

## Severity Table (Confirmed)

| Finding | Severity | Dimension |
|---------|----------|-----------|
| F1 | **HIGH** | Physics-correctness (docstring) |
| F2 | medium | Physics-correctness (docstring) |
| F3 | medium | Physics-correctness (data quality) |
| F4 | medium | Performance |
| F5 | medium | Performance / Correctness |
| F6 | medium | Performance |
| F7 | low | Performance |
| F8 | low | Architecture |
| F9 | low | Architecture (docstring) |
| F10 | medium | Test-gaps |
| NEW-1 | low | Physics-correctness (estimation fallback) |

**Highest confirmed severity: HIGH (F1)**

No census finding was found to be FALSE. All 10 original findings are real.
