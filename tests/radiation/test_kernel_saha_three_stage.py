"""Parity tests for the three-stage + IPD snapshot Saha kernel (bead rs7e).

Audit 2026-06-09 finding 01-F4: the JAX snapshot Saha
(``cflibs/radiation/kernels.py``) was two-stage (I/II) with the *raw*
ionization potential — the missing Debye-Hückel ionization-potential
depression (IPD) made the kernel's ion/neutral ratio ~9 % lower than the
CPU/inversion convention at 0.8 eV / 1e17 cm^-3, and the missing stage III
re-assigned the doubly-ionized population to stage II (Ca II line
intensities inflated ×2.9 at the 1.3 eV manifold edge).

Acceptance (bead CF-LIBS-improved-rs7e): kernel-vs-CPU ionization fractions
agree to <1 % across T ∈ {0.5, 0.8, 1.0, 1.3} eV × n_e ∈ {1e16, 1e17, 1e18}
cm^-3 for Fe, Ca, Mg, Si, Al, Na, Ti, with
:class:`~cflibs.plasma.saha_boltzmann.SahaBoltzmannSolver` as the oracle.

Run on ``JAX_PLATFORMS=cpu`` with x64 enabled by ``conftest.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from cflibs.core.constants import EV_TO_K  # noqa: E402
from cflibs.core.jax_runtime import AtomicSnapshot  # noqa: E402
from cflibs.plasma.state import SingleZoneLTEPlasma  # noqa: E402

PARITY_ELEMENTS = ["Fe", "Ca", "Mg", "Si", "Al", "Na", "Ti"]
PARITY_T_EV = [0.5, 0.8, 1.0, 1.3]
PARITY_NE_CM3 = [1e16, 1e17, 1e18]

# Acceptance bound from the bead: <1 % absolute on every stage fraction.
# The implementation mirrors the CPU adapter exactly (same partition specs,
# same Δχ, same IPD-truncated direct sums), so the measured worst-case
# disagreement is ~3e-7; 1e-3 leaves float headroom while staying an order
# of magnitude inside the acceptance bound.
FRACTION_ATOL = 1e-3


def _db_path() -> str:
    candidates = [
        Path("libs_production.db"),
        Path("ASD_da/libs_production.db"),
        Path(__file__).parents[2] / "libs_production.db",
        Path(__file__).parents[2] / "ASD_da" / "libs_production.db",
    ]
    p = next((str(c) for c in candidates if c.exists()), None)
    if p is None:
        pytest.skip("Production database not found")
    return p


@pytest.fixture(scope="module")
def production_db():
    from cflibs.atomic.database import AtomicDatabase

    return AtomicDatabase(_db_path())


@pytest.fixture(scope="module")
def cpu_solver(production_db):
    from cflibs.plasma.saha_boltzmann import SahaBoltzmannSolver

    return SahaBoltzmannSolver(production_db)


@pytest.fixture(scope="module")
def parity_snapshot(production_db):
    """Snapshot with level arrays so the kernel mirrors the CPU direct sums."""
    return production_db.snapshot(
        elements=PARITY_ELEMENTS,
        wavelength_range=(200.0, 800.0),
        include_levels=True,
    )


def _plasma(T_eV: float, n_e: float) -> SingleZoneLTEPlasma:
    return SingleZoneLTEPlasma(
        T_e=T_eV * EV_TO_K,
        n_e=n_e,
        species={el: 1e15 for el in PARITY_ELEMENTS},
    )


def _kernel_fractions(snapshot, T_eV: float, n_e: float):
    from cflibs.radiation.kernels import snapshot_ionization_fractions

    return snapshot_ionization_fractions(_plasma(T_eV, n_e), snapshot)


@pytest.mark.requires_db
@pytest.mark.requires_jax
@pytest.mark.physics
class TestKernelCpuIonizationParity:
    """Kernel vs CPU-solver ionization fractions across the rs7e grid."""

    @pytest.mark.parametrize("T_eV", PARITY_T_EV)
    @pytest.mark.parametrize("n_e", PARITY_NE_CM3)
    def test_fraction_parity(self, parity_snapshot, cpu_solver, T_eV, n_e):
        kernel = _kernel_fractions(parity_snapshot, T_eV, n_e)
        for el in PARITY_ELEMENTS:
            cpu = cpu_solver.get_ionization_fractions(el, T_eV, n_e)
            for stage in (1, 2, 3):
                diff = abs(kernel[el].get(stage, 0.0) - cpu.get(stage, 0.0))
                assert diff < FRACTION_ATOL, (
                    f"{el} stage {stage} at T={T_eV} eV, n_e={n_e:.0e}: "
                    f"kernel={kernel[el].get(stage, 0.0):.6f} "
                    f"cpu={cpu.get(stage, 0.0):.6f} (diff {diff:.2e})"
                )

    def test_fractions_normalized(self, parity_snapshot):
        kernel = _kernel_fractions(parity_snapshot, 1.0, 1e17)
        for el in PARITY_ELEMENTS:
            total = sum(kernel[el].values())
            assert total == pytest.approx(1.0, abs=1e-9)


@pytest.mark.requires_db
@pytest.mark.requires_jax
@pytest.mark.physics
class TestCaHotEdgeRegression:
    """Regression pins for the ×2.9 Ca II inflation at the hot manifold edge."""

    def test_ca_f2_pinned_to_cpu_at_1p3ev_1e16(self, parity_snapshot, cpu_solver):
        """At 1.3 eV / 1e16 the doubly-ionized stage dominates Ca.

        The two-stage kernel assigned f2 ≈ 1.0 here (everything not neutral
        landed in stage II) — a ×2.9+ Ca II over-emission.  The corrected f2
        must match the CPU oracle (~0.05) and stage III must carry the bulk.
        """
        kernel = _kernel_fractions(parity_snapshot, 1.3, 1e16)
        cpu = cpu_solver.get_ionization_fractions("Ca", 1.3, 1e16)
        assert kernel["Ca"][2] == pytest.approx(cpu.get(2, 0.0), abs=FRACTION_ATOL)
        assert kernel["Ca"][3] == pytest.approx(cpu.get(3, 0.0), abs=FRACTION_ATOL)
        # Hard physical pins (guard against kernel AND oracle drifting
        # together): Ca III dominates, Ca II is a small minority.
        assert kernel["Ca"][3] > 0.85
        assert kernel["Ca"][2] < 0.10

    def test_ca_f3_matches_audit_value_at_1e17(self, parity_snapshot, cpu_solver):
        """Audit 01-F4 measured Ca f3 = 0.65 at 1.3 eV / 1e17 cm^-3."""
        kernel = _kernel_fractions(parity_snapshot, 1.3, 1e17)
        cpu = cpu_solver.get_ionization_fractions("Ca", 1.3, 1e17)
        assert kernel["Ca"][3] == pytest.approx(cpu.get(3, 0.0), abs=FRACTION_ATOL)
        assert 0.60 < kernel["Ca"][3] < 0.70

    def test_no_two_stage_ca_ii_inflation(self, parity_snapshot):
        """The old two-stage f2 was ~2.9× the corrected value at 1.3 eV/1e17."""
        kernel = _kernel_fractions(parity_snapshot, 1.3, 1e17)
        f2 = kernel["Ca"][2]
        f3 = kernel["Ca"][3]
        # Two-stage behaviour folded f3 into f2: f2_two_stage ≈ f2 + f3.
        two_stage_f2 = f2 + f3
        assert two_stage_f2 / f2 > 2.5  # the audit's ×2.9, with slack


@pytest.mark.requires_db
@pytest.mark.requires_jax
@pytest.mark.physics
class TestIpdApplied:
    """The Δχ-lowered Saha exponent (the ~9 % audit discrepancy) is applied."""

    def test_fe_ion_neutral_ratio_matches_cpu_not_raw_ip(
        self, parity_snapshot, cpu_solver, production_db
    ):
        from cflibs.plasma.partition import ionization_potential_depression

        T_eV, n_e = 0.8, 1e17
        kernel = _kernel_fractions(parity_snapshot, T_eV, n_e)
        cpu = cpu_solver.get_ionization_fractions("Fe", T_eV, n_e)

        ratio_kernel = kernel["Fe"][2] / kernel["Fe"][1]
        ratio_cpu = cpu[2] / cpu[1]
        assert ratio_kernel == pytest.approx(ratio_cpu, rel=1e-3)

        # The raw-IP (no-IPD) ratio is exp(Δχ/kT) ≈ 1.089 lower — make sure
        # the kernel is NOT reproducing it (the pre-rs7e behaviour).
        delta_chi = ionization_potential_depression(n_e, T_eV * EV_TO_K)
        no_ipd_factor = float(np.exp(delta_chi / T_eV))
        assert no_ipd_factor > 1.05  # sanity: the audit's ~1.089 scale
        ratio_no_ipd = ratio_cpu / no_ipd_factor
        assert abs(ratio_kernel - ratio_no_ipd) / ratio_no_ipd > 0.05


@pytest.mark.requires_jax
class TestIpdLevelCutoff:
    """Lines whose upper level exceeds the Δχ-lowered IP carry zero population."""

    @staticmethod
    def _synthetic_snapshot() -> AtomicSnapshot:
        """One species (Fe I, ip = 7.9 eV) with a bound and a near-IP line."""
        ln_u = float(np.log(25.0))
        return AtomicSnapshot(
            species=(("Fe", 1), ("Fe", 2)),
            line_wavelengths_nm=jnp.array([400.0, 410.0]),
            line_A_ki=jnp.array([1.0e7, 1.0e7]),
            line_E_k_ev=jnp.array([3.0, 7.88]),
            line_g_k=jnp.array([5.0, 5.0]),
            line_E_i_ev=jnp.array([0.0, 0.0]),
            line_g_i=jnp.array([3.0, 3.0]),
            line_species_index=jnp.array([0, 0], dtype=jnp.int32),
            line_stark_w=jnp.zeros(2),
            line_stark_alpha=jnp.zeros(2),
            line_natural_w=jnp.zeros(2),
            partition_coeffs=jnp.array(
                [[ln_u, 0.0, 0.0, 0.0, 0.0], [ln_u, 0.0, 0.0, 0.0, 0.0]]
            ),
            ionization_potential_ev=jnp.array([7.9, 16.19]),
        )

    def test_near_ip_line_zeroed_at_high_density(self):
        """At 1e18 cm^-3, Δχ ≈ 0.21 eV lowers the 7.9 eV IP below E_k = 7.88."""
        from cflibs.radiation.kernels import _saha_three_stage_populations

        snapshot = self._synthetic_snapshot()
        plasma = SingleZoneLTEPlasma(
            T_e=1.0 * EV_TO_K, n_e=1e18, species={"Fe": 1e16}
        )
        n_upper = np.asarray(_saha_three_stage_populations(plasma, snapshot))
        assert n_upper[0] > 0.0  # E_k = 3.0 eV: bound, populated
        assert n_upper[1] == 0.0  # E_k = 7.88 eV > 7.9 - Δχ: continuum

    def test_near_ip_line_kept_at_low_density(self):
        """At 1e14 cm^-3, Δχ ≈ 0.007 eV keeps E_k = 7.88 below the cutoff."""
        from cflibs.radiation.kernels import _saha_three_stage_populations

        snapshot = self._synthetic_snapshot()
        plasma = SingleZoneLTEPlasma(
            T_e=1.0 * EV_TO_K, n_e=1e14, species={"Fe": 1e16}
        )
        n_upper = np.asarray(_saha_three_stage_populations(plasma, snapshot))
        assert n_upper[0] > 0.0
        assert n_upper[1] > 0.0


@pytest.mark.requires_jax
class TestBackCompat:
    """Pre-rs7e snapshots (no stage-III fields) degrade to two-stage + IPD."""

    def test_legacy_snapshot_runs_and_has_no_stage_three(self):
        from cflibs.radiation.kernels import snapshot_ionization_fractions

        snapshot = TestIpdLevelCutoff._synthetic_snapshot()
        assert snapshot.partition_coeffs_iii is None
        plasma = SingleZoneLTEPlasma(
            T_e=1.0 * EV_TO_K, n_e=1e17, species={"Fe": 1e16}
        )
        fractions = snapshot_ionization_fractions(plasma, snapshot)
        assert fractions["Fe"][3] == 0.0
        assert fractions["Fe"][1] + fractions["Fe"][2] == pytest.approx(1.0, abs=1e-9)

    def test_two_stage_alias_points_at_three_stage(self):
        from cflibs.radiation.kernels import (
            _saha_three_stage_populations,
            _saha_two_stage_populations,
        )

        assert _saha_two_stage_populations is _saha_three_stage_populations


@pytest.mark.requires_db
@pytest.mark.requires_jax
@pytest.mark.physics
class TestManifoldGeneratorFractions:
    """The manifold's Saha-Eggert solver follows the same three-stage system.

    The generator bakes polynomial partition fits (no per-n_e IPD truncation
    of U is possible in its static float32 arrays), so the tolerance is
    looser than the kernel's — 2 % absolute at the smoke conditions.
    """

    def test_smoke_config_fractions_match_cpu(self, cpu_solver):
        from cflibs.manifold.config import ManifoldConfig
        from cflibs.manifold.generator import ManifoldGenerator

        config_path = Path(__file__).parents[2] / "examples" / "manifold_smoke_config.yaml"
        if not config_path.exists():
            pytest.skip("manifold smoke config not found")
        config = ManifoldConfig.from_file(config_path)
        config.db_path = Path(_db_path())
        generator = ManifoldGenerator(config)
        atomic_data = generator.atomic_data

        T_eV, n_e = 1.0, 1e17
        t_k = T_eV * EV_TO_K
        u0, u1, u2 = ManifoldGenerator._calculate_partition_functions(t_k, atomic_data)
        frac0, frac1, frac2, _ = ManifoldGenerator._calculate_saha_fractions(
            T_eV, n_e, u0, u1, u2, atomic_data
        )

        lines_el_idx = np.asarray(atomic_data[6])
        for el_idx, el in enumerate(config.elements):
            line = int(np.argmax(lines_el_idx == el_idx))
            cpu = cpu_solver.get_ionization_fractions(el, T_eV, n_e)
            for stage, frac in ((1, frac0), (2, frac1), (3, frac2)):
                diff = abs(float(np.asarray(frac)[line]) - cpu.get(stage, 0.0))
                assert diff < 0.02, (
                    f"{el} stage {stage}: generator={float(np.asarray(frac)[line]):.4f} "
                    f"cpu={cpu.get(stage, 0.0):.4f}"
                )
