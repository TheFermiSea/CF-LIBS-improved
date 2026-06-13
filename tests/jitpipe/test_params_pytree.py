"""J0 AC6 — ``PipelineParams`` pytree + ``StaticConfig`` hashability.

* ``PipelineParams`` flatten/unflatten round-trips and every continuous
  ``AnalysisPipelineConfig`` knob the stage specs name is present;
* changing any ``PipelineParams`` leaf does NOT retrigger compilation (asserted
  via the per-call jit cache-size probe);
* ``StaticConfig`` is hashable and changing a static field DOES recompile (it
  is the cache key by design).

All CPU-x64 (conftest forces it), well under the 600 s watchdog.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.requires_jax


def test_flatten_unflatten_roundtrip():
    import jax

    from cflibs.jitpipe import PipelineParams

    p = PipelineParams(wavelength_tolerance_nm=0.07, min_snr=12.5, max_iterations=15.0)
    leaves, treedef = jax.tree_util.tree_flatten(p)
    # Every leaf is a scalar value (no static aux among the leaves).
    assert len(leaves) == len(PipelineParams.__dataclass_fields__)
    p2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert p2 == p


def test_tree_map_preserves_type():
    import jax

    from cflibs.jitpipe import PipelineParams

    p = PipelineParams()
    doubled = jax.tree_util.tree_map(lambda x: x * 2.0, p)
    assert isinstance(doubled, PipelineParams)
    assert doubled.min_snr == p.min_snr * 2.0


def test_contains_continuous_analysis_config_knobs():
    """AC6: every continuous knob the stage specs name is on PipelineParams."""
    from cflibs.jitpipe import PipelineParams

    expected = {
        "min_relative_intensity",
        "top_k_per_element",
        "wavelength_tolerance_nm",
        "min_peak_height",
        "peak_width_nm",
        "isolation_wavelength_nm",
        "min_snr",
        "min_energy_spread_ev",
        "min_lines_per_element",
        "max_lines_per_element",
        "residual_shift_scan_nm",
        "global_shift_scan_nm",
        "max_iterations",
        "t_tolerance_k",
        "ne_tolerance_frac",
        "pressure_pa",
        "boltzmann_weight_cap",
        "min_boltzmann_r2",
    }
    fields = set(PipelineParams.__dataclass_fields__)
    missing = expected - fields
    assert not missing, f"PipelineParams missing continuous knobs: {missing}"


def test_from_analysis_config():
    """``from_analysis_config`` pulls continuous knobs; None -> defaults."""
    from cflibs.inversion.pipeline import AnalysisPipelineConfig
    from cflibs.jitpipe import PipelineParams

    cfg = AnalysisPipelineConfig(
        preset="geological",
        elements=["Fe", "Cu"],
        wavelength_tolerance_nm=0.05,
        min_snr=15.0,
        max_iterations=12,
        min_relative_intensity=None,  # -> default 0.0
    )
    p = PipelineParams.from_analysis_config(cfg)
    assert p.wavelength_tolerance_nm == 0.05
    assert p.min_snr == 15.0
    assert p.max_iterations == 12.0
    assert p.min_relative_intensity == 0.0  # None fell back to the default


def test_leaf_change_does_not_recompile():
    """AC6: changing any PipelineParams leaf reuses the compiled graph."""
    import jax
    import jax.numpy as jnp

    from cflibs.jitpipe import PipelineParams

    @jax.jit
    def kernel(params):
        return (
            params.wavelength_tolerance_nm * 2.0
            + params.min_snr
            + jnp.log(params.pressure_pa)
            + params.t_tolerance_k
        )

    jax.clear_caches()
    p0 = PipelineParams()
    kernel(p0)
    assert kernel._cache_size() == 1

    # Perturb several leaves; each call must hit the same compiled graph.
    for tol, snr, pres, ttol in [
        (0.05, 20.0, 5e4, 50.0),
        (0.2, 5.0, 2e5, 200.0),
        (0.01, 100.0, 1.0, 1.0),
    ]:
        kernel(
            PipelineParams(
                wavelength_tolerance_nm=tol,
                min_snr=snr,
                pressure_pa=pres,
                t_tolerance_k=ttol,
            )
        )
    assert (
        kernel._cache_size() == 1
    ), f"changing a PipelineParams leaf recompiled: cache_size={kernel._cache_size()}"


def test_static_config_hashable():
    from cflibs.jitpipe import StaticConfig

    s = StaticConfig(bucket_id=256, n_species=30, level_pad=676)
    # Hashable + usable as a dict key / set member.
    assert hash(s) == hash(StaticConfig(bucket_id=256, n_species=30, level_pad=676))
    assert {s: 1}[s] == 1
    assert s in {s}

    # Distinct statics hash distinctly (cache-key behaviour).
    other = StaticConfig(bucket_id=512, n_species=30, level_pad=676)
    assert s != other
    assert hash(s) != hash(other) or s != other


def test_static_config_keys_jit_cache():
    """A StaticConfig change is a recompile by design (it is the cache key)."""
    import functools

    import jax

    from cflibs.jitpipe import PipelineParams, StaticConfig

    @functools.partial(jax.jit, static_argnums=1)
    def kernel(params, static: StaticConfig):
        # Branch on a static field — legal because it is a Python value, not a
        # traced one.
        scale = 2.0 if static.broadening_mode == "gaussian" else 3.0
        return params.min_snr * scale + static.bucket_id

    jax.clear_caches()
    p = PipelineParams()
    s1 = StaticConfig(bucket_id=256, n_species=30, level_pad=676, broadening_mode="gaussian")
    kernel(p, s1)
    assert kernel._cache_size() == 1

    # Same statics, different params leaf -> no recompile.
    kernel(PipelineParams(min_snr=99.0), s1)
    assert kernel._cache_size() == 1

    # Different static -> recompile.
    s2 = StaticConfig(bucket_id=512, n_species=30, level_pad=676, broadening_mode="doppler")
    kernel(p, s2)
    assert kernel._cache_size() == 2
