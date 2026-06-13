"""End-to-end jittable pipeline orchestration (J8; ADR-0004 §5.1.1, §8.1).

Composes the stage modules into single-spectrum (:func:`run_one`) and batched
(:func:`run_batch`) entry points, mirroring the reference stage order of
:func:`cflibs.inversion.pipeline.run_pipeline`:

    response-multiply -> preprocess -> calibrate -> detect + identify
    -> (self-absorption) -> stark n_e -> fit/solve -> closure.

Architecture (J8 plan §0/§5)
----------------------------
The nine stage *kernels* (preprocess/detect/calibrate/identify/fit/selfabs/
stark/solve) are device-pure and each parity-tested in isolation. J8 owns the
*composition* + the host gather/scatter seam (``host.py``):

* The **front-end** (response -> calibrate -> detect -> identify -> select) is
  driven through :func:`cflibs.jitpipe.host.run_front_end`, which reuses the
  reference ``detect_and_select_lines`` path so the observation set fed to the
  solve spine is byte-faithful to the parity oracle. The detect/identify/
  calibrate jit kernels (J2/J3/J4) are individually parity-tested; threading
  them into a byte-faithful reproduction of the full ``detect_line_observations``
  gate stack is tracked separately (see ``remaining_todo``).
* The **solve spine** (the parity-critical CF-LIBS Saha-Boltzmann math) has two
  routes keyed by the resolved preset (the M1 production-config parity gap
  D-J8-1, J8 plan §5):

  - the **shared-math** ``raw`` preset (``saha_boltzmann_graph=False`` +
    ``stark_ne=False``) runs the device-pure
    :func:`cflibs.jitpipe.solve.scan_solve` kernel, which mirrors the reference
    ``_run_lax_while_loop`` fixed point bit-for-bit (J7 parity), reconstituted
    by :func:`host.cflibs_result_from_loopstate`;
  - the **production geological/metallic** preset
    (``saha_boltzmann_graph=True`` + ``stark_ne=True``) runs
    :func:`cflibs.jitpipe.solve.production_solve` — the ``scan_solve``
    initializer SEEDS the :func:`cflibs.jitpipe.solve.joint_wls_solve`
    production estimator (``sb_graph=True``, ``n_e`` pinned to the Stark
    measurement), mirroring the reference ``_solve_python`` SB-graph +
    Stark-pinned-``n_e`` path the shared-math scan cannot reproduce
    (ADR-0004 §6.1; J7 §4 row 10).

  Both routes gather the padded ``(E, N_max)`` block
  (:func:`host.build_observation_block`), assemble ``LaxKernelInputs`` from the
  baked snapshot (no DB at solve time), and reconstitute a
  :class:`~cflibs.inversion.solve.iterative.CFLIBSResult`.

Failure policy (J8 plan §4, AC4): at zero valid observations the reference
raises ``ValueError`` (``pipeline.py:872``); ``run_one`` instead returns the
SAME all-FN ``CFLIBSResult`` (:func:`host.all_fn_result`) the scoreboard scores
as all false-negative — no crash, no NaN.

The ``PipelineParams`` -> kernel-kwarg fan-out (the documented name-drift map,
J8 plan §3.1) is bridged in :func:`_solve_params` below.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cflibs.jitpipe import host as _host

if TYPE_CHECKING:  # pragma: no cover
    from cflibs.jitpipe.params import PipelineParams, StaticConfig
    from cflibs.jitpipe.snapshot import PipelineSnapshot


__all__ = ["run_one", "run_batch", "solve_stage", "observations_to_result"]


# ---------------------------------------------------------------------------
# PipelineParams -> solve-kernel kwarg fan-out (J8 plan §3.1 name-drift map).
# ---------------------------------------------------------------------------


def _solve_params(params: "PipelineParams") -> dict:
    """Map ``PipelineParams`` continuous knobs to ``scan_solve`` kwargs.

    Bridges the documented name drift (J8 plan §3.1): every field name differs
    between ``PipelineParams`` and the solve kernel signature. This is the exact
    checklist:

    * ``t_tolerance_k`` -> ``t_tol_k``
    * ``ne_tolerance_frac`` -> ``ne_tol_frac``
    * ``min_boltzmann_r2`` -> ``min_r2``
    * ``pressure_pa`` -> ``pressure_pa`` (straight)

    ``max_iterations`` is the traced convergence budget; the *static* scan trip
    count lives in ``StaticConfig.max_iters`` (passed separately by the caller).
    """
    return {
        "t_tol_k": float(params.t_tolerance_k),
        "ne_tol_frac": float(params.ne_tolerance_frac),
        "pressure_pa": float(params.pressure_pa),
        "min_r2": float(params.min_boltzmann_r2),
    }


# ---------------------------------------------------------------------------
# Solve stage — device-pure kernel call + host reconstitution.
# ---------------------------------------------------------------------------


def solve_stage(
    block: "_host.ObservationBlock",
    snapshot: "PipelineSnapshot",
    params: "PipelineParams",
    static: "StaticConfig",
    *,
    ne_stark_cm3: float | None = None,
    production: bool = False,
    ne_scatter_cm3: float = 0.0,
):
    """Run the jit solve spine on a padded observation block + reconstitute.

    Two solve routes (the M1 production-config parity gap D-J8-1, J8 plan §5):

    * **shared-math path** (``production=False``, the ``raw`` preset:
      ``saha_boltzmann_graph=False`` + ``stark_ne=False`` + ``standard``
      closure): the device-pure :func:`scan_solve` kernel, which mirrors the
      reference ``_solve_lax``/``_run_lax_while_loop`` fixed point bit-for-bit.
    * **production geological/metallic path** (``production=True``, the
      ``geological``/``metallic`` presets: ``saha_boltzmann_graph=True`` +
      ``stark_ne=True``): the validated :func:`production_solve` dispatch —
      the :func:`scan_solve` initializer SEEDS the :func:`joint_wls_solve`
      production estimator (``sb_graph=True``, ``n_e`` pinned to the Stark
      measurement), mirroring the reference ``_solve_python`` SB-graph +
      Stark-pinned-``n_e`` path the shared-math scan cannot reproduce
      (ADR-0004 §6.1; J7 §4 row 10).

    Honours the ``StaticConfig`` static knobs (``closure_mode``, ``max_iters``)
    and the ``PipelineParams`` traced knobs (the §3.1 name-drift map).

    Parameters
    ----------
    block : ObservationBlock
        Padded ``(E, N_max)`` Boltzmann block. ``n_observations == 0`` returns
        the all-FN result (failure policy).
    snapshot, params, static
        Atomic snapshot, traced knobs, static config.
    ne_stark_cm3 : float or None
        Stark-measured electron density pinning the warm-start ``n_e`` (J6).
        ``None`` -> the reference default 1e17 cm^-3 (scan) or the joint solve's
        pressure-balance fallback (production).
    production : bool
        Route through :func:`production_solve` (joint WLS seeded by the scan)
        when True; the shared-math :func:`scan_solve` when False.
    ne_scatter_cm3 : float
        Stark multi-line MAD scatter (production path covariance only).

    Returns
    -------
    CFLIBSResult
    """
    import jax.numpy as jnp

    from cflibs.jitpipe.solve import production_solve, scan_solve

    if block.n_observations == 0 or block.x is None:
        return _host.all_fn_result(list(block.elements))

    inp = _host.lax_inputs_from_observation_block(snapshot, block)

    closure_mode = static.closure_mode
    oxide_factors = None
    if closure_mode == "oxide":
        oxide_factors = jnp.asarray(
            _host.oxide_factors_for_elements(snapshot, block.elements), dtype=jnp.float64
        )

    solve_kw = _solve_params(params)

    if production:
        joint = production_solve(
            inp,
            ne_stark_cm3=ne_stark_cm3,
            ne_scatter_cm3=float(ne_scatter_cm3),
            init_T_K=10000.0,
            init_ne_cm3=1.0e17,
            seed_iters=int(static.max_iters),
            n_gn_steps=int(static.max_iters),
            closure_mode=closure_mode,
            oxide_factors=oxide_factors,
            t_tol_k=solve_kw["t_tol_k"],
            ne_tol_frac=solve_kw["ne_tol_frac"],
            pressure_pa=solve_kw["pressure_pa"],
            min_r2=solve_kw["min_r2"],
        )
        return _cflibs_result_from_joint(joint, list(block.elements))

    init_ne = 1.0e17 if ne_stark_cm3 is None else float(ne_stark_cm3)
    final_state = scan_solve(
        inp,
        init_T_K=10000.0,
        init_ne_cm3=init_ne,
        max_iters=int(static.max_iters),
        closure_mode=closure_mode,
        oxide_factors=oxide_factors,
        **solve_kw,
    )
    return _host.cflibs_result_from_loopstate(final_state, list(block.elements))


def _cflibs_result_from_joint(joint, elements: list):
    """Reconstitute a reference ``CFLIBSResult`` from a :class:`JointWLSResult`.

    Mirrors :func:`cflibs.jitpipe.host.cflibs_result_from_loopstate` for the
    production joint-WLS path (D-J8-1): maps the E-indexed simplex back to an
    element-keyed dict, reports the Stark provenance + the joint convergence
    flag (reported hard, ADR-0004 §6.4). Kept in the impure composition module
    so ``solve.py`` stays a DB-free kernel module (AC5).
    """
    import numpy as np

    from cflibs.inversion.solve.iterative import CFLIBSResult

    conc_arr = np.asarray(joint.concentrations)
    concentrations = {el: float(conc_arr[i]) for i, el in enumerate(elements)}
    return CFLIBSResult(
        temperature_K=float(joint.T_K),
        temperature_uncertainty_K=0.0,
        electron_density_cm3=float(joint.n_e_cm3),
        concentrations=concentrations,
        concentration_uncertainties={},
        iterations=int(joint.n_iter),
        converged=bool(joint.converged),
        temperature_corona_K=None,
        quality_metrics={
            "physical": float(bool(joint.physical)),
            "ne_from_stark": float(bool(joint.ne_from_stark)),
            "ne_scatter_cm3": float(joint.ne_scatter_cm3),
        },
        electron_density_uncertainty_cm3=0.0,
        boltzmann_covariance=None,
    )


def _is_production(pipeline) -> bool:
    """Is the resolved config the production geological/metallic preset?

    The divergence D-J8-1 routing key: the reference ``saha_boltzmann_graph``
    and/or ``stark_ne`` flags select the SB-graph + Stark-pinned ``_solve_python``
    path (``ANALYSIS_PRESETS['geological'/'metallic']``). When either is set the
    jit composition routes the solve through :func:`production_solve`
    (joint WLS seeded by the scan); the ``raw`` preset
    (``saha_boltzmann_graph=False`` + ``stark_ne=False``) stays on the
    shared-math :func:`scan_solve`. ``None`` (no config) -> shared-math.
    """
    if pipeline is None:
        return False
    return bool(getattr(pipeline, "saha_boltzmann_graph", False)) or bool(
        getattr(pipeline, "stark_ne", False)
    )


def observations_to_result(
    observations,
    snapshot: "PipelineSnapshot",
    params: "PipelineParams",
    static: "StaticConfig",
    *,
    ne_stark_cm3: float | None = None,
    production: bool = False,
):
    """Solve a ``LineObservation`` list -> :class:`CFLIBSResult` (host gather + kernel).

    The host wrapper behind :func:`cflibs.jitpipe.solve.iterative_solve`: gathers
    the padded ``(E, N_max)`` block (:func:`host.build_observation_block`) and
    runs the jit solve spine via :func:`solve_stage`. Kept here in the impure
    composition module so the kernel module ``solve.py`` stays DB-free (AC5).

    ``production`` selects the geological/metallic joint-WLS dispatch (D-J8-1).
    """
    block = _host.build_observation_block(
        observations, weight_cap=float(params.boltzmann_weight_cap)
    )
    return solve_stage(
        block, snapshot, params, static, ne_stark_cm3=ne_stark_cm3, production=production
    )


# ---------------------------------------------------------------------------
# Single-spectrum end-to-end composition.
# ---------------------------------------------------------------------------


def run_one(
    intensities: Any,
    wavelengths_nm: Any,
    snapshot: "PipelineSnapshot",
    params: "PipelineParams",
    static: "StaticConfig",
    *,
    atomic_db: Any = None,
    pipeline_config: Any = None,
):
    """Run the full jittable inversion pipeline on ONE spectrum (J8).

    Composes every stage in the reference order
    (:func:`cflibs.inversion.pipeline.run_pipeline`): response/preprocess ->
    calibrate -> detect + identify -> (self-absorption) -> stark n_e ->
    fit/solve -> closure. Returns a :class:`CFLIBSResult` — the same dataclass
    the reference returns, so downstream consumers and parity adapters see
    identical types.

    Parameters
    ----------
    intensities : array
        Raw measured intensities, shape ``(N_pix,)``.
    wavelengths_nm : array
        Wavelength axis, nm, shape ``(N_pix,)``.
    snapshot : PipelineSnapshot
        Atomic-data snapshot (host-built, passed as a pytree).
    params : PipelineParams
        Traced continuous knobs.
    static : StaticConfig
        Static config — the jit cache key (``closure_mode``, ``max_iters``,
        ``apply_self_absorption``).
    atomic_db : AtomicDatabase, optional
        Reference DB for the host front-end (detect/identify/calibrate/Stark).
        Defaults to opening ``host.DEFAULT_DB_PATH``.
    pipeline_config : AnalysisPipelineConfig, optional
        Resolved front-end config. When omitted, one is built from ``params`` +
        ``static`` (geological-equivalent knobs) for the elements in the
        snapshot's species axis.

    Returns
    -------
    CFLIBSResult
        Inversion result (T, n_e, concentrations, diagnostics). At zero usable
        observations: the all-FN failure-policy result (AC4), not a crash.
    """
    from cflibs.jitpipe.host import build_observation_block, run_front_end

    if atomic_db is None:
        from cflibs.atomic.database import AtomicDatabase

        atomic_db = AtomicDatabase(_host.DEFAULT_DB_PATH)

    pipeline = pipeline_config
    if pipeline is None:
        pipeline = _pipeline_config_from(params, static, snapshot)

    front = run_front_end(wavelengths_nm, intensities, atomic_db, pipeline)
    if front.n_observations == 0:
        return _host.all_fn_result(list(pipeline.elements))

    block = build_observation_block(
        front.observations, weight_cap=float(params.boltzmann_weight_cap)
    )
    # D-J8-1: route the production geological/metallic preset (SB-graph +
    # Stark-pinned n_e) through the joint-WLS dispatch; keep scan_solve for the
    # shared-math (raw) preset. The route is the resolved config's
    # ``saha_boltzmann_graph``/``stark_ne`` flags (the reference routing key).
    return solve_stage(
        block,
        snapshot,
        params,
        static,
        ne_stark_cm3=front.ne_stark_cm3,
        production=_is_production(pipeline),
    )


def _pipeline_config_from(
    params: "PipelineParams", static: "StaticConfig", snapshot: "PipelineSnapshot"
):
    """Build an ``AnalysisPipelineConfig`` from ``params`` + ``static`` + snapshot.

    Used when ``run_one`` is called without an explicit front-end config:
    resolves a geological-equivalent pipeline over every element in the
    snapshot's species axis, threading the traced ``PipelineParams`` knobs
    and the static ``closure_mode`` / ``apply_self_absorption`` choices.
    """
    from cflibs.inversion.pipeline import build_pipeline_config

    elements = sorted({el for el, _sp in snapshot.species})
    apply_sa = "observable" if static.apply_self_absorption else "off"
    # Build the geological preset faithfully: ``saha_boltzmann_graph=True`` +
    # ``stark_ne=True`` are inherited from the preset (NOT overridden here), so
    # ``run_one`` routes the solve through the production joint-WLS dispatch
    # (:func:`production_solve`) that mirrors the reference ``_solve_python``
    # SB-graph + Stark-pinned path (D-J8-1; the shared-math ``raw`` preset stays
    # on ``scan_solve``). ``closure_mode`` follows the static config.
    overrides = {
        "wavelength_tolerance_nm": float(params.wavelength_tolerance_nm),
        "min_peak_height": float(params.min_peak_height),
        "peak_width_nm": float(params.peak_width_nm),
        "isolation_wavelength_nm": float(params.isolation_wavelength_nm),
        "min_snr": float(params.min_snr),
        "min_energy_spread_ev": float(params.min_energy_spread_ev),
        "min_lines_per_element": int(round(params.min_lines_per_element)),
        "max_lines_per_element": int(round(params.max_lines_per_element)),
        "top_k_per_element": int(round(params.top_k_per_element)),
        "residual_shift_scan_nm": float(params.residual_shift_scan_nm),
        "global_shift_scan_nm": float(params.global_shift_scan_nm),
        "max_iterations": int(round(params.max_iterations)),
        "t_tolerance_k": float(params.t_tolerance_k),
        "ne_tolerance_frac": float(params.ne_tolerance_frac),
        "pressure_pa": float(params.pressure_pa),
        "boltzmann_weight_cap": float(params.boltzmann_weight_cap),
        "min_boltzmann_r2": float(params.min_boltzmann_r2),
        "closure_mode": static.closure_mode,
        "apply_self_absorption": apply_sa,
    }
    return build_pipeline_config(elements, preset="geological", overrides=overrides)


def run_batch(
    intensities: Any,
    wavelengths_nm: Any,
    snapshot: "PipelineSnapshot",
    params: "PipelineParams",
    static: "StaticConfig",
    *,
    atomic_db: Any = None,
    pipeline_configs: Any = None,
):
    """Run the pipeline on a BATCH of spectra (J8 host-loop; vmap spine in J9).

    The host gather/front-end is impure (SQLite, scipy) and stays host-side, so
    the batch axis is a host loop over :func:`run_one` for J8. The jit-graphable
    *solve core* (``scan_solve``) is already ``vmap``-clean (J7 parity test);
    J9 will lift the whole solve spine under ``jit(vmap(...))`` once the
    front-end gather is fully on-device. For M1 this returns a list of
    per-spectrum :class:`CFLIBSResult`.

    Parameters
    ----------
    intensities : array
        Raw intensities, shape ``(B, N_pix)``.
    wavelengths_nm : array
        Wavelength axis, nm, shape ``(N_pix,)`` (shared) or ``(B, N_pix)``.
    snapshot, params, static
        As :func:`run_one` (shared across the batch).
    atomic_db, pipeline_configs : optional
        Shared DB / per-spectrum front-end configs.

    Returns
    -------
    list[CFLIBSResult]
    """
    import numpy as np

    inten = np.asarray(intensities)
    wl = np.asarray(wavelengths_nm)
    B = inten.shape[0]
    shared_wl = wl.ndim == 1
    results = []
    for b in range(B):
        wl_b = wl if shared_wl else wl[b]
        cfg_b = None if pipeline_configs is None else pipeline_configs[b]
        results.append(
            run_one(
                inten[b],
                wl_b,
                snapshot,
                params,
                static,
                atomic_db=atomic_db,
                pipeline_config=cfg_b,
            )
        )
    return results
