# Test-Suite Triage — 2026-06-10 (goal-first program, Phase B)

Classification of all 211 test files against the project goal (accurate autonomous
identification + efficient composition analysis; zero users, no back-compat owed).

Counts: {'mixed': 44, 'physics-truth': 72, 'implementation-pin': 12, 'unit-utility': 60, 'goal-metric': 23}

Judgment rule: would this assertion failing ever indicate the GOAL got worse,
or only that the implementation changed? Pins protect the latter.

## Purge list (pure implementation-pin files)

- `tests/benchmark/test_alias_sweep_workflows_registered.py` — Pins exact kwargs dicts, cell ordering, and r2-gate of the historical 8-cell alias-fix sweep (jaunty-weaving-mist Phase C campaign). Constant==literal tautologies preserving a finished experiment's configuration; measures no accuracy. Experiment archaeology.
- `tests/benchmark/test_jax_workflows_xfail_check.py` — Meta-test asserting a stale @xfail marker string still exists in another test file's source; self-described as delete-when-fixed housekeeping for bead 359q. Pure archaeology, protects nothing.
- `tests/inversion/test_bayesian_back_compat.py` — Pure back-compat shim suite: legacy flat-path names must resolve after the bayesian.py->package split. With zero users no back-compat is owed; test_flat_and_package_paths_agree imports the same module twice and asserts identity (tautological). Any real consumer test covers the imports that matter. PURGE.
- `tests/inversion/test_bayesian_samplers.py` — Entirely back-compat surface pins: alias identity (NumPyroNUTSSampler is MCMCSampler), legacy run() keyword-set preservation, Protocol duck-typing tautologies, frozen-dataclass shape. No assertion failing here would ever mean identification/composition accuracy or runtime got worse — only that the API was renamed.
- `tests/test_boltzmann_outlier_tuning.py` — Pins tuned fitter defaults as exact values (outlier_sigma==2.5, max_iterations==10, huber_epsilon==1.2); remaining asserts are constructor-assignment tautologies and weak smoke already covered by test_boltzmann.py.
- `tests/test_comb_precision.py` — Pins tuned Tier-2 detection thresholds (Mn/Na/K) and exact CombIdentifier defaults (0.12, 2.0) on contrived inputs. Whether these thresholds help F1 is a benchmark question, not a unit invariant.
- `tests/test_comb_recall.py` — Pins the capped-denominator scoring formula (score 2/10 vs old 2/50) under hand-tuned parameters; explicit old-vs-new behavior archaeology. A benchmark-better scoring change would fail it without any recall regression.
- `tests/test_daq_interface.py` — process_spectrum is an explicit placeholder; tests pin its hardcoded dummy plasma parameters (T==12000 K, ne==1e17) — asserting a stub returns its stub values. Nothing goal- or physics-protecting.
- `tests/test_jax_import_hygiene.py` — AST lint enforcing that only a hardcoded carve-out list may use literal 'try: import jax' and kernels.py must not import host.py. Pure architectural-conformance pin (ADR-0001/T1-1 archaeology); failing never indicates the goal got worse.
- `tests/test_kernel_module_layout.py` — Asserts kernels.py module layout exists per ADR and that host modules re-export private kernel symbols 'for back-compat with existing callers'. Module-layout and shim re-export pins; no physics or goal protection.
- `tests/test_repo_health.py` — Enforces an 800-LOC-per-file ceiling on bayesian/*.py (ADR/spec acceptance criterion). Pure code-organization policing; failing can never indicate identification/composition accuracy or runtime got worse.
- `tests/test_vrabel2020_cdsb.py` — Empty tombstone: docstring-only module documenting that the legacy CDSBPlotter tests were removed with the class; coverage now lives in test_self_absorption.py. Zero assertions, pure deleted-feature remnant — delete the file.

## Mixed files (surgical pin removal; pinned tests named)

- `tests/bandit/test_parameter_sweep_bandit_integration.py`: test_bandit_zero_produces_no_arm_fields_in_manifest
- `tests/benchmark/test_alias_high_recall_workflow.py`: test_strict_alias_still_registered_unchanged
- `tests/benchmark/test_bayesian_config.py`: test_bayesian_pipeline_config_applied
- `tests/cli/test_analyze_invert_no_drift.py`: test_helper_and_analyze_default_floor_agree, test_helper_exclude_resonance_tied_to_self_absorption, test_analyze_and_invert_select_identical_lines
- `tests/cli/test_pipeline_defaults.py`: TestPresetResolution::test_default_preset_is_geological_best_validated, TestPresetResolution::test_metallic_preset, TestPresetResolution::test_raw_preset_is_legacy_defaults, TestPresetResolution::test_preset_registry_contents, TestPresetResolution::test_axis_alignment_knobs_default_fixed_raw_keeps_legacy, TestBatchAnalyzeParity::test_default_pipeline_is_measured_best
- `tests/cli/test_response_curve_plumbing.py`: test_run_pipeline_identity_is_bit_identical
- `tests/evolution/test_prompts.py`: test_render_preamble_states_the_hard_constraint, test_render_preamble_clarifies_ml_is_ok_in_optimization
- `tests/inversion/identify/test_alias_jax_boltzmann.py`: TestALIASIdentifierFlag::test_constructor_accepts_flag
- `tests/inversion/identify/test_alias_jax_nnls.py`: TestALIASIdentifierJaxFlags::test_all_flags_accepted, TestALIASIdentifierJaxFlags::test_default_is_cpu
- `tests/inversion/identify/test_alias_presets.py`: test_all_expected_presets_registered, test_strict_preset_uses_fixed_gate, test_v2_preset_uses_adaptive_gates, test_high_recall_v2_is_v2_plus_high_recall, test_consensus_voter_matches_v2_physics
- `tests/inversion/identify/test_candidate_count_invariance.py`: TestStandaloneNNLSCountInvariance::test_sum_normalized_floor_mechanism_is_count_scaling
- `tests/inversion/identify/test_comb_wavelength_tolerance.py`: test_comb_legacy_defaults_unchanged_without_resolving_power
- `tests/inversion/identify/test_residual_gates.py`: TestEdgeRidingScanRegression::test_legacy_mop_up_scan_rides_edge_and_admits_contaminated_match, TestPerPeakOwnership::test_legacy_path_still_double_counts
- `tests/inversion/physics/test_boltzmann_jax_composition.py`: TestBehaviorContracts::test_default_use_jax_is_false
- `tests/inversion/solve/bayesian/test_vmap_chain_method.py`: test_mcmcsampler_uses_vectorized_chain_method, test_mcmcsampler_max_tree_depth_default_is_8
- `tests/inversion/test_bayesian_forward_model_kernel_migration.py`: test_forward_py_body_does_not_call_adapter, test_mcmc_sampler_default_chain_method_is_vectorized
- `tests/inversion/test_bayesian_priors.py`: test_prior_config_defaults, test_noise_parameters_defaults
- `tests/inversion/test_candidate_prefilter_falsy_zero.py`: TestFalsyZeroEstimate::test_zero_estimated_T_is_used_not_fallback, TestFalsyZeroEstimate::test_zero_estimated_ne_is_used_not_fallback
- `tests/inversion/test_comb_fp_reduction.py`: test_tier2_defaults_are_strict, test_tier2_strict_disabled_matches_baseline
- `tests/inversion/test_response_correction.py`: TestSpectralResponseCorrection::test_identity_none_is_bit_identical, test_argon_branching_ratio_stub_raises_with_citation
- `tests/manifold/test_generator_physics_w3.py`: TestDopplerSigmaParity::test_voigt_path_doppler_sigma_matches_profiles
- `tests/manifold/test_sampling_guard.py`: TestNyquistGuard::test_default_pixels_is_18432, TestNyquistGuard::test_yaml_default_pixels_is_18432
- `tests/radiation/test_forward_model_parity.py`: test_bayesian_forward_kernel_parity_deferred
- `tests/radiation/test_kernel_saha_three_stage.py`: TestBackCompat::test_two_stage_alias_points_at_three_stage, TestBackCompat::test_legacy_snapshot_runs_and_has_no_stage_three
- `tests/radiation/test_stark_t_factor_toggle.py`: test_toggle_on_disables_t_factor, test_toggle_value_must_be_exactly_1
- `tests/scripts/test_measure_bhvo2_harness.py`: test_default_knobs_are_cli_defaults, test_maintainer_gate_flags_resolve_to_defaults, test_preset_flag_resolves_bundle, test_script_does_not_hand_build_pipeline
- `tests/test_alias.py`: test_default_detection_threshold, test_max_lines_per_element_parameter
- `tests/test_alias_unit.py`: test_alias_default_is_strict, test_alias_high_recall_lowers_thresholds, test_per_ion_stage_default_off_byte_identical, test_temperature_estimator_legacy_byte_identical, test_r2_gate_fixed_mode_byte_identical, test_default_r2_gate_mode_is_adaptive_t
- `tests/test_closure_strategy.py`: TestSoftmaxClosureParity::test_bit_identical_1d, TestSoftmaxClosureParity::test_bit_identical_batched, TestSoftmaxClosureParity::test_bit_identical_extreme_values, TestILRClosureParity::test_matches_apply_ilr, TestPWLRClosureParity::test_matches_apply_pwlr
- `tests/test_comb.py`: test_default_min_correlation_remains_benchmark_gated, test_max_lines_per_element_parameter, test_coverage_penalty_reduces_score
- `tests/test_correlation_identifier.py`: test_instrument_fwhm_parameter, test_max_lines_per_element_parameter, test_default_min_confidence_lowered
- `tests/test_dirichlet_prior.py`: TestStickBreakingTransform::test_stick_breaking_produces_valid_simplex, TestStickBreakingTransform::test_stick_breaking_with_sparse_alpha, TestStickBreakingTransform::test_stick_breaking_uniform_alpha_is_uniform_on_simplex, TestStickBreakingTransform::test_stick_breaking_peaked_alpha_concentrates_mass, TestDirichletProperties::test_dirichlet_sum_to_one, TestDirichletProperties::test_dirichlet_all_positive, TestDirichletProperties::test_dirichlet_mean_equals_normalized_alpha, TestDirichletProperties::test_asymmetric_alpha_encodes_prior_knowledge
- `tests/test_distributed_mcmc.py`: TestDistributedMCMCConfig::test_default_values, TestDistributedMCMCResult::test_creation, TestGPUConfig::test_gpu_info_dataclass
- `tests/test_element_id.py`: test_identified_line_creation, test_identified_line_defaults, test_element_identification_creation, test_element_identification_result_creation, test_element_identification_result_consistency
- `tests/test_hybrid_consensus_2of3.py`: TestHybridIdentifierDefaultPreservation::test_require_both_default_is_true, TestHybridIdentifierDefaultPreservation::test_new_consensus_class_does_not_replace_hybrid_identifier
- `tests/test_line_selection.py`: TestLineSelectorInit::test_default_parameters, TestIsolationFactor::test_isolation_formula, TestScoringFormula::test_score_formula, TestScoringFormula::test_score_with_default_uncertainty, TestLineScore::test_dataclass_creation, TestLineSelectionResult::test_dataclass_creation
- `tests/test_matrix_effects.py`: TestPsRegimeGating::test_ns_defaults_unchanged, TestPsRegimeGating::test_ns_correction_alters_carbon
- `tests/test_no_hidden_x64_enablement.py`: test_jax_identifier_flags_for_does_not_mutate_x64
- `tests/test_partition_function_provider.py`: test_spec_is_cached_compute_once, test_derive_partition_spec_does_not_refit
- `tests/test_preprocessing_low_snr.py`: test_default_baseline_method_is_median_not_als
- `tests/test_quality.py`: TestQualityAssessorInit::test_default_thresholds, TestQualityMetrics::test_dataclass_creation
- `tests/test_spectral_nnls.py`: TestHybridUnionRecallFloor::test_standalone_default_is_count_invariant, TestHybridUnionRecallFloor::test_hybrid_union_arm_defaults_to_recall_favoring_floor, TestSpectralNNLSIdentifier::test_estimated_params_stored
- `tests/test_spectrum_model.py`: test_spectrum_model_legacy_mode_unchanged
- `tests/test_two_region_fit.py`: test_solver_two_region_flag

Full per-file table: test-triage.json (machine-readable).