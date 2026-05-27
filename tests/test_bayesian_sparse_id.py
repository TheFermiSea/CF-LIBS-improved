"""Tests for cflibs.benchmark.bayesian_sparse_id config + helpers."""

from cflibs.benchmark.bayesian_sparse_id import (
    _empty_result,
    bayesian_sparse_config_name,
    bayesian_sparse_workflow_configs,
)


class TestConfigGrid:
    def test_quick_returns_single_minimal_config(self):
        configs = bayesian_sparse_workflow_configs(quick=True)
        assert len(configs) == 1
        c = configs[0]
        assert c["num_warmup"] == 200
        assert c["num_samples"] == 200
        assert c["num_chains"] == 1

    def test_full_grid_has_multiple_configs(self):
        configs = bayesian_sparse_workflow_configs(quick=False)
        assert len(configs) >= 2

    def test_all_configs_have_required_keys(self):
        for quick in (True, False):
            for c in bayesian_sparse_workflow_configs(quick=quick):
                for key in (
                    "num_warmup",
                    "num_samples",
                    "num_chains",
                    "target_accept_prob",
                    "baseline_degree",
                    "k_max",
                    "presence_floor",
                    "posterior_prob_threshold",
                ):
                    assert key in c, f"missing key {key}"

    def test_full_grid_varies_baseline_degree(self):
        configs = bayesian_sparse_workflow_configs(quick=False)
        baseline_degrees = {c["baseline_degree"] for c in configs}
        assert len(baseline_degrees) >= 2

    def test_presence_floor_and_prob_threshold_reasonable(self):
        for c in bayesian_sparse_workflow_configs(quick=False):
            assert 0.0 < c["presence_floor"] < 1.0
            assert 0.0 < c["posterior_prob_threshold"] <= 1.0


class TestConfigName:
    def test_config_name_format(self):
        config = {"baseline_degree": 3, "k_max": 15, "num_samples": 1000}
        name = bayesian_sparse_config_name(config)
        assert name == "bl3_k15_n1000"

    def test_config_name_with_defaults(self):
        name = bayesian_sparse_config_name({})
        assert name == "bl0_k15_n1000"

    def test_config_name_distinct_per_config(self):
        configs = bayesian_sparse_workflow_configs(quick=False)
        names = {bayesian_sparse_config_name(c) for c in configs}
        assert len(names) == len(configs)


class TestEmptyResult:
    def test_empty_result_has_all_rejected(self):
        candidates = ["Fe", "Ca", "Mg"]
        result = _empty_result(candidates, ["Fe"], {"k_max": 15}, "test error")
        assert result.detected_elements == []
        assert len(result.rejected_elements) == 3
        assert len(result.all_elements) == 3
        for eid in result.all_elements:
            assert eid.detected is False
            assert eid.score == 0.0  # NOSONAR — rejected elements get score literally 0.0
            assert eid.metadata["error"] == "test error"

    def test_empty_result_algorithm_tag(self):
        result = _empty_result(["Fe"], [], {}, "err")
        assert result.algorithm == "bayesian_sparse"

    def test_empty_result_preserves_prefiltered_and_config(self):
        result = _empty_result(["Fe", "Ca"], ["Fe"], {"k_max": 5}, "boom")
        assert result.parameters["prefiltered_elements"] == ["Fe"]
        assert result.parameters["config"] == {"k_max": 5}
        assert result.parameters["error"] == "boom"

    def test_empty_result_no_peaks(self):
        result = _empty_result(["Fe"], [], {}, "err")
        assert result.experimental_peaks == []
        assert result.n_peaks == 0
        assert result.n_matched_peaks == 0
        assert result.n_unmatched_peaks == 0

    def test_empty_result_handles_empty_candidate_list(self):
        result = _empty_result([], [], {}, "no candidates")
        assert result.all_elements == []
        assert result.rejected_elements == []
