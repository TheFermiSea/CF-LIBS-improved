import numpy as np
from typing import Dict
from cflibs.inversion.common.result_base import (
    ResultTableMixin,
    StatisticsMixin,
    TABLE_SEP,
    TABLE_HEADER,
)


class DummyResult(ResultTableMixin):
    def __init__(self, conc_mean: Dict[str, float], conc_std: Dict[str, float]):
        self.concentrations_mean = conc_mean
        self.concentrations_std = conc_std


def test_result_table_mixin_header_footer():
    dummy = DummyResult({}, {})
    header = dummy._format_header("Test Title")
    assert "Test Title" in header
    assert TABLE_HEADER in header

    sep = dummy._format_separator()
    assert sep == TABLE_SEP

    footer = dummy._format_footer()
    assert footer == TABLE_HEADER


def test_result_table_mixin_param_row():
    dummy = DummyResult({}, {})
    row = dummy._format_param_row("T [eV]", 1.2345, 0.1234, ci=(1.1, 1.4))
    assert "T [eV]" in row
    assert "1.2345" in row
    assert "0.1234" in row
    assert "[1.1000, 1.4000]" in row

    row_no_ci = dummy._format_param_row("T [eV]", 1.2345, 0.1234, include_ci=False)
    assert "1.2345" in row_no_ci
    assert "[1.1" not in row_no_ci  # Only verify that CI brackets are missing


def test_result_table_mixin_param_row_exp():
    dummy = DummyResult({}, {})
    row = dummy._format_param_row_exp("Ne [cm-3]", 1.23e17, 0.12e17, ci=(1.1e17, 1.4e17))
    assert "Ne [cm-3]" in row
    assert "1.23e+17" in row

    row_no_std_ci = dummy._format_param_row_exp("Ne [cm-3]", 1.23e17, include_ci=False)
    assert "1.23e+17" in row_no_std_ci
    assert "[1.1" not in row_no_std_ci


def test_result_table_mixin_param_row_int():
    dummy = DummyResult({}, {})
    row = dummy._format_param_row_int("T [K]", 12345, 123, ci=(12000, 12500))
    assert "T [K]" in row
    assert "12345" in row
    assert "123" in row
    assert "[12000, 12500]" in row

    row_no_ci = dummy._format_param_row_int("T [K]", 12345, 123, include_ci=False)
    assert "[12000" not in row_no_ci


def test_result_table_mixin_concentration_table():
    conc_mean = {"Fe": 0.5, "Ni": 0.5}
    conc_std = {"Fe": 0.05, "Ni": 0.05}
    conc_ci = {"Fe": (0.4, 0.6), "Ni": (0.4, 0.6)}

    dummy = DummyResult(conc_mean, conc_std)

    # Use the attributes to call the function to simulate expected use-case.
    # We pass it explicitly since `_format_concentration_table` expects conc_mean and conc_std as parameters.
    # Note: the mixin docs say "Classes using this mixin should have attributes: concentrations_mean, concentrations_std"
    # Although `_format_concentration_table` itself currently takes them as arguments, we fulfill the contract.
    lines = dummy._format_concentration_table(
        dummy.concentrations_mean, dummy.concentrations_std, conc_ci
    )

    assert len(lines) > 0
    assert any("Fe" in line and "0.5000" in line for line in lines)
    assert any("Ni" in line and "0.5000" in line for line in lines)

    # Test without CI
    lines_no_ci = dummy._format_concentration_table(
        dummy.concentrations_mean, dummy.concentrations_std, include_ci=False
    )
    assert not any("[0.4" in line for line in lines_no_ci)

    # Test without passing conc_ci but with include_ci=True
    lines_auto_ci = dummy._format_concentration_table(
        dummy.concentrations_mean, dummy.concentrations_std, include_ci=True
    )
    assert not any("[0.4" in line for line in lines_auto_ci)


def test_statistics_mixin_compute_ci():
    samples = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # Test < 2 samples
    ci_empty = StatisticsMixin.compute_ci(np.array([]))
    assert np.isnan(ci_empty[0]) and np.isnan(ci_empty[1])

    ci_one = StatisticsMixin.compute_ci(np.array([1]))
    assert np.isnan(ci_one[0]) and np.isnan(ci_one[1])

    # Test standard computation
    ci_90 = StatisticsMixin.compute_ci(samples, level=0.90)
    assert ci_90[0] < ci_90[1]

    # Since we test exactly how np.percentile behaves for these edges:
    expected_lower = np.percentile(samples, 5.0)
    expected_upper = np.percentile(samples, 95.0)
    assert np.isclose(ci_90[0], expected_lower)
    assert np.isclose(ci_90[1], expected_upper)


def test_statistics_mixin_compute_ci_68():
    samples = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ci_68 = StatisticsMixin.compute_ci_68(samples)

    expected_lower = np.percentile(samples, 16.0)
    expected_upper = np.percentile(samples, 84.0)
    assert np.isclose(ci_68[0], expected_lower)
    assert np.isclose(ci_68[1], expected_upper)


def test_statistics_mixin_compute_ci_95():
    samples = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ci_95 = StatisticsMixin.compute_ci_95(samples)

    expected_lower = np.percentile(samples, 2.5)
    expected_upper = np.percentile(samples, 97.5)
    assert np.isclose(ci_95[0], expected_lower)
    assert np.isclose(ci_95[1], expected_upper)


def test_statistics_mixin_compute_quantiles():
    samples = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # Test < 2 samples
    q_empty = StatisticsMixin.compute_quantiles(np.array([1]))
    assert all(np.isnan(x) for x in q_empty)

    # Test valid computation
    q = StatisticsMixin.compute_quantiles(samples)
    assert len(q) == 4

    expected_q025 = np.percentile(samples, 2.5)
    expected_q16 = np.percentile(samples, 16.0)
    expected_q84 = np.percentile(samples, 84.0)
    expected_q975 = np.percentile(samples, 97.5)

    assert np.isclose(q[0], expected_q025)
    assert np.isclose(q[1], expected_q16)
    assert np.isclose(q[2], expected_q84)
    assert np.isclose(q[3], expected_q975)
