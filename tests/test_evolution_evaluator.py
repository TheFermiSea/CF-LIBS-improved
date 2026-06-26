import pytest
from cflibs.evolution.evaluator import assert_benchmark_relevance


def test_assert_benchmark_relevance_pass():
    diff = """--- a/cflibs/identification/comb.py
+++ b/cflibs/identification/comb.py
@@ -1,1 +1,1 @@
-old
+new"""
    exercised = {"cflibs/identification/comb.py", "cflibs/inversion/solve/bayesian.py"}
    # Should not raise
    assert_benchmark_relevance(diff, exercised)


def test_assert_benchmark_relevance_fail():
    diff = """--- a/cflibs/identification/comb.py
+++ b/cflibs/identification/comb.py
@@ -1,1 +1,1 @@
-old
+new"""
    exercised = {"cflibs/identification/alias.py", "cflibs/inversion/solve/bayesian.py"}
    with pytest.raises(RuntimeError) as excinfo:
        assert_benchmark_relevance(diff, exercised)
    assert "none of these files are exercised by the current benchmark" in str(excinfo.value)


def test_assert_benchmark_relevance_multiple_files():
    diff = """--- a/cflibs/identification/comb.py
+++ b/cflibs/identification/comb.py
--- a/cflibs/identification/hybrid.py
+++ b/cflibs/identification/hybrid.py"""
    exercised = {"cflibs/identification/hybrid.py"}
    # Should pass because hybrid.py is touched and exercised
    assert_benchmark_relevance(diff, exercised)


def test_assert_benchmark_relevance_binary_diff_not_false_rejected():
    """Binary-diff lines must not trigger a false-rejection (M1-20)."""
    diff = "Binary files a/cflibs/data.bin and b/cflibs/data.bin differ"
    assert_benchmark_relevance(diff, {"cflibs/data.bin"})
