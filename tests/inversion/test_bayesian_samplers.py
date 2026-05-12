"""Unit tests for the T1-6 ``Sampler`` Protocol and back-compat sampler aliases."""

from __future__ import annotations

import inspect

import pytest

from cflibs.inversion.solve.bayesian.samplers import (
    DynestyNestedSampler,
    MCMCSampler,
    NestedSampler,
    NumPyroNUTSSampler,
    Sampler,
    SamplerResult,
)


def test_numpyro_alias_is_mcmc_sampler():
    """The new canonical name aliases the legacy class."""
    assert NumPyroNUTSSampler is MCMCSampler


def test_dynesty_alias_is_nested_sampler():
    """The new canonical name aliases the legacy class."""
    assert DynestyNestedSampler is NestedSampler


def test_legacy_mcmc_sampler_run_signature_preserved():
    """``MCMCSampler.run`` keeps the legacy keyword set."""
    sig = inspect.signature(MCMCSampler.run)
    params = set(sig.parameters)
    expected = {
        "self",
        "observed",
        "num_warmup",
        "num_samples",
        "num_chains",
        "seed",
        "target_accept_prob",
        "max_tree_depth",
        "progress_bar",
    }
    assert expected.issubset(params), f"missing legacy MCMCSampler.run params: {expected - params}"


def test_legacy_nested_sampler_run_signature_preserved():
    """``NestedSampler.run`` keeps the legacy keyword set."""
    sig = inspect.signature(NestedSampler.run)
    params = set(sig.parameters)
    expected = {
        "self",
        "observed",
        "nlive",
        "dlogz",
        "sample",
        "bound",
        "seed",
        "maxiter",
        "maxcall",
        "verbose",
    }
    assert expected.issubset(
        params
    ), f"missing legacy NestedSampler.run params: {expected - params}"


def test_samplers_expose_fit_method():
    """Both samplers expose the new :meth:`Sampler.fit` adapter."""
    assert callable(getattr(MCMCSampler, "fit", None))
    assert callable(getattr(NestedSampler, "fit", None))


def test_sampler_protocol_runtime_check_passes_via_duck_typing():
    """A class with ``fit(model, data, **kwargs)`` satisfies the runtime Protocol."""

    class StubSampler:
        def fit(self, model, data, **kwargs):  # noqa: D401, ARG002
            return SamplerResult(
                posterior_samples={"x": []},  # type: ignore[arg-type]
                log_evidence=None,
                diagnostics={},
                metadata={},
            )

    assert isinstance(StubSampler(), Sampler)


def test_sampler_protocol_runtime_check_rejects_missing_fit():
    """Objects without a ``fit`` method must not pass the runtime check."""

    class Empty:
        pass

    assert not isinstance(Empty(), Sampler)


def test_sampler_result_dataclass_shape():
    """``SamplerResult`` is a frozen dataclass with the documented fields."""
    res = SamplerResult(
        posterior_samples={},
        log_evidence=None,
        diagnostics={"r_hat": {"T_eV": 1.0}},
        metadata={"sampler": "stub"},
    )
    assert res.log_evidence is None
    assert res.metadata["sampler"] == "stub"
    with pytest.raises(Exception):  # noqa: PT011 - dataclasses.FrozenInstanceError
        res.metadata = {"x": 1}  # type: ignore[misc]
