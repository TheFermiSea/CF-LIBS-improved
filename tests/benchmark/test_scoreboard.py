"""Tests for the goal-metric scoreboard (bead A1).

Fast and DB-free: the pipeline is monkeypatched for the scoring-math tests,
and the adapter-contract test only touches each adapter's FIRST yield
(skip-with-log datasets are tolerated).
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from cflibs.benchmark import scoreboard as sb
from cflibs.benchmark import scoreboard_registry as reg
from cflibs.benchmark.scoreboard_registry import (
    SpectrumTruth,
    iter_datasets,
    register_dataset,
    registered_names,
)

_ELEMENT_RE = re.compile(r"^[A-Z][a-z]?$")


def _toy_adapter():
    truth = SpectrumTruth(frozenset({"Fe"}))
    yield "toy_0", np.linspace(200.0, 300.0, 10), np.zeros(10), truth


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_round_trip(self, monkeypatch):
        monkeypatch.setattr(reg, "_REGISTRY", {})
        entry = register_dataset("toy", _toy_adapter, tags=("synthetic", "tiny"))
        assert entry.name == "toy"
        assert entry.tags == frozenset({"synthetic", "tiny"})
        assert registered_names() == ["toy"]
        (got,) = list(iter_datasets())
        assert got.adapter_factory is _toy_adapter
        sid, wl, inten, truth = next(got.adapter_factory())
        assert sid == "toy_0"
        assert truth.elements_present == frozenset({"Fe"})

    def test_duplicate_rejected_unless_replace(self, monkeypatch):
        monkeypatch.setattr(reg, "_REGISTRY", {})
        register_dataset("toy", _toy_adapter)
        with pytest.raises(ValueError, match="already registered"):
            register_dataset("toy", _toy_adapter)
        register_dataset("toy", _toy_adapter, replace=True)  # idempotent path

    def test_filters(self, monkeypatch):
        monkeypatch.setattr(reg, "_REGISTRY", {})
        register_dataset("a", _toy_adapter, tags=("real",))
        register_dataset("b", _toy_adapter, tags=("synthetic",))
        assert [e.name for e in iter_datasets(names=["b"])] == ["b"]
        assert [e.name for e in iter_datasets(tags=["real"])] == ["a"]
        assert [e.name for e in iter_datasets()] == ["a", "b"]
        with pytest.raises(KeyError, match="Unknown scoreboard dataset"):
            list(iter_datasets(names=["nope"]))

    def test_tier_registration(self, monkeypatch):
        monkeypatch.setattr(reg, "_REGISTRY", {})
        assert register_dataset("opt", _toy_adapter).tier == "optimization"
        assert register_dataset("held", _toy_adapter, tier="holdout").tier == "holdout"
        assert register_dataset("locked", _toy_adapter, tier="vault").tier == "vault"
        with pytest.raises(ValueError, match="tier"):
            register_dataset("bad", _toy_adapter, tier="secret")

    def test_truth_basis_derivation(self):
        assert SpectrumTruth(frozenset({"Fe"})).composition_basis == "presence_only"
        quantitative = SpectrumTruth(frozenset({"Fe"}), composition_wt={"Fe": 100.0})
        assert quantitative.composition_basis == "element_wt"
        with pytest.raises(ValueError, match="inconsistent"):
            SpectrumTruth(frozenset({"Fe"}), composition_wt=None, composition_basis="element_wt")
        with pytest.raises(ValueError, match="inconsistent"):
            SpectrumTruth(
                frozenset({"Fe"}), composition_wt={"Fe": 1.0}, composition_basis="presence_only"
            )


# ---------------------------------------------------------------------------
# Adapter contract conformance (every registered core adapter's first yield)
# ---------------------------------------------------------------------------


class TestAdapterContract:
    def test_every_core_adapter_first_yield(self, monkeypatch):
        from cflibs.benchmark.adapters_core import register_core_adapters

        monkeypatch.setattr(reg, "_REGISTRY", {})
        register_core_adapters()
        assert registered_names() == [
            "bhvo2_chemcam",
            "aalto",
            "nist_srm_612",
            "nist_steel",
            "synthetic_fixedforward",
        ]
        # The adoption-gate dataset is holdout TIER at registration (the
        # campaign splits derive their refusal sets from this).
        tiers = {e.name: e.tier for e in iter_datasets()}
        assert tiers["bhvo2_chemcam"] == "holdout"
        assert all(t == "optimization" for n, t in tiers.items() if n != "bhvo2_chemcam")
        for entry in iter_datasets():
            gen = entry.adapter_factory()
            first = next(gen, None)
            gen.close()
            if first is None:
                # skip-with-log datasets (e.g. nist placeholders) are tolerated
                continue
            spectrum_id, wavelength, intensity, truth = first
            assert isinstance(spectrum_id, str) and spectrum_id, entry.name
            assert isinstance(wavelength, np.ndarray), entry.name
            assert isinstance(intensity, np.ndarray), entry.name
            assert wavelength.ndim == intensity.ndim == 1, entry.name
            assert wavelength.shape == intensity.shape, entry.name
            assert np.all(np.isfinite(wavelength)), entry.name
            assert isinstance(truth, SpectrumTruth), entry.name
            assert truth.elements_present, entry.name
            for el in truth.elements_present:
                assert _ELEMENT_RE.match(el), f"{entry.name}: bad element symbol {el!r}"
            if truth.composition_wt is not None:
                assert truth.composition_basis == "element_wt", entry.name
                for el, wt in truth.composition_wt.items():
                    assert _ELEMENT_RE.match(el), entry.name
                    assert 0.0 <= wt <= 100.0, f"{entry.name}: {el}={wt} not wt%"
            else:
                assert truth.composition_basis == "presence_only", entry.name
            assert isinstance(truth.notes, str) and truth.notes, entry.name


# ---------------------------------------------------------------------------
# Scoring math (rigged fake pipeline; no DB)
# ---------------------------------------------------------------------------


class _FakeResult:
    def __init__(self, concentrations):
        self.concentrations = concentrations
        self.temperature_K = 9000.0
        self.electron_density_cm3 = 1e17
        self.converged = True
        self.quality_metrics = {"ne_from_stark": 1.0}


class TestScoringMath:
    def test_presence_confusion(self):
        out = sb.presence_confusion(
            {"Fe": 0.5, "Ag": 0.02, "Si": 0.001},
            frozenset({"Fe", "Si"}),
            ["Ag", "Fe", "Si", "W"],
        )
        assert out["tp"] == ["Fe"]
        assert out["fp"] == ["Ag"]
        assert out["fn"] == ["Si"]  # below the 5e-3 presence eps
        assert out["called_present"] == ["Ag", "Fe"]

    def test_precision_recall_f1(self):
        p, r, f1 = sb.precision_recall_f1(1, 1, 1)
        assert p == pytest.approx(0.5)
        assert r == pytest.approx(0.5)
        assert f1 == pytest.approx(0.5)
        assert sb.precision_recall_f1(0, 0, 0) == (0.0, 0.0, 0.0)

    def test_composition_errors(self):
        rmse, signed = sb.composition_errors({"Fe": 0.50, "Si": 0.001}, {"Fe": 50.0, "Si": 20.0})
        assert signed["Fe"] == pytest.approx(0.0)
        assert signed["Si"] == pytest.approx(0.1 - 20.0)
        assert rmse == pytest.approx(np.sqrt((0.0**2 + 19.9**2) / 2.0))

    def test_run_scoreboard_with_fake_pipeline(self, monkeypatch, tmp_path):
        monkeypatch.setattr(reg, "_REGISTRY", {})
        truth = SpectrumTruth(
            frozenset({"Fe", "Si"}),
            composition_wt={"Fe": 50.0, "Si": 20.0},
            notes="rigged",
        )
        wl = np.linspace(200.0, 300.0, 16)

        def adapter():
            yield "s1", wl, np.ones(16), truth
            yield "s2", wl, np.ones(16), truth

        register_dataset("rigged", adapter, tags=("synthetic",))

        def fake_run_pipeline(wavelength, intensity, atomic_db, pipeline, uncertainty_mode="none"):
            if fake_run_pipeline.calls == 0:
                fake_run_pipeline.calls += 1
                # Fe correct, Ag confounder FP, Si dropped below presence eps.
                return _FakeResult({"Fe": 0.5, "Ag": 0.02, "Si": 0.001}), {
                    "n_observations": 7,
                    "stage_timings": {
                        "calibration_s": 1.0,
                        "detection_id_s": 2.0,
                        "stark_ne_s": 0.5,
                        "solve_s": 3.0,
                        "total_s": 6.5,
                    },
                }
            raise ValueError("No usable spectral lines detected for inversion.")

        fake_run_pipeline.calls = 0
        monkeypatch.setattr(sb, "run_pipeline", fake_run_pipeline)

        board = sb.run_scoreboard(atomic_db=None, datasets=["rigged"])
        (ds,) = board["datasets"]
        assert ds["status"] == "ok"
        assert ds["n_run"] == 2 and ds["n_ok"] == 1 and ds["n_failed"] == 1
        # s1: tp=[Fe], fp=[Ag], fn=[Si]; s2 (failed): fn=[Fe, Si].
        m = ds["id_metrics"]
        assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 3)
        assert m["precision"] == pytest.approx(0.5)
        assert m["recall"] == pytest.approx(0.25)
        assert m["f1"] == pytest.approx(2 * 0.5 * 0.25 / 0.75)
        # Composition: only the ok spectrum is scored.
        comp = ds["composition"]
        assert comp["n_scored"] == 1
        assert comp["rmse_wt_median"] == pytest.approx(np.sqrt((0.0 + 19.9**2) / 2.0))
        assert comp["per_element_signed_mean_wt"]["Si"] == pytest.approx(-19.9)
        # Runtime medians come from the production stage_timings diagnostics.
        assert ds["runtime"]["median_solve_s"] == pytest.approx(3.0)
        assert ds["failures"]["s2"].startswith("ValueError")
        # Candidate-set policy: truth UNION confounders.
        rec1, _ = ds["spectra"]
        assert rec1["candidates"] == sorted({"Fe", "Si", *sb.CONFOUNDER_ELEMENTS})

        # Artifacts render and round-trip.
        json_path, md_path = sb.write_artifacts(board, tmp_path)
        assert json_path.exists() and md_path.exists()
        md = md_path.read_text()
        assert "| rigged |" in md
        assert "Candidate-set policy" in md

    def test_sampling_is_seeded_and_deterministic(self, monkeypatch):
        monkeypatch.setattr(reg, "_REGISTRY", {})
        n = 20

        def adapter():
            for i in range(n):
                yield f"s{i:02d}", np.linspace(200, 300, 4), np.ones(4), SpectrumTruth(
                    frozenset({"Fe"})
                )

        register_dataset("big", adapter)

        def fake_run_pipeline(*args, **kwargs):
            return _FakeResult({"Fe": 1.0}), {"n_observations": 1, "stage_timings": {}}

        monkeypatch.setattr(sb, "run_pipeline", fake_run_pipeline)
        board1 = sb.run_scoreboard(atomic_db=None, max_spectra=5, seed=123)
        board2 = sb.run_scoreboard(atomic_db=None, max_spectra=5, seed=123)
        ids1 = [r["spectrum_id"] for r in board1["datasets"][0]["spectra"]]
        ids2 = [r["spectrum_id"] for r in board2["datasets"][0]["spectra"]]
        assert ids1 == ids2
        assert len(ids1) == 5
        assert board1["datasets"][0]["sampled"] is True
        assert board1["datasets"][0]["n_total"] == n

    def test_holdout_and_vault_tiers_are_gated(self, monkeypatch):
        """altitude#7: tier is a dataset property. Default boards exclude
        holdout and vault; --include-holdout admits holdout only; explicitly
        requesting an excluded dataset is a hard error, never a silent skip."""
        monkeypatch.setattr(reg, "_REGISTRY", {})
        register_dataset("opt", _toy_adapter, tags=("real",))
        register_dataset("gate", _toy_adapter, tags=("real",), tier="holdout")
        register_dataset("locked", _toy_adapter, tags=("real",), tier="vault")

        def fake_run_pipeline(*args, **kwargs):
            return _FakeResult({"Fe": 1.0}), {"n_observations": 1, "stage_timings": {}}

        monkeypatch.setattr(sb, "run_pipeline", fake_run_pipeline)

        board = sb.run_scoreboard(atomic_db=None)
        assert [ds["name"] for ds in board["datasets"]] == ["opt"]
        assert board["include_holdout"] is False
        assert board["datasets"][0]["tier"] == "optimization"

        board = sb.run_scoreboard(atomic_db=None, include_holdout=True)
        assert [ds["name"] for ds in board["datasets"]] == ["opt", "gate"]
        assert "--include-holdout" in sb.render_markdown(board)
        assert {ds["name"]: ds["tier"] for ds in board["datasets"]} == {
            "opt": "optimization",
            "gate": "holdout",
        }

        with pytest.raises(ValueError, match="HOLDOUT"):
            sb.run_scoreboard(atomic_db=None, datasets=["gate"])
        # Vault never runs — not even with include_holdout.
        with pytest.raises(ValueError, match="VAULT"):
            sb.run_scoreboard(atomic_db=None, datasets=["locked"], include_holdout=True)
        board = sb.run_scoreboard(atomic_db=None, include_holdout=True)
        assert "locked" not in [ds["name"] for ds in board["datasets"]]
        # Explicitly requesting holdout WITH the flag is allowed.
        board = sb.run_scoreboard(atomic_db=None, datasets=["gate"], include_holdout=True)
        assert [ds["name"] for ds in board["datasets"]] == ["gate"]

    def test_skipped_dataset_renders(self, monkeypatch, tmp_path):
        monkeypatch.setattr(reg, "_REGISTRY", {})

        def empty_adapter():
            yield from ()

        register_dataset("ghost", empty_adapter, tags=("placeholder",))
        board = sb.run_scoreboard(atomic_db=None)
        (ds,) = board["datasets"]
        assert ds["status"] == "skipped"
        assert ds["n_run"] == 0
        md = sb.render_markdown(board)
        assert "skipped" in md
