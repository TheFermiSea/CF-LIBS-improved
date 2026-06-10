"""Split disjointness + structural holdout refusal."""

import json
from pathlib import Path

import pytest

import splits

MANIFEST_PATH = splits.DEFAULT_MANIFEST_PATH


def _manifest():
    if not MANIFEST_PATH.exists():  # pragma: no cover
        pytest.skip(f"committed split manifest missing: {MANIFEST_PATH}")
    return json.loads(Path(MANIFEST_PATH).read_text())


@pytest.mark.unit
def test_committed_manifest_validates():
    manifest = _manifest()
    splits.validate_manifest(manifest)  # raises on tier overlap / target leaks
    assert manifest["seed"] == splits.SPLIT_SEED


@pytest.mark.unit
def test_tiers_are_disjoint_by_spectrum_and_target():
    manifest = _manifest()
    for name, opt_ids in manifest["optimization"].items():
        held = manifest["holdout"].get(name, [])
        assert not set(opt_ids) & set(held), name
        opt_targets = {splits.target_key(name, sid) for sid in opt_ids}
        held_targets = {splits.target_key(name, sid) for sid in held}
        assert not opt_targets & held_targets, name


@pytest.mark.unit
def test_holdout_and_vault_membership():
    manifest = _manifest()
    assert "bhvo2_chemcam" not in manifest["optimization"]
    assert "emslibs2019" not in manifest["optimization"]
    assert len(manifest["holdout"]["bhvo2_chemcam"]) == 4
    assert "gibbons2024" in manifest["vault"]
    assert "gibbons2024" not in manifest["optimization"]
    assert "gibbons2024" not in manifest["holdout"]


@pytest.mark.unit
def test_objective_refuses_holdout_datasets():
    manifest = _manifest()
    with pytest.raises(splits.HoldoutViolation):
        splits.assert_optimization_only(manifest, ["bhvo2_chemcam"])
    with pytest.raises(splits.HoldoutViolation):
        splits.assert_optimization_only(manifest, ["aalto", "emslibs2019"])
    with pytest.raises(splits.HoldoutViolation):
        splits.assert_optimization_only(manifest, ["gibbons2024"])
    # Vault is refused even with holdout privileges.
    with pytest.raises(splits.HoldoutViolation):
        splits.assert_not_vault(["gibbons2024"])
    # Valid request passes.
    splits.assert_optimization_only(manifest, ["aalto", "chemcam_calib"])


@pytest.mark.unit
def test_objective_refuses_holdout_spectrum_ids():
    manifest = _manifest()
    leaked = manifest["holdout"]["chemcam_calib"][0]
    with pytest.raises(splits.HoldoutViolation):
        splits.assert_optimization_only(manifest, ["chemcam_calib"], {"chemcam_calib": [leaked]})
    ok = manifest["optimization"]["chemcam_calib"][:3]
    splits.assert_optimization_only(manifest, ["chemcam_calib"], {"chemcam_calib": ok})


@pytest.mark.unit
def test_evaluate_overrides_holdout_needs_explicit_flag():
    import objective as objective_mod

    manifest = _manifest()
    ctx = objective_mod.EvalContext(
        db_path=Path("/nonexistent.db"),
        manifest=manifest,
        datasets=("bhvo2_chemcam",),
    )
    # The guard fires BEFORE any data/DB access.
    with pytest.raises(splits.HoldoutViolation):
        objective_mod.evaluate_overrides({}, ctx, section="holdout", allow_restricted=False)
    with pytest.raises(splits.HoldoutViolation):
        objective_mod.evaluate_overrides({}, ctx, section="optimization")
    with pytest.raises(splits.HoldoutViolation):
        objective_mod.evaluate_overrides(
            {}, ctx, section="holdout", datasets=("gibbons2024",), allow_restricted=True
        )


@pytest.mark.unit
def test_target_key():
    assert splits.target_key("chemcam_calib", "chemcam/AGV2/rep1") == "AGV2"
    assert splits.target_key("chemcam_calib", "chemcam/AGV2/rep2") == "AGV2"
    assert splits.target_key("silva2022", "silva2022/field1/sample1") == (
        "silva2022/field1/sample1"
    )
    assert splits.target_key("csa_planetary", "csa/large200/Anorthosite2120") == (
        splits.target_key("csa_planetary", "csa/subset1000/Anortho2120")
    )
