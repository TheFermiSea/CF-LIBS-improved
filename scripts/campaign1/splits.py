"""Campaign 1 dataset splits: optimization vs holdout vs vault (design 2.1).

Tier rules (docs/audit/2026-06-10-goalfirst/optimization-program-design.md):

- **Optimization set** (fitness; queried freely): ``aalto`` (all),
  ``synthetic_fixedforward`` (seeded 74/288 sample),
  ``chemcam_calib``/``csa_planetary``/``silva2022`` train splits
  (~60% of *targets*, never of spectra — multiple spectra of one target are
  correlated, splitting by spectrum leaks truth).
- **Holdout** (adoption gate only; every query ledger-logged):
  ``bhvo2_chemcam`` (all 4), ``emslibs2019`` (all), plus the held-out ~40%
  target splits of the three datasets above.
- **Vault** (end-of-program only): ``gibbons2024``. NEVER evaluated by this
  tooling — :class:`HoldoutViolation` guards refuse it even in holdout mode.

Splits are seeded, persisted as explicit spectrum-id lists under
``docs/benchmarks/manifests/`` and embedded into every study's
``frozen_manifest.json``. The objective refuses holdout/vault material
structurally (:func:`assert_optimization_only` / :func:`assert_not_vault`).

CLI::

    PYTHONPATH=$PWD python scripts/campaign1/splits.py build \
        [--output docs/benchmarks/manifests/campaign1-splits-v1.json]
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_VERSION = "campaign1-splits-v1"
DEFAULT_MANIFEST_PATH = (
    REPO_ROOT / "docs" / "benchmarks" / "manifests" / (MANIFEST_VERSION + ".json")
)

SPLIT_SEED = 20260610
TRAIN_FRACTION = 0.6
SYNTHETIC_SAMPLE_SIZE = 74  # seeded 74/288 sample per the committed baseline

#: Datasets fully inside the optimization set.
FULL_OPTIMIZATION_DATASETS = ("aalto",)
#: Datasets split ~60/40 by target identity (train -> optimization, rest -> holdout).
TARGET_SPLIT_DATASETS = ("chemcam_calib", "csa_planetary", "silva2022")
#: Datasets entirely holdout (the adoption gate; bhvo2 is the headline number).
HOLDOUT_ONLY_DATASETS = ("bhvo2_chemcam", "emslibs2019")
#: Vault: never measured by any tuning loop. One run, ever, by a human.
VAULT_DATASETS = ("gibbons2024",)

OPTIMIZATION_DATASETS = (
    FULL_OPTIMIZATION_DATASETS + ("synthetic_fixedforward",) + (TARGET_SPLIT_DATASETS)
)


class HoldoutViolation(RuntimeError):
    """Raised when holdout/vault material is requested by the optimization loop."""


_HAND_SAMPLE_RE = re.compile(r"^(hand sample\d+)\b")


def target_key(dataset: str, spectrum_id: str) -> str:
    """Sample/target identity for one spectrum (the split unit, design 2.1)."""
    if dataset == "chemcam_calib":
        # "chemcam/{target}/{rep}" — replicates of one target stay together.
        parts = spectrum_id.split("/")
        if len(parts) < 3:
            raise ValueError(f"Unexpected chemcam_calib spectrum id {spectrum_id!r}")
        return parts[1]
    if dataset == "csa_planetary":
        # "csa/{set}/{stem}" — the same target appears in multiple sets and as
        # "hand sampleN ..." variants; normalize via the adapter's own alias map.
        from cflibs.benchmark.datasets.csa_planetary import SUBSET_NAME_ALIASES

        stem = spectrum_id.split("/")[-1]
        stem = SUBSET_NAME_ALIASES.get(stem, stem)
        match = _HAND_SAMPLE_RE.match(stem)
        return match.group(1) if match else stem
    # silva2022: one spectrum per sample id; everything else: identity.
    return spectrum_id


def _register_adapters() -> None:
    from cflibs.benchmark.adapters_core import register_core_adapters
    from cflibs.benchmark.adapters_extended import register_extended_adapters

    register_core_adapters()
    register_extended_adapters()


def _dataset_ids(name: str) -> list[str]:
    from cflibs.benchmark.scoreboard_registry import iter_datasets

    entries = list(iter_datasets(names=[name]))
    if not entries:
        raise KeyError(f"Dataset {name!r} not registered")
    ids = [sid for sid, _wl, _inten, _truth in entries[0].adapter_factory()]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Dataset {name!r} yielded duplicate spectrum ids")
    return ids


def build_splits(seed: int = SPLIT_SEED) -> dict[str, Any]:
    """Generate the split manifest by iterating the registered adapters."""
    import numpy as np

    _register_adapters()
    optimization: dict[str, list[str]] = {}
    holdout: dict[str, list[str]] = {}
    targets: dict[str, dict[str, list[str]]] = {}

    for name in FULL_OPTIMIZATION_DATASETS:
        optimization[name] = _dataset_ids(name)

    synth_ids = _dataset_ids("synthetic_fixedforward")
    if len(synth_ids) > SYNTHETIC_SAMPLE_SIZE:
        rng = np.random.default_rng(seed)
        idx = sorted(
            int(i) for i in rng.choice(len(synth_ids), size=SYNTHETIC_SAMPLE_SIZE, replace=False)
        )
        optimization["synthetic_fixedforward"] = [synth_ids[i] for i in idx]
    else:
        optimization["synthetic_fixedforward"] = synth_ids

    for name in TARGET_SPLIT_DATASETS:
        ids = _dataset_ids(name)
        by_target: dict[str, list[str]] = {}
        for sid in ids:
            by_target.setdefault(target_key(name, sid), []).append(sid)
        unique_targets = sorted(by_target)
        rng = np.random.default_rng([seed, len(name), *map(ord, name)])
        rng.shuffle(unique_targets)
        n_train = int(round(TRAIN_FRACTION * len(unique_targets)))
        train_targets = set(unique_targets[:n_train])
        optimization[name] = sorted(sid for tgt in train_targets for sid in by_target[tgt])
        holdout[name] = sorted(sid for tgt in unique_targets[n_train:] for sid in by_target[tgt])
        targets[name] = {
            "train": sorted(train_targets),
            "holdout": sorted(set(unique_targets) - train_targets),
        }

    for name in HOLDOUT_ONLY_DATASETS:
        holdout[name] = _dataset_ids(name)

    vault = {name: _dataset_ids(name) for name in VAULT_DATASETS}

    return {
        "version": MANIFEST_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "train_fraction": TRAIN_FRACTION,
        "synthetic_sample_size": SYNTHETIC_SAMPLE_SIZE,
        "rule": (
            "Splits are by sample/target identity, never by spectrum "
            "(design 2.1). bhvo2_chemcam and emslibs2019 are holdout-only; "
            "gibbons2024 is vault (never evaluated by campaign tooling)."
        ),
        "optimization": optimization,
        "holdout": holdout,
        "vault": vault,
        "targets": targets,
        "counts": {
            "optimization": {k: len(v) for k, v in optimization.items()},
            "holdout": {k: len(v) for k, v in holdout.items()},
            "vault": {k: len(v) for k, v in vault.items()},
        },
    }


def load_manifest(path: Path | str = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    """Load and sanity-check a committed split manifest."""
    manifest = json.loads(Path(path).read_text())
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError(
            f"Split manifest version {manifest.get('version')!r} != {MANIFEST_VERSION!r}"
        )
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Structural invariants: tier disjointness and target-identity hygiene."""
    optimization = manifest["optimization"]
    holdout = manifest["holdout"]
    vault = manifest["vault"]
    for name in VAULT_DATASETS:
        if name in optimization or name in holdout:
            raise HoldoutViolation(f"Vault dataset {name!r} appears in a queryable tier")
    for name in HOLDOUT_ONLY_DATASETS:
        if name in optimization:
            raise HoldoutViolation(f"Holdout-only dataset {name!r} appears in optimization")
    for name, opt_ids in optimization.items():
        overlap = set(opt_ids) & set(holdout.get(name, []))
        if overlap:
            raise HoldoutViolation(
                f"{name}: {len(overlap)} spectrum ids in BOTH optimization and holdout"
            )
        opt_targets = {target_key(name, sid) for sid in opt_ids}
        held_targets = {target_key(name, sid) for sid in holdout.get(name, [])}
        shared = opt_targets & held_targets
        if shared:
            raise HoldoutViolation(
                f"{name}: target identities shared across tiers: {sorted(shared)[:5]}"
            )
    for name, ids in vault.items():
        for tier in (optimization, holdout):
            for other, other_ids in tier.items():
                if other == name and set(ids) & set(other_ids):
                    raise HoldoutViolation(f"Vault ids of {name!r} leaked into a tier")


def assert_not_vault(datasets: Iterable[str]) -> None:
    """Vault datasets are refused unconditionally (design 2.1)."""
    requested = set(datasets)
    forbidden = requested & set(VAULT_DATASETS)
    if forbidden:
        raise HoldoutViolation(
            f"VAULT dataset(s) {sorted(forbidden)} can never be evaluated by campaign "
            "tooling (design 2.1: gibbons2024 is the end-of-program figure)."
        )


def assert_optimization_only(
    manifest: Mapping[str, Any],
    datasets: Iterable[str],
    spectrum_ids: Optional[Mapping[str, Iterable[str]]] = None,
) -> None:
    """Refuse any holdout-tagged dataset or holdout spectrum id.

    This is the structural guard the objective calls on EVERY evaluation:
    BHVO-2 and emslibs2019 (and the 40% holdout target splits) never enter
    the optimization loop.
    """
    requested = list(datasets)
    assert_not_vault(requested)
    optimization = manifest["optimization"]
    holdout = manifest["holdout"]
    for name in requested:
        if name in HOLDOUT_ONLY_DATASETS:
            raise HoldoutViolation(
                f"Dataset {name!r} is HOLDOUT-only (adoption gate); the optimization "
                "objective refuses it."
            )
        if name not in optimization:
            raise HoldoutViolation(
                f"Dataset {name!r} is not in the optimization tier of the split manifest."
            )
    if spectrum_ids is not None:
        for name, ids in spectrum_ids.items():
            held = set(holdout.get(name, []))
            leaked = set(ids) & held
            if leaked:
                raise HoldoutViolation(
                    f"{name}: {len(leaked)} requested spectrum ids belong to the "
                    f"holdout split (e.g. {sorted(leaked)[:3]})."
                )
            allowed = set(manifest["optimization"].get(name, []))
            unknown = set(ids) - allowed
            if unknown:
                raise HoldoutViolation(
                    f"{name}: {len(unknown)} requested spectrum ids are outside the "
                    f"optimization split (e.g. {sorted(unknown)[:3]})."
                )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Generate the split manifest from the adapters")
    build.add_argument("--seed", type=int, default=SPLIT_SEED)
    build.add_argument("--output", type=Path, default=DEFAULT_MANIFEST_PATH)
    args = parser.parse_args(argv)

    if args.command == "build":
        import cflibs

        print(f"cflibs={Path(cflibs.__file__).resolve().parent}")
        manifest = build_splits(seed=args.seed)
        validate_manifest(manifest)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2))
        print(f"Wrote {args.output}")
        for tier in ("optimization", "holdout", "vault"):
            print(f"  {tier}: {manifest['counts'][tier]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
