# CF-LIBS Agent Notes

This file is a concise companion to `AGENTS.md` and should stay aligned with repo-verified workflows.

## Core Setup
- `uv venv --python 3.12`
- `pip install -e ".[dev]"`
- `uv pip install -e ".[local]"`
- `uv pip install -e ".[cluster]"`

## Quality Gates
- `ruff check cflibs/ tests/`
- `black --check cflibs/`
- `mypy cflibs/`
- `pytest tests/ -v`
- `pytest tests/ -v --benchmark-only`
- `JAX_PLATFORMS=cpu pytest tests/`
- `sphinx-build -b html docs docs/_build/html`

## CLI Workflows
- `cflibs generate-db`
- `cflibs forward examples/config_example.yaml --output spectrum.csv`
- `cflibs invert spectrum.csv --elements Fe Cu --config examples/inversion_config_example.yaml`
- `cflibs generate-manifold examples/manifold_config_example.yaml --progress`

## Data And Validation Workflows
- `python datagen_v2.py`
- `nohup python datagen_v2.py &`
- `python scripts/validate_nist_parity.py --element Fe --T 0.8 --ne 1e17 --wl-min 220 --wl-max 265 --resolving-power 1000`
- `python scripts/run_nist_validation.py --db ASD_da/libs_production.db --output output/validation/nist_crosscheck_report.json`
- `python scripts/validate_real_data.py --datasets steel_245nm FeNi_380nm --no-plots`
- `python scripts/calibrate_alias.py --db-path ASD_da/libs_production.db --data-dir data --output-dir output/calibration`
- `python scripts/generate_model_library.py chunk --chunk-id 0 --n-chunks 8 --output-dir output/model_library`
- `python scripts/generate_model_library.py consolidate --output-dir output/model_library`
- `python scripts/generate_model_library.py build-index --output-dir output/model_library`
- `python scripts/generate_model_library.py submit --n-chunks 32 --output-dir output/model_library`

## Cluster Notes
- `mpirun -np 3 --hostfile hosts.txt python manifold-generator.py`
- `srun -N 3 --gpus-per-node=1 python manifold-generator.py`
