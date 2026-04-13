default:
    @just --list

uv_cache := "${TMPDIR:-/tmp}/cflibs-uv-cache"

setup:
    mkdir -p "{{uv_cache}}"
    UV_CACHE_DIR="{{uv_cache}}" uv venv .venv --python 3.12
    UV_CACHE_DIR="{{uv_cache}}" uv pip install --python .venv/bin/python -e ".[dev]"

setup-codex:
    mkdir -p "{{uv_cache}}"
    UV_CACHE_DIR="{{uv_cache}}" uv venv .venv --python 3.12
    UV_CACHE_DIR="{{uv_cache}}" uv pip install --python .venv/bin/python -e ".[dev,jax-cpu,hdf5]"

setup-ci:
    mkdir -p "{{uv_cache}}"
    UV_CACHE_DIR="{{uv_cache}}" uv venv .venv --python 3.12
    UV_CACHE_DIR="{{uv_cache}}" uv pip install --python .venv/bin/python -e ".[dev,ci]"

fmt:
    .venv/bin/black cflibs tests

fmt-check:
    .venv/bin/black --check cflibs tests

fmt-ruff:
    .venv/bin/ruff format cflibs tests

fmt-ruff-check:
    .venv/bin/ruff format --check cflibs tests

lint:
    .venv/bin/ruff check cflibs tests

lint-fix:
    .venv/bin/ruff check --fix cflibs tests

typecheck:
    .venv/bin/mypy cflibs

typecheck-ty:
    .venv/bin/ty check cflibs --exit-zero

test:
    .venv/bin/pytest tests/ -v

test-fast:
    .venv/bin/pytest -m "not requires_db and not requires_bayesian and not requires_rust and not requires_jax and not slow" tests/ -v

test-unit:
    .venv/bin/pytest -m "unit and not requires_db and not requires_rust and not requires_jax" tests/ -v

benchmark:
    .venv/bin/pytest tests/ -v --benchmark-only

test-rust:
    cargo test --manifest-path native/cflibs-core/Cargo.toml

test-rust-nextest:
    cargo nextest run --manifest-path native/cflibs-core/Cargo.toml

check:
    just lint
    just typecheck
    just test-fast
