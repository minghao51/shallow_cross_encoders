# Shallow Cross Encoders — common development tasks

default:
    @just --list

# Install dependencies
install:
    uv sync

# Install with dev dependencies
install-dev:
    uv sync --extra dev

# Run all tests (excluding llm + slow)
test:
    uv run pytest tests/ -x -q

# Run quick tests (unit only, excluding slow)
test-quick:
    uv run pytest tests/unit/ -x -q -m "not slow"

# Run unit tests only
test-unit:
    uv run pytest tests/unit/ -x -q

# Run integration tests only
test-integration:
    uv run pytest tests/integration/ -x -q

# Run e2e tests only
test-e2e:
    uv run pytest tests/e2e/ -x -q

# Run benchmark (slow) tests only
test-slow:
    uv run pytest tests/ -x -q -m slow

# Run ALL tests including slow ones
test-full:
    uv run pytest tests/ -x -q -m "not llm"

# Run tests with coverage
test-cov:
    uv run pytest tests/ --cov=src --cov-report=term-missing -q -m "not llm and not slow"

# Lint with ruff
lint:
    uv run ruff check src/ tests/ scripts/

# Format check with ruff
format-check:
    uv run ruff format --check src/ tests/ scripts/

# Auto-fix lint issues
lint-fix:
    uv run ruff check --fix src/ tests/ scripts/

# Type check with mypy
typecheck:
    uv run mypy src/reranker/

# Run lint + typecheck
check: lint typecheck

# Train all strategies sequentially
train-all:
    uv run reranker train hybrid
    uv run reranker train distilled
    uv run reranker train binary
    uv run reranker train late_interaction

# Run quick synthetic benchmark
benchmark-quick:
    uv run benchmarks/run.py synthetic --quick

# Run full benchmark suite
benchmark-full:
    uv run benchmarks/run.py full

# Run benchmark with profiling (memory + CPU)
benchmark-viz:
    uv run reranker benchmark run --quick --profile

# Run benchmark and generate plots
benchmark-compare strategy_a strategy_b results:
    uv run reranker benchmark compare {{strategy_a}} {{strategy_b}} --results {{results}}

# Run YAML sweep benchmark
benchmark-sweep config:
    uv run benchmarks/run_sweep.py --config {{config}}

# Run multi-dataset BEIR sweep (trec-covid, nfcorpus, scidocs, fiqa-qa, arguana)
benchmark-beir:
    uv run scripts/benchmark_beir_multi.py

# Generate all synthetic datasets
generate-data:
    uv run reranker generate pairs --count 100
    uv run reranker generate preferences --count 100
    uv run reranker generate contradictions --count 50

# Evaluate a strategy
eval strategy:
    uv run reranker eval run {{strategy}}

# Check dependency and config health
doctor:
    uv run reranker doctor check

# Serve MkDocs dev server
docs-serve:
    uv run --group docs mkdocs serve

# Build docs site
docs-build:
    uv run --group docs mkdocs build

# Draft changelog entries from recent conventional commits
changelog-draft:
    @echo "=== Recent conventional commits (for CHANGELOG.md) ==="
    @git log --oneline --no-merges -30

# Run pre-commit on all files
pre-commit:
    uv run pre-commit run --all-files
