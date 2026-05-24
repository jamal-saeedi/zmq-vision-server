.PHONY: install install-torch install-tf test lint format type-check clean help

## install: Install core + dev dependencies (no ML backend)
install:
	pip install -e ".[dev]"

## install-torch: Install PyTorch backend + dev dependencies
install-torch:
	pip install -e ".[torch,dev]"

## install-tf: Install TensorFlow backend + dev dependencies
install-tf:
	pip install -e ".[tf,dev]"

## test: Run the full test suite with coverage
test:
	pytest tests/ -v --cov=zmq_inference --cov-report=term-missing

## test-fast: Run tests without coverage (faster iteration)
test-fast:
	pytest tests/ -v -x

## lint: Run ruff linter on package and tests
lint:
	ruff check zmq_inference/ tests/

## format: Auto-format code with black
format:
	black zmq_inference/ tests/

## format-check: Verify formatting without modifying files
format-check:
	black --check zmq_inference/ tests/

## type-check: Run mypy static type analysis
type-check:
	mypy zmq_inference/

## clean: Remove build artifacts, caches, and generated files
clean:
	find . -type d -name __pycache__ | xargs rm -rf
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info htmlcov coverage.xml .coverage

## help: Show this help message
help:
	@grep -E '^## ' Makefile | sed 's/## //'
