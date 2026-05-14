.PHONY: install-dev test lint benchmark-basic benchmark-optional build release-check release-check-python clean-artifacts

install-dev:
	python -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

benchmark-basic:
	humanize-pl-benchmark --engines basic --mode standard --allow-fallback --fail-on-status

benchmark-optional:
	humanize-pl-benchmark --engines nlp,hybrid --mode standard --offline-models --require-models --fail-on-status

build:
	python -m build --wheel --no-isolation

release-check: test lint benchmark-basic build

release-check-python:
	humanize-pl-release-check

clean-artifacts:
	rm -rf docs_tests/results dist build *.egg-info .pytest_cache .ruff_cache
