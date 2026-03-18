.PHONY: tests, lints, docs, format

tests:
	uv run pytest -v --doctest-modules --cov=./sbijax --cov-report=xml sbijax

lints:
	uv run ruff check sbijax examples

format:
	uv run ruff check --fix sbijax examples
	uv run ruff format sbijax examples

docs:
	cd docs && make html
