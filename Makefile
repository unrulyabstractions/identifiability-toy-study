.PHONY: notebook test lint format sanity
notebook:
	uv run jupyter lab
test:
	uv run pytest -q
lint:
	uv run ruff check .
	uv run black --check .
format:
	uv run ruff check --fix .
	uv run black .
sanity:
	uv run python scripts/sanity.py
