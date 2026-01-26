.PHONY: notebook lint format sanity run
notebook:
	uv run jupyter lab
run:
	uv run python main.py
lint:
	uv run ruff check .
	uv run black --check .
format:
	uv run ruff check --fix .
	uv run black .
sanity:
	uv run python scripts/sanity.py
