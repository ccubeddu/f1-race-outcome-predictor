.PHONY: setup test lint data practice audit

setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -e '.[ml,dev]'

test:
	.venv/bin/pytest

lint:
	.venv/bin/ruff check .

data:
	.venv/bin/f1predict data sync --start 2018 --end 2025

practice:
	.venv/bin/f1predict data sync-practice --start 2018 --end 2025

audit:
	.venv/bin/f1predict data audit
