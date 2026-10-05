export VOLGATE_CONFIG ?= configs/dev.yaml

.PHONY: prepare base combos gate table evaluate r1 devreport test

prepare:
	uv run python scripts/02_prepare.py

base:
	uv run python scripts/03_base_forecasts.py

combos:
	uv run python scripts/04_combinations.py

gate:
	uv run python scripts/05_gate.py

table:
	uv run python scripts/06_fz0_table.py

evaluate:
	uv run python scripts/07_evaluate.py

r1:
	uv run python scripts/08_r1.py

devreport:
	uv run python scripts/09_dev_report.py

test:
	uv run pytest -q
