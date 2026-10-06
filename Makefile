export VOLGATE_CONFIG ?= configs/dev.yaml

.PHONY: download prepare base combos gate table evaluate r1 devreport r2 compare hypotheses test

download:
	uv run python scripts/01_download.py

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

r2:
	uv run python scripts/10_r2.py

compare:
	uv run python scripts/11_dev_compare.py

hypotheses:
	uv run python scripts/12_hypotheses.py

test:
	uv run pytest -q
