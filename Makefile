PYTHON = python

.PHONY: help setup data test backtest run clean

help:
	@echo "Available commands:"
	@echo "  make setup     - Install Python dependencies"
	@echo "  make data      - Generate synthetic cohort dataset"
	@echo "  make test      - Run edge case tests"
	@echo "  make backtest  - Run temporal holdout backtesting"
	@echo "  make run       - Launch Streamlit stakeholder dashboard"

setup:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) -m data.synthetic_data_generator --output data/synthetic_cohort.csv --students 500 --weeks 16 --seed 42

test:
	$(PYTHON) -m pytest tests/test_edge_cases.py -v

backtest:
	$(PYTHON) -m src.backtest

run:
	$(PYTHON) -m streamlit run dashboard/app.py
