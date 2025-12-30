.PHONY: all fetch synth_data preprocess train_tiny eval_tiny test clean help

PYTHON ?= python3
DEVICE ?= cpu

# Default target
all: help

# Fetch official GRU4Rec implementation
fetch:
	$(PYTHON) scripts/fetch_official.py

# Generate synthetic data for CI/testing
synth_data:
	$(PYTHON) scripts/make_synth_data.py --output data/synth_sessions.tsv --n_sessions 1000 --n_items 500

# Preprocess data with temporal split
preprocess:
	$(PYTHON) scripts/preprocess_sessions.py --input data/synth_sessions.tsv --output_dir data --train_ratio 0.8

# Train tiny model for testing
train_tiny: fetch
	$(PYTHON) scripts/run_gru4rec.py train \
		--data data/train.tsv \
		--model results/model_tiny.pt \
		--layers 64 \
		--epochs 3 \
		--batch_size 32 \
		--device $(DEVICE)

# Evaluate tiny model
eval_tiny:
	$(PYTHON) scripts/run_gru4rec.py eval \
		--model results/model_tiny.pt \
		--test data/test.tsv \
		--cutoffs 5 10 20 \
		--device $(DEVICE)

# Run baselines
baselines:
	$(PYTHON) -c "from src.baselines.popularity import PopularityBaseline; \
		from src.baselines.markov import MarkovBaseline; \
		import pandas as pd; \
		train = pd.read_csv('data/train.tsv', sep='\t'); \
		test = pd.read_csv('data/test.tsv', sep='\t'); \
		pop = PopularityBaseline(); pop.fit(train); \
		print('Popularity:', pop.evaluate(test, k=[5,10,20])); \
		mk = MarkovBaseline(); mk.fit(train); \
		print('Markov:', mk.evaluate(test, k=[5,10,20]))"

# Generate comparison report
report:
	$(PYTHON) src/report.py --results_dir results --output results/comparison.png

# Run tests
test:
	$(PYTHON) -m pytest tests/ -v

# CI pipeline (full reproducibility check)
ci: fetch synth_data preprocess train_tiny eval_tiny
	@echo "CI pipeline completed successfully"

# Clean generated files
clean:
	rm -rf vendor/
	rm -f data/*.tsv
	rm -f results/*.pt results/*.json results/*.png
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

help:
	@echo "GRU4Rec Reproduction Study"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  fetch        Clone official GRU4Rec to vendor/"
	@echo "  synth_data   Generate synthetic session data"
	@echo "  preprocess   Split data into train/test (temporal)"
	@echo "  train_tiny   Train a small model for testing"
	@echo "  eval_tiny    Evaluate the tiny model"
	@echo "  baselines    Run popularity and Markov baselines"
	@echo "  report       Generate comparison report"
	@echo "  test         Run pytest"
	@echo "  ci           Full CI pipeline"
	@echo "  clean        Remove generated files"
	@echo ""
	@echo "Options:"
	@echo "  DEVICE=cuda:0  Use GPU (default: cpu)"
