#!/usr/bin/env bash
# Reproduce all experiments for:
# "Cost-Sensitive Online Window Size Selection for Portfolio Management"
# Yi-Chen Liu, Chung-Han Hsieh (2024)
#
# Experiments 1–3 use synthetic data (no API keys required).
# Experiments 4–5 require DATALOADER_API_KEY and MASSIVE_API_KEY in .env.
#
# Usage:
#   bash reproduce.sh               # all experiments
#   bash reproduce.sh synthetic     # experiments 1, 2, 3 only

set -o errexit

echo "=== Installing dependencies ==="
pip install -r requirements.txt -q

echo "=== Running tests ==="
python -m pytest tests/ -v --tb=short

echo "=== Experiment 1: Synthetic regime-shift comparison ==="
python -m exp.exp1

echo "=== Experiment 2: Robustness to regime-switch timing and expert-set size ==="
python -m exp.exp2

echo "=== Experiment 3: Comparison with online portfolio-selection algorithms (synthetic) ==="
python -m exp.exp3

if [ "${1:-}" = "synthetic" ]; then
  echo "Stopping after synthetic experiments (--synthetic flag)."
  exit 0
fi

echo "=== Experiment 4: DJIA historical backtest ==="
python -m exp.exp4

echo "=== Experiment 5: S&P 500 historical backtest (reduced universe — see results/RESULTS.md) ==="
python -m exp.exp5

echo "=== All experiments complete. Results in results/ ==="
