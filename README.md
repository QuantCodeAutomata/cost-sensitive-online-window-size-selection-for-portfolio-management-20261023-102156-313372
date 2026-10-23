# Cost-Sensitive Online Window Size Selection for Portfolio Management

**Paper:** Cost-Sensitive Online Window Size Selection for Portfolio Management  
**Authors:** Yi-Chen Liu, Chung-Han Hsieh  
**Year:** 2026  
**arXiv:** https://arxiv.org/abs/2609.29887

---

## Overview

This repository replicates the experiments from the paper. The framework maintains a pool
of sliding-window mean-variance portfolio **experts** (window sizes in {5, 10, 21, …, 126}
trading days) and combines them via the **Hedge** or **Fixed Share** online learning
algorithms, incorporating proportional transaction costs (c = 0.001) into both the expert
objectives and the tracking-regret definition (Section 2 / Problem 2.4).

The experiments cover:
1. Synthetic regime-shift comparison of Hedge vs. Fixed Share (Section 4.1, Figures 1–2).
2. Robustness to alternative regime-switch timings and an eight-expert set (Appendix B.2).
3. Comparison with Universal Portfolio, EG, PAMR, CWMR, and OLMAR on synthetic data (Appendix B.3).
4. Historical backtest on 30 Dow Jones Industrial Average constituents + BIL (Appendix B.4, Table 1).
5. Historical backtest on fixed S&P 500 constituents + BIL (Appendix B.5, Table 3).

---

## Repository layout

```
.
├── README.md
├── reproduce.sh            # Master reproduction script
├── requirements.txt
├── .gitignore
├── .env.example            # API key placeholders
├── src/                    # Shared library modules
│   ├── config.py           # Constants derived from the paper
│   ├── experts.py          # Cost-sensitive MV expert (Problem 2.1)
│   ├── online_learning.py  # Hedge and Fixed Share (Section 3.2)
│   ├── regret.py           # Tracking regret and DP comparator (Section 2.2)
│   ├── metrics.py          # Performance metrics (Appendix B.1)
│   ├── baselines.py        # UP, EG, PAMR, CWMR, OLMAR
│   └── plotting.py         # Figures
├── exp/                    # Experiment runners
│   ├── exp1.py             # Synthetic regime-shift (Figures 1–2)
│   ├── exp2.py             # Robustness variants (Figures 3–4)
│   ├── exp3.py             # Comparison with baselines – synthetic (Figure 5)
│   ├── exp4.py             # DJIA backtest (Figures 6–9, Table 1)
│   └── exp5.py             # S&P 500 backtest (Figures 10–13, Table 3)
├── data/
│   ├── generate_synthetic.py  # Synthetic DGP (Section 4.1)
│   └── data_loader.py         # QuantX Dataloader wrapper
├── tests/
│   └── test_core.py
└── results/
    └── RESULTS.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in the API keys (required for experiments 4–5):

```bash
cp .env.example .env
# edit .env and set DATALOADER_API_KEY and MASSIVE_API_KEY
```

---

## Running experiments

Each experiment is a standalone script run from the repository root:

```bash
# Synthetic experiments (no API keys required)
python -m exp.exp1
python -m exp.exp2
python -m exp.exp3

# Historical backtests (require DATALOADER_API_KEY and MASSIVE_API_KEY in .env)
python -m exp.exp4   # DJIA — 30 stocks, 2021-03-04 to 2024-02-25
python -m exp.exp5   # S&P 500 — reduced to 100 stocks (see RESULTS.md for details)

# Full 481-stock S&P 500 (very slow — see RESULTS.md)
python -m exp.exp5 --full
```

Reproduce everything:

```bash
bash reproduce.sh
```

---

## Running tests

```bash
python -m pytest tests/ -v
```

---

## Results

All output files (plots, CSV tables, JSON summaries) are saved in `results/`.  
See `results/RESULTS.md` for a detailed comparison with the paper's reported values and a
list of deviations.

---

## Data sources

Market data: QuantX Dataloader — https://quantx.limex.pro/

Price history in this project is loaded through the QuantX Dataloader, a
research-ready market-data service available through a REST API and as an
MCP server for AI agents. It is built on market data from Massive
(https://massive.com/) and prepares every stock history for backtesting:
  - prices adjusted for dividends, ready for total-return research;
  - one continuous history when a company changes its ticker;
  - clean separation when a ticker has belonged to different companies;
  - careful corporate-action handling and fine-grained data polishing.

Get your DATALOADER_API_KEY at https://quantx.limex.pro/ and your
MASSIVE_API_KEY at https://massive.com/.
