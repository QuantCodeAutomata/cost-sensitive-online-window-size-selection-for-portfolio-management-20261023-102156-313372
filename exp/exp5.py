"""
Experiment 5: S&P 500 constituent historical backtest (Appendix B.5).

Reproduces Table 3, Figures 10–13 from the paper.

Universe: Fixed S&P 500 constituents from Appendix C + BIL (risk-free asset).
Period: 2020-01-02 through 2026-08-01 (the last available trading day prior
        to the stated end date is used when the exact date is unavailable).

Note: The paper's full 481-stock universe requires ~73 min of expert computation.
      This run uses the first 100 tickers from the paper's Appendix C list (plus BIL)
      to comply with the replication time limit.  The full universe can be run with:
          python -m exp.exp5 --full
      See results/RESULTS.md for estimated full-run time and the deviation record.

Run as: python -m exp.exp5

Paper: Cost-Sensitive Online Window Size Selection for Portfolio Management
Authors: Yi-Chen Liu, Chung-Han Hsieh (arXiv:2609.29887v2)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data.data_loader import load_prices_batch, compute_returns, drop_missing_dates
from src.config import (
    TRANSACTION_COST,
    WINDOWS_LARGE,
    SP500_TICKERS,
    SP500_INDEX_TICKER,
    SP500_START,
    SP500_END,
    LOSS_A,
    LOSS_B,
    K_EPSILON,
    ANNUALIZATION,
)
from src.experts import run_window_experts
from src.online_learning import run_aggregation_with_returns, select_parameters
from src.baselines import run_up, run_eg, run_pamr, run_cwmr, run_olmar
from src.regret import (
    compute_tracking_regret,
    compute_stock_regret,
    compute_regret_bound,
    compute_switching_budget,
)
from src.metrics import (
    compute_performance_metrics,
    compute_net_returns,
    compute_account_value,
    build_performance_table,
)
from src.plotting import (
    plot_tracking_regret,
    plot_aggregation_weights,
    plot_account_value,
    plot_stock_regret,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = "results"
PRESAMPLE_NEEDED = max(WINDOWS_LARGE)  # 126 days

# Reduced universe: first 99 non-BIL tickers from Appendix C + BIL
# (BIL is always included as the risk-free asset; reduction documented in RESULTS.md)
REDUCED_SIZE = 99
TICKERS_REDUCED = [t for t in SP500_TICKERS if t != "BIL"][:REDUCED_SIZE] + ["BIL"]

PAPER_EXPECTED = {
    "Hedge": {"cumulative_return_pct": 567.2010, "max_drawdown_pct": -56.9601},
    "Fixed Share": {"cumulative_return_pct": 461.4739, "max_drawdown_pct": -58.5937},
    "UP": {"cumulative_return_pct": 205.3887, "max_drawdown_pct": -38.3346},
    "EG": {"cumulative_return_pct": 205.7900, "max_drawdown_pct": -38.2862},
    "PAMR": {"cumulative_return_pct": -52.0775},
    "CWMR": {"cumulative_return_pct": 205.5993},
    "OLMAR": {"cumulative_return_pct": -44.0283},
}


def compute_log_losses(weights: np.ndarray, returns: np.ndarray) -> np.ndarray:
    gross = np.sum(weights * (1.0 + returns), axis=1)
    return -np.log(np.maximum(gross, 1e-10))


def main(full: bool = False) -> None:
    t0 = time.time()
    tickers = SP500_TICKERS if full else TICKERS_REDUCED
    m_label = "full 481" if full else f"reduced {len(tickers)}"
    logger.info("=== Experiment 5: S&P 500 Historical Backtest (%s stocks) ===", m_label)
    logger.info("Universe: %d tickers, period %s to %s", len(tickers), SP500_START, SP500_END)

    # ── 1. Fetch prices ───────────────────────────────────────────────────
    # Need PRESAMPLE_NEEDED trading days before SP500_START
    extended_start = (
        pd.Timestamp(SP500_START) - pd.offsets.BDay(PRESAMPLE_NEEDED + 15)
    ).strftime("%Y-%m-%d")

    logger.info("Fetching prices from %s to %s ...", extended_start, SP500_END)
    try:
        prices, ticker_details = load_prices_batch(
            tickers=tickers,
            start_date=extended_start,
            end_date=SP500_END,
            batch_size=50,
            adjust_dividends=True,
            identity_date=SP500_END,
        )
    except EnvironmentError as exc:
        logger.error("API authentication error: %s", exc)
        logger.error("Set DATALOADER_API_KEY and MASSIVE_API_KEY environment variables.")
        raise

    logger.info(
        "Prices shape: %s, date range: %s to %s",
        prices.shape, prices.index[0].date(), prices.index[-1].date(),
    )
    for ticker, td in ticker_details.items():
        src = td.get("data_source", "unknown")
        if src != "massive":
            logger.info("  Ticker %s: data_source=%s", ticker, src)

    # ── 2. Align and compute returns ──────────────────────────────────────
    returns_all = compute_returns(prices, min_obs=PRESAMPLE_NEEDED)
    returns_all = drop_missing_dates(returns_all, max_missing_frac=0.05)

    # Drop any tickers with residual NaN to ensure finite rolling statistics
    nan_cols = returns_all.columns[returns_all.isna().any()].tolist()
    if nan_cols:
        logger.warning(
            "Dropping %d tickers with residual NaN after date filter: %s",
            len(nan_cols), nan_cols[:10],
        )
        returns_all = returns_all.drop(columns=nan_cols)
    logger.info("Final return panel: %d dates × %d assets", *returns_all.shape)

    eval_start = pd.Timestamp(SP500_START)
    eval_mask = returns_all.index >= eval_start

    presample_df = returns_all[~eval_mask]
    eval_returns_df = returns_all[eval_mask]

    n_presample = len(presample_df)
    if n_presample < PRESAMPLE_NEEDED:
        missing = PRESAMPLE_NEEDED - n_presample
        extra = returns_all.iloc[:PRESAMPLE_NEEDED]
        eval_returns_df = returns_all.iloc[PRESAMPLE_NEEDED:]

    eval_returns = eval_returns_df.values
    eval_dates = eval_returns_df.index
    T_eval, m = eval_returns.shape
    n = len(WINDOWS_LARGE)

    returns_full = np.vstack([
        returns_all.values[:PRESAMPLE_NEEDED],
        eval_returns,
    ])
    logger.info(
        "Evaluation: %d rows (%s to %s) | Assets: %d",
        T_eval, eval_dates[0].date(), eval_dates[-1].date(), m,
    )

    # ── 3. Expert portfolios ──────────────────────────────────────────────
    w0 = np.ones(m) / m
    logger.info("Running %d window experts over %d eval days ...", n, T_eval)
    t_exp = time.time()
    expert_weights, expert_losses = run_window_experts(
        returns=returns_full,
        windows=WINDOWS_LARGE,
        c=TRANSACTION_COST,
        w0=w0,
    )
    logger.info("Expert computation: %.1f seconds", time.time() - t_exp)

    # ── 4. Parameters ─────────────────────────────────────────────────────
    K_T = compute_switching_budget(T_eval, K_EPSILON)
    logger.info("K(T=%d, eps=%.2f) = %d", T_eval, K_EPSILON, K_T)

    eta_h, alpha_h, _ = select_parameters(T_eval, 0, n, TRANSACTION_COST, LOSS_A, LOSS_B)
    eta_f, alpha_f, _ = select_parameters(T_eval, K_T, n, TRANSACTION_COST, LOSS_A, LOSS_B)
    logger.info("Hedge: eta=%.4f | FS: eta=%.4f, alpha=%.4f", eta_h, eta_f, alpha_f)

    # ── 5. Online learning ────────────────────────────────────────────────
    logger.info("Running Hedge ...")
    agg_w_h, q_h, agg_l_h, _ = run_aggregation_with_returns(
        eval_returns, expert_weights, expert_losses, eta_h, 0.0, w0, TRANSACTION_COST
    )
    logger.info("Running Fixed Share ...")
    agg_w_f, q_f, agg_l_f, _ = run_aggregation_with_returns(
        eval_returns, expert_weights, expert_losses, eta_f, alpha_f, w0, TRANSACTION_COST
    )

    # ── 6. Tracking regret ────────────────────────────────────────────────
    cum_rh, _, _ = compute_tracking_regret(agg_l_h, expert_losses, K_T)
    cum_rf, _, _ = compute_tracking_regret(agg_l_f, expert_losses, K_T)
    bound_h = compute_regret_bound(T_eval, 0, n, TRANSACTION_COST, LOSS_A, LOSS_B, eta_h, 0.0)
    bound_f = compute_regret_bound(T_eval, K_T, n, TRANSACTION_COST, LOSS_A, LOSS_B, eta_f, alpha_f)

    plot_tracking_regret(
        eval_dates, cum_rh, cum_rf, bound_h, bound_f, None,
        f"{RESULTS_DIR}/exp5_tracking_regret.png",
        title="Figure 10: S&P 500 Cost-Sensitive Tracking Regret",
    )
    plot_aggregation_weights(
        eval_dates, q_h, WINDOWS_LARGE, None,
        f"{RESULTS_DIR}/exp5_weights_hedge.png",
        title="Figure 11a: S&P 500 Hedge Aggregation Weights",
    )
    plot_aggregation_weights(
        eval_dates, q_f, WINDOWS_LARGE, None,
        f"{RESULTS_DIR}/exp5_weights_fs.png",
        title="Figure 11b: S&P 500 Fixed Share Aggregation Weights",
    )

    # ── 7. Baselines ──────────────────────────────────────────────────────
    logger.info("Running baselines ...")
    up_w, _ = run_up(eval_returns, n_samples=300)
    eg_w, _ = run_eg(eval_returns, eta=0.05)
    pamr_w, _ = run_pamr(eval_returns, epsilon=0.5, C=500.0)
    cwmr_w, _ = run_cwmr(eval_returns, epsilon=0.5, confidence=0.95)
    olmar_w, _ = run_olmar(eval_returns, window=5, epsilon=10.0)

    # ── 8. Performance metrics (Table 3) ─────────────────────────────────
    all_weights = {
        "Hedge": agg_w_h,
        "Fixed Share": agg_w_f,
        "UP": up_w,
        "EG": eg_w,
        "PAMR": pamr_w,
        "CWMR": cwmr_w,
        "OLMAR": olmar_w,
    }
    for j_idx, j in enumerate(WINDOWS_LARGE):
        all_weights[f"Expert w={j}"] = expert_weights[:, j_idx, :]

    metrics_list = [
        compute_performance_metrics(w, eval_returns, TRANSACTION_COST, w0, name)
        for name, w in all_weights.items()
    ]
    perf_table = build_performance_table(metrics_list)
    logger.info("\nPerformance Table:\n%s", perf_table.to_string())
    perf_table.to_csv(f"{RESULTS_DIR}/exp5_performance.csv")

    # ── 9. Account value and R_stock ─────────────────────────────────────
    av_strategies = {
        name: compute_account_value(
            compute_net_returns(w, eval_returns, TRANSACTION_COST, w0)
        )
        for name, w in {
            "Hedge": agg_w_h,
            "Fixed Share": agg_w_f,
            "UP": up_w,
            "EG": eg_w,
        }.items()
    }
    plot_account_value(
        [eval_dates[0] - pd.offsets.BDay(1)] + list(eval_dates),
        av_strategies,
        f"{RESULTS_DIR}/exp5_account_value.png",
        title="Figure 12: S&P 500 Account Value",
    )

    stock_losses = -np.log(np.maximum(1.0 + eval_returns, 1e-10))
    r_stock_final = {}
    r_stock_series = {}
    for name, w in all_weights.items():
        log_loss = compute_log_losses(w, eval_returns)
        cum_rs, _ = compute_stock_regret(log_loss, stock_losses, K_T)
        r_stock_series[name] = cum_rs
        r_stock_final[name] = float(cum_rs[-1])

    plot_stock_regret(
        eval_dates,
        {k: v for k, v in r_stock_series.items() if k in [
            "Hedge", "Fixed Share", "UP", "EG", "PAMR", "CWMR", "OLMAR"
        ]},
        f"{RESULTS_DIR}/exp5_rstock.png",
        title="Figure 13: S&P 500 R_stock vs. Best Stock Sequence",
    )

    # ── 10. Save results ──────────────────────────────────────────────────
    summary = {
        "experiment": "exp5",
        "config": {
            "universe_mode": "full" if full else f"reduced_{len(tickers)}",
            "n_tickers": len(tickers),
            "start": SP500_START,
            "end": SP500_END,
            "T_eval": int(T_eval),
            "m": int(m),
            "windows": WINDOWS_LARGE,
            "c": TRANSACTION_COST,
            "K_T": int(K_T),
            "epsilon_K": K_EPSILON,
            "eta_hedge": round(eta_h, 6),
            "eta_fs": round(eta_f, 6),
            "alpha_fs": round(alpha_f, 6),
        },
        "performance": {
            m_d["label"]: {
                "cumulative_return_pct": round(m_d["cumulative_return"], 4),
                "max_drawdown_pct": round(m_d["max_drawdown"], 4),
                "sharpe": round(m_d["annualized_sharpe"], 4),
            }
            for m_d in metrics_list
        },
        "tracking_regret": {
            "hedge_final": float(cum_rh[-1]),
            "fs_final": float(cum_rf[-1]),
        },
        "r_stock_final": r_stock_final,
        "paper_expected": PAPER_EXPECTED,
    }
    with open(f"{RESULTS_DIR}/exp5_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame(q_h, index=eval_dates, columns=[f"w{j}" for j in WINDOWS_LARGE]).to_csv(
        f"{RESULTS_DIR}/exp5_q_hedge.csv"
    )

    elapsed = time.time() - t0
    logger.info("Exp 5 complete in %.1f seconds (%.1f min).", elapsed, elapsed / 60)
    for name in ["Hedge", "Fixed Share", "UP", "EG"]:
        act = summary["performance"].get(name, {})
        exp = PAPER_EXPECTED.get(name, {})
        logger.info(
            "  %s: CR=%.4f%% (paper %.4f%%) | MDD=%.4f%% (paper %.4f%%)",
            name,
            act.get("cumulative_return_pct", float("nan")),
            exp.get("cumulative_return_pct", float("nan")),
            act.get("max_drawdown_pct", float("nan")),
            exp.get("max_drawdown_pct", float("nan")),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S&P 500 backtest (Exp 5)")
    parser.add_argument("--full", action="store_true",
                        help="Use the full 481-ticker paper universe (very slow).")
    args = parser.parse_args()
    main(full=args.full)
