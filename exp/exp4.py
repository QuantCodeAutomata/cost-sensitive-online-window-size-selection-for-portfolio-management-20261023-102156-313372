"""
Experiment 4: Dow Jones Industrial Average historical backtest (Appendix B.4).

Reproduces Table 1, Figures 6–9 from the paper.

Run as: python -m exp.exp4

Paper: Cost-Sensitive Online Window Size Selection for Portfolio Management
Authors: Yi-Chen Liu, Chung-Han Hsieh (arXiv:2609.29887v2)
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data.data_loader import load_prices_batch, compute_returns, drop_missing_dates
from src.config import (
    TRANSACTION_COST,
    WINDOWS_LARGE,
    DJIA_TICKERS,
    DJIA_INDEX_TICKER,
    DJIA_START,
    DJIA_END,
    LOSS_A,
    LOSS_B,
    K_EPSILON,
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
PRESAMPLE_NEEDED = max(WINDOWS_LARGE)  # 126 days before first evaluation date


def compute_log_losses(weights: np.ndarray, returns: np.ndarray) -> np.ndarray:
    gross = np.sum(weights * (1.0 + returns), axis=1)
    return -np.log(np.maximum(gross, 1e-10))


def main() -> None:
    t0 = time.time()
    logger.info("=== Experiment 4: DJIA Historical Backtest ===")
    logger.info("Universe: %d tickers, period %s to %s", len(DJIA_TICKERS), DJIA_START, DJIA_END)

    # ── 1. Load data ──────────────────────────────────────────────────────
    # We need at least PRESAMPLE_NEEDED trading days before DJIA_START.
    # Fetch from far enough before the start date to ensure presample availability.
    extended_start = (
        pd.Timestamp(DJIA_START) - pd.offsets.BDay(PRESAMPLE_NEEDED + 10)
    ).strftime("%Y-%m-%d")

    logger.info("Fetching prices from %s to %s ...", extended_start, DJIA_END)
    try:
        prices, ticker_details = load_prices_batch(
            tickers=DJIA_TICKERS,
            start_date=extended_start,
            end_date=DJIA_END,
            batch_size=31,
            adjust_dividends=True,
            identity_date=DJIA_END,
        )
    except EnvironmentError as exc:
        logger.error("API authentication error: %s", exc)
        logger.error("Set DATALOADER_API_KEY and MASSIVE_API_KEY environment variables.")
        raise

    # Log data details
    logger.info("Prices shape: %s, date range: %s to %s",
                prices.shape, prices.index[0].date(), prices.index[-1].date())
    for ticker, td in ticker_details.items():
        src = td.get("data_source", "unknown")
        if src != "massive":
            logger.info("  Ticker %s: data_source=%s", ticker, src)

    # ── 2. Compute returns and align ──────────────────────────────────────
    returns_all = compute_returns(prices, min_obs=PRESAMPLE_NEEDED)
    returns_all = drop_missing_dates(returns_all, max_missing_frac=0.0)

    # Identify the first evaluation date (first day >= DJIA_START after presample)
    eval_start = pd.Timestamp(DJIA_START)
    eval_mask = returns_all.index >= eval_start
    presample_mask = ~eval_mask

    presample_returns = returns_all[presample_mask].values
    eval_returns_df = returns_all[eval_mask]

    # Ensure we have enough presample
    if len(presample_returns) < PRESAMPLE_NEEDED:
        # Borrow from early eval period
        short = PRESAMPLE_NEEDED - len(presample_returns)
        presample_returns = returns_all.values[:PRESAMPLE_NEEDED]
        eval_returns_df = returns_all.iloc[PRESAMPLE_NEEDED:]

    eval_returns = eval_returns_df.values
    eval_dates = eval_returns_df.index
    T_eval, m = eval_returns.shape
    n = len(WINDOWS_LARGE)

    # Concatenate presample + eval for rolling statistics
    returns_full = np.vstack([presample_returns[-PRESAMPLE_NEEDED:], eval_returns])
    logger.info(
        "Presample: %d rows | Evaluation: %d rows (%s to %s) | Assets: %d",
        PRESAMPLE_NEEDED, T_eval, eval_dates[0].date(), eval_dates[-1].date(), m,
    )

    # ── 3. Expert portfolios ─────────────────────────────────────────────
    w0 = np.ones(m) / m
    logger.info("Running %d window experts over %d evaluation periods ...", n, T_eval)
    t_exp = time.time()
    expert_weights, expert_losses = run_window_experts(
        returns=returns_full,
        windows=WINDOWS_LARGE,
        c=TRANSACTION_COST,
        w0=w0,
    )
    logger.info("Expert computation: %.1f seconds", time.time() - t_exp)

    # ── 4. Compute K(T) and select parameters ────────────────────────────
    K_T = compute_switching_budget(T_eval, K_EPSILON)
    logger.info("K(T=%d, eps=%.1f) = %d", T_eval, K_EPSILON, K_T)

    eta_h, alpha_h, bound_h = select_parameters(T_eval, 0, n, TRANSACTION_COST, LOSS_A, LOSS_B)
    eta_f, alpha_f, bound_f = select_parameters(T_eval, K_T, n, TRANSACTION_COST, LOSS_A, LOSS_B)
    logger.info("Hedge: eta=%.4f | Fixed Share: eta=%.4f, alpha=%.4f (K=%d)", eta_h, eta_f, alpha_f, K_T)

    # ── 5. Run Hedge ─────────────────────────────────────────────────────
    logger.info("Running Hedge ...")
    agg_w_h, q_h, agg_l_h, _ = run_aggregation_with_returns(
        eval_returns, expert_weights, expert_losses, eta_h, 0.0, w0, TRANSACTION_COST
    )

    # ── 6. Run Fixed Share ────────────────────────────────────────────────
    logger.info("Running Fixed Share ...")
    agg_w_f, q_f, agg_l_f, _ = run_aggregation_with_returns(
        eval_returns, expert_weights, expert_losses, eta_f, alpha_f, w0, TRANSACTION_COST
    )

    # ── 7. Tracking regret (Figure 6) ────────────────────────────────────
    logger.info("Computing tracking regret K=%d ...", K_T)
    cum_rh, _, _ = compute_tracking_regret(agg_l_h, expert_losses, K_T)
    cum_rf, _, _ = compute_tracking_regret(agg_l_f, expert_losses, K_T)

    bound_h_series = compute_regret_bound(T_eval, 0, n, TRANSACTION_COST, LOSS_A, LOSS_B, eta_h, 0.0)
    bound_f_series = compute_regret_bound(T_eval, K_T, n, TRANSACTION_COST, LOSS_A, LOSS_B, eta_f, alpha_f)

    plot_tracking_regret(
        eval_dates, cum_rh, cum_rf, bound_h_series, bound_f_series, None,
        f"{RESULTS_DIR}/exp4_tracking_regret.png",
        title="Figure 6: DJIA Cost-Sensitive Tracking Regret",
    )
    plot_aggregation_weights(
        eval_dates, q_h, WINDOWS_LARGE, None,
        f"{RESULTS_DIR}/exp4_weights_hedge.png",
        title="Figure 7a: DJIA Hedge Aggregation Weights",
    )
    plot_aggregation_weights(
        eval_dates, q_f, WINDOWS_LARGE, None,
        f"{RESULTS_DIR}/exp4_weights_fs.png",
        title="Figure 7b: DJIA Fixed Share Aggregation Weights",
    )

    # ── 8. Baselines ──────────────────────────────────────────────────────
    logger.info("Running baselines ...")
    up_w, _ = run_up(eval_returns, n_samples=300)
    eg_w, _ = run_eg(eval_returns, eta=0.05)
    pamr_w, _ = run_pamr(eval_returns, epsilon=0.5, C=500.0)
    cwmr_w, _ = run_cwmr(eval_returns, epsilon=0.5, confidence=0.95)
    olmar_w, _ = run_olmar(eval_returns, window=5, epsilon=10.0)

    # ── 9. Performance metrics (Table 1) ─────────────────────────────────
    all_weights = {
        "Hedge": agg_w_h,
        "Fixed Share": agg_w_f,
        "UP": up_w,
        "EG": eg_w,
        "PAMR": pamr_w,
        "CWMR": cwmr_w,
        "OLMAR": olmar_w,
    }
    # Add individual window experts
    for j_idx, j in enumerate(WINDOWS_LARGE):
        all_weights[f"Expert w={j}"] = expert_weights[:, j_idx, :]

    metrics_list = [
        compute_performance_metrics(w, eval_returns, TRANSACTION_COST, w0, name)
        for name, w in all_weights.items()
    ]
    perf_table = build_performance_table(metrics_list)
    logger.info("\nPerformance Table:\n%s", perf_table.to_string())
    perf_table.to_csv(f"{RESULTS_DIR}/exp4_performance.csv")

    # ── 10. Account value plot (Figure 8) ────────────────────────────────
    # Include +1 initial date
    av_dates = [eval_dates[0] - pd.offsets.BDay(1)] + list(eval_dates)
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
        av_dates, av_strategies,
        f"{RESULTS_DIR}/exp4_account_value.png",
        title="Figure 8: DJIA Account Value",
    )

    # ── 11. R_stock (Figure 9) ────────────────────────────────────────────
    stock_losses = -np.log(np.maximum(1.0 + eval_returns, 1e-10))
    r_stock_series = {}
    r_stock_final = {}
    for name, w in all_weights.items():
        log_loss = compute_log_losses(w, eval_returns)
        cum_rs, _ = compute_stock_regret(log_loss, stock_losses, K_T)
        r_stock_series[name] = cum_rs
        r_stock_final[name] = float(cum_rs[-1])
    plot_stock_regret(
        eval_dates, {k: v for k, v in r_stock_series.items() if k in [
            "Hedge", "Fixed Share", "UP", "EG", "PAMR", "CWMR", "OLMAR"
        ]},
        f"{RESULTS_DIR}/exp4_rstock.png",
        title="Figure 9: DJIA R_stock vs. Best Stock Sequence",
    )

    # ── 12. Save JSON summary ─────────────────────────────────────────────
    summary = {
        "experiment": "exp4",
        "config": {
            "tickers": DJIA_TICKERS,
            "start": DJIA_START,
            "end": DJIA_END,
            "T_eval": int(T_eval),
            "m": int(m),
            "windows": WINDOWS_LARGE,
            "c": TRANSACTION_COST,
            "K_T": int(K_T),
            "epsilon_K": K_EPSILON,
            "eta_hedge": round(eta_h, 6),
            "eta_fs": round(eta_f, 6),
            "alpha_fs": round(alpha_f, 6),
            "loss_a": LOSS_A,
            "loss_b": LOSS_B,
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
        "paper_expected": {
            "Hedge": {"cumulative_return_pct": 21.1950, "max_drawdown_pct": -23.5123},
            "Fixed Share": {"cumulative_return_pct": 17.7014, "max_drawdown_pct": -23.0759},
            "UP": {"cumulative_return_pct": 48.7250, "max_drawdown_pct": -20.2783},
            "EG": {"cumulative_return_pct": 48.2584, "max_drawdown_pct": -20.2783},
        },
    }
    with open(f"{RESULTS_DIR}/exp4_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save series
    pd.DataFrame({
        "date": eval_dates,
        "hedge_regret": cum_rh,
        "fs_regret": cum_rf,
    }).to_csv(f"{RESULTS_DIR}/exp4_regret_series.csv", index=False)

    pd.DataFrame(q_h, index=eval_dates, columns=[f"w{j}" for j in WINDOWS_LARGE]).to_csv(
        f"{RESULTS_DIR}/exp4_q_hedge.csv"
    )

    elapsed = time.time() - t0
    logger.info("Exp 4 complete in %.1f seconds.", elapsed)

    # Paper comparison
    for name in ["Hedge", "Fixed Share", "UP", "EG"]:
        perf_actual = summary["performance"].get(name, {})
        perf_paper = summary["paper_expected"].get(name, {})
        logger.info(
            "  %s: Realized CR=%.4f%% (paper: %.4f%%) | MDD=%.4f%% (paper: %.4f%%)",
            name,
            perf_actual.get("cumulative_return_pct", float("nan")),
            perf_paper.get("cumulative_return_pct", float("nan")),
            perf_actual.get("max_drawdown_pct", float("nan")),
            perf_paper.get("max_drawdown_pct", float("nan")),
        )


if __name__ == "__main__":
    main()
