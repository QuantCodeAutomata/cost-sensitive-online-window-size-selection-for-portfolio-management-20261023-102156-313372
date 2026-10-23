"""
Experiment 3: Synthetic comparison with standard online portfolio-selection algorithms
(Appendix B.3, Figure 5).

Run as: python -m exp.exp3
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd

from data.generate_synthetic import generate_synthetic_returns
from src.config import (
    TRANSACTION_COST,
    WINDOWS_SMALL,
    SYN_N_STOCKS,
    SYN_T,
    SYN_REGIME_CHANGE,
    SYN_J1,
    SYN_J2,
    SYN_START_DATE,
    SYN_MU_DAILY,
    SYN_ALPHA,
    SYN_SIGMA,
    SYN_SEED,
    LOSS_A,
    LOSS_B,
)
from src.experts import run_window_experts
from src.online_learning import run_aggregation_with_returns, select_parameters
from src.baselines import run_up, run_eg, run_pamr, run_cwmr, run_olmar
from src.regret import compute_stock_regret
from src.metrics import compute_performance_metrics, build_performance_table
from src.plotting import plot_stock_regret

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = "results"
K_STOCK = 5  # switching budget for stock comparator (assumed; paper: unspecified)


def compute_log_losses(weights: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """
    Compute per-period negative log-return loss: -log(max(w'(1+y), 1e-10)).

    This is the loss used for R_stock comparator evaluation (Appendix B.3).

    Parameters
    ----------
    weights : np.ndarray, shape (T, m)
    returns : np.ndarray, shape (T, m)

    Returns
    -------
    losses : np.ndarray, shape (T,)
    """
    gross = np.sum(weights * (1.0 + returns), axis=1)
    return -np.log(np.maximum(gross, 1e-10))


def main() -> None:
    t0 = time.time()
    logger.info("=== Experiment 3: Comparison with Online Portfolio Algorithms (Synthetic) ===")

    # 1. Generate synthetic returns (same as exp1)
    df_ret, eval_dates = generate_synthetic_returns(
        n_stocks=SYN_N_STOCKS,
        T=SYN_T,
        regime_change=SYN_REGIME_CHANGE,
        j1=SYN_J1,
        j2=SYN_J2,
        mu_daily=SYN_MU_DAILY,
        alpha=SYN_ALPHA,
        sigma=SYN_SIGMA,
        seed=SYN_SEED,
        start_date=SYN_START_DATE,
        windows=WINDOWS_SMALL,
    )
    returns_full = df_ret.values
    n_presample = max(WINDOWS_SMALL)
    eval_returns = returns_full[n_presample:]
    T_eval, m = eval_returns.shape
    n = len(WINDOWS_SMALL)
    w0 = np.ones(m) / m

    logger.info("Generated %d evaluation rows, %d assets", T_eval, m)

    # 2. Run window experts
    logger.info("Running window experts (%s) ...", WINDOWS_SMALL)
    expert_weights, expert_losses = run_window_experts(
        returns_full, WINDOWS_SMALL, TRANSACTION_COST, w0
    )

    # 3. Run Hedge and Fixed Share
    eta_h, _, _ = select_parameters(T_eval, 0, n, TRANSACTION_COST, LOSS_A, LOSS_B)
    eta_f, alpha_f, _ = select_parameters(T_eval, K_STOCK, n, TRANSACTION_COST, LOSS_A, LOSS_B)

    logger.info("Running Hedge and Fixed Share ...")
    agg_w_h, _, _, _ = run_aggregation_with_returns(
        eval_returns, expert_weights, expert_losses, eta_h, 0.0, w0, TRANSACTION_COST
    )
    agg_w_f, _, _, _ = run_aggregation_with_returns(
        eval_returns, expert_weights, expert_losses, eta_f, alpha_f, w0, TRANSACTION_COST
    )

    # 4. Run baselines
    logger.info("Running baselines (UP, EG, PAMR, CWMR, OLMAR) ...")
    up_w, _ = run_up(eval_returns, n_samples=500, seed=SYN_SEED)
    eg_w, _ = run_eg(eval_returns, eta=0.05)
    pamr_w, _ = run_pamr(eval_returns, epsilon=0.5, C=500.0)
    cwmr_w, _ = run_cwmr(eval_returns, epsilon=0.5, confidence=0.95)
    olmar_w, _ = run_olmar(eval_returns, window=5, epsilon=10.0)

    # 5. Compute R_stock for all strategies
    # Individual stock log-return losses: loss_i(t) = -log(max(1 + y_i(t), 1e-10))
    stock_losses = -np.log(np.maximum(1.0 + eval_returns, 1e-10))  # shape (T, m)
    logger.info("Computing R_stock with K=%d ...", K_STOCK)

    strategy_weights = {
        "Hedge": agg_w_h,
        "Fixed Share": agg_w_f,
        "UP": up_w,
        "EG": eg_w,
        "PAMR": pamr_w,
        "CWMR": cwmr_w,
        "OLMAR": olmar_w,
    }
    regret_series = {}
    regret_final = {}
    for name, w_strat in strategy_weights.items():
        log_losses = compute_log_losses(w_strat, eval_returns)
        cum_rs, best_loss = compute_stock_regret(log_losses, stock_losses, K_STOCK)
        regret_series[name] = cum_rs
        regret_final[name] = float(cum_rs[-1])
        logger.info("  %s: final R_stock = %.4f", name, cum_rs[-1])

    # 6. Performance metrics
    all_metrics = []
    for name, w_strat in strategy_weights.items():
        met = compute_performance_metrics(w_strat, eval_returns, TRANSACTION_COST, w0, name)
        all_metrics.append(met)
    perf_table = build_performance_table(all_metrics)
    logger.info("\n%s", perf_table.to_string())

    # 7. Plot R_stock (Figure 5)
    plot_stock_regret(
        dates=eval_dates,
        regret_series=regret_series,
        filename=f"{RESULTS_DIR}/exp3_rstock.png",
        title="Figure 5: R_stock — Regret vs. Best Stock Sequence (Synthetic)",
    )

    # 8. Save results
    summary = {
        "experiment": "exp3",
        "config": {
            "n_stocks": SYN_N_STOCKS,
            "T": T_eval,
            "K_stock": K_STOCK,
            "windows": WINDOWS_SMALL,
            "c": TRANSACTION_COST,
            "assumed_params": {
                "mu_daily": SYN_MU_DAILY,
                "alpha_dgp": SYN_ALPHA,
                "sigma": SYN_SIGMA,
            },
        },
        "r_stock_final": regret_final,
        "performance": {
            m_dict["label"]: {
                "cumulative_return_pct": m_dict["cumulative_return"],
                "max_drawdown_pct": m_dict["max_drawdown"],
                "sharpe": m_dict["annualized_sharpe"],
            }
            for m_dict in all_metrics
        },
        "verdict": {
            "hedge_lower_rstock_than_baselines": bool(
                regret_final["Hedge"] < min(
                    regret_final[b] for b in ["UP", "EG", "PAMR", "CWMR", "OLMAR"]
                )
            ),
            "fs_lower_rstock_than_baselines": bool(
                regret_final["Fixed Share"] < min(
                    regret_final[b] for b in ["UP", "EG", "PAMR", "CWMR", "OLMAR"]
                )
            ),
        },
    }
    with open(f"{RESULTS_DIR}/exp3_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save regret time series
    rs_df = pd.DataFrame(regret_series, index=eval_dates)
    rs_df.to_csv(f"{RESULTS_DIR}/exp3_rstock_series.csv")
    perf_table.to_csv(f"{RESULTS_DIR}/exp3_performance.csv")

    logger.info("Exp 3 complete in %.1f seconds.", time.time() - t0)


if __name__ == "__main__":
    main()
