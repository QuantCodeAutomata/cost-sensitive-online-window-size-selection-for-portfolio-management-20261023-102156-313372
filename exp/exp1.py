"""
Experiment 1: Synthetic regime-shift comparison of Hedge and Fixed Share (Section 4.1).

Reproduces Figures 1 and 2 from the paper.

Paper: Cost-Sensitive Online Window Size Selection for Portfolio Management
Authors: Yi-Chen Liu, Chung-Han Hsieh (arXiv:2609.29887v2)

Run as: python -m exp.exp1
"""

from __future__ import annotations

import json
import logging
import sys
import time

import numpy as np
import pandas as pd

from data.generate_synthetic import generate_synthetic_returns, get_regime_change_date
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
from src.online_learning import (
    run_aggregation_with_returns,
    select_parameters,
)
from src.regret import (
    compute_tracking_regret,
    compute_regret_bound,
)
from src.metrics import compute_performance_metrics, build_performance_table
from src.plotting import plot_tracking_regret, plot_aggregation_weights

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = "results"
K_SYNTHETIC = 1  # one switch (one regime change); assumed


def main() -> None:
    t0 = time.time()
    logger.info("=== Experiment 1: Synthetic Regime-Shift Comparison ===")
    logger.info(
        "DGP: n_stocks=%d, T=%d, regime_change=%d, j1=%d→j2=%d, "
        "mu=%.4f, alpha=%.2f, sigma=%.3f, seed=%d",
        SYN_N_STOCKS, SYN_T, SYN_REGIME_CHANGE,
        SYN_J1, SYN_J2, SYN_MU_DAILY, SYN_ALPHA, SYN_SIGMA, SYN_SEED,
    )

    # 1. Generate synthetic returns
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
    returns_full = df_ret.values  # shape (n_presample + T, n_stocks)
    n_presample = max(WINDOWS_SMALL)
    eval_returns = returns_full[n_presample:]  # shape (T, m)
    T_eval, m = eval_returns.shape
    n = len(WINDOWS_SMALL)

    logger.info(
        "Generated %d presample + %d evaluation rows, %d assets",
        n_presample, T_eval, m,
    )

    # 2. Run window experts
    logger.info("Computing expert portfolios for windows %s ...", WINDOWS_SMALL)
    w0 = np.ones(m) / m
    expert_weights, expert_losses = run_window_experts(
        returns=returns_full,
        windows=WINDOWS_SMALL,
        c=TRANSACTION_COST,
        w0=w0,
    )
    logger.info("Expert losses range: [%.4f, %.4f]", expert_losses.min(), expert_losses.max())

    # 3. Select optimal parameters (Proposition 3.5)
    # Hedge: K=0 (static regret baseline)
    eta_hedge, alpha_hedge, bound_hedge_val = select_parameters(
        T=T_eval, K=0, n=n, c=TRANSACTION_COST, a=LOSS_A, b=LOSS_B
    )
    # Fixed Share: K=1 (one switch due to one regime change)
    eta_fs, alpha_fs, bound_fs_val = select_parameters(
        T=T_eval, K=K_SYNTHETIC, n=n, c=TRANSACTION_COST, a=LOSS_A, b=LOSS_B
    )
    logger.info(
        "Hedge: eta=%.4f, alpha=0 | Fixed Share: eta=%.4f, alpha=%.4f (K=%d)",
        eta_hedge, eta_fs, alpha_fs, K_SYNTHETIC,
    )

    # 4. Run Hedge
    logger.info("Running Hedge ...")
    agg_w_hedge, q_hedge, agg_losses_hedge, log_q_hedge = run_aggregation_with_returns(
        returns=eval_returns,
        expert_weights=expert_weights,
        expert_losses=expert_losses,
        eta=eta_hedge,
        alpha=0.0,
        w_hat_prev=w0,
        c=TRANSACTION_COST,
    )

    # 5. Run Fixed Share
    logger.info("Running Fixed Share ...")
    agg_w_fs, q_fs, agg_losses_fs, log_q_fs = run_aggregation_with_returns(
        returns=eval_returns,
        expert_weights=expert_weights,
        expert_losses=expert_losses,
        eta=eta_fs,
        alpha=alpha_fs,
        w_hat_prev=w0,
        c=TRANSACTION_COST,
    )

    # 6. Compute regret (Eq. 4)
    logger.info("Computing tracking regret with K=%d ...", K_SYNTHETIC)
    cumreg_hedge, best_loss_hedge, best_seq_hedge = compute_tracking_regret(
        agg_full_losses=agg_losses_hedge,
        expert_losses=expert_losses,
        K=K_SYNTHETIC,
    )
    cumreg_fs, best_loss_fs, best_seq_fs = compute_tracking_regret(
        agg_full_losses=agg_losses_fs,
        expert_losses=expert_losses,
        K=K_SYNTHETIC,
    )

    # 7. Theoretical bounds (Corollary 3.4)
    bound_hedge_series = compute_regret_bound(
        T=T_eval, K=0, n=n, c=TRANSACTION_COST, a=LOSS_A, b=LOSS_B,
        eta=eta_hedge, alpha=0.0,
    )
    bound_fs_series = compute_regret_bound(
        T=T_eval, K=K_SYNTHETIC, n=n, c=TRANSACTION_COST, a=LOSS_A, b=LOSS_B,
        eta=eta_fs, alpha=alpha_fs,
    )

    # 8. Adaptation time: first post-change date where 5-day expert dominates
    regime_change_idx = SYN_REGIME_CHANGE  # first post-change index (0-based)
    w5_idx = WINDOWS_SMALL.index(SYN_J2)   # index of the 5-day expert
    def adaptation_time(q_hist: np.ndarray, start_idx: int) -> int:
        """Days after start_idx until w5_idx is the largest-weight expert."""
        for i in range(start_idx, T_eval):
            if np.argmax(q_hist[i]) == w5_idx:
                return i - start_idx
        return T_eval - start_idx  # never

    adapt_hedge = adaptation_time(q_hedge, regime_change_idx)
    adapt_fs = adaptation_time(q_fs, regime_change_idx)
    logger.info(
        "Adaptation time (5-day expert dominant): Hedge=%d days, Fixed Share=%d days",
        adapt_hedge, adapt_fs,
    )

    # 9. Performance metrics
    metrics_hedge = compute_performance_metrics(agg_w_hedge, eval_returns, TRANSACTION_COST, w0, "Hedge")
    metrics_fs = compute_performance_metrics(agg_w_fs, eval_returns, TRANSACTION_COST, w0, "Fixed Share")
    perf_table = build_performance_table([metrics_hedge, metrics_fs])
    logger.info("\n%s", perf_table.to_string())

    # 10. Save results
    regime_change_date = get_regime_change_date(SYN_START_DATE, SYN_REGIME_CHANGE)
    logger.info("Regime change date: %s", regime_change_date)

    # Plots
    plot_tracking_regret(
        dates=eval_dates,
        hedge_regret=cumreg_hedge,
        fs_regret=cumreg_fs,
        bound_hedge=bound_hedge_series,
        bound_fs=bound_fs_series,
        regime_change_date=regime_change_date,
        filename=f"{RESULTS_DIR}/exp1_tracking_regret.png",
        title="Figure 1: Cost-Sensitive Tracking Regret — Synthetic Regime Shift",
    )
    plot_aggregation_weights(
        dates=eval_dates,
        q_history=q_hedge,
        windows=WINDOWS_SMALL,
        regime_change_date=regime_change_date,
        filename=f"{RESULTS_DIR}/exp1_weights_hedge.png",
        title="Figure 2a: Hedge Aggregation Weights — Synthetic",
    )
    plot_aggregation_weights(
        dates=eval_dates,
        q_history=q_fs,
        windows=WINDOWS_SMALL,
        regime_change_date=regime_change_date,
        filename=f"{RESULTS_DIR}/exp1_weights_fs.png",
        title="Figure 2b: Fixed Share Aggregation Weights — Synthetic",
    )

    # Save metrics JSON
    summary = {
        "experiment": "exp1",
        "description": "Synthetic regime-shift comparison: Hedge vs Fixed Share",
        "config": {
            "n_stocks": SYN_N_STOCKS,
            "T": T_eval,
            "regime_change": SYN_REGIME_CHANGE,
            "j1": SYN_J1,
            "j2": SYN_J2,
            "windows": WINDOWS_SMALL,
            "c": TRANSACTION_COST,
            "K": K_SYNTHETIC,
            "eta_hedge": round(eta_hedge, 6),
            "alpha_hedge": 0.0,
            "eta_fs": round(eta_fs, 6),
            "alpha_fs": round(alpha_fs, 6),
            "seed": SYN_SEED,
            "assumed_params": {
                "mu_daily": SYN_MU_DAILY,
                "alpha_dgp": SYN_ALPHA,
                "sigma": SYN_SIGMA,
                "loss_a": LOSS_A,
                "loss_b": LOSS_B,
            },
        },
        "results": {
            "hedge": {
                "final_tracking_regret": float(cumreg_hedge[-1]),
                "theoretical_bound_hedge": float(bound_hedge_series[-1]),
                "adaptation_time_days": int(adapt_hedge),
                "cumulative_return_pct": round(metrics_hedge["cumulative_return"], 4),
                "max_drawdown_pct": round(metrics_hedge["max_drawdown"], 4),
                "sharpe": round(metrics_hedge["annualized_sharpe"], 4),
            },
            "fixed_share": {
                "final_tracking_regret": float(cumreg_fs[-1]),
                "theoretical_bound_fs": float(bound_fs_series[-1]),
                "adaptation_time_days": int(adapt_fs),
                "cumulative_return_pct": round(metrics_fs["cumulative_return"], 4),
                "max_drawdown_pct": round(metrics_fs["max_drawdown"], 4),
                "sharpe": round(metrics_fs["annualized_sharpe"], 4),
            },
        },
        "verdict": {
            "fixed_share_adapts_faster": bool(adapt_fs < adapt_hedge),
            "regret_within_bound_hedge": bool(cumreg_hedge[-1] <= bound_hedge_series[-1] * 1.01),
            "regret_within_bound_fs": bool(cumreg_fs[-1] <= bound_fs_series[-1] * 1.01),
        },
    }
    out_path = f"{RESULTS_DIR}/exp1_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Save cumulative regret series
    pd.DataFrame({
        "date": eval_dates,
        "hedge_regret": cumreg_hedge,
        "fs_regret": cumreg_fs,
        "bound_hedge": bound_hedge_series,
        "bound_fs": bound_fs_series,
    }).to_csv(f"{RESULTS_DIR}/exp1_regret_series.csv", index=False)

    # Save q_history
    q_hedge_df = pd.DataFrame(q_hedge, index=eval_dates, columns=[f"w{j}" for j in WINDOWS_SMALL])
    q_fs_df = pd.DataFrame(q_fs, index=eval_dates, columns=[f"w{j}" for j in WINDOWS_SMALL])
    q_hedge_df.to_csv(f"{RESULTS_DIR}/exp1_q_hedge.csv")
    q_fs_df.to_csv(f"{RESULTS_DIR}/exp1_q_fs.csv")
    perf_table.to_csv(f"{RESULTS_DIR}/exp1_performance.csv")

    elapsed = time.time() - t0
    logger.info("Exp 1 complete in %.1f seconds.", elapsed)
    logger.info("Key results:")
    logger.info("  Hedge final regret: %.4f | bound: %.4f | bound satisfied: %s",
                cumreg_hedge[-1], bound_hedge_series[-1],
                summary["verdict"]["regret_within_bound_hedge"])
    logger.info("  Fixed Share final regret: %.4f | bound: %.4f | bound satisfied: %s",
                cumreg_fs[-1], bound_fs_series[-1],
                summary["verdict"]["regret_within_bound_fs"])
    logger.info("  Fixed Share adapts faster: %s (FS=%d days, Hedge=%d days)",
                summary["verdict"]["fixed_share_adapts_faster"], adapt_fs, adapt_hedge)


if __name__ == "__main__":
    main()
