"""
Experiment 2: Synthetic robustness to regime-switch timing and expert-set size
(Appendix B.2, Figures 3 and 4).

Run as: python -m exp.exp2
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd

from data.generate_synthetic import generate_synthetic_returns, get_regime_change_date
from src.config import (
    TRANSACTION_COST,
    WINDOWS_SMALL,
    WINDOWS_LARGE,
    SYN_N_STOCKS,
    SYN_T,
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
from src.regret import compute_tracking_regret
from src.plotting import plot_aggregation_weights, plot_tracking_regret

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = "results"
K_SYNTHETIC = 1

# Alternative regime-switch timings (Appendix B.2, Figure 3; exact values assumed)
TIMING_VARIANTS = [500, 1000, 1500]  # first observation in second regime (1-indexed)

# Expert-set size scenarios (Appendix B.2, Figure 4)
EXPERT_SET_VARIANTS = [WINDOWS_SMALL, WINDOWS_LARGE]


def run_scenario(
    regime_change: int,
    windows: list[int],
    shared_innovations: np.ndarray,
    label: str,
) -> dict:
    """Run one robustness scenario and return a results dictionary."""
    n_presample = max(windows)
    total = n_presample + SYN_T

    # Regenerate from shared innovations but with scenario-specific j_t schedule
    rng = np.random.default_rng(SYN_SEED)
    mu = np.full(SYN_N_STOCKS, SYN_MU_DAILY)

    all_returns = np.zeros((total, SYN_N_STOCKS))
    all_returns[:n_presample] = shared_innovations[:n_presample]

    j_schedule = np.full(SYN_T, SYN_J2, dtype=int)
    j_schedule[:regime_change] = SYN_J1

    for t_eval in range(SYN_T):
        t_abs = n_presample + t_eval
        j_t = int(j_schedule[t_eval])
        past_slice = all_returns[t_abs - j_t : t_abs]
        rolling_mean = past_slice.mean(axis=0)
        eps = shared_innovations[n_presample + t_eval]
        all_returns[t_abs] = mu + SYN_ALPHA * (rolling_mean - mu) + eps

    eval_returns = all_returns[n_presample:]
    T_eval, m = eval_returns.shape
    n = len(windows)
    w0 = np.ones(m) / m

    # Expert portfolios
    logger.info("[%s] Computing %d expert portfolios...", label, n)
    expert_weights, expert_losses = run_window_experts(all_returns, windows, TRANSACTION_COST, w0)

    # Parameters
    eta_h, _, _ = select_parameters(T_eval, 0, n, TRANSACTION_COST, LOSS_A, LOSS_B)
    eta_f, alpha_f, _ = select_parameters(T_eval, K_SYNTHETIC, n, TRANSACTION_COST, LOSS_A, LOSS_B)

    # Run Hedge
    _, q_hedge, agg_l_hedge, _ = run_aggregation_with_returns(
        eval_returns, expert_weights, expert_losses, eta_h, 0.0, w0, TRANSACTION_COST
    )
    # Run Fixed Share
    _, q_fs, agg_l_fs, _ = run_aggregation_with_returns(
        eval_returns, expert_weights, expert_losses, eta_f, alpha_f, w0, TRANSACTION_COST
    )

    cumreg_h, _, _ = compute_tracking_regret(agg_l_hedge, expert_losses, K_SYNTHETIC)
    cumreg_f, _, _ = compute_tracking_regret(agg_l_fs, expert_losses, K_SYNTHETIC)

    # Adaptation time: post-regime-change dominance of SYN_J2 expert
    w2_idx = windows.index(SYN_J2) if SYN_J2 in windows else 0

    def adapt_time(q_hist: np.ndarray, start: int) -> int:
        for i in range(start, T_eval):
            if np.argmax(q_hist[i]) == w2_idx:
                return i - start
        return T_eval - start

    adapt_h = adapt_time(q_hedge, regime_change)
    adapt_f = adapt_time(q_fs, regime_change)

    return {
        "label": label,
        "regime_change": regime_change,
        "windows": windows,
        "n_experts": n,
        "final_regret_hedge": float(cumreg_h[-1]),
        "final_regret_fs": float(cumreg_f[-1]),
        "adaptation_hedge": int(adapt_h),
        "adaptation_fs": int(adapt_f),
        "fixed_share_adapts_faster": bool(adapt_f < adapt_h),
        "q_hedge": q_hedge,
        "q_fs": q_fs,
        "eval_dates": pd.bdate_range(start=SYN_START_DATE, periods=T_eval),
        "regime_change_date": get_regime_change_date(SYN_START_DATE, regime_change),
    }


def main() -> None:
    t0 = time.time()
    logger.info("=== Experiment 2: Robustness to Regime-Switch Timing and Expert-Set Size ===")

    # Generate shared innovations so timing variants are comparable
    rng_base = np.random.default_rng(SYN_SEED)
    n_presample_max = max(WINDOWS_LARGE)
    total = n_presample_max + SYN_T
    innovations = rng_base.normal(scale=SYN_SIGMA, size=(total, SYN_N_STOCKS))
    # Also pre-generate presample initial values (same as generate_synthetic)
    innovations[:n_presample_max] = rng_base.normal(
        loc=SYN_MU_DAILY, scale=SYN_SIGMA, size=(n_presample_max, SYN_N_STOCKS)
    )

    all_results = []

    # Scenario group 1: Timing sensitivity with N={5,10,21,63}
    logger.info("--- Timing sensitivity (Figure 3) ---")
    for rc in TIMING_VARIANTS:
        label = f"timing_rc{rc}_N4"
        logger.info("Scenario: %s", label)
        result = run_scenario(rc, WINDOWS_SMALL, innovations, label)
        all_results.append(result)

        # Plot weight paths
        plot_aggregation_weights(
            dates=result["eval_dates"],
            q_history=result["q_hedge"],
            windows=WINDOWS_SMALL,
            regime_change_date=result["regime_change_date"],
            filename=f"{RESULTS_DIR}/exp2_{label}_hedge_weights.png",
            title=f"Figure 3a: Hedge Weights — Regime Change t={rc} (N=4)",
        )
        plot_aggregation_weights(
            dates=result["eval_dates"],
            q_history=result["q_fs"],
            windows=WINDOWS_SMALL,
            regime_change_date=result["regime_change_date"],
            filename=f"{RESULTS_DIR}/exp2_{label}_fs_weights.png",
            title=f"Figure 3b: Fixed Share Weights — Regime Change t={rc} (N=4)",
        )

    # Scenario group 2: Expert-set size N=8 (midpoint timing)
    logger.info("--- Expert-set size N=8 (Figure 4) ---")
    label8 = "expert_set_N8_rc1000"
    logger.info("Scenario: %s", label8)
    result8 = run_scenario(1000, WINDOWS_LARGE, innovations, label8)
    all_results.append(result8)

    plot_aggregation_weights(
        dates=result8["eval_dates"],
        q_history=result8["q_hedge"],
        windows=WINDOWS_LARGE,
        regime_change_date=result8["regime_change_date"],
        filename=f"{RESULTS_DIR}/exp2_{label8}_hedge_weights.png",
        title="Figure 4a: Hedge Weights — N=8 Expert Set",
    )
    plot_aggregation_weights(
        dates=result8["eval_dates"],
        q_history=result8["q_fs"],
        windows=WINDOWS_LARGE,
        regime_change_date=result8["regime_change_date"],
        filename=f"{RESULTS_DIR}/exp2_{label8}_fs_weights.png",
        title="Figure 4b: Fixed Share Weights — N=8 Expert Set",
    )

    # Save summary JSON
    def _to_serializable(v: object) -> object:
        if isinstance(v, (np.ndarray, pd.DatetimeIndex)):
            return None
        if isinstance(v, (pd.Timestamp,)):
            return str(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        return v

    summary_rows = [
        {k: _to_serializable(v) for k, v in r.items()
         if not isinstance(v, (np.ndarray, pd.DatetimeIndex))}
        for r in all_results
    ]
    out_path = f"{RESULTS_DIR}/exp2_results.json"
    with open(out_path, "w") as f:
        json.dump({"experiment": "exp2", "scenarios": summary_rows}, f, indent=2,
                  default=_to_serializable)

    elapsed = time.time() - t0
    logger.info("Exp 2 complete in %.1f seconds.", elapsed)
    for r in all_results:
        logger.info(
            "  %s | adapt Hedge=%d Fixed Share=%d | FS faster=%s",
            r["label"], r["adaptation_hedge"], r["adaptation_fs"],
            r["fixed_share_adapts_faster"],
        )


if __name__ == "__main__":
    main()
