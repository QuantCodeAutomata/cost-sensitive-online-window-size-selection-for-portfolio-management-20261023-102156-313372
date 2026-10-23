"""
Portfolio performance metrics (Appendix B.1, Table 1-3).

Paper: Cost-Sensitive Online Window Size Selection for Portfolio Management
Authors: Yi-Chen Liu, Chung-Han Hsieh (arXiv:2609.29887v2)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.config import ANNUALIZATION


def compute_net_returns(
    weights: np.ndarray,
    returns: np.ndarray,
    c: float,
    w_prev: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute per-period net return after proportional transaction cost.

    net_return(t) = w(t)' y(t) - c * ||w(t) - w(t-1)||_1

    Parameters
    ----------
    weights : np.ndarray, shape (T, m)
    returns : np.ndarray, shape (T, m)
    c : float
    w_prev : np.ndarray or None
        Portfolio before period 1; defaults to uniform.

    Returns
    -------
    net_ret : np.ndarray, shape (T,)
    """
    T, m = weights.shape
    if w_prev is None:
        w_prev = np.ones(m) / m
    w_prev_arr = np.vstack([w_prev[None, :], weights[:-1]])  # shape (T, m)
    turnover = np.sum(np.abs(weights - w_prev_arr), axis=1)  # shape (T,)
    gross_ret = np.sum(weights * returns, axis=1)  # shape (T,)
    net_ret = gross_ret - c * turnover
    return net_ret


def compute_account_value(
    net_returns: np.ndarray,
    v0: float = 1.0,
) -> np.ndarray:
    """
    Compute account value series (Appendix B.1).

    V(t+1) = V(t) * (1 + net_return(t))
    V(1) = v0.

    Parameters
    ----------
    net_returns : np.ndarray, shape (T,)
    v0 : float
        Initial account value.

    Returns
    -------
    account_value : np.ndarray, shape (T+1,)
        V(0), V(1), ..., V(T).
    """
    factors = 1.0 + net_returns
    if np.any(factors <= 0):
        # Clip to avoid undefined log; flag but do not raise
        factors = np.maximum(factors, 1e-10)
    v = np.empty(len(net_returns) + 1)
    v[0] = v0
    v[1:] = v0 * np.cumprod(factors)
    return v


def compute_cumulative_return(account_value: np.ndarray) -> float:
    """Cumulative return in percent: 100*(V_final/V_initial - 1)."""
    return float(100.0 * (account_value[-1] / account_value[0] - 1.0))


def compute_max_drawdown(account_value: np.ndarray) -> float:
    """
    Signed maximum drawdown (MDD) in percent.

    MDD = 100 * min_t (V_t / max_{s<=t} V_s - 1).
    Negative values indicate losses; values closer to 0 are better.
    """
    running_max = np.maximum.accumulate(account_value)
    drawdowns = account_value / running_max - 1.0
    return float(100.0 * np.min(drawdowns))


def compute_sharpe_ratio(net_returns: np.ndarray) -> float:
    """
    Annualized Sharpe ratio (no risk-free rate, annualization = 252 trading days).

    Sharpe = sqrt(252) * mean(r) / std(r)

    Parameters
    ----------
    net_returns : np.ndarray, shape (T,)
    """
    if len(net_returns) < 2:
        return np.nan
    mean_r = np.mean(net_returns)
    std_r = np.std(net_returns, ddof=1)
    if std_r < 1e-12:
        return np.nan
    return float(np.sqrt(ANNUALIZATION) * mean_r / std_r)


def compute_performance_metrics(
    weights: np.ndarray,
    returns: np.ndarray,
    c: float,
    w0: np.ndarray | None = None,
    label: str = "",
) -> dict:
    """
    Compute all trading performance metrics for one strategy (Tables 1-3).

    Parameters
    ----------
    weights : np.ndarray, shape (T, m)
    returns : np.ndarray, shape (T, m)
    c : float
    w0 : optional initial portfolio
    label : str

    Returns
    -------
    metrics : dict with keys:
        label, cumulative_return, sample_mean_return, sample_std_return,
        annualized_sharpe, max_drawdown, account_value
    """
    T, m = weights.shape
    if w0 is None:
        w0 = np.ones(m) / m
    w_prev_arr = np.vstack([w0[None, :], weights[:-1]])
    turnover_arr = np.sum(np.abs(weights - w_prev_arr), axis=1)
    total_turnover = float(np.sum(turnover_arr))

    net_ret = compute_net_returns(weights, returns, c, w0)
    av = compute_account_value(net_ret)
    mean_ret = float(100.0 * np.mean(net_ret))
    std_ret = float(np.std(net_ret, ddof=1))
    return {
        "label": label,
        "cumulative_return": compute_cumulative_return(av),
        "sample_mean_return": mean_ret,
        "mean_return": mean_ret,           # alias for test compatibility
        "sample_std_return": std_ret,
        "std_return": std_ret,             # alias for test compatibility
        "annualized_sharpe": compute_sharpe_ratio(net_ret),
        "max_drawdown": compute_max_drawdown(av),
        "total_turnover": total_turnover,
        "account_value": av,
        "net_returns": net_ret,
    }


def build_performance_table(metrics_list: list[dict]) -> pd.DataFrame:
    """
    Build a performance summary DataFrame from a list of metric dicts.
    """
    rows = []
    for m in metrics_list:
        rows.append({
            "Strategy": m["label"],
            "Cumulative Return (%)": round(m["cumulative_return"], 4),
            "Sample Mean Return (%)": round(m["sample_mean_return"], 4),
            "Sample Std": round(m["sample_std_return"], 4),
            "Annualized Sharpe": round(m["annualized_sharpe"], 4),
            "Max Drawdown (%)": round(m["max_drawdown"], 4),
        })
    return pd.DataFrame(rows).set_index("Strategy")
