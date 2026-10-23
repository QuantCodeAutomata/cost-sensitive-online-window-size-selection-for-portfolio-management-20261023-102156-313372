"""
Visualization utilities (Figures 1-2, Appendix B figures).

Paper: Cost-Sensitive Online Window Size Selection for Portfolio Management
Authors: Yi-Chen Liu, Chung-Han Hsieh (arXiv:2609.29887v2)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Consistent color palette across experiments
_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f"]


def plot_tracking_regret(
    dates: pd.DatetimeIndex | np.ndarray,
    hedge_regret: np.ndarray,
    fs_regret: np.ndarray,
    bound_hedge: np.ndarray | None,
    bound_fs: np.ndarray | None,
    regime_change_date: str | None,
    filename: str,
    title: str = "Cost-Sensitive Tracking Regret (Figure 1)",
) -> None:
    """
    Plot cumulative tracking regret for Hedge and Fixed Share (Figure 1 / Appendix B).

    Parameters
    ----------
    dates : DatetimeIndex or array of dates (length T)
    hedge_regret, fs_regret : np.ndarray, shape (T,)
    bound_hedge, bound_fs : optional theoretical upper bounds (T,)
    regime_change_date : optional vertical line date
    filename : output path
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, hedge_regret, label="Hedge", color=_COLORS[0], lw=1.5)
    ax.plot(dates, fs_regret, label="Fixed Share", color=_COLORS[1], lw=1.5)
    if bound_hedge is not None:
        ax.plot(dates, bound_hedge, "--", label="Theoretical Bound (Hedge)", color=_COLORS[0], alpha=0.6)
    if bound_fs is not None:
        ax.plot(dates, bound_fs, ":", label="Theoretical Bound (Fixed Share)", color=_COLORS[1], alpha=0.6)
    if regime_change_date is not None:
        ax.axvline(pd.Timestamp(regime_change_date), color="k", linestyle="--", alpha=0.7, label="Regime Change")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Cost-Sensitive Tracking Regret")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_aggregation_weights(
    dates: pd.DatetimeIndex | np.ndarray,
    q_history: np.ndarray,
    windows: list[int],
    regime_change_date: str | None,
    filename: str,
    title: str = "Aggregation Weights (Figure 2)",
) -> None:
    """
    Stacked area plot of aggregation weights q_j(t) (Figure 2 / Appendix B).

    Parameters
    ----------
    dates : length T
    q_history : np.ndarray, shape (T, n)
    windows : list of n window sizes
    regime_change_date : optional vertical line
    filename : output path
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = [f"w={j}" for j in windows]
    colors = _COLORS[:len(windows)]
    ax.stackplot(dates, q_history.T, labels=labels, colors=colors, alpha=0.8)
    if regime_change_date is not None:
        ax.axvline(pd.Timestamp(regime_change_date), color="k", linestyle="--", alpha=0.9, label="Regime Change")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Aggregation Weight q_j(t)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8, ncol=min(len(windows), 4))
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_account_value(
    dates: pd.DatetimeIndex | np.ndarray,
    strategies: dict[str, np.ndarray],
    filename: str,
    title: str = "Account Value",
) -> None:
    """
    Plot account value series for multiple strategies.

    Parameters
    ----------
    dates : length T+1 (including initial date)
    strategies : dict mapping label -> account_value array (T+1,)
    filename : output path
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, av) in enumerate(strategies.items()):
        ax.plot(dates, av, label=label, color=_COLORS[i % len(_COLORS)], lw=1.5)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (initial = 1)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_stock_regret(
    dates: pd.DatetimeIndex | np.ndarray,
    regret_series: dict[str, np.ndarray],
    filename: str,
    title: str = "R_stock — Regret vs. Best Stock Sequence",
) -> None:
    """
    Plot cumulative R_stock for all strategies (Figure 5 / Appendix B.3).
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, rs) in enumerate(regret_series.items()):
        ax.plot(dates, rs, label=label, color=_COLORS[i % len(_COLORS)], lw=1.5)
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative R_stock")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
