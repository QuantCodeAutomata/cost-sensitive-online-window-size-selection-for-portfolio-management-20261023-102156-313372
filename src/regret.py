"""
Cost-sensitive tracking regret computation (Section 2.2, Problem 2.4, Corollary 3.4).

Paper: Cost-Sensitive Online Window Size Selection for Portfolio Management
Authors: Yi-Chen Liu, Chung-Han Hsieh (arXiv:2609.29887v2)
"""

from __future__ import annotations

import numpy as np
from src.online_learning import compute_C_func, compute_A_T


def compute_best_comparator_dp(
    losses: np.ndarray,
    K: int,
) -> tuple[float, np.ndarray]:
    """
    Dynamic program to find the best expert sequence with at most K switches.

    Recurrence (Section 2.2, Definition 2.2):
        D[t, j, k] = losses[t, j] + min(D[t-1, j, k], min_{h!=j} D[t-1, h, k-1])
    Initialized at D[0, j, k] = losses[0, j] for all j, k.

    Parameters
    ----------
    losses : np.ndarray, shape (T, n)
        tilde_l_j(t) for each period t and expert j.
    K : int
        Maximum number of switches allowed.

    Returns
    -------
    best_loss : float
        Minimum cumulative loss over comparator class J_{T,K}.
    best_sequence : np.ndarray, shape (T,)
        Indices into experts for the optimal sequence (0-indexed).
    """
    T, n = losses.shape
    K = min(K, T - 1)  # K can be at most T-1

    # DP table: D[t, j, k] = min cumulative loss ending at expert j with k switches so far
    INF = 1e18
    # Use two rolling arrays for memory efficiency: current and previous timestep
    # Shape: (n, K+1)
    D_prev = losses[0][:, None] * np.ones((n, K + 1))  # D[0, j, k] = losses[0, j] for all k

    # For backtracking: store argmin at each step
    # choice[t, j, k] = (best_j_prev, k_prev) — tuples stored as flat index for speed
    choice = np.zeros((T, n, K + 1, 2), dtype=np.int32)  # [prev_j, prev_k]

    for t in range(1, T):
        D_curr = np.full((n, K + 1), INF)
        for j in range(n):
            for k in range(K + 1):
                # Option 1: stay on expert j (no switch)
                if D_prev[j, k] < INF:
                    val_stay = D_prev[j, k]
                else:
                    val_stay = INF
                best_prev_j, best_prev_k = j, k

                # Option 2: switch from some h != j (requires k >= 1)
                if k >= 1:
                    # Find min over h != j of D_prev[h, k-1]
                    best_switch_val = INF
                    best_h = -1
                    for h in range(n):
                        if h != j and D_prev[h, k - 1] < best_switch_val:
                            best_switch_val = D_prev[h, k - 1]
                            best_h = h
                    if best_switch_val < val_stay:
                        val_stay = best_switch_val
                        best_prev_j, best_prev_k = best_h, k - 1

                D_curr[j, k] = losses[t, j] + val_stay
                choice[t, j, k, 0] = best_prev_j
                choice[t, j, k, 1] = best_prev_k
        D_prev = D_curr

    # Find minimal terminal state
    best_loss = INF
    best_j_final = 0
    best_k_final = 0
    for j in range(n):
        for k in range(K + 1):
            if D_prev[j, k] < best_loss:
                best_loss = D_prev[j, k]
                best_j_final = j
                best_k_final = k

    # Backtrack to recover sequence
    sequence = np.zeros(T, dtype=np.int32)
    sequence[T - 1] = best_j_final
    cur_j, cur_k = best_j_final, best_k_final
    for t in range(T - 1, 0, -1):
        prev_j = choice[t, cur_j, cur_k, 0]
        prev_k = choice[t, cur_j, cur_k, 1]
        sequence[t - 1] = prev_j
        cur_j, cur_k = prev_j, prev_k

    return float(best_loss), sequence


def compute_best_comparator_dp_fast(
    losses: np.ndarray,
    K: int,
) -> tuple[float, np.ndarray]:
    """
    Vectorised DP for the K-switch comparator (faster for large n and T).

    Uses column-wise min operations to avoid the inner loops.

    Parameters
    ----------
    losses : np.ndarray, shape (T, n)
    K : int

    Returns
    -------
    best_loss : float
    best_sequence : np.ndarray, shape (T,)
    """
    T, n = losses.shape
    K = min(K, T - 1)
    INF = np.float64(1e18)

    # D[j, k] = min cumulative loss ending at expert j with k switches used so far
    D = np.full((n, K + 1), INF, dtype=np.float64)
    D[:, :] = losses[0, :, None]  # broadcast: D[j, k] = losses[0, j] for all k

    # Backtracking arrays: back_j[t, j, k] and back_k[t, j, k]
    back_j = np.zeros((T, n, K + 1), dtype=np.int32)
    back_k = np.zeros((T, n, K + 1), dtype=np.int32)

    for t in range(1, T):
        D_new = np.full((n, K + 1), INF, dtype=np.float64)
        bj = np.full((n, K + 1), -1, dtype=np.int32)
        bk = np.full((n, K + 1), -1, dtype=np.int32)

        # Option 1: stay — D_new[j, k] = D[j, k] + losses[t, j]
        stay_val = D + losses[t, :, None]  # shape (n, K+1)

        # Option 2: switch from best h != j at k-1
        # For each k >= 1, compute min over h for each target j
        switch_val = np.full((n, K + 1), INF, dtype=np.float64)
        switch_from_j = np.zeros((n, K + 1), dtype=np.int32)
        if K >= 1:
            for k in range(1, K + 1):
                col = D[:, k - 1]  # shape (n,)
                # For target j, best source h != j
                total_min = col.min()
                total_argmin = col.argmin()
                for j in range(n):
                    if total_argmin != j:
                        switch_val[j, k] = total_min + losses[t, j]
                        switch_from_j[j, k] = total_argmin
                    else:
                        # best excluding j
                        if n > 1:
                            masked = col.copy()
                            masked[j] = INF
                            best_h = masked.argmin()
                            if masked[best_h] < INF:
                                switch_val[j, k] = masked[best_h] + losses[t, j]
                                switch_from_j[j, k] = best_h

        # Choose best option for each (j, k)
        use_switch = (switch_val < stay_val) & (switch_val < INF)
        D_new = np.where(use_switch, switch_val, stay_val)
        # Record backtrack
        for j in range(n):
            for k in range(K + 1):
                if use_switch[j, k]:
                    bj[j, k] = switch_from_j[j, k]
                    bk[j, k] = k - 1
                else:
                    bj[j, k] = j
                    bk[j, k] = k

        back_j[t] = bj
        back_k[t] = bk
        D = D_new

    # Find best terminal state
    best_idx = np.unravel_index(np.argmin(D), D.shape)
    best_j_final, best_k_final = int(best_idx[0]), int(best_idx[1])
    best_loss = float(D[best_j_final, best_k_final])

    # Backtrack
    sequence = np.zeros(T, dtype=np.int32)
    sequence[T - 1] = best_j_final
    cur_j, cur_k = best_j_final, best_k_final
    for t in range(T - 1, 0, -1):
        prev_j = int(back_j[t, cur_j, cur_k])
        prev_k = int(back_k[t, cur_j, cur_k])
        sequence[t - 1] = prev_j
        cur_j, cur_k = prev_j, prev_k

    return float(best_loss), sequence


def compute_tracking_regret(
    agg_full_losses: np.ndarray,
    expert_losses: np.ndarray,
    K: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Compute cost-sensitive tracking regret (Problem 2.4, Eq. 4).

    R_track^c(T, K) = sum_t [l(w_hat(t),y(t)) + c*||w_hat(t)-w_hat(t-1)||_1]
                    - min_{j: S_T(j)<=K} sum_t tilde_l_{j_t}(t)

    Parameters
    ----------
    agg_full_losses : np.ndarray, shape (T,)
        Per-period cost-sensitive loss of aggregate portfolio.
    expert_losses : np.ndarray, shape (T, n)
        tilde_l_j(t) for each expert.
    K : int

    Returns
    -------
    cumulative_regret : np.ndarray, shape (T,)
        Cumulative regret up to each period.
    best_comparator_loss : float
    best_sequence : np.ndarray, shape (T,)
    """
    T = len(agg_full_losses)
    best_loss, best_seq = compute_best_comparator_dp_fast(expert_losses, K)
    # Comparator cumulative loss per period
    comparator_cum = np.cumsum(expert_losses[np.arange(T), best_seq])
    agg_cum = np.cumsum(agg_full_losses)
    cumulative_regret = agg_cum - comparator_cum
    return cumulative_regret, float(best_loss), best_seq


def compute_regret_bound(
    T: int,
    K: int,
    n: int,
    c: float,
    a: float,
    b: float,
    eta: float,
    alpha: float,
) -> np.ndarray:
    """
    Compute the theoretical regret bound as a cumulative series (Corollary 3.4, Proposition 3.5).

    For K=0 (Hedge): bound = 2*sqrt(A_T*log(n))
    For K>=1 (Fixed Share): bound = 2*sqrt(A_T*C(K,alpha,n,T)) + 2*(1-1/n)*alpha*C_T

    The bound grows monotonically. We return a cumulative approximation by scaling linearly.

    Parameters
    ----------
    T, K, n, c, a, b, eta, alpha : as defined in Proposition 3.5.

    Returns
    -------
    bound_series : np.ndarray, shape (T,)
        Cumulative theoretical upper bound up to each period t.
    """
    A_T = compute_A_T(T, c, a, b)
    C_T = c * T

    if K == 0 or alpha == 0.0:
        total_bound = 2.0 * np.sqrt(A_T * np.log(n))
    else:
        C_val = compute_C_func(K, alpha, n, T)
        total_bound = (
            2.0 * np.sqrt(A_T * C_val)
            + 2.0 * (1.0 - 1.0 / n) * alpha * C_T
        )

    # Bound is a finite-horizon quantity; display as linearly increasing series
    return np.linspace(0, total_bound, T)


def compute_stock_regret(
    agg_full_losses: np.ndarray,
    stock_log_losses: np.ndarray,
    K: int,
) -> tuple[np.ndarray, float]:
    """
    Compute R_stock: tracking regret against the best stock sequence.

    Parameters
    ----------
    agg_full_losses : np.ndarray, shape (T,)
        Per-period cost-sensitive log-return loss of aggregate portfolio.
    stock_log_losses : np.ndarray, shape (T, m)
        Per-period log-return loss for each individual stock:
        loss_i(t) = -log(1 + y_i(t)).
    K : int
        Switching budget for the stock comparator.

    Returns
    -------
    cumulative_stock_regret : np.ndarray, shape (T,)
    best_comparator_loss : float
    """
    best_loss, best_seq = compute_best_comparator_dp_fast(stock_log_losses, K)
    T = len(agg_full_losses)
    comparator_cum = np.cumsum(stock_log_losses[np.arange(T), best_seq])
    agg_cum = np.cumsum(agg_full_losses)
    return agg_cum - comparator_cum, float(best_loss)


def compute_switching_budget(T: int, epsilon: float = 0.1) -> int:
    """
    Compute K(T) = floor(T / (log(T) * (log(log(T)))^(1+epsilon))) (Appendix B.4).

    Parameters
    ----------
    T : int
    epsilon : float
        Positive constant (assumed 0.1).

    Returns
    -------
    K : int
    """
    if T <= 2:
        return 0
    log_T = np.log(T)
    log_log_T = np.log(log_T) if log_T > 1 else 1e-10
    if log_log_T <= 0:
        return 0
    denom = log_T * (log_log_T ** (1.0 + epsilon))
    K = int(np.floor(T / denom))
    return max(K, 0)
