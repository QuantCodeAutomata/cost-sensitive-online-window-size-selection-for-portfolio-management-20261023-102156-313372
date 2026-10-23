"""
Hedge and Fixed Share online learning algorithms (Section 3.2, Eqs. 9-10).

Paper: Cost-Sensitive Online Window Size Selection for Portfolio Management
Authors: Yi-Chen Liu, Chung-Han Hsieh (arXiv:2609.29887v2)
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Core update rules
# ---------------------------------------------------------------------------

def hedge_update(
    log_q: np.ndarray,
    losses: np.ndarray,
    eta: float,
) -> np.ndarray:
    """
    Hedge update (Section 3.2, Remark 3.2): Fixed Share with alpha=0.

    q_j(t+1) = q_j(t) * exp(-eta * tilde_l_j(t)) / Z  (Eq. 9)

    Parameters
    ----------
    log_q : np.ndarray, shape (n,)
        Log aggregation weights at current time.
    losses : np.ndarray, shape (n,)
        tilde_l_j(t) for each expert.
    eta : float
        Learning rate.

    Returns
    -------
    log_q_next : np.ndarray, shape (n,)
        Updated log weights (normalized).
    """
    log_unnorm = log_q - eta * losses
    log_z = logsumexp(log_unnorm)
    return log_unnorm - log_z


def fixed_share_update(
    log_q: np.ndarray,
    losses: np.ndarray,
    eta: float,
    alpha: float,
    n: int,
) -> np.ndarray:
    """
    Fixed Share update (Section 3.2, Eqs. 9-10).

    Step 1 — Exponential weighting (Eq. 9):
        q_j^exp(t) = q_j(t) * exp(-eta * tilde_l_j(t)) / Z

    Step 2 — Sharing (Eq. 10):
        q_j(t+1) = (1 - alpha) * q_j^exp(t) + alpha / n

    Parameters
    ----------
    log_q : np.ndarray, shape (n,)
    losses : np.ndarray, shape (n,)
    eta : float
    alpha : float
        Mixing parameter in [0, 1].
    n : int
        Number of experts.

    Returns
    -------
    log_q_next : np.ndarray, shape (n,)
    """
    if alpha == 0.0:
        return hedge_update(log_q, losses, eta)
    # Step 1: exponential weighting (log-sum-exp stabilised)
    log_unnorm = log_q - eta * losses
    log_z = logsumexp(log_unnorm)
    q_exp = np.exp(log_unnorm - log_z)  # shape (n,)
    # Step 2: sharing
    q_next = (1.0 - alpha) * q_exp + alpha / n
    # Renormalize to handle floating-point errors
    q_next = np.maximum(q_next, 0.0)
    q_next /= q_next.sum()
    return np.log(q_next)


# ---------------------------------------------------------------------------
# Parameter selection (Proposition 3.5)
# ---------------------------------------------------------------------------

def compute_A_T(
    T: int,
    c: float,
    a: float,
    b: float,
) -> float:
    """
    Compute A_T used in Proposition 3.5 for constant transaction cost.

    A_T = (1/8) * sum_{t=1}^T (b-a+2c)^2 + sum_{t=2}^T c*(b-a+2c(t-1))
    For constant c: A_T = (1/8)*T*(b-a+2c)^2 + (T-1)*c*(b-a+2c)

    Parameters
    ----------
    T, c, a, b : scalars as defined in Proposition 3.5.
    """
    span = b - a + 2.0 * c
    A_T = (T / 8.0) * span ** 2 + (T - 1) * c * span
    return float(A_T)


def compute_C_func(
    K: int,
    alpha: float,
    n: int,
    T: int,
) -> float:
    """
    Compute C(K, alpha, n, T) from Corollary 3.4.

    C = (K+1)*log(n) + K*log(1/alpha) + (T-K-1)*log(1/(1-alpha))  (alpha > 0)
    C = log(n)  if K == 0 and alpha == 0
    C = inf     if K > 0 and alpha == 0
    """
    if alpha == 0.0:
        return float(np.log(n)) if K == 0 else float("inf")
    part1 = (K + 1) * np.log(n)
    part2 = K * np.log(1.0 / alpha)
    part3 = (T - K - 1) * np.log(1.0 / (1.0 - alpha))
    return float(part1 + part2 + part3)


def select_parameters(
    T: int,
    K: int,
    n: int,
    c: float,
    a: float,
    b: float,
) -> tuple[float, float, float]:
    """
    Select optimal eta and alpha via Proposition 3.5.

    Returns
    -------
    eta : float
    alpha : float
    bound : float
        Theoretical tracking regret upper bound.
    """
    A_T = compute_A_T(T, c, a, b)
    C_T = c * T  # constant c

    if C_T == 0.0:
        # Case 1: no transaction costs
        alpha_star = K / (T - 1) if T > 1 else 0.0
        if K == 0 or T <= 1:
            H_val = 0.0
        else:
            x = alpha_star
            H_val = -x * np.log(x) - (1.0 - x) * np.log(1.0 - x)
        C_arg = (K + 1) * np.log(n) + (T - 1) * H_val
        span = b - a
        eta_star = (1.0 / span) * np.sqrt(8.0 * C_arg / T)
        bound = span * np.sqrt(T / 2.0 * C_arg)
        return float(eta_star), float(alpha_star), float(bound)

    if K == 0:
        # Case 2: c > 0, K = 0
        alpha_star = 0.0
        eta_star = np.sqrt(np.log(n) / A_T)
        bound = 2.0 * np.sqrt(A_T * np.log(n))
        return float(eta_star), float(alpha_star), float(bound)

    # Case 3: c > 0, K >= 1
    # Solve stationarity equation numerically (Proposition 3.5, Case 3)
    # (K-(T-1)*alpha)/(alpha*(1-alpha)) = 2*(1-1/n)*C_T*sqrt(C(K,alpha,n,T)/A_T)
    alpha_upper = K / (T - 1) - 1e-10

    def equation(alpha: float) -> float:
        C_val = compute_C_func(K, alpha, n, T)
        lhs = (K - (T - 1) * alpha) / (alpha * (1.0 - alpha))
        rhs = 2.0 * (1.0 - 1.0 / n) * C_T * np.sqrt(C_val / A_T)
        return lhs - rhs

    alpha_low = 1e-10
    try:
        fa = equation(alpha_low)
        fb = equation(alpha_upper)
        if fa * fb > 0:
            # Both same sign — use boundary that gives smaller bound
            alpha_star = alpha_upper if abs(alpha_upper) < abs(alpha_low) else alpha_low
        else:
            alpha_star = brentq(equation, alpha_low, alpha_upper, xtol=1e-12, maxiter=200)
    except Exception:
        alpha_star = alpha_upper / 2.0

    C_star = compute_C_func(K, alpha_star, n, T)
    eta_star = np.sqrt(C_star / A_T)
    bound = (
        2.0 * np.sqrt(A_T * C_star)
        + 2.0 * (1.0 - 1.0 / n) * alpha_star * C_T
    )
    return float(eta_star), float(alpha_star), float(bound)


# ---------------------------------------------------------------------------
# Full simulation: run Hedge or Fixed Share over T steps
# ---------------------------------------------------------------------------

def run_aggregation(
    expert_weights: np.ndarray,
    expert_losses: np.ndarray,
    eta: float,
    alpha: float,
    w_hat_prev: np.ndarray,
    c: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run Hedge (alpha=0) or Fixed Share (alpha>0) over all T periods.

    Parameters
    ----------
    expert_weights : np.ndarray, shape (T, n, m)
    expert_losses : np.ndarray, shape (T, n)
        tilde_l_j(t) for each t and j.
    eta : float
    alpha : float
    w_hat_prev : np.ndarray, shape (m,)
        Initial aggregate portfolio w_hat(0).
    c : float

    Returns
    -------
    agg_weights : np.ndarray, shape (T, m)
        Aggregate portfolio w_hat(t) at each period.
    q_history : np.ndarray, shape (T, n)
        Aggregation weights q(t) BEFORE update at each period.
    agg_losses : np.ndarray, shape (T,)
        Cost-sensitive loss of aggregate: -w_hat(t)'y(t) + c*||w_hat(t)-w_hat(t-1)||_1.
    log_q_history : np.ndarray, shape (T+1, n)
        Full log-weight history including initial q(1).
    """
    T, n, m = expert_weights.shape
    log_q = np.log(np.ones(n) / n)  # q_j(1) = 1/n

    agg_weights = np.zeros((T, m))
    q_history = np.zeros((T, n))
    agg_losses = np.zeros(T)
    log_q_history = np.zeros((T + 1, n))
    log_q_history[0] = log_q

    w_hat_t_minus_1 = w_hat_prev.copy()

    for t in range(T):
        q_t = np.exp(log_q)  # current aggregation weights
        q_history[t] = q_t

        # Aggregate portfolio w_hat(t) = sum_j q_j(t) * w_j(t) (Section 2.1)
        w_hat_t = q_t @ expert_weights[t]  # shape (m,)
        w_hat_t = np.maximum(w_hat_t, 0.0)
        w_hat_t /= w_hat_t.sum()
        agg_weights[t] = w_hat_t

        # Cost-sensitive aggregate loss: l(w_hat(t), y(t)) + c*||w_hat(t)-w_hat(t-1)||_1
        # Note: y(t) is inferred from expert_losses and expert_weights via loss convention
        # We compute directly from: agg_loss_t = sum_j q_j(t)*tilde_l_j(t) + aggregate turnover term
        # But the actual regret uses: sum_t [l(w_hat,y) + c*||w_hat(t)-w_hat(t-1)||_1]
        # We store the aggregate turnover; prediction loss is implicit in regret formula
        agg_turnover = np.sum(np.abs(w_hat_t - w_hat_t_minus_1))
        agg_losses[t] = agg_turnover  # store only the turnover part here (see regret computation)

        # Update weights after observing y(t)
        if alpha == 0.0:
            log_q = hedge_update(log_q, expert_losses[t], eta)
        else:
            log_q = fixed_share_update(log_q, expert_losses[t], eta, alpha, n)
        log_q_history[t + 1] = log_q

        w_hat_t_minus_1 = w_hat_t

    return agg_weights, q_history, agg_losses, log_q_history


def run_aggregation_with_returns(
    returns: np.ndarray,
    expert_weights: np.ndarray,
    expert_losses: np.ndarray,
    eta: float,
    alpha: float,
    w_hat_prev: np.ndarray,
    c: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run aggregation and compute full cost-sensitive aggregate loss with realized returns.

    Parameters
    ----------
    returns : np.ndarray, shape (T, m)
        Realized returns for evaluation period.
    expert_weights : np.ndarray, shape (T, n, m)
    expert_losses : np.ndarray, shape (T, n)
    eta, alpha : float
    w_hat_prev : np.ndarray, shape (m,)
    c : float

    Returns
    -------
    agg_weights : np.ndarray, shape (T, m)
    q_history : np.ndarray, shape (T, n)
    agg_full_losses : np.ndarray, shape (T,)
        l(w_hat(t), y(t)) + c*||w_hat(t)-w_hat(t-1)||_1 per period.
    log_q_history : np.ndarray, shape (T+1, n)
    """
    T, n, m = expert_weights.shape
    log_q = np.log(np.ones(n) / n)

    agg_weights = np.zeros((T, m))
    q_history = np.zeros((T, n))
    agg_full_losses = np.zeros(T)
    log_q_history = np.zeros((T + 1, n))
    log_q_history[0] = log_q

    w_hat_prev_ = w_hat_prev.copy()

    for t in range(T):
        q_t = np.exp(log_q)
        q_history[t] = q_t

        w_hat_t = q_t @ expert_weights[t]
        w_hat_t = np.maximum(w_hat_t, 0.0)
        w_hat_t /= w_hat_t.sum()
        agg_weights[t] = w_hat_t

        # Full cost-sensitive aggregate loss (Problem 2.4, Eq. 4)
        pred_loss = -np.dot(w_hat_t, returns[t])
        turnover = np.sum(np.abs(w_hat_t - w_hat_prev_))
        agg_full_losses[t] = pred_loss + c * turnover

        if alpha == 0.0:
            log_q = hedge_update(log_q, expert_losses[t], eta)
        else:
            log_q = fixed_share_update(log_q, expert_losses[t], eta, alpha, n)
        log_q_history[t + 1] = log_q
        w_hat_prev_ = w_hat_t

    return agg_weights, q_history, agg_full_losses, log_q_history
