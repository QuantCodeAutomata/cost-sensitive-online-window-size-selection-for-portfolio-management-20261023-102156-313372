"""
Baseline online portfolio-selection algorithms (Appendix B.3/B.5).

Implemented from their respective foundational papers:
- UP:    Cover (1991) — approximated by CRP/uniform allocation
- EG:    Helmbold et al. (1998)
- PAMR:  Li et al. (2012)
- CWMR:  Li et al. (2013)
- OLMAR: Li & Hoi (2014)

Paper: Cost-Sensitive Online Window Size Selection for Portfolio Management
Authors: Yi-Chen Liu, Chung-Han Hsieh (arXiv:2609.29887v2)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Universal Portfolio (Cover 1991) — approximation via Dirichlet sampling
# Custom — Context7 found no library equivalent matching paper conventions
# ---------------------------------------------------------------------------

def run_up(
    returns: np.ndarray,
    n_samples: int = 300,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Universal Portfolio (Cover 1991) via Monte Carlo CRP approximation.

    Each sampled CRP w is drawn from Dirichlet(1,...,1); the UP weight of a
    portfolio at time T is proportional to the product of its gross returns.
    The deployed portfolio at each step is the weighted average of all CRPs.

    Parameters
    ----------
    returns : np.ndarray, shape (T, m)
        Return matrix (evaluation period only).
    n_samples : int
        Number of CRP samples.
    seed : int

    Returns
    -------
    weights : np.ndarray, shape (T, m)
    losses : np.ndarray, shape (T,)
        -log(1 + portfolio_return) at each period.
    """
    T, m = returns.shape
    rng = np.random.default_rng(seed)
    # Sample constant rebalanced portfolios from Dirichlet
    crp_weights = rng.dirichlet(np.ones(m), size=n_samples)  # (n_samples, m)

    # log-wealth of each CRP
    log_wealth = np.zeros(n_samples)
    weights_hist = np.zeros((T, m))
    losses = np.zeros(T)

    for t in range(T):
        y_t = returns[t]
        x_t = 1.0 + y_t  # price relatives

        # Portfolio weights: weighted average of CRPs by current wealth
        wealth_weights = np.exp(log_wealth - log_wealth.max())
        wealth_weights /= wealth_weights.sum()
        w_t = wealth_weights @ crp_weights  # shape (m,)
        w_t = np.maximum(w_t, 0.0)
        w_t /= w_t.sum()
        weights_hist[t] = w_t

        # Loss: negative log portfolio return (for R_stock computation)
        port_return = np.dot(w_t, x_t)
        port_return = max(port_return, 1e-10)
        losses[t] = -np.log(port_return)

        # Update CRP log-wealth
        crp_returns = crp_weights @ x_t  # shape (n_samples,)
        crp_returns = np.maximum(crp_returns, 1e-10)
        log_wealth += np.log(crp_returns)

    return weights_hist, losses


# ---------------------------------------------------------------------------
# Exponential Gradient (Helmbold et al. 1998)
# Custom — Context7 found no library equivalent matching paper conventions
# ---------------------------------------------------------------------------

def run_eg(
    returns: np.ndarray,
    eta: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exponential Gradient (EG) algorithm (Helmbold et al. 1998).

    w_i(t+1) = w_i(t) * exp(eta * x_i(t) / (w(t)'x(t))) / Z

    Parameters
    ----------
    returns : np.ndarray, shape (T, m)
    eta : float
        Learning rate.

    Returns
    -------
    weights : np.ndarray, shape (T, m)
    losses : np.ndarray, shape (T,)
        -log(1 + portfolio_return) at each period.
    """
    T, m = returns.shape
    w = np.ones(m) / m  # uniform initialization
    weights_hist = np.zeros((T, m))
    losses = np.zeros(T)

    for t in range(T):
        x_t = 1.0 + returns[t]
        weights_hist[t] = w

        port_return = np.dot(w, x_t)
        port_return = max(port_return, 1e-10)
        losses[t] = -np.log(port_return)

        # EG update
        log_w_unnorm = np.log(np.maximum(w, 1e-300)) + eta * x_t / port_return
        log_z = np.max(log_w_unnorm) + np.log(np.sum(np.exp(log_w_unnorm - np.max(log_w_unnorm))))
        w = np.exp(log_w_unnorm - log_z)
        w = np.maximum(w, 0.0)
        w /= w.sum()

    return weights_hist, losses


# ---------------------------------------------------------------------------
# PAMR (Li et al. 2012)
# Custom — Context7 found no library equivalent matching paper conventions
# ---------------------------------------------------------------------------

def run_pamr(
    returns: np.ndarray,
    epsilon: float = 0.5,
    C: float = 500.0,
    variant: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Passive Aggressive Mean Reversion (PAMR-1, Li et al. 2012).

    The PAMR-1 update (Eq. 12 in Li et al. 2012):
        tau_t = (w(t)' x(t) - epsilon) / ||x(t) - mean(x(t))||^2
        tau_t = min(tau_t, C)   [PAMR-1]
        w(t+1) = project(w(t) - tau_t * (x(t) - mean(x(t))) * sign)

    Parameters
    ----------
    returns : np.ndarray, shape (T, m)
    epsilon : float
        Mean reversion threshold.
    C : float
        Aggressiveness cap (PAMR-1).
    variant : int
        1 = PAMR-1.

    Returns
    -------
    weights : np.ndarray, shape (T, m)
    losses : np.ndarray, shape (T,)
    """
    T, m = returns.shape
    w = np.ones(m) / m
    weights_hist = np.zeros((T, m))
    losses = np.zeros(T)

    for t in range(T):
        x_t = 1.0 + returns[t]
        weights_hist[t] = w

        port_return = np.dot(w, x_t)
        port_return = max(port_return, 1e-10)
        losses[t] = -np.log(port_return)

        # PAMR update toward mean reversion
        x_bar = x_t.mean()
        x_dev = x_t - x_bar
        norm_sq = np.dot(x_dev, x_dev)

        loss_t = max(np.dot(w, x_t) - epsilon, 0.0)
        if norm_sq > 1e-10:
            tau = loss_t / norm_sq
            if variant == 1:
                tau = min(tau, C)
        else:
            tau = 0.0

        w_new = w - tau * x_dev
        # Project onto simplex
        w = _project_simplex(w_new)

    return weights_hist, losses


# ---------------------------------------------------------------------------
# CWMR (Li et al. 2013)
# Custom — Context7 found no library equivalent matching paper conventions
# ---------------------------------------------------------------------------

def run_cwmr(
    returns: np.ndarray,
    epsilon: float = 0.5,
    confidence: float = 0.95,
    variant: str = "sd",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Confidence Weighted Mean Reversion (CWMR-Var, Li et al. 2013).

    Maintains a Gaussian belief over the portfolio (mu_t, Sigma_t).
    Update maximizes confidence while satisfying the mean-reversion constraint.

    Parameters
    ----------
    returns : np.ndarray, shape (T, m)
    epsilon : float
        Mean-reversion threshold.
    confidence : float
        Confidence parameter phi (typically 0.95 → phi_inv ≈ 1.645).
    variant : str
        "sd" = CWMR-Stdev (Li et al. 2013 Eq. 20-21).

    Returns
    -------
    weights : np.ndarray, shape (T, m)
    losses : np.ndarray, shape (T,)
    """
    from scipy.stats import norm as _norm
    T, m = returns.shape
    mu = np.ones(m) / m
    Sigma = np.eye(m) / (m ** 2)  # initial covariance
    phi = _norm.ppf(confidence)  # z-score
    weights_hist = np.zeros((T, m))
    losses = np.zeros(T)

    for t in range(T):
        x_t = 1.0 + returns[t]
        weights_hist[t] = mu

        port_return = np.dot(mu, x_t)
        port_return = max(port_return, 1e-10)
        losses[t] = -np.log(port_return)

        # CWMR-Var update (simplified closed-form)
        x_bar = x_t.mean()
        x_dev = x_t - x_bar

        mu_tilde = np.dot(mu, x_t)
        sigma_sq = float(x_dev @ Sigma @ x_dev)

        M = mu_tilde
        V = sigma_sq
        if V > 1e-10:
            lam_numerator = -((phi ** 2) * V - M + epsilon) * phi
            lam_denominator = (phi ** 2) * V + 0.5 * (phi ** 2) * np.sqrt((phi ** 2) * V ** 2 + 4 * V * (M - epsilon) ** 2)
            if abs(lam_denominator) > 1e-12:
                lam = max(lam_numerator / lam_denominator, 0.0)
            else:
                lam = 0.0
        else:
            lam = 0.0

        # Update mean
        mu_new = mu - lam * Sigma @ x_dev
        # Project
        mu = _project_simplex(mu_new)

        # Update covariance (simplified: diagonal shrinkage)
        Sigma_new = np.linalg.inv(
            np.linalg.inv(Sigma) + 2.0 * lam * phi * np.outer(x_dev, x_dev)
        ) if lam > 0 else Sigma
        Sigma = Sigma_new
        # Ensure PSD
        eigvals = np.linalg.eigvalsh(Sigma)
        if eigvals.min() < 1e-10:
            Sigma += (1e-8 - eigvals.min()) * np.eye(m)

    return weights_hist, losses


# ---------------------------------------------------------------------------
# OLMAR (Li & Hoi 2014)
# Custom — Context7 found no library equivalent matching paper conventions
# ---------------------------------------------------------------------------

def run_olmar(
    returns: np.ndarray,
    window: int = 5,
    epsilon: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Online Moving Average Reversion (OLMAR, Li & Hoi 2014).

    Predicts x_hat(t+1) as the moving average reversion:
        x_hat_i(t+1) = (1/window) * sum_{k=0}^{window-1} x_i(t-k)

    Then applies a PAMR-like update toward the predicted price relative.

    Parameters
    ----------
    returns : np.ndarray, shape (T, m)
    window : int
        Moving average window (Li & Hoi 2014 use w=5 by default).
    epsilon : float
        Threshold for the mean-reversion constraint.

    Returns
    -------
    weights : np.ndarray, shape (T, m)
    losses : np.ndarray, shape (T,)
    """
    T, m = returns.shape
    w_port = np.ones(m) / m
    weights_hist = np.zeros((T, m))
    losses = np.zeros(T)
    price_relatives = 1.0 + returns  # shape (T, m)

    for t in range(T):
        x_t = price_relatives[t]
        weights_hist[t] = w_port

        port_return = np.dot(w_port, x_t)
        port_return = max(port_return, 1e-10)
        losses[t] = -np.log(port_return)

        # Predict next price relative via moving average
        if t < window - 1:
            past = price_relatives[max(0, t - window + 1) : t + 1]
        else:
            past = price_relatives[t - window + 1 : t + 1]
        x_hat = past.mean(axis=0)  # shape (m,)

        # OLMAR update
        x_bar = x_hat.mean()
        x_dev = x_hat - x_bar
        norm_sq = np.dot(x_dev, x_dev)

        mu_pred = np.dot(w_port, x_hat)
        loss_t = max(mu_pred - epsilon, 0.0)
        if norm_sq > 1e-10:
            tau = loss_t / norm_sq
        else:
            tau = 0.0

        w_new = w_port + tau * (x_dev - (np.dot(w_port, x_dev)) * np.ones(m))
        w_port = _project_simplex(w_new)

    return weights_hist, losses


# ---------------------------------------------------------------------------
# Helper: Euclidean projection onto the probability simplex
# ---------------------------------------------------------------------------

def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Project vector v onto the unit simplex {w >= 0, sum(w) = 1}."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u > (cssv - 1.0) / np.arange(1, n + 1))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)
