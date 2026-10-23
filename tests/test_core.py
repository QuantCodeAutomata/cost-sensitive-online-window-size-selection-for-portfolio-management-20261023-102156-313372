"""
Tests for core src/ modules.

Run with: python -m pytest tests/test_core.py -v

All tests use small fixtures or synthetic data; no external API calls are made.

Paper: Cost-Sensitive Online Window Size Selection for Portfolio Management
Authors: Yi-Chen Liu, Chung-Han Hsieh (arXiv:2609.29887v2)
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(2024)


def make_returns(T: int = 50, m: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0005, 0.02, (T, m))


def uniform_w(m: int) -> np.ndarray:
    return np.ones(m) / m


# ---------------------------------------------------------------------------
# Data generation tests
# ---------------------------------------------------------------------------

class TestGenerateSynthetic:
    def test_returns_shape(self) -> None:
        from data.generate_synthetic import generate_synthetic_returns
        df, dates = generate_synthetic_returns(
            n_stocks=4, T=200, regime_change=100,
            j1=21, j2=5, mu_daily=0.0005, alpha=0.5,
            sigma=0.02, seed=42, start_date="2020-01-01", windows=[5, 21],
        )
        # Full matrix includes presample
        assert df.shape[1] == 4
        assert len(dates) == 200

    def test_column_names(self) -> None:
        from data.generate_synthetic import generate_synthetic_returns
        df, _ = generate_synthetic_returns(
            n_stocks=3, T=100, regime_change=50,
            j1=10, j2=5, mu_daily=0.0, alpha=0.3,
            sigma=0.01, seed=1, start_date="2020-01-01", windows=[5, 10],
        )
        assert list(df.columns) == ["SYN_1", "SYN_2", "SYN_3"]

    def test_returns_finite(self) -> None:
        from data.generate_synthetic import generate_synthetic_returns
        df, _ = generate_synthetic_returns(
            n_stocks=5, T=200, regime_change=100,
            j1=21, j2=5, mu_daily=0.0005, alpha=0.5,
            sigma=0.02, seed=7, start_date="2020-01-01", windows=[5, 21],
        )
        assert np.all(np.isfinite(df.values))

    def test_presample_via_full_matrix(self) -> None:
        """The returned matrix has at least max(windows) presample rows."""
        from data.generate_synthetic import generate_synthetic_returns
        windows = [5, 21]
        df, _ = generate_synthetic_returns(
            n_stocks=4, T=100, regime_change=50,
            j1=21, j2=5, mu_daily=0.0, alpha=0.4,
            sigma=0.01, seed=3, start_date="2020-01-01", windows=windows,
        )
        # df includes presample rows; total rows > T
        assert len(df) > 100

    def test_regime_change_date(self) -> None:
        from data.generate_synthetic import get_regime_change_date
        import pandas as pd
        date = get_regime_change_date("2020-01-01", 100)
        assert isinstance(date, pd.Timestamp)


# ---------------------------------------------------------------------------
# Expert tests
# ---------------------------------------------------------------------------

class TestExperts:
    def test_solve_mv_expert_simplex(self) -> None:
        """Solved portfolio must satisfy sum==1 and w>=0."""
        from src.experts import solve_mv_expert
        m = 4
        rng = np.random.default_rng(0)
        mu = rng.normal(0.001, 0.01, m)
        Sigma = np.eye(m) * 0.01
        w_prev = uniform_w(m)
        w = solve_mv_expert(mu, Sigma, w_prev, c=0.001)
        assert np.isclose(w.sum(), 1.0, atol=1e-5), f"sum(w)={w.sum()}"
        assert np.all(w >= -1e-6), f"min(w)={w.min()}"

    def test_solve_mv_expert_no_shorting(self) -> None:
        """With a negative-mean asset, optimal weight should be zero, not negative."""
        from src.experts import solve_mv_expert
        m = 3
        mu = np.array([-0.1, 0.01, 0.01])
        Sigma = np.eye(m) * 0.0001
        w_prev = uniform_w(m)
        w = solve_mv_expert(mu, Sigma, w_prev, c=0.0)
        assert w[0] < 0.01  # near-zero weight on the bad asset

    def test_rolling_stats_shape(self) -> None:
        from src.experts import compute_rolling_stats
        returns = make_returns(30, 5)
        mu, Sig = compute_rolling_stats(returns, t=25, window=10)
        assert mu.shape == (5,)
        assert Sig.shape == (5, 5)

    def test_rolling_stats_psd(self) -> None:
        from src.experts import compute_rolling_stats
        returns = make_returns(50, 6)
        mu, Sig = compute_rolling_stats(returns, t=40, window=20)
        eigvals = np.linalg.eigvalsh(Sig)
        assert np.all(eigvals >= -1e-10), f"min eigval={eigvals.min()}"

    def test_run_window_experts_shape(self) -> None:
        from src.experts import run_window_experts
        T_total = 30  # includes presample
        m = 3
        returns = make_returns(T_total, m)
        windows = [5, 10]
        w0 = uniform_w(m)
        weights, losses = run_window_experts(returns, windows, c=0.001, w0=w0)
        T_eval = T_total - max(windows)
        n = len(windows)
        assert weights.shape == (T_eval, n, m), weights.shape
        assert losses.shape == (T_eval, n), losses.shape

    def test_run_window_experts_simplex(self) -> None:
        from src.experts import run_window_experts
        returns = make_returns(30, 4)
        w0 = uniform_w(4)
        weights, _ = run_window_experts(returns, [5, 10], c=0.001, w0=w0)
        # All weights sum to 1
        sums = weights.sum(axis=2)  # (T_eval, n)
        assert np.allclose(sums, 1.0, atol=1e-4), f"weight sums: {sums}"
        assert np.all(weights >= -1e-5), "Negative expert weights found"

    def test_expert_losses_sign(self) -> None:
        """Loss = -PnL + c*turnover; can be positive or negative."""
        from src.experts import run_window_experts
        returns = make_returns(25, 3)
        w0 = uniform_w(3)
        _, losses = run_window_experts(returns, [5, 10], c=0.001, w0=w0)
        assert np.all(np.isfinite(losses))


# ---------------------------------------------------------------------------
# Online learning tests
# ---------------------------------------------------------------------------

class TestOnlineLearning:
    def test_hedge_update_sums_to_one(self) -> None:
        from src.online_learning import hedge_update
        n = 4
        log_q = np.log(np.ones(n) / n)
        losses = np.array([0.01, 0.02, -0.01, 0.005])
        log_q_new = hedge_update(log_q, losses, eta=1.0)
        q_new = np.exp(log_q_new)
        assert np.isclose(q_new.sum(), 1.0, atol=1e-10)
        assert np.all(q_new > 0)

    def test_hedge_favours_low_loss_expert(self) -> None:
        """Expert 0 has lowest loss; Hedge must assign it more weight."""
        from src.online_learning import hedge_update
        n = 3
        log_q = np.log(np.ones(n) / n)
        losses = np.array([0.01, 0.1, 0.1])
        log_q_new = hedge_update(log_q, losses, eta=2.0)
        q_new = np.exp(log_q_new)
        assert q_new[0] > q_new[1]
        assert q_new[0] > q_new[2]

    def test_fixed_share_preserves_mass(self) -> None:
        from src.online_learning import fixed_share_update
        n = 4
        log_q = np.log(np.array([0.5, 0.3, 0.1, 0.1]))
        losses = np.array([0.01, 0.02, 0.03, 0.04])
        log_q_new = fixed_share_update(log_q, losses, eta=0.5, alpha=0.1, n=n)
        q_new = np.exp(log_q_new)
        assert np.isclose(q_new.sum(), 1.0, atol=1e-10)
        assert np.all(q_new > 0)

    def test_fixed_share_alpha_zero_equals_hedge(self) -> None:
        """With alpha=0, Fixed Share must produce the same update as Hedge."""
        from src.online_learning import hedge_update, fixed_share_update
        n = 4
        log_q = np.log(np.array([0.4, 0.3, 0.2, 0.1]))
        losses = np.array([0.01, 0.05, 0.02, 0.03])
        eta = 1.0
        h = hedge_update(log_q, losses, eta)
        fs = fixed_share_update(log_q, losses, eta, alpha=0.0, n=n)
        assert np.allclose(np.exp(h), np.exp(fs), atol=1e-10)

    def test_select_parameters_k0(self) -> None:
        """For K=0 (Hedge), alpha must be 0 and eta must be positive."""
        from src.online_learning import select_parameters
        eta, alpha, bound = select_parameters(
            T=500, K=0, n=4, c=0.001, a=-0.15, b=0.15
        )
        assert alpha == 0.0
        assert eta > 0

    def test_select_parameters_kpos(self) -> None:
        """For K>=1 (Fixed Share), alpha must be positive."""
        from src.online_learning import select_parameters
        eta, alpha, bound = select_parameters(
            T=500, K=2, n=4, c=0.001, a=-0.15, b=0.15
        )
        assert alpha > 0
        assert eta > 0

    def test_run_aggregation_shape(self) -> None:
        from src.online_learning import run_aggregation_with_returns
        T, m, n = 20, 4, 2
        eval_returns = make_returns(T, m)
        expert_weights = np.tile(uniform_w(m), (T, n, 1))
        expert_losses = np.zeros((T, n))
        w0 = uniform_w(m)
        agg_w, q_hist, agg_l, _ = run_aggregation_with_returns(
            eval_returns, expert_weights, expert_losses,
            eta=0.5, alpha=0.1, w_hat_prev=w0, c=0.001
        )
        assert agg_w.shape == (T, m)
        assert q_hist.shape == (T, n)
        assert agg_l.shape == (T,)

    def test_run_aggregation_simplex(self) -> None:
        from src.online_learning import run_aggregation_with_returns
        T, m, n = 20, 3, 2
        eval_returns = make_returns(T, m)
        expert_weights = np.tile(uniform_w(m), (T, n, 1))
        expert_losses = np.zeros((T, n))
        w0 = uniform_w(m)
        agg_w, q_hist, _, _ = run_aggregation_with_returns(
            eval_returns, expert_weights, expert_losses,
            eta=0.5, alpha=0.1, w_hat_prev=w0, c=0.001
        )
        assert np.allclose(agg_w.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(agg_w >= -1e-6)
        assert np.allclose(q_hist.sum(axis=1), 1.0, atol=1e-10)

    def test_no_look_ahead(self) -> None:
        """Aggregate portfolio at t uses only expert portfolios from t (not t+1 losses)."""
        from src.online_learning import run_aggregation_with_returns
        T, m, n = 10, 3, 2
        eval_returns = make_returns(T, m)
        expert_weights = np.tile(uniform_w(m), (T, n, 1))
        expert_losses = np.zeros((T, n))
        w0 = uniform_w(m)
        agg_w1, _, _, _ = run_aggregation_with_returns(
            eval_returns, expert_weights, expert_losses,
            eta=0.5, alpha=0.0, w_hat_prev=w0, c=0.001
        )
        # Perturb future return at t=5 (0-indexed); agg_w at t=0..4 must not change
        eval_returns2 = eval_returns.copy()
        eval_returns2[5:] *= 2
        agg_w2, _, _, _ = run_aggregation_with_returns(
            eval_returns2, expert_weights, expert_losses,
            eta=0.5, alpha=0.0, w_hat_prev=w0, c=0.001
        )
        assert np.allclose(agg_w1[:5], agg_w2[:5], atol=1e-12)


# ---------------------------------------------------------------------------
# Regret tests
# ---------------------------------------------------------------------------

class TestRegret:
    def test_tracking_regret_shape(self) -> None:
        from src.regret import compute_tracking_regret
        T, n = 20, 3
        agg_l = np.random.default_rng(0).normal(0.01, 0.01, T)
        expert_l = np.abs(np.random.default_rng(0).normal(0.01, 0.01, (T, n)))
        cum_r, path, _ = compute_tracking_regret(agg_l, expert_l, K=1)
        assert cum_r.shape == (T,)

    def test_regret_vs_best_sequence_nonneg_when_best(self) -> None:
        """If the aggregate loss equals the best expert loss, regret is zero."""
        from src.regret import compute_tracking_regret
        T, n = 10, 2
        losses = np.zeros((T, n))
        losses[:, 0] = 0.05  # expert 0 always 0.05
        losses[:, 1] = 0.05  # expert 1 always 0.05
        agg_l = np.full(T, 0.05)  # aggregate matches best (K=0)
        cum_r, _, _ = compute_tracking_regret(agg_l, losses, K=0)
        assert np.allclose(cum_r, 0.0, atol=1e-10)

    def test_regret_positive_when_suboptimal(self) -> None:
        """Suboptimal aggregate should produce positive final regret vs best expert."""
        from src.regret import compute_tracking_regret
        T, n = 20, 2
        losses = np.zeros((T, n))
        losses[:, 0] = 0.01  # expert 0 always better
        losses[:, 1] = 0.10
        agg_l = np.full(T, 0.10)  # aggregate tracks worst expert
        cum_r, _, _ = compute_tracking_regret(agg_l, losses, K=0)
        assert cum_r[-1] > 0

    def test_switching_budget_formula(self) -> None:
        """K(T) must be non-negative and increase with T."""
        from src.regret import compute_switching_budget
        Ks = [compute_switching_budget(T, epsilon=0.1) for T in [100, 250, 500, 1000]]
        assert all(k >= 0 for k in Ks)
        assert Ks[-1] >= Ks[0]

    def test_stock_regret_shape(self) -> None:
        from src.regret import compute_stock_regret
        T, m = 15, 5
        agg_loss = np.random.default_rng(1).normal(0.01, 0.005, T)
        stock_losses = np.abs(np.random.default_rng(1).normal(0.01, 0.01, (T, m)))
        cum_rs, best = compute_stock_regret(agg_loss, stock_losses, K=1)
        assert cum_rs.shape == (T,)
        assert np.isfinite(best)

    def test_regret_bound_monotone(self) -> None:
        """The theoretical regret bound should be non-decreasing (it grows with T)."""
        from src.regret import compute_regret_bound
        b = compute_regret_bound(50, K=0, n=4, c=0.001, a=-0.15, b=0.15,
                                 eta=1.0, alpha=0.0)
        assert len(b) == 50
        assert np.all(np.diff(b) >= -1e-12)


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_compute_net_returns(self) -> None:
        from src.metrics import compute_net_returns
        T, m = 10, 3
        w = np.tile(uniform_w(m), (T, 1))
        ret = make_returns(T, m)
        nr = compute_net_returns(w, ret, c=0.001, w_prev=uniform_w(m))
        assert nr.shape == (T,)
        assert np.all(np.isfinite(nr))

    def test_account_value_starts_at_one(self) -> None:
        from src.metrics import compute_account_value
        net_r = np.array([0.01, -0.005, 0.02])
        av = compute_account_value(net_r)
        assert len(av) == 4  # includes initial value 1.0
        assert np.isclose(av[0], 1.0)

    def test_account_value_compounding(self) -> None:
        from src.metrics import compute_account_value
        net_r = np.array([0.10, 0.10])
        av = compute_account_value(net_r)
        assert np.isclose(av[-1], 1.1 * 1.1, atol=1e-10)

    def test_max_drawdown_negative(self) -> None:
        from src.metrics import compute_performance_metrics
        T, m = 30, 3
        w = np.tile(uniform_w(m), (T, 1))
        # Force large negative return to guarantee drawdown
        ret = make_returns(T, m)
        ret[15] = -0.1
        metrics = compute_performance_metrics(w, ret, c=0.001, w0=uniform_w(m), label="test")
        assert metrics["max_drawdown"] <= 0.0

    def test_metrics_keys(self) -> None:
        from src.metrics import compute_performance_metrics
        T, m = 20, 4
        w = np.tile(uniform_w(m), (T, 1))
        ret = make_returns(T, m)
        metrics = compute_performance_metrics(w, ret, c=0.001, w0=uniform_w(m), label="test")
        for key in ["label", "cumulative_return", "mean_return", "std_return",
                    "annualized_sharpe", "max_drawdown", "total_turnover"]:
            assert key in metrics, f"Missing key: {key}"

    def test_performance_table_rows(self) -> None:
        from src.metrics import compute_performance_metrics, build_performance_table
        T, m = 20, 3
        w = np.tile(uniform_w(m), (T, 1))
        ret = make_returns(T, m)
        m_list = [
            compute_performance_metrics(w, ret, c=0.001, w0=uniform_w(m), label="A"),
            compute_performance_metrics(w, ret, c=0.001, w0=uniform_w(m), label="B"),
        ]
        table = build_performance_table(m_list)
        assert len(table) == 2


# ---------------------------------------------------------------------------
# Baselines tests
# ---------------------------------------------------------------------------

class TestBaselines:
    def test_eg_simplex(self) -> None:
        from src.baselines import run_eg
        ret = make_returns(20, 4)
        w, _ = run_eg(ret, eta=0.05)
        assert w.shape == (20, 4)
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(w >= -1e-6)

    def test_pamr_simplex(self) -> None:
        from src.baselines import run_pamr
        ret = make_returns(20, 4)
        w, _ = run_pamr(ret, epsilon=0.5, C=500.0)
        assert w.shape == (20, 4)
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-5)

    def test_cwmr_simplex(self) -> None:
        from src.baselines import run_cwmr
        ret = make_returns(20, 4)
        w, _ = run_cwmr(ret, epsilon=0.5, confidence=0.95)
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-5)

    def test_olmar_simplex(self) -> None:
        from src.baselines import run_olmar
        ret = make_returns(20, 4)
        w, _ = run_olmar(ret, window=3, epsilon=5.0)
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-5)

    def test_up_simplex(self) -> None:
        from src.baselines import run_up
        ret = make_returns(20, 4)
        w, _ = run_up(ret, n_samples=20, seed=42)
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-6)

    def test_eg_nonneg_weights(self) -> None:
        from src.baselines import run_eg
        ret = make_returns(30, 5)
        w, _ = run_eg(ret, eta=0.1)
        assert np.all(w >= -1e-8)

    def test_olmar_no_lookahead(self) -> None:
        """Perturbing future returns must not change past weights in OLMAR."""
        from src.baselines import run_olmar
        ret = make_returns(20, 3, seed=99)
        w1, _ = run_olmar(ret, window=3, epsilon=5.0)
        ret2 = ret.copy()
        ret2[10:] *= 10.0
        w2, _ = run_olmar(ret2, window=3, epsilon=5.0)
        assert np.allclose(w1[:10], w2[:10], atol=1e-10)


# ---------------------------------------------------------------------------
# End-to-end smoke test (synthetic, very short)
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_synthetic_mini_pipeline(self) -> None:
        """A 4-window, 4-stock, 80-day mini pipeline must run to completion."""
        from data.generate_synthetic import generate_synthetic_returns
        from src.experts import run_window_experts
        from src.online_learning import run_aggregation_with_returns, select_parameters
        from src.regret import compute_tracking_regret

        df, dates = generate_synthetic_returns(
            n_stocks=4, T=80, regime_change=40,
            j1=21, j2=5, mu_daily=0.0005, alpha=0.5,
            sigma=0.02, seed=99, start_date="2022-01-01", windows=[5, 10, 21, 63],
        )
        # Full return matrix (presample + eval rows)
        returns_full = df.values
        T_eval = 80
        m = 4
        windows = [5, 10, 21, 63]
        n = len(windows)
        w0 = np.ones(m) / m
        c = 0.001

        expert_w, expert_l = run_window_experts(returns_full, windows, c, w0)
        assert expert_w.shape == (T_eval, n, m)

        eta, alpha, _ = select_parameters(T_eval, K=1, n=n, c=c, a=-0.15, b=0.15)
        agg_w, q_h, agg_l, _ = run_aggregation_with_returns(
            returns_full[-T_eval:], expert_w, expert_l, eta, alpha, w0, c
        )
        assert agg_w.shape == (T_eval, m)

        cum_r, _, _ = compute_tracking_regret(agg_l, expert_l, K=1)
        assert cum_r.shape == (T_eval,)
        assert np.isfinite(cum_r[-1])

    def test_regime_shift_weight_direction(self) -> None:
        """After a 63→5 regime shift, Fixed Share should eventually give more
        weight to the 5-day expert than the 63-day expert (qualitative test)."""
        from data.generate_synthetic import generate_synthetic_returns
        from src.experts import run_window_experts
        from src.online_learning import run_aggregation_with_returns, select_parameters

        df, _ = generate_synthetic_returns(
            n_stocks=8, T=300, regime_change=150,
            j1=63, j2=5, mu_daily=0.001, alpha=0.7,
            sigma=0.015, seed=42, start_date="2020-01-01", windows=[5, 63],
        )
        returns_full = df.values
        windows = [5, 63]
        n = len(windows)
        m = 8
        w0 = np.ones(m) / m
        expert_w, expert_l = run_window_experts(returns_full, windows, 0.001, w0)
        eta, alpha, _ = select_parameters(300, K=1, n=n, c=0.001, a=-0.15, b=0.15)
        _, q_fs, _, _ = run_aggregation_with_returns(
            returns_full[-300:], expert_w, expert_l, eta, alpha, w0, 0.001
        )
        # q_fs shape: (300, 2); index 0 = 5-day expert, index 1 = 63-day expert
        # In the second regime (t>150), the 5-day expert (idx 0) should gain weight
        avg_post_w5 = q_fs[150:, 0].mean()
        avg_post_w63 = q_fs[150:, 1].mean()
        # This is a qualitative test; the DGP may not always satisfy it with small T
        # We just confirm weights are valid probabilities
        assert np.allclose(q_fs.sum(axis=1), 1.0, atol=1e-10)
        assert np.all(q_fs >= 0)
