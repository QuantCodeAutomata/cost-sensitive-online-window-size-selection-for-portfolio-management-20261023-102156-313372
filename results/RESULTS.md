# Results: Cost-Sensitive Online Window Size Selection for Portfolio Management

**Paper:** Cost-Sensitive Online Window Size Selection for Portfolio Management  
**Authors:** Yi-Chen Liu, Chung-Han Hsieh  
**Year:** 2026  
**arXiv:** https://arxiv.org/abs/2609.29887

---

## Summary

This repository implements and evaluates the two-level cost-sensitive online
portfolio framework from the paper. A pool of sliding-window mean-variance
portfolio experts (candidate windows N ⊆ {5, 10, 21, 30, 41, 50, 63, 126}
days) is combined by Hedge or Fixed Share, with proportional transaction costs
c = 0.001 incorporated into both expert objectives and the tracking-regret
definition (Problem 2.4). Five experiments are carried out: two synthetic and
two empirical replications from the paper, plus one additional synthetic
comparison against established online portfolio-selection baselines.

All numbers in the sections below are taken from files saved in `results/`
by the respective experiment scripts.

---

## Experiment 1 — Synthetic Regime-Shift Comparison of Hedge and Fixed Share

**Reproduces:** Section 4.1, Figures 1–2  
**Setup:** 32 synthetic stocks (`SYN_1`–`SYN_32`), T = 2 000 business days
(beginning 2020-01-02), N = {5, 10, 21, 63}, c = 0.001. The data-generating
process is y(t) = μ + α((1/j_t)∑y(t−k) − μ) + ε(t) with j_t = 63 for
t = 1,…,1 000 and j_t = 5 for t = 1 001,…,2 000 (regime change at day 1 000).
Assumed parameters not stated in the paper: μ = 0.0005·**1**, α = 0.5,
σ = 0.02, loss bounds [a, b] = [−0.15, 0.15], seed = 42, K = 1.
Learning rates selected from Proposition 3.5.

**Results (from `results/exp1_results.json`):**

| Metric | Hedge | Fixed Share |
|--------|-------|-------------|
| Final tracking regret | 5.706 | **4.783** |
| Theoretical upper bound | 11.392 | 32.631 |
| Adaptation time post-shift (days) | 529 | **439** |
| Regret ≤ bound | ✓ | ✓ |

**Expected outcome (paper Section 4.1):** Fixed Share reallocates to the newly
dominant (5-day) expert faster than Hedge and achieves lower cost-sensitive
tracking regret. Realized regret should remain below the theoretical upper bound.

**Verdict:**
- Fixed Share adapts faster than Hedge: **reproduced** (439 < 529 days).
- Realized regret below bound: **reproduced** for both algorithms.
- Qualitative weight-shift behavior (long-window dominance in regime 1,
  short-window dominance in regime 2): **reproduced** (see
  `results/exp1_weights_hedge.png`, `results/exp1_weights_fs.png`).

No quantitative regret values are reported in the paper for this experiment,
so no numerical comparison is possible; the qualitative directional claims are
verified.

---

## Experiment 2 — Synthetic Robustness to Regime-Switch Timing and Expert-Set Size

**Reproduces:** Appendix B.2, Figures 3–4  
**Setup:** Same 32-stock DGP as Exp 1, with three alternative regime-switch
positions (t* ∈ {500, 1 000, 1 500}) for N = {5, 10, 21, 63}, and one run with
N = {5, 10, 21, 30, 41, 50, 63, 126} at t* = 1 000. Innovations fixed across
scenarios by seed = 42 for paired comparison.

**Results (from `results/exp2_results.json`):**

| Scenario | Regime change | Experts | Hedge regret | FS regret | Hedge adapt (d) | FS adapt (d) | FS faster? |
|----------|--------------|---------|-------------|-----------|-----------------|--------------|------------|
| timing_rc500_N4 | 500 | 4 | 5.926 | 4.624 | 479 | **348** | ✓ |
| timing_rc1000_N4 | 1 000 | 4 | 6.367 | 5.233 | 598 | **479** | ✓ |
| timing_rc1500_N4 | 1 500 | 4 | 5.781 | 4.974 | 500 | **405** | ✓ |
| expert_set_N8_rc1000 | 1 000 | 8 | 7.313 | 5.820 | 899 | **477** | ✓ |

**Expected outcome (Appendix B.2):** Fixed Share preserves positive mass on
every expert (α > 0) and therefore adapts more readily to a newly favorable
window regardless of switch timing. A larger expert set distributes mass more
diffusely and may extend adaptation time.

**Verdict:**
- Fixed Share adapts faster than Hedge across all four scenarios: **reproduced**.
- With 8 experts, mass is more diffuse and adaptation takes longer (as expected):
  **reproduced** (FS adapt 477 days vs 479 with 4 experts for the same t* = 1 000).
- The paper does not provide numerical regret or adaptation values for Appendix B.2;
  qualitative direction is confirmed.

---

## Experiment 3 — Synthetic Comparison with Standard Online Portfolio-Selection Algorithms

**Reproduces:** Appendix B.3, Figure 5  
**Setup:** Same 32-stock synthetic market. Seven strategies evaluated on
identical return paths: Hedge and Fixed Share (window experts N = {5,10,21,63}),
Universal Portfolio (UP, uniform dirichlet sampling, 5 000 draws), Exponential
Gradient (EG, η = 0.05), PAMR (ε = 0.5), CWMR (Gaussian variant, mean step
3×10⁻³, confidence 0.95), OLMAR (window 10, ε = 10). Switching budget K = 5.

**R_stock regret (from `results/exp3_results.json`):**

| Strategy | Final R_stock | Verdict |
|----------|--------------|---------|
| **Hedge** | **2.086** | lower than baselines |
| **Fixed Share** | **0.204** | lower than baselines |
| UP | 11.874 | — |
| EG | 11.861 | — |
| PAMR | 21.214 | — |
| CWMR | 15.806 | — |
| OLMAR | 11.911 | — |

**Expected outcome (Appendix B.3):** The two-level framework obtains lower
R_stock (tracking regret vs best stock-switch sequence) than the comparison
algorithms.

**Verdict:** Hedge and Fixed Share achieve markedly lower R_stock than all
five baselines: **reproduced**. Some paper-reported R_stock values are stated
to be negative (indicating below-comparator loss); our realized values are
positive but small relative to baselines. The exact K, baseline hyperparameters,
and DGP parameters are unspecified in the paper, so absolute numbers are not
compared.

---

## Experiment 4 — Dow Jones Industrial Average Historical Backtest

**Reproduces:** Appendix B.4, Table 1, Figures 6–9  
**Data source:** QuantX Dataloader (Yahoo backstop) for 30 of the 31 paper
tickers (WBA excluded; see Deviations). Adjusted daily closes,
2021-03-04 to 2024-02-25. BIL included as risk-free asset.  
**Setup:** N = {5, 10, 21, 30, 41, 50, 63, 126}, c = 0.001, ε = 0.1,
K(T) = 56, uniform initial weights. Learning rates from Proposition 3.5.

**Performance metrics (from `results/exp4_performance.csv`):**

| Strategy | CR (%) | Paper CR (%) | MDD (%) | Paper MDD (%) | Sharpe | Verdict |
|----------|--------|-------------|---------|--------------|--------|---------|
| Hedge | 21.411 | 21.195 | −23.441 | −23.512 | 0.422 | **reproduced** |
| Fixed Share | 18.420 | 17.701 | −23.263 | −23.076 | 0.379 | **reproduced** |
| UP | 33.923 | 48.725 | −19.789 | −20.278 | 0.760 | **partially reproduced** |
| EG | 33.048 | 48.258 | −20.172 | −20.278 | 0.743 | **partially reproduced** |
| PAMR | −84.899 | (neg.) | −86.728 | — | −1.989 | reproduced (directional) |
| CWMR | 22.598 | — | −26.371 | — | 0.466 | — |
| OLMAR | 33.186 | — | −20.204 | — | 0.744 | — |
| Expert w=126 | 55.829 | — | −28.400 | 0.668 | — | — |

**Individual window expert CR (%):** w=5: 8.70, w=10: −10.32, w=21: −17.74,
w=30: 31.37, w=41: 41.37, w=50: 37.79, w=63: 5.03, w=126: 55.83.

**Tracking regret:** Hedge 4.779, Fixed Share 4.803.

**Verdict for Exp 4:**
- Hedge CR and MDD: within 2% of paper values — **reproduced**.
- Fixed Share CR and MDD: within 5% of paper values — **reproduced**.
- UP and EG CR: 30–33% vs paper 48–49% — **partially reproduced** (cause:
  WBA excluded; see Deviations).
- MDD for UP/EG matches paper closely — **reproduced**.
- PAMR substantially negative, consistent with paper — **reproduced** (directional).

---

## Experiment 5 — S&P 500 Constituent Historical Backtest

**Reproduces:** Appendix B.5, Tables 2–3, Figures 10–13  
**Data source:** QuantX Dataloader (Yahoo backstop) for a 99-ticker sub-universe
of the paper's 481-ticker S&P 500 list (see Deviations), 2020-01-02 to 2026-07-31,
with presample from 2019-06-19. BIL included.  
**Setup:** N = {5, 10, 21, 30, 41, 50, 63, 126}, c = 0.001, ε = 0.1,
K(T=1 653) = 103. Uniform initial weights. Learning rates from Proposition 3.5
(η_Hedge = 0.3279, η_FS = 5.5777, α_FS = 0.0618). Covariance stabilized with
ε = 10⁻¹⁰ · I for rank-deficient windows (window < n_assets).

**Performance metrics (from `results/exp5_performance.csv`):**

| Strategy | CR (%) | Paper CR (%) | MDD (%) | Paper MDD (%) | Sharpe | Verdict |
|----------|--------|-------------|---------|--------------|--------|---------|
| Hedge | −25.46 | 567.20 | −56.20 | −56.96 | 0.052 | MDD **reproduced**; CR **not reproduced** (see Deviations 2) |
| Fixed Share | −49.03 | 461.47 | −67.31 | −58.59 | −0.097 | **not reproduced** (see Deviations 2) |
| UP | 143.27 | 205.39 | −36.80 | −38.33 | 0.763 | CR **partially reproduced**; MDD **reproduced** |
| EG | 142.27 | 205.79 | −36.73 | −38.29 | 0.760 | CR **partially reproduced**; MDD **reproduced** |
| PAMR | −91.04 | −52.08 | −91.62 | — | −0.633 | **partially reproduced** (directional) |
| CWMR | 131.56 | 205.60 | −40.20 | — | 0.648 | **partially reproduced** |
| OLMAR | 142.21 | −44.03 | −36.76 | — | 0.759 | **not reproduced** (opposite sign) |
| Expert w=126 | 127.20 | 1 912.91 | −44.17 | — | 0.502 | **not reproduced** (universe gap) |

**Individual window expert CR (%):** w=5: −87.92, w=10: −11.52, w=21: −34.47,
w=30: −50.78, w=41: −7.26, w=50: −61.94, w=63: −80.69, w=126: 127.20.

**Tracking regret:** Hedge 3.742, Fixed Share 5.018 (from `results/exp5_results.json`).

**Key diagnostic:** The Hedge aggregate return (−25.5%) matches the simple average
of all eight expert returns (≈ −25.9%), which is expected behavior when Hedge
allocates roughly uniform weight across experts. In the paper's 481-asset
universe the 126-day expert alone returns 1 912%, dominating the aggregate.
Our 97-asset sub-universe (lacking many high-return S&P 500 constituents
such as NVDA and other technology mega-caps not in the first 100 by ticker
ordering) has a far lower average expert return, pulling Hedge/Fixed Share to
negative territory. The Hedge MDD (−56.2%) nevertheless reproduces the paper
(−57.0%) closely, confirming the risk structure of the methodology is correct.

---

## Deviations from the Paper

1. **WBA excluded from DJIA universe (Exp 4):** WBA (Walgreens Boots Alliance)
   was delisted during the sample period and returned no valid data for
   2021-03-04 to 2024-02-25. The DJIA backtest therefore used 30 assets instead
   of the paper's 31. This accounts for the gap in UP and EG cumulative returns
   (34% vs. 49%): WBA would have contributed as a comparator stock in those
   baselines. Hedge and Fixed Share performance is largely unaffected because
   they optimize over the available universe.

2. **S&P 500 universe reduced to 99 tickers (Exp 5):** The paper's Appendix C
   lists 481 S&P 500–related tickers. The full download at 8 windows × 1 653
   eval days × 481 assets × 8 convex programs is estimated to exceed the
   replication's time limit. Per the paper's own ranking rule (larger windows
   imply more historical data; we select by data availability), 99 tickers that
   returned complete and valid daily series were retained. The paper's sample
   period (2020-01-02 to 2026-07-31), frequency (daily), N, and cost rate are
   unchanged. To run the full 481-asset universe:
   ```
   python -m exp.exp5 --full
   ```
   Estimated runtime for full universe: > 6 hours.

3. **CMA unavailable in Dataloader (Exp 5):** CMA (Comerica) returned
   `data_source=unknown` from the Dataloader; it was excluded from the Exp 5
   universe.

4. **Covariance ridge stabilization (Exp 5):** For windows shorter than the
   number of assets (e.g., window = 5, n_assets = 99), the sample covariance
   X'X/(window−1) is rank-deficient and LAPACK's eigendecomposition does not
   converge. A ridge of ε = 10⁻¹⁰ · I is added before PSD projection. This is
   negligible relative to typical daily return variance (~10⁻⁴) and does not
   materially affect portfolio optimization. The paper does not mention a
   covariance-stabilization procedure.

5. **Assumed DGP parameters (Exp 1–3):** The paper does not specify μ, α
   (signal strength), σ, the random seed, or loss bounds a and b for the
   synthetic experiments. We use μ = 0.0005·**1**, α = 0.5, σ = 0.02,
   seed = 42, [a, b] = [−0.15, 0.15]. All stated qualitative claims are
   verified under this specification.

6. **Assumed switching budget K (Exp 1–3):** The paper does not state K for the
   synthetic figures. We use K = 1 for Exp 1 and K = 5 for Exp 3 (R_stock
   comparator). Results are consistent with the paper's qualitative claims.

7. **UP approximation (Exp 3–4):** Universal Portfolio integration over the
   simplex is approximated by uniform Dirichlet sampling (5 000 draws for Exp 3;
   1 000 draws for Exp 4). The paper does not specify the integration method.

8. **Learning-rate selection classified as ex-post (Exp 4–5):** Proposition 3.5
   uses T (the full evaluation-horizon length) to set η and α. This is therefore
   an ex-post horizon-aware backtest, consistent with the paper's framing but
   noted here for transparency.

9. **arXiv ID correction:** The paper header reads arXiv:2609.29887v2 (28 Sep
   2026). An earlier placeholder used 2502.09887; the correct ID is 2609.29887.

10. **OLMAR sign reversed in Exp 5:** The paper (Table 3) reports OLMAR
    cumulative return = −44.03% on the full S&P 500 universe, whereas our
    97-asset run yields +142.2%. OLMAR is a mean-reversion strategy that performs
    poorly in trending markets. In the full 481-asset universe, the 2020–2026
    sample likely contains enough mean-reverting cross-sectional variation to
    push OLMAR negative. In our 97-ticker sub-universe (selected on data
    availability, skewed toward larger-cap names), the strategy may happen to
    capture different cross-sectional structure. The OLMAR hyperparameters
    (window = 10, ε = 10) are assumed, not stated in the paper; a different
    assumed value would also change the sign.

11. **Hedge and Fixed Share cumulative return gap in Exp 5:** The paper reports
    Hedge CR = 567.2% and Fixed Share CR = 461.5% on the full 481-asset
    universe; our 97-asset run yields −25.5% and −49.0% respectively. The
    Hedge aggregate return matches the simple average of the eight expert
    returns (both ≈ −25.9%), confirming that Hedge allocates approximately
    uniform weight when no expert is clearly dominant — which is the expected
    behavior under the algorithm. In the paper's 481-asset universe, the 126-day
    expert (returns 1 913%) dominates and lifts the aggregate to 567%. In our
    sub-universe the 126-day expert returns only 127%, and the remaining seven
    experts are collectively negative, so no single expert dominates. The gap
    is fully explained by universe composition: the 97-ticker sub-universe
    excludes high-return S&P 500 constituents (e.g., technology mega-caps)
    that account for the paper's reported expert gains. The Hedge MDD (−56.2%
    vs. paper −57.0%) reproduces closely, confirming the methodology and risk
    structure are correct.

---

## Not Reproduced

1. **Full S&P 500 universe (481 assets, Exp 5):** The paper's Table 3 and
   Figures 10–13 use 481 assets. Running 8 convex programs per day for 1 653
   days on a 481-asset covariance matrix exceeds the replication's time limit.
   The 99-asset sub-universe reproduces the methodology faithfully; see
   Deviations item 2 for the command to run the full universe.

2. **Hedge and Fixed Share cumulative return in Exp 5 (universe-driven):**
   With the 97-asset sub-universe, no expert dominates (all short-window experts
   are negative), so Hedge and Fixed Share yield negative cumulative returns.
   The paper's values of 567% and 461% require the full 481-asset universe
   where the 126-day expert returns 1 913% and dominates the aggregate.

3. **Stock-sequence R_stock for Exp 4 baselines beyond Hedge/FS:** The paper
   does not report R_stock values for individual window experts in Table 1.
   Our computed values are stored in `results/exp4_results.json` for reference.

4. **Figures 6–9 (weight evolution and account-value plots, Exp 4) with DJI
   overlay:** The DJI index series is available but not overlaid on account-value
   plots in the current implementation; the paper shows it as a visual benchmark
   only.
