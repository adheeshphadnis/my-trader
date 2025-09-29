# Core Financial Concepts (What We'll Measure & Why)

This project evaluates strategies using **historical backtests** and **Monte Carlo simulations** to understand the *distribution* of outcomes (not just a single point estimate). Below are the key concepts, with simple math you can sanity-check.

---

## Returns

**Simple return** (daily or monthly):
- r_t = P_t / P_{t-1} - 1

**Log return** (additive over time):
- g_t = ln(P_t / P_{t-1})

We’ll use simple returns for interpretability; log returns are convenient for math but not required.

**Cumulative return** over T periods (simple returns):
- (1 + r_1) * (1 + r_2) * ... * (1 + r_T) - 1

**CAGR** (Compounded Annual Growth Rate) for N years:
- CAGR = (Ending / Beginning)^{1/N} - 1

---

## Volatility (Risk) & Drawdowns

**Volatility (annualized)** from daily returns r_t:
- σ_annual ≈ std(r_daily) * √252
(from monthly returns: √12)

**Max Drawdown (MDD)**:
- Largest peak-to-trough drop of cumulative equity curve:
  MDD = max over time of (peak_to_date - equity) / peak_to_date

This captures the “worst pain” you’d have felt holding the strategy.

---

## Risk-Adjusted Returns

**Sharpe ratio** (excess return per unit of volatility):
- Sharpe = (E[R] - R_f) / σ
  - R_f = risk-free rate (often set ≈ 0 for short horizons; we’ll make this configurable)
  - Higher is better; sensitive to non-normality and downside asymmetry.

**Sortino ratio** (penalizes downside only):
- Sortino = (E[R] - R_f) / σ_downside
  - σ_downside = std of negative returns only
  - More appropriate when upside volatility isn’t “bad.”

---

## Monte Carlo Simulation

We estimate the *distribution* of 3-year outcomes by simulating many future paths:

1) **Bootstrap** historical returns (randomly sample with replacement).
2) Optionally, fit a simple distribution or use block bootstrap to preserve some autocorrelation.
3) For each path, compound returns over the horizon to get an ending value.
4) Aggregate thousands of paths → summarize **median**, **mean**, **5th/95th percentiles**, **worst/best**, and **probability of loss**.

Why this matters: point forecasts hide tail risks. Distributions help you see *how often* you might get a bad outcome.

---

## 200-Day Moving Average (Trend) Strategy

**Rule**: Invest in SPY when price > 200-day simple moving average; otherwise stay in cash (≈ risk-free).
- Intuition: avoid major downtrends; accept whipsaws.
- Implementation details:
  - Compute 200-day SMA on **prior data** (no lookahead).
  - Use next-day close or same-day close after signal generation *consistently*.
  - Account for transaction costs (we’ll parameterize; default can be 0 at first).
  - Cash return can be 0% initially; we can add T-bill yield later.

**Trade-offs**:
- Pros: historically reduces severe drawdowns.
- Cons: can underperform during sharp V-shaped recoveries and sideways markets.

---

## What “Good” Looks Like

We’ll compare strategies on:
- **CAGR**
- **Volatility**
- **Sharpe / Sortino**
- **Max Drawdown**
- **Distribution percentiles** from Monte Carlo

We’ll choose based on your risk tolerance (7/10), focusing on drawdown control vs expected return.

> **Note**: Past performance is not a guarantee of future results. Our simulations are tools to reason about risk, not promises.
