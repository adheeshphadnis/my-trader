#!/usr/bin/env python3
"""
Builds a compact comparison report for Buy&Hold vs SMA200+Cash:

1. Backtest metrics (historical)
2. Monte Carlo before-tax (3y horizon, $10k)
3. Monte Carlo after-tax (3y horizon, $10k)

Outputs:
- reports/comparison/strategy_report.csv
- reports/comparison/strategy_report.md   (pretty Markdown table for reading)
"""

from pathlib import Path
import numpy as np
import pandas as pd

from quant_sim_lab.data.loader import (
    fetch_spy_history, load_spy_csv,
    fetch_tbill_3m_daily, load_tbill_daily
)
from quant_sim_lab.strategies.buy_and_hold import signal_buy_and_hold
from quant_sim_lab.strategies.sma_trend import signal_sma
from quant_sim_lab.backtest.engine import run_backtest
from quant_sim_lab.risk.metrics import (
    cagr_from_series, annualized_vol, sharpe_ratio, sortino_ratio, max_drawdown
)
from quant_sim_lab.sim.monte_carlo import simulate_iid, summarize_mc
from quant_sim_lab.risk.tax_annual import apply_annual_tax_to_path, buyhold_tax_end_only
from quant_sim_lab.config.tax import SHORT_TERM_RATE, LONG_TERM_RATE

# --- Config ---
HORIZON_YEARS = 3
HORIZON_DAYS = int(252 * HORIZON_YEARS)
N_PATHS = 10_000
START_VALUE = 10_000.0
SEED = 99
OUT_DIR = Path("reports/comparison")
# -------------

def summarize_backtest(name, equity, returns):
    return {
        "strategy": name,
        "cagr_hist": cagr_from_series(equity),
        "vol_hist": annualized_vol(returns),
        "sharpe_hist": sharpe_ratio(returns),
        "sortino_hist": sortino_ratio(returns),
        "max_dd_hist": max_drawdown(equity),
        "last_equity_hist": float(equity.iloc[-1]),
    }

def main():
    # --- Load data ---
    spy_csv = fetch_spy_history()
    tbill_csv = fetch_tbill_3m_daily()
    df = load_spy_csv(spy_csv)
    close = df["Close"]
    cash_daily = load_tbill_daily(tbill_csv).reindex(close.index).ffill().fillna(0.0)

    # --- Signals ---
    sig_bh = signal_buy_and_hold(close)
    sig_sma = signal_sma(close, window=200)

    # --- Backtests (pre-tax) ---
    res_bh = run_backtest(close, sig_bh, fee_bps=0.0, slippage_bps=0.0, cash_daily=0.0)
    res_sma = run_backtest(close, sig_sma, fee_bps=1.0, slippage_bps=2.0, cash_daily=cash_daily)

    backtests = [
        summarize_backtest("Buy&Hold", res_bh.equity, res_bh.strat_returns),
        summarize_backtest("SMA200+Cash", res_sma.equity, res_sma.strat_returns),
    ]
    backtests_df = pd.DataFrame(backtests)

    # --- Monte Carlo before-tax ---
    mc_bh = simulate_iid(res_bh.strat_returns, HORIZON_DAYS, n_paths=N_PATHS, start_value=START_VALUE, seed=SEED)
    mc_sma = simulate_iid(res_sma.strat_returns, HORIZON_DAYS, n_paths=N_PATHS, start_value=START_VALUE, seed=SEED+1)

    mc_summary = pd.DataFrame([
        {"strategy": "Buy&Hold", "scenario": "Before-tax", **summarize_mc(mc_bh.ending_values, START_VALUE)},
        {"strategy": "SMA200+Cash", "scenario": "Before-tax", **summarize_mc(mc_sma.ending_values, START_VALUE)},
    ])

    # --- Monte Carlo after-tax (realistic annual tax model) ---
    # Build synthetic business-day calendar
    start_date = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(start=start_date, periods=HORIZON_DAYS)

    from quant_sim_lab.sim.monte_carlo import _iid_bootstrap_daily

    # Draw bootstrap samples of daily returns
    r_bh  = _iid_bootstrap_daily(res_bh.strat_returns,  HORIZON_DAYS, N_PATHS, np.random.default_rng(SEED+2))
    r_sma = _iid_bootstrap_daily(res_sma.strat_returns, HORIZON_DAYS, N_PATHS, np.random.default_rng(SEED+3))

    end_bh_at  = np.empty(N_PATHS, dtype=float)
    end_sma_at = np.empty(N_PATHS, dtype=float)

    for i in range(N_PATHS):
        # Buy&Hold: tax once at end
        eq_bh_at  = buyhold_tax_end_only(r_bh[i], START_VALUE, LONG_TERM_RATE)
        end_bh_at[i] = eq_bh_at[-1]

        # SMA: annual tax with carryforward
        eq_sma_at = apply_annual_tax_to_path(r_sma[i], dates, START_VALUE, SHORT_TERM_RATE)
        end_sma_at[i] = eq_sma_at[-1]

    mc_after_tax_summary = pd.DataFrame([
        {"strategy": "Buy&Hold", "scenario": "After-tax", **summarize_mc(end_bh_at, START_VALUE)},
        {"strategy": "SMA200+Cash", "scenario": "After-tax", **summarize_mc(end_sma_at, START_VALUE)},
    ])


    # --- Merge all results ---
    report = backtests_df.merge(
        pd.concat([mc_summary, mc_after_tax_summary], ignore_index=True),
        on="strategy",
        how="outer"
    )

    # Make labels crystal clear
    report["label"] = report["strategy"] + " — " + report["scenario"]

    # Order columns for readability
    cols_hist = ["strategy", "cagr_hist", "vol_hist", "sharpe_hist", "sortino_hist", "max_dd_hist", "last_equity_hist"]
    cols_mc   = ["scenario", "start_value", "mean_end", "median_end", "p05_end", "p25_end", "p75_end", "p95_end", "prob_end_below_start", "min_end", "max_end", "num_paths"]
    display_cols = ["label"] + cols_hist[1:] + cols_mc[1:]  # drop duplicate 'strategy' and 'scenario'

    report = report.sort_values(["strategy", "scenario"]).reset_index(drop=True)

    # Also provide a compact pivot (Before vs After side-by-side)
    pivot_metrics = ["mean_end", "median_end", "p05_end", "p95_end", "prob_end_below_start"]
    pivot = (
        report.pivot_table(index="strategy", columns="scenario", values=pivot_metrics)
        .sort_index(axis=1)  # columns: ('mean_end','After-tax') etc.
    )

    # Save outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "strategy_report.csv"
    md_path  = OUT_DIR / "strategy_report.md"
    md_pivot = OUT_DIR / "strategy_report_pivot.md"

    # Main flat table (explicit 'label' column)
    report[["label"] + cols_hist[1:] + cols_mc[1:]].to_csv(csv_path, index=False)
    with open(md_path, "w") as f:
        f.write(report[["label"] + cols_hist[1:] + cols_mc[1:]].to_markdown(index=False))

    # Pivoted comparison: Before-tax vs After-tax side-by-side
    with open(md_pivot, "w") as f:
        f.write("# Side-by-side (Before-tax vs After-tax)\n\n")
        f.write(pivot.to_markdown())

    # Console prints
    print("\n=== Strategy Comparison Report (labeled) ===")
    print(report[["label"] + cols_hist[1:] + cols_mc[1:]].to_string(index=False))

    print("\n=== Side-by-side (Before-tax vs After-tax) ===")
    print(pivot.to_string())
    print(f"\n✅ Saved flat table: {csv_path}")
    print(f"✅ Saved Markdown table: {md_path}")
    print(f"✅ Saved side-by-side Markdown: {md_pivot}")


if __name__ == "__main__":
    main()
