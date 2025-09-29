#!/usr/bin/env python3
"""
Monte Carlo simulation of 3-year outcomes for:
- Buy & Hold (SPY)
- SMA-200 trend-following

Outputs:
- reports/montecarlo/summary.csv  (rows per strategy)
- reports/montecarlo/buyhold_hist.png
- reports/montecarlo/sma200_hist.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from quant_sim_lab.data.loader import fetch_spy_history, load_spy_csv
from quant_sim_lab.strategies.buy_and_hold import signal_buy_and_hold
from quant_sim_lab.strategies.sma_trend import signal_sma
from quant_sim_lab.backtest.engine import run_backtest
from quant_sim_lab.sim.monte_carlo import simulate_iid, summarize_mc, TRADING_DAYS_PER_YEAR

# ------- Config -------
HORIZON_YEARS = 3
HORIZON_DAYS = int(HORIZON_YEARS * TRADING_DAYS_PER_YEAR)   # ≈ 756
N_PATHS = 10000
START_VALUE = 10_000.0
SEED = 123
# ----------------------


def save_histogram(values: np.ndarray, out_path: Path, title: str, bins: int = 60):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel("Ending Value ($)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    # Ensure data exists
    csv_path = fetch_spy_history()  # idempotent
    df = load_spy_csv(csv_path)
    close = df["Close"].copy()

    # Signals
    sig_bh = signal_buy_and_hold(close)
    sig_sma = signal_sma(close, window=200)

    # Backtests to get daily strategy returns (already cost-aware per engine config)
    res_bh = run_backtest(close, sig_bh, fee_bps=0.0, slippage_bps=0.0)
    res_sma = run_backtest(close, sig_sma, fee_bps=1.0, slippage_bps=2.0)

    # Monte Carlo (i.i.d. bootstrap)
    mc_bh = simulate_iid(res_bh.strat_returns, horizon_days=HORIZON_DAYS, n_paths=N_PATHS, start_value=START_VALUE, seed=SEED)
    mc_sma = simulate_iid(res_sma.strat_returns, horizon_days=HORIZON_DAYS, n_paths=N_PATHS, start_value=START_VALUE, seed=SEED + 1)

    # Summaries
    sum_bh = summarize_mc(mc_bh.ending_values, START_VALUE)
    sum_sma = summarize_mc(mc_sma.ending_values, START_VALUE)

    summary_df = pd.DataFrame([
        {"strategy": "Buy&Hold", **sum_bh},
        {"strategy": "SMA200", **sum_sma},
    ])

    out_dir = Path("reports/montecarlo")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\n=== Monte Carlo (3-year) Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\n✅ Saved summary: {summary_csv}")

    # Plots
    save_histogram(mc_bh.ending_values, out_dir / "buyhold_hist.png", f"Buy&Hold — Ending Values (3y, n={N_PATHS})")
    save_histogram(mc_sma.ending_values, out_dir / "sma200_hist.png", f"SMA-200 — Ending Values (3y, n={N_PATHS})")

    print(f"✅ Saved histograms to: {out_dir}\n")


if __name__ == "__main__":
    main()
