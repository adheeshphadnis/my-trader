#!/usr/bin/env python3
"""
Monte Carlo simulation with cash yield included for SMA strategy.

Outputs:
- reports/montecarlo/cash/buyhold_hist.png
- reports/montecarlo/cash/sma200_hist.png
- reports/montecarlo/cash/buyhold_cone.png
- reports/montecarlo/cash/sma200_cone.png
- reports/montecarlo/cash/summary.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from quant_sim_lab.data.loader import (
    fetch_spy_history, load_spy_csv,
    fetch_tbill_3m_daily, load_tbill_daily
)
from quant_sim_lab.strategies.buy_and_hold import signal_buy_and_hold
from quant_sim_lab.strategies.sma_trend import signal_sma
from quant_sim_lab.backtest.engine import run_backtest
from quant_sim_lab.sim.monte_carlo import (
    simulate_iid,
    simulate_block_paths,
    percentile_cone,
    summarize_mc,
)

# ---- Config ----
HORIZON_YEARS = 3
HORIZON_DAYS = int(252 * HORIZON_YEARS)
N_PATHS = 10000
START_VALUE = 10_000.0
SEED = 2025
BLOCK_SIZE = 20
PCTS = [5, 25, 50, 75, 95]
OUT_DIR = Path("reports/montecarlo/cash")
# ----------------


def plot_hist(vals: np.ndarray, out_path: Path, title: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(vals, bins=80, color="steelblue", alpha=0.7)
    plt.axvline(START_VALUE, color="red", linestyle="--", label="Start")
    plt.title(title)
    plt.xlabel("Ending value ($)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_cone(cone_df: pd.DataFrame, out_path: Path, title: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.fill_between(cone_df.index, cone_df["p25"], cone_df["p75"], alpha=0.3, label="IQR 25–75%")
    plt.fill_between(cone_df.index, cone_df["p05"], cone_df["p95"], alpha=0.15, label="5–95%")
    plt.plot(cone_df.index, cone_df["p50"], linewidth=2, label="Median")
    plt.axhline(START_VALUE, linestyle="--", color="red", linewidth=1)
    plt.title(title)
    plt.xlabel("Trading days (0 ≈ today)")
    plt.ylabel("Portfolio value ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    # Load data
    spy_csv = fetch_spy_history()
    tbill_csv = fetch_tbill_3m_daily()
    df = load_spy_csv(spy_csv)
    close = df["Close"]
    cash_daily = load_tbill_daily(tbill_csv).reindex(close.index).ffill().fillna(0.0)

    # Build signals
    sig_bh = signal_buy_and_hold(close)
    sig_sma = signal_sma(close, window=200)

    # Backtests
    res_bh = run_backtest(close, sig_bh, fee_bps=0.0, slippage_bps=0.0, cash_daily=0.0)
    res_sma = run_backtest(close, sig_sma, fee_bps=1.0, slippage_bps=2.0, cash_daily=cash_daily)

    # Monte Carlo IID endings
    mc_bh = simulate_iid(res_bh.strat_returns, HORIZON_DAYS, n_paths=N_PATHS, start_value=START_VALUE, seed=SEED)
    mc_sma = simulate_iid(res_sma.strat_returns, HORIZON_DAYS, n_paths=N_PATHS, start_value=START_VALUE, seed=SEED+1)

    plot_hist(mc_bh.ending_values, OUT_DIR / "buyhold_hist.png", "Buy&Hold — 3y Ending Value Dist")
    plot_hist(mc_sma.ending_values, OUT_DIR / "sma200_hist.png", "SMA200+Cash — 3y Ending Value Dist")

    # Monte Carlo block bootstrap cones
    paths_bh  = simulate_block_paths(res_bh.strat_returns,  HORIZON_DAYS, n_paths=N_PATHS, start_value=START_VALUE, block_size=BLOCK_SIZE, seed=SEED)
    paths_sma = simulate_block_paths(res_sma.strat_returns, HORIZON_DAYS, n_paths=N_PATHS, start_value=START_VALUE, block_size=BLOCK_SIZE, seed=SEED+1)

    cone_bh  = percentile_cone(paths_bh,  percentiles=PCTS)
    cone_sma = percentile_cone(paths_sma, percentiles=PCTS)

    plot_cone(cone_bh,  OUT_DIR / "buyhold_cone.png", "Buy&Hold — 3y Percentile Cone (with cash)")
    plot_cone(cone_sma, OUT_DIR / "sma200_cone.png", "SMA200+Cash — 3y Percentile Cone")

    # Summary CSV
    summary = pd.DataFrame([
        {"strategy": "Buy&Hold",  **summarize_mc(mc_bh.ending_values, START_VALUE)},
        {"strategy": "SMA200+Cash", **summarize_mc(mc_sma.ending_values, START_VALUE)},
    ])
    summary_path = OUT_DIR / "summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n=== Monte Carlo with Cash (3y horizon, $10k start) ===")
    print(summary.to_string(index=False))
    print(f"\n✅ Saved histograms, cones, summary: {OUT_DIR}")


if __name__ == "__main__":
    main()
