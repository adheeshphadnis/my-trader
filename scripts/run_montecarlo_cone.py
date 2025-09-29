#!/usr/bin/env python3
"""
Monte Carlo percentile cones over 3 years, using block bootstrap to preserve streaks.

Outputs:
- reports/montecarlo/cones/buyhold_cone.png
- reports/montecarlo/cones/sma200_cone.png
- reports/montecarlo/cones/summary_block.csv  (median, p5, p95, prob under $10k & $8k)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from quant_sim_lab.data.loader import fetch_spy_history, load_spy_csv
from quant_sim_lab.strategies.buy_and_hold import signal_buy_and_hold
from quant_sim_lab.strategies.sma_trend import signal_sma
from quant_sim_lab.backtest.engine import run_backtest
from quant_sim_lab.sim.monte_carlo import (
    simulate_block_paths,
    percentile_cone,
    TRADING_DAYS_PER_YEAR,
)

# ----- Config -----
HORIZON_YEARS = 3
HORIZON_DAYS = int(252 * HORIZON_YEARS)
N_PATHS = 10000
START_VALUE = 10_000.0
SEED = 2025
BLOCK_SIZE = 20        # ~ 1 trading month blocks
PCTS = [5, 25, 50, 75, 95]
# Risk checkpoints for your 7/10 risk tolerance:
CHECK_LEVELS = [10_000.0, 8_000.0]   # prob of ending below $10k and $8k
# ------------------


def make_cone_plot(cone_df: pd.DataFrame, out_path: Path, title: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    # Fan: fill between p25 and p75, p05 and p95; plot median
    if "p25" in cone_df and "p75" in cone_df:
        plt.fill_between(cone_df.index, cone_df["p25"], cone_df["p75"], alpha=0.3, label="IQR (25-75%)")
    if "p05" in cone_df and "p95" in cone_df:
        plt.fill_between(cone_df.index, cone_df["p05"], cone_df["p95"], alpha=0.15, label="5-95%")
    if "p50" in cone_df:
        plt.plot(cone_df.index, cone_df["p50"], label="Median", linewidth=2)

    # Starting capital line
    plt.axhline(START_VALUE, linestyle="--", linewidth=1)

    plt.title(title)
    plt.xlabel("Trading days (0 ≈ today)")
    plt.ylabel("Portfolio value ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def summarize_tail_risk(paths: np.ndarray, levels: list[float]) -> dict:
    """
    Compute probabilities of ending below certain dollar levels.
    paths: (n_paths, n_days+1)
    """
    end_vals = paths[:, -1]
    out = {}
    for L in levels:
        out[f"prob_end_below_{int(L)}"] = float(np.mean(end_vals < L))
    out["median_end"] = float(np.median(end_vals))
    out["p05_end"] = float(np.percentile(end_vals, 5))
    out["p95_end"] = float(np.percentile(end_vals, 95))
    return out


def main():
    # Ensure data
    csv_path = fetch_spy_history()
    df = load_spy_csv(csv_path)
    close = df["Close"]

    # Signals & backtests (to get daily strategy returns)
    sig_bh = signal_buy_and_hold(close)
    sig_sma = signal_sma(close, window=200)

    res_bh = run_backtest(close, sig_bh, fee_bps=0.0, slippage_bps=0.0)
    res_sma = run_backtest(close, sig_sma, fee_bps=1.0, slippage_bps=2.0)

    # Block bootstrap equity paths
    paths_bh  = simulate_block_paths(res_bh.strat_returns,  HORIZON_DAYS, n_paths=N_PATHS, start_value=START_VALUE, block_size=BLOCK_SIZE, seed=SEED)
    paths_sma = simulate_block_paths(res_sma.strat_returns, HORIZON_DAYS, n_paths=N_PATHS, start_value=START_VALUE, block_size=BLOCK_SIZE, seed=SEED+1)

    # Percentile cones
    cone_bh  = percentile_cone(paths_bh,  percentiles=PCTS)
    cone_sma = percentile_cone(paths_sma, percentiles=PCTS)

    out_dir = Path("reports/montecarlo/cones")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plots
    make_cone_plot(cone_bh,  out_dir / "buyhold_cone.png", "Buy&Hold — 3y Percentile Cone (block bootstrap)")
    make_cone_plot(cone_sma, out_dir / "sma200_cone.png",  "SMA-200 — 3y Percentile Cone (block bootstrap)")

    # Tail-risk summary aligned to your tolerance
    sum_bh  = {"strategy": "Buy&Hold", **summarize_tail_risk(paths_bh, CHECK_LEVELS)}
    sum_sma = {"strategy": "SMA200",   **summarize_tail_risk(paths_sma, CHECK_LEVELS)}

    summary = pd.DataFrame([sum_bh, sum_sma])
    summary_csv = out_dir / "summary_block.csv"
    summary.to_csv(summary_csv, index=False)

    print("\n=== Monte Carlo (Block Bootstrap) Tail-Risk Summary (3y) ===")
    print(summary.to_string(index=False))
    print(f"\n✅ Saved cone plots + summary to: {out_dir}\n")


if __name__ == "__main__":
    main()
