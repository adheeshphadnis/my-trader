#!/usr/bin/env python3
"""
Backtest the Monte Carlo simulator itself.

Flow:
1) Load SPY and (optionally) T-bill cash series.
2) Build strategy signals on the FULL price history (properly shifted to avoid look-ahead).
3) Split into TRAIN and TEST by a cutoff date (default: 2018-01-01).
4) Backtest TRAIN and TEST segments to get:
   - TRAIN: daily strategy returns (used for MC bootstrap)
   - TEST:  actual daily strategy returns (to compound to actual end value)
5) Monte Carlo: simulate the TEST horizon using TRAIN daily returns (i.i.d. bootstrap).
6) Compare the actual TEST ending value vs the simulated distribution.
7) Save histogram + CSV summary.

Strategies included:
- Buy&Hold
- SMA200 (+ cash yield when out of market)

Usage:
    source .venv/bin/activate
    python scripts/backtest_mc_predictive.py --split-date 2018-01-01 --n-paths 10000
"""

from pathlib import Path
import argparse
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
from quant_sim_lab.sim.monte_carlo import simulate_iid, summarize_mc, TRADING_DAYS_PER_YEAR

OUT_DIR = Path("reports/mc_backtest")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compound_to_value(start_value: float, daily_returns: pd.Series) -> float:
    r = daily_returns.fillna(0.0).to_numpy(dtype=float)
    return float(start_value * np.prod(1.0 + r))


def summarize_predictive_check(
    strategy_name: str,
    start_value: float,
    mc_endings: np.ndarray,
    actual_end: float,
    train_start: str, train_end: str,
    test_start: str, test_end: str,
) -> dict:
    mc = summarize_mc(mc_endings, start_value)
    # Percentile rank of actual within MC endings
    pct_rank = float(np.mean(mc_endings <= actual_end))
    row = {
        "strategy": strategy_name,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "horizon_days": len(mc_endings) and None,  # filled below
        "start_value": start_value,
        "actual_end": actual_end,
        "actual_return_pct": (actual_end / start_value - 1.0),
        "mc_mean_end": mc["mean_end"],
        "mc_median_end": mc["median_end"],
        "mc_p05_end": mc["p05_end"],
        "mc_p95_end": mc["p95_end"],
        "mc_prob_end_below_start": mc["prob_end_below_start"],
        "mc_percentile_of_actual": pct_rank,  # e.g., 0.50 means actual ~ median
        "num_paths": mc["num_paths"],
    }
    return row


def plot_hist_with_actual(values: np.ndarray, actual: float, out_path: Path, title: str, start_value: float):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.hist(values, bins=80, alpha=0.8)
    plt.axvline(start_value, linestyle="--", linewidth=1, label="Start")
    plt.axvline(actual, linestyle="-", linewidth=2, label=f"Actual end (${actual:,.0f})")
    plt.title(title)
    plt.xlabel("Ending value ($)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-date", type=str, default="2018-01-01",
                    help="Date where TRAIN ends and TEST begins (YYYY-MM-DD).")
    ap.add_argument("--n-paths", type=int, default=10_000, help="Monte Carlo paths.")
    ap.add_argument("--start-value", type=float, default=10_000.0, help="Initial capital.")
    ap.add_argument("--sma-window", type=int, default=200, help="SMA window for trend strategy.")
    ap.add_argument("--seed", type=int, default=1234, help="RNG seed.")
    args = ap.parse_args()

    split_date = pd.Timestamp(args.split_date)

    # 1) Ensure data
    spy_csv = fetch_spy_history()
    tbill_csv = fetch_tbill_3m_daily()

    df = load_spy_csv(spy_csv)
    close = df["Close"].copy()
    cash_daily = load_tbill_daily(tbill_csv).reindex(close.index).ffill().fillna(0.0)

    # 2) Signals on FULL history (shifted inside signal functions)
    sig_bh = signal_buy_and_hold(close)
    sig_sma = signal_sma(close, window=args.sma_window)

    # 3) TRAIN/TEST split
    train_mask = close.index < split_date
    test_mask  = close.index >= split_date

    # Sanity: require at least one year in both sets
    if train_mask.sum() < 252 or test_mask.sum() < 252:
        raise ValueError("Need at least ~1 year of data in both TRAIN and TEST. Adjust --split-date.")

    # Slice series
    close_tr, close_te = close[train_mask], close[test_mask]
    sig_bh_tr, sig_bh_te = sig_bh[train_mask], sig_bh[test_mask]
    sig_sma_tr, sig_sma_te = sig_sma[train_mask], sig_sma[test_mask]
    cash_tr, cash_te = cash_daily[train_mask], cash_daily[test_mask]

    # 4) Backtests to get TRAIN returns (for MC) and TEST actual (for comparison)
    # Buy&Hold ignores cash
    res_bh_tr = run_backtest(close_tr, sig_bh_tr, fee_bps=0.0, slippage_bps=0.0, cash_daily=0.0)
    res_bh_te = run_backtest(close_te, sig_bh_te, fee_bps=0.0, slippage_bps=0.0, cash_daily=0.0)

    # SMA uses cash yield when flat
    res_sma_tr = run_backtest(close_tr, sig_sma_tr, fee_bps=1.0, slippage_bps=2.0, cash_daily=cash_tr)
    res_sma_te = run_backtest(close_te, sig_sma_te, fee_bps=1.0, slippage_bps=2.0, cash_daily=cash_te)

    # Actual test ending values (compound the test returns from start_value)
    actual_bh_end  = compound_to_value(args.start_value, res_bh_te.strat_returns)
    actual_sma_end = compound_to_value(args.start_value, res_sma_te.strat_returns)

    # 5) Monte Carlo from TRAIN daily returns over TEST horizon
    horizon_days = len(res_bh_te.strat_returns)  # both strategies have same length in test
    mc_bh = simulate_iid(res_bh_tr.strat_returns, horizon_days, n_paths=args.n_paths, start_value=args.start_value, seed=args.seed)
    mc_sma = simulate_iid(res_sma_tr.strat_returns, horizon_days, n_paths=args.n_paths, start_value=args.start_value, seed=args.seed + 1)

    # 6) Summaries comparing MC vs actual
    row_bh = summarize_predictive_check(
        "Buy&Hold",
        args.start_value,
        mc_bh.ending_values,
        actual_bh_end,
        train_start=close_tr.index[0].date().isoformat(),
        train_end=close_tr.index[-1].date().isoformat(),
        test_start=close_te.index[0].date().isoformat(),
        test_end=close_te.index[-1].date().isoformat(),
    )
    row_bh["horizon_days"] = horizon_days

    row_sma = summarize_predictive_check(
        f"SMA{args.sma_window}+Cash",
        args.start_value,
        mc_sma.ending_values,
        actual_sma_end,
        train_start=close_tr.index[0].date().isoformat(),
        train_end=close_tr.index[-1].date().isoformat(),
        test_start=close_te.index[0].date().isoformat(),
        test_end=close_te.index[-1].date().isoformat(),
    )
    row_sma["horizon_days"] = horizon_days

    summary = pd.DataFrame([row_bh, row_sma])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "summary.csv"
    summary.to_csv(csv_path, index=False)

    print("\n=== MC Predictive Check (TRAIN→TEST) ===")
    print(summary.to_string(index=False))
    print(f"\n✅ Saved: {csv_path}")

    # 7) Plots
    plot_hist_with_actual(mc_bh.ending_values, actual_bh_end, OUT_DIR / "buyhold_hist_vs_actual.png",
                          f"Buy&Hold — Simulated Test Endings vs Actual ({row_bh['test_start']}→{row_bh['test_end']})",
                          args.start_value)
    plot_hist_with_actual(mc_sma.ending_values, actual_sma_end, OUT_DIR / "sma_hist_vs_actual.png",
                          f"SMA{args.sma_window}+Cash — Simulated Test Endings vs Actual ({row_sma['test_start']}→{row_sma['test_end']})",
                          args.start_value)
    print(f"✅ Saved histograms in: {OUT_DIR}\n")


if __name__ == "__main__":
    main()
