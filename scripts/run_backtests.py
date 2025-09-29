#!/usr/bin/env python3
"""
Run two strategies on SPY:
- Buy & Hold
- 200-day SMA trend-following

Outputs:
- reports/backtests/metrics.csv
- reports/figures/equity_curves.png
- reports/figures/drawdowns.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from quant_sim_lab.data.loader import fetch_spy_history, load_spy_csv
from quant_sim_lab.strategies.buy_and_hold import signal_buy_and_hold
from quant_sim_lab.strategies.sma_trend import signal_sma
from quant_sim_lab.backtest.engine import run_backtest
from quant_sim_lab.risk.metrics import (
    cagr_from_series, annualized_vol, sharpe_ratio, sortino_ratio,
    max_drawdown, drawdown_series
)


def summarize_metrics(name: str, equity: pd.Series, returns: pd.Series, rf_annual: float = 0.0) -> dict:
    # Ensure datetime-ish labels (safety; loader should already enforce this)
    try:
        start_lbl = equity.index[0].date().isoformat()
        end_lbl = equity.index[-1].date().isoformat()
    except Exception:
        # Fallback if index is not datetime-like
        start_lbl = str(equity.index[0])
        end_lbl = str(equity.index[-1])

    return {
        "strategy": name,
        "cagr": cagr_from_series(equity),
        "vol_annual": annualized_vol(returns),
        "sharpe": sharpe_ratio(returns, rf=rf_annual),
        "sortino": sortino_ratio(returns, rf=rf_annual),
        "max_drawdown": max_drawdown(equity),
        "last_equity": float(equity.iloc[-1]),
        "samples": int(len(returns)),
        "start": start_lbl,
        "end": end_lbl,
    }



def main():
    # Ensure data exists
    csv_path = fetch_spy_history()  # idempotent
    df = load_spy_csv(csv_path)

    close = df["Close"].copy()

    # Build signals
    sig_bh = signal_buy_and_hold(close)
    sig_sma = signal_sma(close, window=200)

    # Run backtests (use small cost defaults; you can tune later)
    res_bh = run_backtest(close, sig_bh, fee_bps=0.0, slippage_bps=0.0)
    res_sma = run_backtest(close, sig_sma, fee_bps=1.0, slippage_bps=2.0)  # example: 1bp fee + 2bp slippage on switches

    # Metrics
    metrics = []
    metrics.append(summarize_metrics("Buy&Hold", res_bh.equity, res_bh.strat_returns))
    metrics.append(summarize_metrics("SMA200", res_sma.equity, res_sma.strat_returns))
    met_df = pd.DataFrame(metrics)

    # Save metrics
    out_dir = Path("reports/backtests")
    out_dir.mkdir(parents=True, exist_ok=True)
    met_csv = out_dir / "metrics.csv"
    met_df.to_csv(met_csv, index=False)

    print("\n=== Metrics ===")
    print(met_df.to_string(index=False))
    print(f"\n✅ Saved metrics: {met_csv}")

    # Plot equity curves
    fig_eq = Path("reports/figures/equity_curves.png")
    plt.figure(figsize=(10, 5))
    (res_bh.equity.rename("Buy&Hold") * 1.0).plot()
    (res_sma.equity.rename("SMA200") * 1.0).plot()
    plt.title("Equity Curves (Starting at 1.0)")
    plt.xlabel("Date"); plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout(); plt.savefig(fig_eq); plt.close()
    print(f"✅ Saved equity curves: {fig_eq}")

    # Plot drawdowns
    fig_dd = Path("reports/figures/drawdowns.png")
    dd_bh = drawdown_series(res_bh.equity).rename("Buy&Hold")
    dd_sma = drawdown_series(res_sma.equity).rename("SMA200")
    plt.figure(figsize=(10,5))
    dd_bh.plot()
    dd_sma.plot()
    plt.title("Drawdowns")
    plt.xlabel("Date"); plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout(); plt.savefig(fig_dd); plt.close()
    print(f"✅ Saved drawdowns: {fig_dd}\n")


if __name__ == "__main__":
    main()
