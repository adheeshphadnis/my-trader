#!/usr/bin/env python3
"""
Run Buy&Hold and SMA trend with cash yield when out of market.
Also sweep SMA windows and compare metrics.

Outputs:
- reports/backtests/metrics_with_cash.csv
- reports/backtests/sma_sweep_metrics.csv
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
from quant_sim_lab.risk.metrics import (
    cagr_from_series, annualized_vol, sharpe_ratio, sortino_ratio,
    max_drawdown
)

OUT_DIR = Path("reports/backtests")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def summarize(name: str, equity: pd.Series, returns: pd.Series, rf_annual: float = 0.0) -> dict:
    return {
        "strategy": name,
        "cagr": cagr_from_series(equity),
        "vol_annual": annualized_vol(returns),
        "sharpe": sharpe_ratio(returns, rf=rf_annual),
        "sortino": sortino_ratio(returns, rf=rf_annual),
        "max_drawdown": max_drawdown(equity),
        "last_equity": float(equity.iloc[-1]),
        "samples": int(len(returns)),
        "start": equity.index[0].date().isoformat() if hasattr(equity.index[0], "date") else str(equity.index[0]),
        "end": equity.index[-1].date().isoformat() if hasattr(equity.index[-1], "date") else str(equity.index[-1]),
    }

def main():
    # Ensure data exists
    spy_csv = fetch_spy_history()
    tbill_csv = fetch_tbill_3m_daily()

    df = load_spy_csv(spy_csv)
    close = df["Close"]
    cash_daily = load_tbill_daily(tbill_csv).reindex(close.index).fillna(method="ffill").fillna(0.0)

    # --- Baseline strategies with cash yield ---
    sig_bh = signal_buy_and_hold(close)  # buy&hold ignores cash (always in)
    sig_sma200 = signal_sma(close, window=200)

    res_bh = run_backtest(close, sig_bh, fee_bps=0.0, slippage_bps=0.0, cash_daily=0.0)
    res_sma200 = run_backtest(close, sig_sma200, fee_bps=1.0, slippage_bps=2.0, cash_daily=cash_daily)

    metrics = [
        summarize("Buy&Hold", res_bh.equity, res_bh.strat_returns),
        summarize("SMA200+Cash", res_sma200.equity, res_sma200.strat_returns),
    ]
    met_df = pd.DataFrame(metrics)
    met_path = OUT_DIR / "metrics_with_cash.csv"
    met_df.to_csv(met_path, index=False)
    print("\n=== Metrics (with cash yield for SMA) ===")
    print(met_df.to_string(index=False))
    print(f"\n✅ Saved: {met_path}")

    # --- SMA window sweep with cash yield when out of market ---
    windows = [100, 150, 200, 250]
    sweep_rows = []
    for w in windows:
        sig = signal_sma(close, window=w)
        res = run_backtest(close, sig, fee_bps=1.0, slippage_bps=2.0, cash_daily=cash_daily)
        sweep_rows.append(summarize(f"SMA{w}+Cash", res.equity, res.strat_returns))

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_path = OUT_DIR / "sma_sweep_metrics.csv"
    sweep_df.to_csv(sweep_path, index=False)
    print("\n=== SMA Sweep (with cash) ===")
    print(sweep_df[["strategy","cagr","vol_annual","sharpe","sortino","max_drawdown","last_equity"]].to_string(index=False))
    print(f"\n✅ Saved: {sweep_path}")

if __name__ == "__main__":
    main()
