#!/usr/bin/env python3
"""
Sanity check:
- Activates the loader
- Writes data/raw/spy.csv
- Computes daily returns + a 200-day MA
- Saves a figure to reports/figures/spy_price_ma.png
- Prints a tiny metrics summary
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import from our package
from quant_sim_lab.data.loader import fetch_spy_history, load_spy_csv


def main():
    # 1) Fetch & save SPY history (idempotent)
    csv_path = fetch_spy_history()

    # 2) Load the CSV
    df = load_spy_csv(csv_path)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])

    # 3) Compute returns and 200-day MA on Close
    df = df.copy()
    if "Close" not in df.columns:
        # If using auto_adjust=True, Close is adjusted; ensure column exists
        raise RuntimeError("Expected 'Close' column missing in downloaded data.")

    df["ret_daily"] = df["Close"].pct_change(fill_method=None)  # no implicit pad
    df["ma200"] = df["Close"].rolling(200, min_periods=200).mean()

    # Drop first 200 days with NaN MA to avoid misleading stats
    df2 = df.dropna().copy()

    # 4) Quick metrics (for your intuition)
    # Annualize daily stats with ~252 trading days
    mean_daily = df2["ret_daily"].mean()
    vol_daily = df2["ret_daily"].std()

    cagr = (df2["Close"].iloc[-1] / df2["Close"].iloc[0]) ** (252 / len(df2)) - 1
    vol_annual = vol_daily * np.sqrt(252)

    print("\n=== SPY Sanity Check ===")
    print(f"Samples: {len(df2):,} daily bars")
    print(f"CAGR (approx): {cagr:.2%}")
    print(f"Annualized Vol: {vol_annual:.2%}")
    print(f"Mean Daily: {mean_daily:.4%},  Std Daily: {vol_daily:.4%}")

    # 5) Plot price + MA and save
    fig_out = Path("reports/figures/spy_price_ma.png")
    fig_out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    df[["Close", "ma200"]].plot(ax=plt.gca())
    plt.title("SPY Close with 200-Day Moving Average")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(fig_out)
    plt.close()

    print(f"✅ Wrote data to: {csv_path}")
    print(f"✅ Saved figure to: {fig_out}\n")


if __name__ == "__main__":
    # Ensure we run from project root so relative paths work
    root = Path(__file__).resolve().parents[1]
    if Path.cwd() != root:
        # Re-run from root
        os_exec = sys.executable
        script = Path(__file__).resolve()
        import subprocess
        subprocess.check_call([os_exec, str(script)], cwd=root)
        sys.exit(0)
    main()
