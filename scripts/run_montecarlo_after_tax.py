#!/usr/bin/env python3
"""
Monte Carlo simulation (3y, $10k) with a more realistic after-tax model:

- Buy&Hold: compound pre-tax, then tax ONE TIME at the end at LTCG rate.
- SMA200+Cash-like: once PER YEAR:
    * compute $ change for the year,
    * offset with loss carryforward,
    * tax net positive at SHORT-TERM rate,
    * reduce equity (tax drag on compounding).

Outputs:
- reports/montecarlo/after_tax/summary.csv
- reports/montecarlo/after_tax/buyhold_hist_after_tax.png
- reports/montecarlo/after_tax/sma200_hist_after_tax.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from quant_sim_lab.config.tax import SHORT_TERM_RATE, LONG_TERM_RATE
from quant_sim_lab.data.loader import (
    fetch_spy_history, load_spy_csv,
    fetch_tbill_3m_daily, load_tbill_daily
)
from quant_sim_lab.strategies.buy_and_hold import signal_buy_and_hold
from quant_sim_lab.strategies.sma_trend import signal_sma
from quant_sim_lab.backtest.engine import run_backtest
from quant_sim_lab.sim.monte_carlo import (
    simulate_iid_paths,
    summarize_mc,
    TRADING_DAYS_PER_YEAR,
)
from quant_sim_lab.risk.tax_annual import apply_annual_tax_to_path, buyhold_tax_end_only

# ---- Config ----
HORIZON_YEARS = 3
HORIZON_DAYS  = int(252 * HORIZON_YEARS)
N_PATHS       = 10_000
START_VALUE   = 10_000.0
SEED          = 777
OUT_DIR       = Path("reports/montecarlo/after_tax")
# ----------------


def plot_hist(vals: np.ndarray, out_path: Path, title: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8,5))
    plt.hist(vals, bins=80)
    plt.axvline(START_VALUE, linestyle="--")
    plt.title(title)
    plt.xlabel("Ending value ($)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    # 1) Build historical daily strategy returns (pre-tax)
    spy_csv   = fetch_spy_history()
    tbill_csv = fetch_tbill_3m_daily()
    df        = load_spy_csv(spy_csv)
    close     = df["Close"]
    cash_daily = load_tbill_daily(tbill_csv).reindex(close.index).ffill().fillna(0.0)

    sig_bh  = signal_buy_and_hold(close)
    sig_sma = signal_sma(close, window=200)

    res_bh  = run_backtest(close, sig_bh, fee_bps=0.0, slippage_bps=0.0, cash_daily=0.0)
    res_sma = run_backtest(close, sig_sma, fee_bps=1.0, slippage_bps=2.0, cash_daily=cash_daily)

    # 2) Simulate IID PATHS (pre-tax daily returns)
    #    We need full paths to apply annual taxes path-wise.
    rng_seed_bh  = SEED
    rng_seed_sma = SEED + 1

    # Simulate returns paths, then compound inside the tax engine
    # simulate_iid_paths returns equity paths if we pass daily returns,
    # but here we need the DATES for annual boundaries; so we synthesize a date index.
    # We'll assume 252 trading days/year starting "today".
    # For realism you could align to real calendar, but this approximation is fine for tax timing.
    start_date = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(start=start_date, periods=HORIZON_DAYS)  # business days

    # Draw IID daily returns matrices (n_paths, n_days)
    # We'll re-use the bootstrap you already have by sampling the returns,
    # then apply annual tax over the simulated path.
    from quant_sim_lab.sim.monte_carlo import _iid_bootstrap_daily
    r_bh  = _iid_bootstrap_daily(res_bh.strat_returns,  HORIZON_DAYS, N_PATHS, np.random.default_rng(rng_seed_bh))
    r_sma = _iid_bootstrap_daily(res_sma.strat_returns, HORIZON_DAYS, N_PATHS, np.random.default_rng(rng_seed_sma))

    # 3) Apply tax engines path-by-path
    end_bh_at  = np.empty(N_PATHS, dtype=float)
    end_sma_at = np.empty(N_PATHS, dtype=float)

    for i in range(N_PATHS):
        # Buy&Hold: tax at end only
        eq_bh_at  = buyhold_tax_end_only(r_bh[i], START_VALUE, LONG_TERM_RATE)
        end_bh_at[i] = eq_bh_at[-1]

        # SMA-like: annual tax with carryforward
        eq_sma_at = apply_annual_tax_to_path(r_sma[i], dates, START_VALUE, SHORT_TERM_RATE, treat_all_as_short_term=True)
        end_sma_at[i] = eq_sma_at[-1]

    # 4) Summaries
    sum_bh_at  = summarize_mc(end_bh_at, START_VALUE)
    sum_sma_at = summarize_mc(end_sma_at, START_VALUE)

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([
        {"strategy": "Buy&Hold (after-tax end)", **sum_bh_at},
        {"strategy": "SMA200+Cash (annual after-tax)", **sum_sma_at},
    ])
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n=== Monte Carlo (More Realistic After-Tax, 3y, $10k) ===")
    print(summary.to_string(index=False))
    print(f"\n✅ Saved summary: {summary_path}")

    # 5) Histograms
    plot_hist(end_bh_at,  out_dir / "buyhold_hist_after_tax.png", "Buy&Hold — Ending Values (after-tax)")
    plot_hist(end_sma_at, out_dir / "sma200_hist_after_tax.png", "SMA200+Cash — Ending Values (annual after-tax)")

    print(f"✅ Saved histograms: {out_dir}\n")


if __name__ == "__main__":
    main()
