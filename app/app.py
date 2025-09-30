# app/app.py
# ------------------------------------------------------------
# Quant Sim Lab — Streamlit UI
# ------------------------------------------------------------
# IMPORTANT: keep this path hack at the very top so local src/ imports work
import sys
from pathlib import Path
sys.path.append(str((Path(__file__).resolve().parents[1] / "src").resolve()))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Friendly dependency check (helps if the wrong Python env is used)
try:
    import yfinance  # noqa: F401
except ImportError as e:
    raise SystemExit(
        "Missing dependency 'yfinance'.\n"
        "➡ Activate your venv and install deps:\n"
        "   source .venv/bin/activate && pip install -r requirements.txt\n"
        "➡ Then launch: python -m streamlit run app/app.py"
    ) from e

from quant_sim_lab.data.loader import (
    fetch_spy_history, load_spy_csv,
    fetch_tbill_3m_daily, load_tbill_daily,
)
from quant_sim_lab.strategies.buy_and_hold import signal_buy_and_hold
from quant_sim_lab.strategies.sma_trend import signal_sma
from quant_sim_lab.backtest.engine import run_backtest
from quant_sim_lab.risk.metrics import (
    cagr_from_series, annualized_vol, sharpe_ratio, sortino_ratio, max_drawdown,
)
from quant_sim_lab.sim.monte_carlo import (
    simulate_iid, simulate_block_paths, percentile_cone, summarize_mc,
    _iid_bootstrap_daily,  # internal sampler used for after-tax pathwise calc
)
from quant_sim_lab.risk.tax_annual import (
    apply_annual_tax_to_path, buyhold_tax_end_only,
)
from quant_sim_lab.config.tax import SHORT_TERM_RATE, LONG_TERM_RATE

# ------------------------------------------------------------
# Streamlit page config
# ------------------------------------------------------------
st.set_page_config(page_title="Quant Sim Lab", layout="wide")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_market_data():
    """Fetch SPY and 3m T-bill (^IRX), return (df_prices, cash_daily)."""
    spy_csv = fetch_spy_history()
    tbill_csv = fetch_tbill_3m_daily()
    df = load_spy_csv(spy_csv)  # has 'Close'
    cash = load_tbill_daily(tbill_csv).reindex(df.index).ffill().fillna(0.0)
    return df, cash

def summarize_backtest(name, equity, returns):
    return {
        "strategy": name,
        "cagr": cagr_from_series(equity),
        "vol_annual": annualized_vol(returns),
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "max_drawdown": max_drawdown(equity),
        "last_equity": float(equity.iloc[-1]),
        "samples": int(len(returns)),
        "start": equity.index[0],
        "end": equity.index[-1],
    }

def plot_equity_curves(curves: dict[str, pd.Series], title: str):
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, ser in curves.items():
        ax.plot(ser.index, ser.values, label=name)
    ax.set_title(title)
    ax.set_ylabel("Equity (× start)")
    ax.legend()
    st.pyplot(fig)

def plot_hist(values: np.ndarray, start_value: float, actual: float | None, title: str):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(values, bins=80, alpha=0.8)
    ax.axvline(start_value, linestyle="--", linewidth=1, label="Start")
    if actual is not None:
        ax.axvline(actual, linestyle="-", linewidth=2, label=f"Actual end (${actual:,.0f})")
    ax.set_title(title)
    ax.set_xlabel("Ending value ($)")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

def plot_cone(cone_df: pd.DataFrame, start_value: float, title: str):
    fig, ax = plt.subplots(figsize=(9, 4))
    if all(c in cone_df for c in ["p25", "p75"]):
        ax.fill_between(cone_df.index, cone_df["p25"], cone_df["p75"], alpha=0.3, label="IQR 25–75%")
    if all(c in cone_df for c in ["p05", "p95"]):
        ax.fill_between(cone_df.index, cone_df["p05"], cone_df["p95"], alpha=0.15, label="5–95%")
    if "p50" in cone_df:
        ax.plot(cone_df.index, cone_df["p50"], linewidth=2, label="Median")
    ax.axhline(start_value, linestyle="--", linewidth=1, label="Start")
    ax.set_title(title)
    ax.set_xlabel("Trading days from today")
    ax.set_ylabel("Portfolio value ($)")
    ax.legend()
    st.pyplot(fig)

def compound_to_value(start: float, rets: pd.Series) -> float:
    r = rets.fillna(0.0).to_numpy(dtype=float)
    return float(start * np.prod(1.0 + r))

# ------------------------------------------------------------
# Sidebar controls (assumptions)
# ------------------------------------------------------------
st.sidebar.header("Assumptions")
START_VALUE = st.sidebar.number_input("Starting capital ($)", min_value=1000, value=10_000, step=500)
HORIZON_YEARS = st.sidebar.slider("Horizon (years)", 1, 10, 3)
TRADING_DAYS = int(252 * HORIZON_YEARS)
N_PATHS = st.sidebar.slider("Monte Carlo paths", 1000, 50_000, 10_000, step=1000)
SEED = st.sidebar.number_input("Random seed", value=2025, step=1)
SMA_WIN = st.sidebar.selectbox("SMA window (days)", [50, 100, 150, 200, 250], index=3)
FEE_BPS = st.sidebar.number_input("Fee (bps per trade)", value=1.0, step=0.5)
SLIP_BPS = st.sidebar.number_input("Slippage (bps per trade)", value=2.0, step=0.5)

st.sidebar.divider()
st.sidebar.subheader("Bootstrap options")
BOOT_KIND = st.sidebar.radio("Bootstrap", ["IID", "Block"], index=1)
BLOCK_SIZE = st.sidebar.slider("Block size (days)", 5, 60, 20, step=5)

st.sidebar.divider()
st.sidebar.subheader("Tax assumptions")
st_stcg = st.sidebar.number_input("Short-term rate (SMA & cash)", min_value=0.0, max_value=0.99, value=float(SHORT_TERM_RATE), step=0.01, format="%.2f")
st_ltcg = st.sidebar.number_input("Long-term rate (Buy&Hold end)",  min_value=0.0, max_value=0.99, value=float(LONG_TERM_RATE),  step=0.01, format="%.2f")

st.sidebar.divider()
st.sidebar.subheader("Train/Test split for MC backtest")
split_date = st.sidebar.date_input("Split date (train ends, test begins)", pd.to_datetime("2018-01-01")).strftime("%Y-%m-%d")

# ------------------------------------------------------------
# Load data & prepare backtests
# ------------------------------------------------------------
df, cash_daily = load_market_data()
close = df["Close"]

sig_bh = signal_buy_and_hold(close)
sig_sma = signal_sma(close, window=SMA_WIN)

# Backtests on full history
res_bh  = run_backtest(close, sig_bh, fee_bps=0.0, slippage_bps=0.0, cash_daily=0.0)
res_sma = run_backtest(close, sig_sma, fee_bps=FEE_BPS, slippage_bps=SLIP_BPS, cash_daily=cash_daily)

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Assumptions & Methodology",
    "Backtests (Historical)",
    "Monte Carlo — Before Tax",
    "Monte Carlo — After Tax (Annual)",
    "MC Predictive Check (Train→Test)",
])

# -----------------------
# Overview
# -----------------------
with tab0:
    st.markdown("## Quant Sim Lab — Dashboard")
    st.write("Use the sidebar to tweak assumptions. Each tab explains the reasoning in plain English.")

# -----------------------
# Assumptions & Methodology
# -----------------------
with tab1:
    st.markdown("### Plain-English Assumptions & Methodology")
    st.markdown("""
**Data**
- SPY adjusted close for price; 3-month T-bill (^IRX) as cash yield proxy (converted to daily return).

**Strategies**
- **Buy & Hold**: always invested.
- **SMA Trend**: if price > SMA(window) → long; else in cash and earn T-bill yield. Costs apply on position changes.

**Backtest math**
- Daily strategy return = position × asset return + (1 − position) × cash daily − trading costs.
- Equity curve = cumulative product of (1 + daily return), normalized to 1 at start.

**Monte Carlo (Before Tax)**
- Resample historical **strategy daily returns** to build thousands of alternative futures.
- **IID** = days sampled independently; **Block** = sample contiguous blocks (preserve streaks).

**After-Tax (Annual Netting)**
- **Buy & Hold**: tax once at the end at long-term rate.
- **SMA+Cash**: each year, net gains/losses; apply short-term tax to net positive; carry losses forward; reduce equity at year-end.

**Predictive Check**
- Split history at a **split date**: use **train** returns to simulate **test** horizon; compare actual vs simulated distribution.
""")

# -----------------------
# Backtests (Historical)
# -----------------------
with tab2:
    st.markdown("### Backtests (Full History)")

    with st.expander("🔎 What do these metrics mean?"):
        st.markdown("""
- **CAGR**: the smoothed annual growth rate (what constant annual % would get you from start to finish).
- **Vol_annual**: annualized volatility; how bumpy the ride is.
- **Sharpe**: return per unit of total volatility.
- **Sortino**: return per unit of downside volatility (ignores upside bumps).
- **Max drawdown**: worst peak-to-trough loss (historical pain).
- **Last equity**: how many times your starting $1 became (e.g., 7.9 ⇒ $1 → $7.9).
        """)

    met_df = pd.DataFrame([
        summarize_backtest("Buy&Hold", res_bh.equity, res_bh.strat_returns),
        summarize_backtest(f"SMA{SMA_WIN}+Cash", res_sma.equity, res_sma.strat_returns),
    ])
    st.dataframe(
        met_df[["strategy","cagr","vol_annual","sharpe","sortino","max_drawdown","last_equity","samples"]],
        use_container_width=True
    )

    plot_equity_curves({
        "Buy&Hold": res_bh.equity,
        f"SMA{SMA_WIN}+Cash": res_sma.equity
    }, "Equity Curves (normalized to 1.0)")

# -----------------------
# Monte Carlo — Before Tax
# -----------------------
with tab3:
    st.markdown("### Monte Carlo — Before Tax")

    with st.expander("🔎 Why simulate? What do the columns mean?"):
        st.markdown("""
- We **resample** historical strategy daily returns to create many alternate futures for your chosen horizon.
- **Median_end** is the 50/50 outcome; **Mean_end** is the average (skewed by big winners).
- **p05 / p95** bracket bad/great scenarios; **Prob_end_below_start** = chance you end below your starting capital.
        """)

    if BOOT_KIND == "IID":
        mc_bh  = simulate_iid(res_bh.strat_returns,  TRADING_DAYS, n_paths=N_PATHS, start_value=START_VALUE, seed=SEED)
        mc_sma = simulate_iid(res_sma.strat_returns, TRADING_DAYS, n_paths=N_PATHS, start_value=START_VALUE, seed=SEED+1)
        endings_bh, endings_sma = mc_bh.ending_values, mc_sma.ending_values
    else:
        paths_bh  = simulate_block_paths(res_bh.strat_returns,  TRADING_DAYS, n_paths=N_PATHS, start_value=START_VALUE, block_size=BLOCK_SIZE, seed=SEED)
        paths_sma = simulate_block_paths(res_sma.strat_returns, TRADING_DAYS, n_paths=N_PATHS, start_value=START_VALUE, block_size=BLOCK_SIZE, seed=SEED+1)
        endings_bh, endings_sma = paths_bh[:, -1], paths_sma[:, -1]

    sum_bh  = summarize_mc(endings_bh, START_VALUE)
    sum_sma = summarize_mc(endings_sma, START_VALUE)

    st.dataframe(pd.DataFrame([
        {"strategy":"Buy&Hold", **sum_bh},
        {"strategy":f"SMA{SMA_WIN}+Cash", **sum_sma},
    ]), use_container_width=True)

    colA, colB = st.columns(2)
    with colA:
        plot_hist(endings_bh, START_VALUE, None, "Buy&Hold — Ending Values (Before Tax)")
    with colB:
        plot_hist(endings_sma, START_VALUE, None, f"SMA{SMA_WIN}+Cash — Ending Values (Before Tax)")

# -----------------------
# Monte Carlo — After Tax (Annual)
# -----------------------
with tab4:
    st.markdown("### Monte Carlo — After Tax (Annual Netting + Carryforward)")

    with st.expander("🔎 How does tax change the picture?"):
        st.markdown(f"""
- **Buy & Hold**: we tax once at the end at long-term rate (**{st_ltcg:.0%}**). Daily compounding is untouched.
- **SMA+Cash**: more trading → taxable events. We tax **net yearly gains** at short-term rate (**{st_stcg:.0%}**), allow loss carryforward, and reduce equity at year-end.
- Result: the **right tail** (big wins) is trimmed more than the left tail (bad years often owe no tax).
        """)

    # Build synthetic biz-day calendar for the horizon
    start_date = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(start=start_date, periods=TRADING_DAYS)

    # Draw iid daily returns arrays; apply annual taxes path-wise
    rng1, rng2 = np.random.default_rng(SEED), np.random.default_rng(SEED+1)
    r_bh  = _iid_bootstrap_daily(res_bh.strat_returns,  TRADING_DAYS, N_PATHS, rng1)
    r_sma = _iid_bootstrap_daily(res_sma.strat_returns, TRADING_DAYS, N_PATHS, rng2)

    end_bh_at  = np.empty(N_PATHS)
    end_sma_at = np.empty(N_PATHS)
    for i in range(N_PATHS):
        eq_bh  = buyhold_tax_end_only(r_bh[i],  START_VALUE, st_ltcg)
        eq_sma = apply_annual_tax_to_path(r_sma[i], dates, START_VALUE, st_stcg)
        end_bh_at[i], end_sma_at[i] = eq_bh[-1], eq_sma[-1]

    sum_bh_at  = summarize_mc(end_bh_at, START_VALUE)
    sum_sma_at = summarize_mc(end_sma_at, START_VALUE)

    st.dataframe(pd.DataFrame([
        {"strategy":"Buy&Hold (after-tax)", **sum_bh_at},
        {"strategy":f"SMA{SMA_WIN}+Cash (after-tax)", **sum_sma_at},
    ]), use_container_width=True)

    colA, colB = st.columns(2)
    with colA:
        plot_hist(end_bh_at, START_VALUE, None, "Buy&Hold — Ending Values (After Tax)")
    with colB:
        plot_hist(end_sma_at, START_VALUE, None, f"SMA{SMA_WIN}+Cash — Ending Values (After Tax)")

    # Optional: cones (for visuals; before-tax cones shown; after-tax cones require heavier path logic)
    if st.checkbox("Show percentile cones (Before Tax, Block Bootstrap)", value=False):
        paths_bh  = simulate_block_paths(res_bh.strat_returns,  TRADING_DAYS, n_paths=min(N_PATHS, 5000), start_value=START_VALUE, block_size=BLOCK_SIZE, seed=SEED+11)
        paths_sma = simulate_block_paths(res_sma.strat_returns, TRADING_DAYS, n_paths=min(N_PATHS, 5000), start_value=START_VALUE, block_size=BLOCK_SIZE, seed=SEED+12)
        cone_bh  = percentile_cone(paths_bh)
        cone_sma = percentile_cone(paths_sma)
        plot_cone(cone_bh, START_VALUE, "Buy&Hold — Percentile Cone (Before Tax, Block Bootstrap)")
        plot_cone(cone_sma, START_VALUE, f"SMA{SMA_WIN}+Cash — Percentile Cone (Before Tax, Block Bootstrap)")

# -----------------------
# MC Predictive Check (Train→Test)
# -----------------------
with tab5:
    st.markdown("### Predictive Check (Train→Test)")

    with st.expander("🔎 What are we testing here?"):
        st.markdown("""
- We split the data into **TRAIN** (before the split date) and **TEST** (after).
- Using TRAIN strategy returns, we simulate the TEST horizon to get a distribution.
- Then we see where the **actual TEST ending** lands within that distribution.
- If the model is reasonable, actual outcomes should land inside the 5–95% band most of the time.
        """)

    split = pd.Timestamp(split_date)
    mask_tr = close.index < split
    mask_te = close.index >= split

    if mask_tr.sum() < 252 or mask_te.sum() < 252:
        st.warning("Need ≥ 1 year in both TRAIN and TEST. Adjust split date.")
    else:
        # Slices
        close_tr, close_te = close[mask_tr], close[mask_te]
        sig_bh_tr,  sig_bh_te  = sig_bh[mask_tr],  sig_bh[mask_te]
        sig_sma_tr, sig_sma_te = sig_sma[mask_tr], sig_sma[mask_te]
        cash_tr,    cash_te    = cash_daily[mask_tr], cash_daily[mask_te]

        # Backtests per slice (pre-tax)
        res_bh_tr  = run_backtest(close_tr, sig_bh_tr, fee_bps=0.0, slippage_bps=0.0, cash_daily=0.0)
        res_bh_te  = run_backtest(close_te, sig_bh_te, fee_bps=0.0, slippage_bps=0.0, cash_daily=0.0)
        res_sma_tr = run_backtest(close_tr, sig_sma_tr, fee_bps=FEE_BPS, slippage_bps=SLIP_BPS, cash_daily=cash_tr)
        res_sma_te = run_backtest(close_te, sig_sma_te, fee_bps=FEE_BPS, slippage_bps=SLIP_BPS, cash_daily=cash_te)

        # Actual endings on TEST
        actual_bh  = compound_to_value(START_VALUE, res_bh_te.strat_returns)
        actual_sma = compound_to_value(START_VALUE, res_sma_te.strat_returns)

        # Simulate TEST horizon from TRAIN returns (pre-tax)
        horizon = len(res_bh_te.strat_returns)
        mc_bh  = simulate_iid(res_bh_tr.strat_returns,  horizon, n_paths=N_PATHS, start_value=START_VALUE, seed=SEED+21)
        mc_sma = simulate_iid(res_sma_tr.strat_returns, horizon, n_paths=N_PATHS, start_value=START_VALUE, seed=SEED+22)

        s_bh  = summarize_mc(mc_bh.ending_values, START_VALUE)
        s_sma = summarize_mc(mc_sma.ending_values, START_VALUE)
        pct_bh  = float(np.mean(mc_bh.ending_values  <= actual_bh))
        pct_sma = float(np.mean(mc_sma.ending_values <= actual_sma))

        st.dataframe(pd.DataFrame([
            {"strategy":"Buy&Hold",
             "train": f"{close_tr.index[0].date()} → {close_tr.index[-1].date()}",
             "test":  f"{close_te.index[0].date()} → {close_te.index[-1].date()}",
             "actual_end": actual_bh,
             "mc_median_end": s_bh["median_end"],
             "mc_p05_end": s_bh["p05_end"],
             "mc_p95_end": s_bh["p95_end"],
             "actual_percentile_in_MC": pct_bh},
            {"strategy":f"SMA{SMA_WIN}+Cash",
             "train": f"{close_tr.index[0].date()} → {close_tr.index[-1].date()}",
             "test":  f"{close_te.index[0].date()} → {close_te.index[-1].date()}",
             "actual_end": actual_sma,
             "mc_median_end": s_sma["median_end"],
             "mc_p05_end": s_sma["p05_end"],
             "mc_p95_end": s_sma["p95_end"],
             "actual_percentile_in_MC": pct_sma},
        ]), use_container_width=True)

        st.markdown("**Distributions vs Actual**")
        colA, colB = st.columns(2)
        with colA:
            plot_hist(mc_bh.ending_values, START_VALUE, actual_bh, "Buy&Hold — Simulated Test Endings vs Actual")
        with colB:
            plot_hist(mc_sma.ending_values, START_VALUE, actual_sma, f"SMA{SMA_WIN}+Cash — Simulated Test Endings vs Actual")
