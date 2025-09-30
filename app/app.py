# app/app.py
# ------------------------------------------------------------
# Quant Sim Lab — Streamlit UI (Multi-Ticker, Date Window)
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
    import yfinance as yf  # noqa: F401
except ImportError as e:
    raise SystemExit(
        "Missing dependency 'yfinance'.\n"
        "➡ Activate your venv and install deps:\n"
        "   source .venv/bin/activate && pip install -r requirements.txt\n"
        "➡ Then launch: python -m streamlit run app/app.py"
    ) from e

from quant_sim_lab.data.loader import fetch_tbill_3m_daily, load_tbill_daily  # cash series (^IRX)
from quant_sim_lab.strategies.buy_and_hold import signal_buy_and_hold
from quant_sim_lab.strategies.sma_trend import signal_sma
from quant_sim_lab.backtest.engine import run_backtest
from quant_sim_lab.risk.metrics import (
    cagr_from_series, annualized_vol, sharpe_ratio, sortino_ratio, max_drawdown,
)
from quant_sim_lab.sim.monte_carlo import (
    simulate_iid, simulate_block_paths, percentile_cone, summarize_mc,
    _iid_bootstrap_daily,
)
from quant_sim_lab.risk.tax_annual import apply_annual_tax_to_path, buyhold_tax_end_only
from quant_sim_lab.config.tax import SHORT_TERM_RATE, LONG_TERM_RATE

# ------------------------------------------------------------
# Streamlit page config
# ------------------------------------------------------------
st.set_page_config(page_title="Quant Sim Lab", layout="wide")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_cash_series() -> pd.Series:
    """Fetch 3m T-bill (^IRX) daily cash return series and return as pandas Series."""
    tbill_csv = fetch_tbill_3m_daily()
    cash = load_tbill_daily(tbill_csv)
    return cash

@st.cache_data(show_spinner=False)
def load_prices(tickers: tuple[str, ...], start_date, end_date) -> pd.DataFrame:
    """
    Download Adjusted Close (or Close) for given tickers using yfinance,
    between [start_date, end_date). Returns a DataFrame (index=dates, columns=tickers).
    Robust to single- vs multi-index columns from yfinance.
    """
    import yfinance as yf  # local import for cache hashing
    if not tickers:
        raise ValueError("No tickers supplied.")

    start_str = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    end_str   = pd.Timestamp(end_date).strftime("%Y-%m-%d")

    tickers_arg = tickers[0] if len(tickers) == 1 else list(tickers)
    data = yf.download(
        tickers_arg,
        auto_adjust=True,     # 'Close' becomes adjusted
        progress=False,
        threads=True,
        interval="1d",
        start=start_str,
        end=end_str,
    )
    if data is None or len(data) == 0:
        raise ValueError(f"No data returned for {tickers} in {start_str}→{end_str}.")

    def extract_close_like(df: pd.DataFrame) -> pd.Series:
        for col in ("Close", "Adj Close", "close", "adj close"):
            if col in df.columns:
                return df[col].rename("Close")
        raise KeyError("Neither 'Close' nor 'Adj Close' found in columns.")

    # Handle MultiIndex vs single-index columns
    if isinstance(data.columns, pd.MultiIndex):
        # Try slice by field level first
        close = None
        for lev in (1, 0):  # commonly fields at level=1
            try:
                if "Close" in data.columns.get_level_values(lev):
                    close = data.xs("Close", axis=1, level=lev)
                    break
                if "Adj Close" in data.columns.get_level_values(lev):
                    close = data.xs("Adj Close", axis=1, level=lev)
                    break
            except Exception:
                pass
        if close is None:
            # Fallback: per-ticker extraction
            pieces = {}
            uniq0 = set(map(str, data.columns.get_level_values(0)))
            uniq1 = set(map(str, data.columns.get_level_values(1)))
            score0 = len(uniq0 & set(tickers))
            score1 = len(uniq1 & set(tickers))
            ticker_level = 0 if score0 >= score1 else 1
            for t in tickers:
                try:
                    df_t = data.xs(t, axis=1, level=ticker_level, drop_level=False)
                    df_t = df_t.droplevel(ticker_level, axis=1)
                    ser = extract_close_like(df_t).rename(t)
                    pieces[t] = ser
                except Exception:
                    continue
            if not pieces:
                raise KeyError("Could not extract Close/Adj Close from MultiIndex data.")
            close = pd.concat(pieces, axis=1)
        found = [c for c in close.columns if c in tickers]
        close = close[found]
        if len(tickers) == 1 and close.shape[1] == 1 and close.columns[0] != tickers[0]:
            close.columns = [tickers[0]]
    else:
        ser = extract_close_like(data)
        close = ser.to_frame(name=tickers[0])

    close = close.sort_index().dropna(how="all").ffill()
    empty_cols = [c for c in close.columns if close[c].isna().all()]
    if empty_cols:
        raise ValueError(f"No valid price data for: {empty_cols} in {start_str}→{end_str}.")
    return close

def normalize_weights(df_weights: pd.DataFrame) -> pd.Series:
    """Return weights Series (index=ticker) normalized to 1.0, negatives clipped to 0."""
    w = df_weights.copy()
    w["weight"] = pd.to_numeric(w["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    tot = w["weight"].sum()
    if tot <= 0:
        w["weight"] = 1.0 / len(w)
    else:
        w["weight"] = w["weight"] / tot
    return w.set_index("ticker")["weight"]

def build_portfolio_series(prices: pd.DataFrame, weights: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Daily-rebalanced portfolio from ticker prices + weights → (portfolio_close_index, portfolio_daily_returns)."""
    w = weights.reindex(prices.columns).fillna(0.0)
    rets = prices.pct_change().fillna(0.0)
    port_rets = (rets * w).sum(axis=1)
    close_idx = 100.0 * (1.0 + port_rets).cumprod()
    close_idx.name = "Close"
    return close_idx, port_rets

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
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (× start)")  # unitless multiple
    ax.legend()
    st.pyplot(fig)

def plot_hist(values: np.ndarray, start_value: float, actual: float | None, title: str):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(values, bins=80, alpha=0.8)
    ax.axvline(start_value, linestyle="--", linewidth=1, label=f"Start ({fmt_dollar(start_value)})")
    if actual is not None:
        ax.axvline(actual, linestyle="-", linewidth=2, label=f"Actual end ({fmt_dollar(actual)})")
    ax.set_title(title)
    ax.set_xlabel("Ending value (USD)")
    ax.set_ylabel("Frequency (paths)")
    _currency_axis(ax, decimals=0)
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
    ax.axhline(start_value, linestyle="--", linewidth=1, label=f"Start ({fmt_dollar(start_value)})")
    ax.set_title(title)
    ax.set_xlabel("Trading days from today")
    ax.set_ylabel("Portfolio value (USD)")
    _currency_axis(ax, decimals=0)
    ax.legend()
    st.pyplot(fig)

def compound_to_value(start: float, rets: pd.Series) -> float:
    r = rets.fillna(0.0).to_numpy(dtype=float)
    return float(start * np.prod(1.0 + r))
# === Benchmark helpers ===
def equal_weight_series(tickers: list[str]) -> pd.Series:
    if len(tickers) == 0:
        return pd.Series(dtype=float)
    w = 1.0 / len(tickers)
    return pd.Series({t: w for t in tickers})

def annualize_mean_std(daily_rets: pd.Series | np.ndarray) -> tuple[float, float]:
    r = np.asarray(daily_rets, dtype=float)
    mu_ann = float(np.nanmean(r) * 252)
    vol_ann = float(np.nanstd(r, ddof=0) * np.sqrt(252))
    return mu_ann, vol_ann

def capm_stats(port_rets: pd.Series, bench_rets: pd.Series) -> dict:
    """Daily CAPM-style stats of portfolio vs benchmark; alpha annualized."""
    # align and drop NaNs
    df = pd.concat([port_rets, bench_rets], axis=1).dropna()
    if df.shape[0] < 2:
        return {"beta": np.nan, "alpha_ann": np.nan, "r2": np.nan}
    pr = df.iloc[:, 0].to_numpy(float)
    br = df.iloc[:, 1].to_numpy(float)
    var_b = np.var(br)
    if var_b == 0.0:
        return {"beta": np.nan, "alpha_ann": np.nan, "r2": np.nan}
    cov_pb = np.cov(pr, br, ddof=0)[0,1]
    beta = cov_pb / var_b
    # daily alpha = mean(port - beta*bench)
    alpha_daily = np.mean(pr - beta * br)
    alpha_ann = alpha_daily * 252
    # r^2 from simple linear regression port ~ a + b*bench
    # r = corr(pr, br); r^2:
    r = np.corrcoef(pr, br)[0,1] if np.std(pr) > 0 and np.std(br) > 0 else np.nan
    r2 = r*r if not np.isnan(r) else np.nan
    return {"beta": float(beta), "alpha_ann": float(alpha_ann), "r2": float(r2)}

from matplotlib.ticker import FuncFormatter, PercentFormatter

# ---------- Number formatting ----------
def fmt_pct(x, decimals=2):
    if pd.isna(x): return ""
    return f"{x*100:.{decimals}f}%"

def fmt_pct_val(x, decimals=2):
    """When the value is already in percent space (e.g., 12.3 for 12.3%), not as a fraction."""
    if pd.isna(x): return ""
    return f"{x:.{decimals}f}%"

def fmt_dollar(x, decimals=0):
    if pd.isna(x): return ""
    return f"${x:,.{decimals}f}"

def fmt_num(x, decimals=2):
    if pd.isna(x): return ""
    return f"{x:.{decimals}f}"

# For matplotlib axes
def _currency_axis(ax, decimals=0):
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_dollar(v, decimals)))

def _percent_axis(ax, decimals=1):
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=decimals))


# ------------------------------------------------------------
# Sidebar — Universe, Weights, Data Window, Assumptions
# ------------------------------------------------------------
# --- Universe & Portfolio: Ticker selection with custom additions ---
st.sidebar.header("Universe & Portfolio")

DEFAULT_UNIVERSE = [
    "SPY","QQQ","VOO","IVV","DIA",
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA",
    "BRK-B","UNH","XOM","JPM","V","MA","HD","PG"
]

# Persist selection
if "selected_tickers" not in st.session_state:
    st.session_state.selected_tickers = ["SPY"]

# Build dynamic options = default universe U previously selected
dynamic_options = sorted(set(DEFAULT_UNIVERSE) | set(st.session_state.selected_tickers))

# Multiselect with safe default (intersection)
safe_default = [t for t in st.session_state.selected_tickers if t in dynamic_options]
selected = st.sidebar.multiselect(
    "Select tickers",
    options=dynamic_options,
    default=safe_default,
    help="Type to search, use checkboxes to add/remove.",
)

# Add custom ticker box
st.sidebar.caption("Don’t see your ticker? Add it below ⤵")
c1, c2 = st.sidebar.columns([2,1])
with c1:
    custom_symbol = st.text_input("Add custom ticker", value="", label_visibility="collapsed", placeholder="e.g., TLT or JOBY")
with c2:
    if st.button("Add"):
        sym = (custom_symbol or "").strip().upper()
        if sym:
            # update session selection and options
            if sym not in st.session_state.selected_tickers:
                st.session_state.selected_tickers.append(sym)
            # force re-run so multiselect options include the new symbol
            if hasattr(st, "rerun"):
                st.rerun()
            else:
                st.experimental_rerun()

# Persist final selection from multiselect
st.session_state.selected_tickers = selected

# Guard: need at least one
if not selected:
    st.sidebar.warning("Please select at least one ticker (e.g., SPY).")
    st.stop()

# Live weights editor
st.sidebar.markdown("**Weights** (edit values; auto-normalized):")
if "weights_df" not in st.session_state:
    st.session_state.weights_df = pd.DataFrame({"ticker": selected, "weight": [1/len(selected)]*len(selected)})
else:
    prev = st.session_state.weights_df.set_index("ticker") if "ticker" in st.session_state.weights_df else pd.DataFrame(index=[])
    rows = []
    for t in selected:
        w = prev["weight"].get(t, 1.0/len(selected)) if len(prev) else 1.0/len(selected)
        rows.append({"ticker": t, "weight": float(w)})
    st.session_state.weights_df = pd.DataFrame(rows)

weights_df = st.sidebar.data_editor(
    st.session_state.weights_df,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    key="weights_editor",
)
weights = normalize_weights(weights_df)

st.sidebar.markdown("**Portfolio basket (normalized):**")
basket_df = pd.DataFrame({"weight": weights}).reset_index().rename(columns={"index":"ticker"})
st.sidebar.dataframe(basket_df, hide_index=True, use_container_width=True)
st.sidebar.caption(" | ".join([f"{t}: {w:.1%}" for t, w in weights.items()]))

# Finalize tickers for downstream code
tickers = tuple(selected)

# --- Benchmark selection ---
st.sidebar.header("Benchmark")
DEFAULT_BENCH_UNIVERSE = ["SPY","QQQ","VOO","IVV","DIA","IWM","TLT","IEF"]
bench_sel = st.sidebar.multiselect(
    "Benchmark tickers (equal-weighted)",
    options=DEFAULT_BENCH_UNIVERSE,
    default=["SPY"],
    help="Pick one or more. We build an equal-weight benchmark basket."
)
if not bench_sel:
    st.sidebar.warning("Select at least one benchmark (e.g., SPY).")
    st.stop()
bench_tickers = tuple(bench_sel)
bench_weights = equal_weight_series(list(bench_tickers))


# Data window (user-selected)
st.sidebar.subheader("Data window")
default_start = pd.to_datetime("2005-01-01").date()
default_end = pd.Timestamp.today().date()
start_date = st.sidebar.date_input("Start date", value=default_start)
end_date   = st.sidebar.date_input("End date",   value=default_end)
if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

# General assumptions
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
# Load data & build portfolio series
# ------------------------------------------------------------
with st.spinner("Loading prices & cash series..."):
    prices = load_prices(tickers, start_date, end_date)
    bench_prices = load_prices(bench_tickers, start_date, end_date)
    cash_daily = load_cash_series()

# align cash
cash_daily = cash_daily.reindex(prices.index).ffill().fillna(0.0)

# portfolio & benchmark series (both daily-rebalanced to their weights)
portfolio_close, portfolio_rets = build_portfolio_series(prices, weights)
bench_close, bench_rets = build_portfolio_series(bench_prices, bench_weights)


# Warn on short windows
if len(prices) < 60:
    st.warning(f"Only {len(prices)} daily bars in the selected window. "
               "Risk metrics like Sharpe/Sortino may be unstable. Choose a longer period.")

# Prepare signals & backtests on the portfolio index
sig_bh = signal_buy_and_hold(portfolio_close)
sig_sma = signal_sma(portfolio_close, window=SMA_WIN)

res_bh  = run_backtest(portfolio_close, sig_bh, fee_bps=0.0,   slippage_bps=0.0,   cash_daily=0.0)
res_sma = run_backtest(portfolio_close, sig_sma, fee_bps=FEE_BPS, slippage_bps=SLIP_BPS, cash_daily=cash_daily)

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab0, tab1, tab_bench, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Assumptions & Methodology",
    "Benchmark",
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
    st.write("Use the sidebar to tweak tickers, weights, the date window, taxes, and simulation knobs. Each tab explains the reasoning in plain English.")
    st.write(f"**Universe**: {', '.join(tickers)}  |  **Weights**: " + ", ".join(f"{t}:{w:.1%}" for t,w in weights.items()))
    st.write(f"**Data window**: {start_date} → {end_date}  |  **Bars**: {len(prices)}")

# -----------------------
# Assumptions & Methodology
# -----------------------
with tab1:
    st.markdown("### Plain-English Assumptions & Methodology")
    st.markdown(f"""
**Data**
- Adjusted Close via Yahoo Finance for: {', '.join(tickers)}.
- 3-month T-bill (^IRX) as cash yield proxy (converted to daily return).
- Window selected by you: **{start_date} → {end_date}**.

**Portfolio Construction**
- We compute **daily returns per ticker** and **rebalance daily** to your weights ({', '.join(f'{t}:{w:.1%}' for t,w in weights.items())}).
- A synthetic portfolio **close index** is built from those returns (starts at 100).

**Strategies (applied on the portfolio index)**
- **Buy & Hold**: always invested in the portfolio.
- **SMA Trend**: invest in the portfolio when the index > SMA({SMA_WIN}); otherwise in cash earning T-bill yield. Costs apply on position changes.

**Backtest math**
- Daily strategy return = position × portfolio return + (1 − position) × cash daily − trading costs.
- Equity curve = cumulative product of (1 + daily return), normalized to 1 at start.

**Monte Carlo (Before Tax)**
- Resample historical **strategy daily returns** to build thousands of alternative futures.
- **IID** = days sampled independently; **Block** = sample contiguous blocks (preserve streaks).

**After-Tax (Annual Netting)**
- **Buy & Hold**: tax once at the end at long-term rate.
- **SMA+Cash**: each year, net gains/losses; tax net positive at short-term rate; carry losses forward; reduce equity at year-end.

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
- **CAGR**: the smoothed annual growth rate (what constant annual % gets you from start to finish).
- **Vol_annual**: annualized volatility; how bumpy the ride is.
- **Sharpe**: return per unit of total volatility.
- **Sortino**: return per unit of downside volatility (ignores upside bumps).
- **Max drawdown**: worst peak-to-trough loss (historical pain).
- **Last equity**: how many times your starting $1 became (e.g., 3.1 ⇒ $1 → $3.10).
        """)

    met_df = pd.DataFrame([
        summarize_backtest("Buy & Hold (Portfolio)", res_bh.equity, res_bh.strat_returns),
        summarize_backtest(f"SMA{SMA_WIN} + Cash (Portfolio)", res_sma.equity, res_sma.strat_returns),
    ])

    # Rename + compute display columns
    bt_display = pd.DataFrame({
        "Strategy": met_df["strategy"],
        "CAGR (%)": met_df["cagr"].apply(lambda x: fmt_pct(x, 2)),
        "Annual Volatility (%)": met_df["vol_annual"].apply(lambda x: fmt_pct(x, 2)),
        "Sharpe": met_df["sharpe"].apply(lambda x: fmt_num(x, 2)),
        "Sortino": met_df["sortino"].apply(lambda x: fmt_num(x, 2)),
        "Max Drawdown (%)": met_df["max_drawdown"].apply(lambda x: fmt_pct(x, 2)),
        "Ending multiple (×)": met_df["last_equity"].apply(lambda x: fmt_num(x, 2)),
        "Samples (days)": met_df["samples"].astype(int),
    })

    st.dataframe(bt_display, use_container_width=True)


    plot_equity_curves({
        "Buy&Hold (Portfolio)": res_bh.equity,
        f"SMA{SMA_WIN}+Cash (Portfolio)": res_sma.equity
    }, "Equity Curves (normalized to 1.0)")


# -----------------------
# Benchmark
# -----------------------
with tab_bench:
    st.markdown("### Benchmark Comparison (Buy&Hold vs Your Portfolio Buy&Hold)")

    with st.expander("🔎 What is this?"):
        st.markdown(f"""
- We construct a benchmark as an **equal-weight daily-rebalanced** basket of your selected tickers: {', '.join(bench_tickers)}.
- We compare **Buy&Hold (Portfolio)** vs **Buy&Hold (Benchmark)** on the same date window.
- We also compute **tracking error**, **information ratio**, and **CAPM beta/alpha** (daily model, alpha annualized).
""")

    # Buy&Hold for portfolio (already have returns via portfolio_rets)
    # Create "always invested" strategy returns = portfolio_rets
    # Equity (normalized to 1):
    eq_port = (1.0 + portfolio_rets).cumprod()
    eq_bench = (1.0 + bench_rets).cumprod()
    eq_port.index.name = eq_bench.index.name = "Date"

    # Metrics
    port_cagr = cagr_from_series(eq_port)
    bench_cagr = cagr_from_series(eq_bench)
    port_vol = annualized_vol(pd.Series(portfolio_rets, index=eq_port.index))
    bench_vol = annualized_vol(pd.Series(bench_rets, index=eq_bench.index))
    port_sharpe = sharpe_ratio(pd.Series(portfolio_rets, index=eq_port.index))
    bench_sharpe = sharpe_ratio(pd.Series(bench_rets, index=eq_bench.index))
    port_sortino = sortino_ratio(pd.Series(portfolio_rets, index=eq_port.index))
    bench_sortino = sortino_ratio(pd.Series(bench_rets, index=eq_bench.index))
    port_mdd = max_drawdown(eq_port)
    bench_mdd = max_drawdown(eq_bench)

    # Active stats
    active = pd.Series(portfolio_rets, index=eq_port.index) - pd.Series(bench_rets, index=eq_bench.index)
    active_mu_ann, active_vol_ann = annualize_mean_std(active)
    tracking_error = active_vol_ann  # by definition
    info_ratio = (active_mu_ann / tracking_error) if tracking_error > 0 else np.nan

    capm = capm_stats(pd.Series(portfolio_rets, index=eq_port.index),
                      pd.Series(bench_rets, index=eq_bench.index))

    # Table
    bench_table = pd.DataFrame([
        {"metric": "CAGR", "Portfolio": port_cagr, "Benchmark": bench_cagr},
        {"metric": "Volatility (ann.)", "Portfolio": port_vol, "Benchmark": bench_vol},
        {"metric": "Sharpe", "Portfolio": port_sharpe, "Benchmark": bench_sharpe},
        {"metric": "Sortino", "Portfolio": port_sortino, "Benchmark": bench_sortino},
        {"metric": "Max Drawdown", "Portfolio": port_mdd, "Benchmark": bench_mdd},
        {"metric": "Tracking Error (ann.)", "Portfolio": tracking_error, "Benchmark": np.nan},
        {"metric": "Information Ratio", "Portfolio": info_ratio, "Benchmark": np.nan},
        {"metric": "Beta (vs benchmark)", "Portfolio": capm["beta"], "Benchmark": 1.0},
        {"metric": "Alpha (annualized)", "Portfolio": capm["alpha_ann"], "Benchmark": 0.0},
        {"metric": "R²", "Portfolio": capm["r2"], "Benchmark": 1.0},
    ])
    bench_disp = bench_table.copy()

    def _fmt_col(metric, val):
        if metric in ["CAGR", "Volatility (ann.)", "Max Drawdown"]:
            return fmt_pct(val, 2)
        if metric in ["Sharpe", "Sortino", "Information Ratio", "Beta (vs benchmark)", "R²"]:
            return fmt_num(val, 2)
        if metric in ["Tracking Error (ann.)"]:
            return fmt_pct(val, 2)
        if metric in ["Alpha (annualized)"]:
            return fmt_pct(val, 2)  # alpha_ann is in fraction/year; show as %
        return val

    for col in ["Portfolio", "Benchmark"]:
        bench_disp[col] = [
            _fmt_col(m, v) for m, v in zip(bench_disp["metric"], bench_disp[col])
        ]

    # Rename columns for clarity
    bench_disp = bench_disp.rename(columns={
        "metric": "Metric",
        "Portfolio": "Your portfolio (BH)",
        "Benchmark": "Benchmark (BH)"
    })
    st.dataframe(bench_disp, use_container_width=True)


    # Glossary for metrics
    METRIC_EXPLAIN = {
        "CAGR": "Compound Annual Growth Rate: the steady yearly % return that gets you from start to finish.",
        "Volatility (ann.)": "How bumpy the ride is, scaled to yearly terms.",
        "Sharpe": "Return per unit of total volatility (risk-adjusted return).",
        "Sortino": "Return per unit of downside volatility (ignores upside bumps).",
        "Max Drawdown": "Worst peak-to-trough loss (historical pain point).",
        "Tracking Error (ann.)": "How much your portfolio’s returns deviate from the benchmark (annualized).",
        "Information Ratio": "Excess return vs benchmark divided by tracking error.",
        "Beta (vs benchmark)": "How sensitive your portfolio is to moves in the benchmark (1.2 = moves 20% more).",
        "Alpha (annualized)": "Average extra return not explained by beta, annualized.",
        "R²": "How well the benchmark explains your portfolio’s returns (1.0 = perfectly explained)."
    }
    # After showing st.dataframe(bench_table, ...)
    st.markdown("### Glossary")
    for metric, expl in METRIC_EXPLAIN.items():
        st.markdown(f"**{metric}** — {expl}")


    # Plots
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(eq_port.index, eq_port.values, label="Portfolio (BH)")
        ax.plot(eq_bench.index, eq_bench.values, label="Benchmark (BH)")
        ax.set_title("Equity Curves — Buy&Hold")
        ax.set_ylabel("Growth (× start)")
        ax.legend()
        ax.set_xlabel("Date")
        ax.set_ylabel("Growth (× start)")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(active.index, (1.0 + active).cumprod() - 1.0)
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_title("Cumulative Active Return (Portfolio − Benchmark)")
        ax.set_ylabel("Active return")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative active return (%)")
        _percent_axis(ax, decimals=1)  # after plotting line (which is in fraction space)
        st.pyplot(fig)

    # Optional: rolling stats
    show_roll = st.checkbox("Show rolling 252d tracking error & active return", value=False)
    if show_roll and len(active) > 252:
        roll_te = active.rolling(252).std() * np.sqrt(252)
        roll_active_ann = active.rolling(252).mean() * 252

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(roll_te.index, roll_te.values)
        ax.set_title("Rolling 252d Tracking Error (annualized)")
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(roll_active_ann.index, roll_active_ann.values)
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_title("Rolling 252d Active Return (annualized)")
        _percent_axis(ax)
        st.pyplot(fig)


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

    def mc_to_display(name, s):
        return {
            "Strategy": name,
            "Start ($)": fmt_dollar(s["start_value"]),
            "Mean end ($)": fmt_dollar(s["mean_end"]),
            "Median end ($)": fmt_dollar(s["median_end"]),
            "5th pct ($)": fmt_dollar(s["p05_end"]),
            "25th pct ($)": fmt_dollar(s["p25_end"]),
            "75th pct ($)": fmt_dollar(s["p75_end"]),
            "95th pct ($)": fmt_dollar(s["p95_end"]),
            "Prob end < start": fmt_pct(s["prob_end_below_start"], 1),
            "Min end ($)": fmt_dollar(s["min_end"]),
            "Max end ($)": fmt_dollar(s["max_end"]),
            "Paths": int(s["num_paths"]),
        }

    mc_display = pd.DataFrame([
        mc_to_display("Buy & Hold (Portfolio)", sum_bh),
        mc_to_display(f"SMA{SMA_WIN} + Cash (Portfolio)", sum_sma),
    ])

    st.dataframe(mc_display, use_container_width=True)


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

    # Synthetic biz-day calendar for horizon
    start_now = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(start=start_now, periods=TRADING_DAYS)

    rng1, rng2 = np.random.default_rng(SEED), np.random.default_rng(SEED+1)
    r_bh  = _iid_bootstrap_daily(res_bh.strat_returns,  TRADING_DAYS, N_PATHS, rng1)
    r_sma = _iid_bootstrap_daily(res_sma.strat_returns, TRADING_DAYS, N_PATHS, rng2)

    end_bh_at  = np.empty(N_PATHS)
    end_sma_at = np.empty(N_PATHS)
    for i in range(N_PATHS):
        eq_bh  = buyhold_tax_end_only(r_bh[i],  START_VALUE, st_ltcg)
        eq_sma = apply_annual_tax_to_path(r_sma[i], dates, START_VALUE, st_stcg)
        end_bh_at[i], end_sma_at[i] = eq_bh[-1], eq_sma[-1]

    sum_bh  = summarize_mc(endings_bh, START_VALUE)
    sum_sma = summarize_mc(endings_sma, START_VALUE)

    def mc_to_display(name, s):
        return {
            "Strategy": name,
            "Start ($)": fmt_dollar(s["start_value"]),
            "Mean end ($)": fmt_dollar(s["mean_end"]),
            "Median end ($)": fmt_dollar(s["median_end"]),
            "5th pct ($)": fmt_dollar(s["p05_end"]),
            "25th pct ($)": fmt_dollar(s["p25_end"]),
            "75th pct ($)": fmt_dollar(s["p75_end"]),
            "95th pct ($)": fmt_dollar(s["p95_end"]),
            "Prob end < start": fmt_pct(s["prob_end_below_start"], 1),
            "Min end ($)": fmt_dollar(s["min_end"]),
            "Max end ($)": fmt_dollar(s["max_end"]),
            "Paths": int(s["num_paths"]),
        }

    mc_display = pd.DataFrame([
        mc_to_display("Buy & Hold (Portfolio)", sum_bh),
        mc_to_display(f"SMA{SMA_WIN} + Cash (Portfolio)", sum_sma),
    ])

    st.dataframe(mc_display, use_container_width=True)


    colA, colB = st.columns(2)
    with colA:
        plot_hist(end_bh_at, START_VALUE, None, "Buy&Hold — Ending Values (After Tax)")
    with colB:
        plot_hist(end_sma_at, START_VALUE, None, f"SMA{SMA_WIN}+Cash — Ending Values (After Tax)")

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
    mask_tr = portfolio_close.index < split
    mask_te = portfolio_close.index >= split

    if mask_tr.sum() < 252 or mask_te.sum() < 252:
        st.warning("Need ≥ 1 year in both TRAIN and TEST. Adjust split date or select tickers with longer histories.")
    else:
        close_tr, close_te = portfolio_close[mask_tr], portfolio_close[mask_te]
        sig_bh_tr,  sig_bh_te  = sig_bh[mask_tr],  sig_bh[mask_te]
        sig_sma_tr, sig_sma_te = sig_sma[mask_tr], sig_sma[mask_te]
        cash_tr,    cash_te    = cash_daily[mask_tr], cash_daily[mask_te]

        res_bh_tr  = run_backtest(close_tr, sig_bh_tr, fee_bps=0.0,   slippage_bps=0.0,   cash_daily=0.0)
        res_bh_te  = run_backtest(close_te, sig_bh_te, fee_bps=0.0,   slippage_bps=0.0,   cash_daily=0.0)
        res_sma_tr = run_backtest(close_tr, sig_sma_tr, fee_bps=FEE_BPS, slippage_bps=SLIP_BPS, cash_daily=cash_tr)
        res_sma_te = run_backtest(close_te, sig_sma_te, fee_bps=FEE_BPS, slippage_bps=SLIP_BPS, cash_daily=cash_te)

        actual_bh  = compound_to_value(START_VALUE, res_bh_te.strat_returns)
        actual_sma = compound_to_value(START_VALUE, res_sma_te.strat_returns)

        # Horizon = length of TEST returns
        horizon = len(res_bh_te.strat_returns)

        if BOOT_KIND == "IID":
            mc_bh  = simulate_iid(res_bh_tr.strat_returns,  horizon, n_paths=N_PATHS, start_value=START_VALUE, seed=SEED+21)
            mc_sma = simulate_iid(res_sma_tr.strat_returns, horizon, n_paths=N_PATHS, start_value=START_VALUE, seed=SEED+22)
            endings_bh, endings_sma = mc_bh.ending_values, mc_sma.ending_values
        else:
            paths_bh  = simulate_block_paths(res_bh_tr.strat_returns,  horizon, n_paths=N_PATHS, start_value=START_VALUE, block_size=BLOCK_SIZE, seed=SEED+21)
            paths_sma = simulate_block_paths(res_sma_tr.strat_returns, horizon, n_paths=N_PATHS, start_value=START_VALUE, block_size=BLOCK_SIZE, seed=SEED+22)
            endings_bh, endings_sma = paths_bh[:, -1], paths_sma[:, -1]

        s_bh  = summarize_mc(endings_bh, START_VALUE)
        s_sma = summarize_mc(endings_sma, START_VALUE)
        pct_bh  = float(np.mean(endings_bh  <= actual_bh))
        pct_sma = float(np.mean(endings_sma <= actual_sma))

        pred_disp = pd.DataFrame([
            {"Strategy": "Buy & Hold (Portfolio)",
            "Train window": f"{close_tr.index[0].date()} → {close_tr.index[-1].date()}",
            "Test window": f"{close_te.index[0].date()} → {close_te.index[-1].date()}",
            "Actual end ($)": fmt_dollar(actual_bh),
            "MC median end ($)": fmt_dollar(s_bh["median_end"]),
            "MC 5th pct ($)": fmt_dollar(s_bh["p05_end"]),
            "MC 95th pct ($)": fmt_dollar(s_bh["p95_end"]),
            "Actual percentile in MC": fmt_pct_val(np.mean(mc_bh.ending_values <= actual_bh)*100, 1)},
            {"Strategy": f"SMA{SMA_WIN} + Cash (Portfolio)",
            "Train window": f"{close_tr.index[0].date()} → {close_tr.index[-1].date()}",
            "Test window": f"{close_te.index[0].date()} → {close_te.index[-1].date()}",
            "Actual end ($)": fmt_dollar(actual_sma),
            "MC median end ($)": fmt_dollar(s_sma["median_end"]),
            "MC 5th pct ($)": fmt_dollar(s_sma["p05_end"]),
            "MC 95th pct ($)": fmt_dollar(s_sma["p95_end"]),
            "Actual percentile in MC": fmt_pct_val(np.mean(mc_sma.ending_values <= actual_sma)*100, 1)},
        ])
        st.dataframe(pred_disp, use_container_width=True)

        colA, colB = st.columns(2)
        with colA:
            plot_hist(mc_bh.ending_values, START_VALUE, actual_bh, "Buy&Hold — Simulated Test Endings vs Actual")
        with colB:
            plot_hist(mc_sma.ending_values, START_VALUE, actual_sma, f"SMA{SMA_WIN}+Cash — Simulated Test Endings vs Actual")
