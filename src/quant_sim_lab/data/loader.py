from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf


def fetch_spy_history(
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    dst_csv: str | Path = "data/raw/spy.csv",
) -> Path:
    """
    Download SPY OHLCV with dividends/splits adjusted Close via yfinance and save to CSV.

    Parameters
    ----------
    start : str | None
        ISO date "YYYY-MM-DD". If None, defaults to 20 years ago.
    end : str | None
        ISO date "YYYY-MM-DD". If None, defaults to today.
    interval : str
        yfinance interval (e.g., "1d", "1wk", "1mo").
    dst_csv : str | Path
        Output CSV path.

    Returns
    -------
    Path
        Path to the written CSV.
    """
    if end is None:
        end = datetime.now().date().isoformat()
    if start is None:
        start = (datetime.now().date() - timedelta(days=365 * 20)).isoformat()

    # Download SPY
    df = yf.download("SPY", start=start, end=end, interval=interval, auto_adjust=True, progress=False)

    if df.empty:
        raise RuntimeError("Downloaded SPY dataframe is empty. Check internet/connectivity or ticker.")

    # Normalize columns and write
    df = df.rename(columns=str.title)  # Open, High, Low, Close, Adj Close -> Close already auto_adjusted
    dst_path = Path(dst_csv)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst_path, index=True)

    return dst_path


def load_spy_csv(src_csv: str | Path = "data/raw/spy.csv") -> pd.DataFrame:
    """
    Load the SPY CSV written by fetch_spy_history with strict parsing and numeric coercion.
    """
    src = Path(src_csv)
    if not src.exists():
        raise FileNotFoundError(f"{src} not found. Run fetch_spy_history() first.")

    # Parse first column (index) as datetime; don't let pandas guess everything
    df = pd.read_csv(src, parse_dates=[0], index_col=0)

    # Ensure expected columns exist
    expected = {"Open", "High", "Low", "Close", "Volume"}
    missing = expected - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in CSV: {missing}. CSV columns = {list(df.columns)}")

    # Force numeric; if anything is non-numeric, coerce to NaN so we can drop it safely
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with NaNs in Close (should be rare)
    df = df.dropna(subset=["Close"])

    return df

