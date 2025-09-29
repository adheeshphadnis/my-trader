from __future__ import annotations
import pandas as pd

def signal_sma(prices: pd.Series, window: int = 200) -> pd.Series:
    """
    1 when price > SMA(window); else 0. Uses prior data (no lookahead) via shift(1).
    """
    sma = prices.rolling(window).mean()
    raw = (prices > sma).astype(float)
    return raw.shift(1).fillna(0.0)
