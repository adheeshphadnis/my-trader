from __future__ import annotations
import pandas as pd

def signal_buy_and_hold(prices: pd.Series) -> pd.Series:
    """
    Always invested = 1 (after first bar to avoid lookahead).
    """
    sig = pd.Series(1.0, index=prices.index)
    # Shift by 1 to simulate entering after first observable bar (conservative)
    return sig.shift(1).fillna(0.0)
