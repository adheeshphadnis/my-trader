from __future__ import annotations
import numpy as np
import pandas as pd

def after_tax_sma_daily(returns: pd.Series, st_rate: float) -> pd.Series:
    """
    Apply short-term tax to SMA (and cash) daily returns:
    - Tax only positive daily returns: r_after = r * (1 - st_rate) if r > 0 else r
    - Losses are not taxed here (offsets ignored in this simplified model).
    """
    r = returns.astype(float).copy()
    pos = r > 0
    r.loc[pos] = r.loc[pos] * (1.0 - st_rate)
    return r

def after_tax_buyhold_endings(ending_values: np.ndarray, start_value: float, lt_rate: float) -> np.ndarray:
    """
    Apply long-term capital gains tax to Buy&Hold at the end only.
    Ending_after_tax = start + (ending - start) * (1 - lt_rate)
    """
    end = np.asarray(ending_values, dtype=float)
    gains = np.maximum(end - start_value, 0.0)
    end_after = start_value + gains * (1.0 - lt_rate)
    # If end < start (a loss), tax does not apply in this simplified model.
    end_after[end < start_value] = end[end < start_value]
    return end_after
