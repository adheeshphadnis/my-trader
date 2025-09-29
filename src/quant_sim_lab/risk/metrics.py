from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252

def to_series(x) -> pd.Series:
    return x if isinstance(x, pd.Series) else pd.Series(x)

def cagr_from_series(equity: pd.Series) -> float:
    equity = to_series(equity).dropna()
    if len(equity) < 2:
        return np.nan
    total_return = equity.iloc[-1] / equity.iloc[0]
    years = len(equity) / TRADING_DAYS
    if years <= 0 or total_return <= 0:
        return np.nan
    return total_return ** (1 / years) - 1

def annualized_vol(returns: pd.Series) -> float:
    returns = to_series(returns).dropna()
    return returns.std() * np.sqrt(TRADING_DAYS)

def downside_deviation(returns: pd.Series, mar: float = 0.0) -> float:
    r = to_series(returns).dropna()
    downside = r[r < mar]
    if len(downside) == 0:
        return 0.0
    return downside.std() * np.sqrt(TRADING_DAYS)

def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    r = to_series(returns).dropna()
    # Convert annual rf to daily approx
    daily_rf = (1 + rf) ** (1 / TRADING_DAYS) - 1
    excess = r - daily_rf
    denom = r.std()
    return np.nan if denom == 0 else (excess.mean() * TRADING_DAYS) / (denom * np.sqrt(TRADING_DAYS))

def sortino_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    r = to_series(returns).dropna()
    daily_rf = (1 + rf) ** (1 / TRADING_DAYS) - 1
    excess = r - daily_rf
    dd = downside_deviation(r, mar=daily_rf)
    return np.nan if dd == 0 else (excess.mean() * TRADING_DAYS) / dd

def drawdown_series(equity: pd.Series) -> pd.Series:
    eq = to_series(equity).dropna()
    peaks = eq.cummax()
    return (eq - peaks) / peaks

def max_drawdown(equity: pd.Series) -> float:
    dd = drawdown_series(equity)
    return float(dd.min()) if len(dd) else np.nan
