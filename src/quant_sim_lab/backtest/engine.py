from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class BacktestResult:
    equity: pd.Series           # equity curve starting at 1.0
    strat_returns: pd.Series    # daily strategy returns (after costs)
    signal: pd.Series           # 0/1 position

def run_backtest(
    close: pd.Series,
    signal: pd.Series,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    cash_daily: pd.Series | float | None = None,
) -> BacktestResult:
    """
    Backtest a 0/1 long-only strategy on a single price series.

    - close: adjusted close prices
    - signal: 0/1 position (already shifted to avoid lookahead)
    - fee_bps/slippage_bps: charged on position *changes* (turnover) as one-time hit.
    - cash_daily: optional daily simple return earned when out of market (signal==0).
        * If Series, it will be aligned to 'close' index (forward-filled).
        * If float, treated as a constant daily return.
        * If None, treated as 0.0 (no yield).
    """
    df = pd.concat({"close": close, "sig": signal}, axis=1).dropna()
    df["ret"] = df["close"].pct_change().fillna(0.0)

    # Cash daily return handling
    if cash_daily is None:
        cd = pd.Series(0.0, index=df.index)
    elif isinstance(cash_daily, (int, float)):
        cd = pd.Series(float(cash_daily), index=df.index)
    else:
        cd = pd.Series(cash_daily).reindex(df.index).fillna(method="ffill").fillna(0.0)

    pos = df["sig"].clip(0, 1)

    # Strategy return before costs: invested gets asset return, otherwise cash yield
    strat_r = pos * df["ret"] + (1.0 - pos) * cd

    # Costs on position changes
    pos_change = pos.diff().abs().fillna(pos.abs())
    cost_rate = (fee_bps + slippage_bps) / 1e4
    costs = pos_change * cost_rate

    strat_r_after = strat_r - costs
    equity = (1 + strat_r_after).cumprod()

    return BacktestResult(equity=equity, strat_returns=strat_r_after, signal=pos)
