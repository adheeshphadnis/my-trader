from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class BacktestResult:
    equity: pd.Series           # equity curve starting at 1.0
    strat_returns: pd.Series    # daily strategy returns
    signal: pd.Series           # 0/1 position

def run_backtest(
    close: pd.Series,
    signal: pd.Series,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> BacktestResult:
    """
    Backtest a 0/1 long-only strategy on a single price series with simple daily compounding.
    - close: price series (adjusted close recommended)
    - signal: 0/1 position (already shifted to avoid lookahead)
    - fee_bps/slippage_bps: charged on position *changes* (turnover) as one-time hit.

    Returns equity starting at 1.0.
    """
    df = pd.concat({"close": close, "sig": signal}, axis=1).dropna()
    # Daily simple returns
    df["ret"] = df["close"].pct_change().fillna(0.0)

    # Position is today's signal (already shifted in signal function)
    pos = df["sig"]

    # Strategy return before costs
    strat_r = pos * df["ret"]

    # Apply trading frictions when position changes: cost on notional change
    pos_change = pos.diff().abs().fillna(pos.abs())  # first day enter if pos>0
    cost_rate = (fee_bps + slippage_bps) / 1e4
    costs = pos_change * cost_rate

    strat_r_after = strat_r - costs

    equity = (1 + strat_r_after).cumprod()
    return BacktestResult(equity=equity, strat_returns=strat_r_after, signal=pos)
