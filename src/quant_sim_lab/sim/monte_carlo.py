from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass
class MCResult:
    """Holds Monte Carlo results for one strategy."""
    ending_values: np.ndarray        # shape: (n_paths,)
    ending_cagrs: np.ndarray         # shape: (n_paths,)
    # Optional: could add full equity paths later for fan charts


def _iid_bootstrap_daily(returns: pd.Series, n_days: int, n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """
    Simple i.i.d. bootstrap of daily returns.
    returns: pd.Series of daily simple returns (e.g., 0.0012 for +0.12%)
    Returns an array shape (n_paths, n_days).
    """
    r = returns.dropna().to_numpy()
    if r.size == 0:
        raise ValueError("Empty returns series provided to bootstrap.")
    idx = rng.integers(0, r.size, size=(n_paths, n_days))
    return r[idx]


def simulate_iid(
    daily_strategy_returns: pd.Series,
    horizon_days: int,
    n_paths: int = 10000,
    start_value: float = 10_000.0,
    seed: int | None = 42,
) -> MCResult:
    """
    Monte Carlo via i.i.d. bootstrap of historical daily strategy returns.

    - daily_strategy_returns: series of daily simple returns from backtest (already includes trading rules & costs)
    - horizon_days: typically ~3 years * 252 = 756
    - n_paths: number of scenarios to simulate
    - start_value: starting capital (e.g., 10_000)
    - seed: RNG seed for reproducibility

    Returns MCResult with per-path ending values and implied ending CAGRs.
    """
    rng = np.random.default_rng(seed)
    paths = _iid_bootstrap_daily(daily_strategy_returns, horizon_days, n_paths, rng)  # (n_paths, n_days)

    # compound each path: start_value * prod(1 + r_t)
    compounded = start_value * np.prod(1.0 + paths, axis=1)  # (n_paths,)

    years = horizon_days / TRADING_DAYS_PER_YEAR
    with np.errstate(invalid="ignore"):
        cagrs = (compounded / start_value) ** (1.0 / years) - 1.0

    return MCResult(ending_values=compounded, ending_cagrs=cagrs)


def summarize_mc(ending_values: np.ndarray, start_value: float) -> dict:
    """
    Produce a compact summary dict of the ending value distribution.
    """
    ev = np.asarray(ending_values, dtype=float)
    summary = {
        "start_value": start_value,
        "mean_end": float(np.mean(ev)),
        "median_end": float(np.median(ev)),
        "p05_end": float(np.percentile(ev, 5)),
        "p25_end": float(np.percentile(ev, 25)),
        "p75_end": float(np.percentile(ev, 75)),
        "p95_end": float(np.percentile(ev, 95)),
        "prob_end_below_start": float(np.mean(ev < start_value)),
        "min_end": float(np.min(ev)),
        "max_end": float(np.max(ev)),
        "num_paths": int(ev.size),
    }
    return summary

# --- NEW CODE BELOW (append to existing file) ---

def _compound_paths(paths: np.ndarray, start_value: float) -> np.ndarray:
    """
    Given returns array shape (n_paths, n_days), return equity paths shape (n_paths, n_days+1),
    including the initial capital at t=0.
    """
    n_paths, n_days = paths.shape
    eq = np.empty((n_paths, n_days + 1), dtype=float)
    eq[:, 0] = start_value
    # cumulative product over days
    eq[:, 1:] = start_value * np.cumprod(1.0 + paths, axis=1)
    return eq


def simulate_iid_paths(
    daily_strategy_returns: pd.Series,
    horizon_days: int,
    n_paths: int = 10000,
    start_value: float = 10_000.0,
    seed: int | None = 42,
) -> np.ndarray:
    """
    IID bootstrap returning full equity paths (n_paths, horizon_days+1).
    """
    rng = np.random.default_rng(seed)
    returns = _iid_bootstrap_daily(daily_strategy_returns, horizon_days, n_paths, rng)
    return _compound_paths(returns, start_value)


def _block_bootstrap_daily(
    returns: pd.Series,
    n_days: int,
    n_paths: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simple non-overlapping circular block bootstrap of daily returns.
    - We pick block starting indices uniformly at random and stitch blocks.
    - 'Circular' means we wrap around at the end of the series.
    """
    r = returns.dropna().to_numpy()
    T = r.size
    if T == 0:
        raise ValueError("Empty returns series provided to block bootstrap.")
    if block_size <= 0:
        raise ValueError("block_size must be >= 1")

    n_blocks = int(np.ceil(n_days / block_size))
    out = np.empty((n_paths, n_blocks * block_size), dtype=float)

    # For each path, draw n_blocks starting positions
    starts = rng.integers(0, T, size=(n_paths, n_blocks))
    for p in range(n_paths):
        pos = 0
        for b in range(n_blocks):
            s = starts[p, b]
            # slice with wrap-around
            end = s + block_size
            if end <= T:
                out[p, pos:pos+block_size] = r[s:end]
            else:
                k = end - T
                out[p, pos:pos+block_size] = np.concatenate([r[s:], r[:k]])
            pos += block_size

    # truncate to exactly n_days
    return out[:, :n_days]


def simulate_block_paths(
    daily_strategy_returns: pd.Series,
    horizon_days: int,
    n_paths: int = 10000,
    start_value: float = 10_000.0,
    block_size: int = 20,
    seed: int | None = 123,
) -> np.ndarray:
    """
    Block bootstrap returning full equity paths (n_paths, horizon_days+1).
    'block_size' ~ number of trading days per block (e.g., 10-20).
    """
    rng = np.random.default_rng(seed)
    returns = _block_bootstrap_daily(daily_strategy_returns, horizon_days, n_paths, block_size, rng)
    return _compound_paths(returns, start_value)


def percentile_cone(
    equity_paths: np.ndarray,
    percentiles: list[int] = [5, 25, 50, 75, 95],
) -> pd.DataFrame:
    """
    equity_paths: (n_paths, n_days+1)
    Returns a DataFrame with index = day 0..n_days, columns like 'p05','p25','p50','p75','p95'.
    """
    qs = np.percentile(equity_paths, q=percentiles, axis=0)  # shape = (len(percentiles), n_days+1)
    cols = [f"p{str(p).zfill(2)}" for p in percentiles]
    df = pd.DataFrame(qs.T, columns=cols)
    df.index.name = "day"
    return df
