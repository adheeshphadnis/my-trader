from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252

def _year_boundaries(index: pd.DatetimeIndex) -> list[tuple[int, int, int]]:
    """
    Return list of (year, start_idx, end_idx_inclusive) for contiguous slices in a DatetimeIndex.
    Assumes index is sorted.
    """
    years = index.year
    bounds = []
    start = 0
    cur = years[0]
    for i in range(1, len(index)):
        if years[i] != cur:
            bounds.append((int(cur), start, i - 1))
            start = i
            cur = years[i]
    bounds.append((int(cur), start, len(index) - 1))
    return bounds


def apply_annual_tax_to_path(
    daily_returns: np.ndarray,
    dates: pd.DatetimeIndex,
    start_value: float,
    st_rate: float,
    *,
    treat_all_as_short_term: bool = True,
) -> np.ndarray:
    """
    Apply a realistic after-tax scheme to a SINGLE path:
      - Compute the path's pre-tax equity via compounding.
      - Once per calendar year, compute that year's $ gain/loss (vs equity at year start).
      - If annual net > 0: pay tax = st_rate * net (short-term assumption for SMA-like strategies).
      - If annual net <= 0: no tax; accumulate $loss carryforward to offset future positive years.
      - Loss carryforward offsets future positive years BEFORE computing tax.
      - Taxes are paid by reducing equity at year-end (drag on compounding).

    Notes:
      - For Buy&Hold, keep pre-tax path and tax only once at the end (use the dedicated BH function).
      - Here we assume all gains are short-term (conservative for SMA). You can relax later.

    Returns the AFTER-TAX equity path as a numpy array shape (n_days+1,)
    (value at day 0 included).
    """
    n = len(daily_returns)
    eq = np.empty(n + 1, dtype=float)
    eq[0] = start_value

    # Build pre-tax path first
    for t in range(n):
        eq[t + 1] = eq[t] * (1.0 + daily_returns[t])

    # Annual tax with loss carryforward
    carry_loss = 0.0
    eq_at = eq.copy()

    bounds = _year_boundaries(dates)
    # We levy tax at the end of each year segment (end_idx)
    for _, start_i, end_i in bounds:
        eq_start = eq_at[start_i]          # equity at year's start (after prior tax if any)
        eq_end   = eq_at[end_i + 1]        # equity at year's end (pre-tax for this year)

        # This year's pre-tax change in dollars
        delta = eq_end - eq_start  # can be positive or negative

        # Apply carryforward offset only to positive years
        taxable_base = delta
        if taxable_base > 0 and carry_loss < 0:
            # Offset positive delta with prior losses
            offset = min(taxable_base, -carry_loss)
            taxable_base -= offset
            carry_loss += offset  # reduces the negative carry toward zero

        if taxable_base > 0:
            # tax as short-term (ordinary rate approximation)
            tax = taxable_base * st_rate
            eq_at[end_i + 1] -= tax
        else:
            # accumulate loss carryforward (increase negative value)
            carry_loss += taxable_base  # adding negative delta makes carry_loss more negative

        # Equity remains adjusted in subsequent periods automatically

    return eq_at


def buyhold_tax_end_only(
    daily_returns: np.ndarray,
    start_value: float,
    lt_rate: float,
) -> np.ndarray:
    """
    Build pre-tax path for Buy&Hold and tax ONLY the final net gain at long-term rate.
    Returns AFTER-TAX equity path (n_days+1).
    """
    n = len(daily_returns)
    eq = np.empty(n + 1, dtype=float)
    eq[0] = start_value
    for t in range(n):
        eq[t + 1] = eq[t] * (1.0 + daily_returns[t])

    gain = eq[-1] - start_value
    if gain > 0:
        tax = gain * lt_rate
        eq[-1] -= tax
    # if loss, no tax in this simple model
    return eq
