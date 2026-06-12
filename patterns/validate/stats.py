"""Performance statistics. Small, hand-checkable, no dependencies on the rest."""

from __future__ import annotations

import numpy as np
from jaxtyping import Float


def sharpe(returns: Float[np.ndarray, " N"], periods_per_year: float) -> float:
    """Annualized Sharpe from a per-period return series. NaN if undefined."""
    if len(returns) < 2:
        return float("nan")
    sd = float(np.std(returns, ddof=1))
    if sd == 0.0:
        return float("nan")
    return float(np.mean(returns)) / sd * float(np.sqrt(periods_per_year))


def max_drawdown(equity: Float[np.ndarray, " N"]) -> float:
    """Largest peak-to-trough decline, as a positive fraction of the peak."""
    if len(equity) == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    drawdowns = (peaks - equity) / peaks
    return float(np.max(drawdowns))


def hit_rate(returns: Float[np.ndarray, " N"]) -> float:
    if len(returns) == 0:
        return float("nan")
    return float(np.mean(returns > 0))
