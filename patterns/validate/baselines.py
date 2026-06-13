"""Mandatory baselines: every strategy number is judged against these.

Random baseline, time-of-day matched: intraday volatility has strong
open/lunch/close structure, so random entries are drawn to match the
strategy's entry minute-of-day distribution exactly — same trade count,
same minutes, same horizon, same costs, just no pattern knowledge.
If the strategy can't beat THIS, it discovered the clock, not a pattern.

p-value is add-one: p = (1 + #{resamples >= strategy}) / (n + 1) — never
zero, never overconfident.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from patterns.validate import stats

NY = "America/New_York"
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class RandomBaselineResult:
    p_value: float
    strategy_mean: float
    random_means: np.ndarray    # (n_resamples,) mean net return of each random run
    n_resamples: int
    n_trades: int


@dataclass(frozen=True)
class BuyHoldResult:
    total_return: float
    sharpe: float
    max_drawdown: float


def _minute_of_day(ts: pd.Series) -> np.ndarray:
    local = ts.dt.tz_convert(NY)
    out: np.ndarray = (local.dt.hour * 60 + local.dt.minute).to_numpy()
    return out


def random_baseline(
    bars: pd.DataFrame,
    decision_ts: list[pd.Timestamp],
    strategy_mean: float,
    horizon: int,
    cost_bps: float,
    min_history_bars: int,
    n_resamples: int = 1000,
    seed: int = 42,
) -> RandomBaselineResult:
    """Distribution of mean net return for random TOD-matched entries."""
    if not decision_ts:
        raise ValueError("random_baseline needs at least one strategy decision")
    rng = np.random.default_rng(seed)
    opens = bars["open"].to_numpy(dtype=np.float64)
    minute = _minute_of_day(bars["ts"])

    session = pd.Series(bars["ts"].dt.tz_convert(NY).dt.date)
    last_pos = session.groupby(session).transform(lambda s: s.index[-1]).to_numpy()
    bars_left = last_pos - np.arange(len(bars))

    # eligible decision bars, bucketed by minute-of-day
    eligible = (np.arange(len(bars)) >= min_history_bars) & (bars_left >= horizon + 1)
    buckets: dict[int, np.ndarray] = {
        m: np.flatnonzero(eligible & (minute == m)) for m in np.unique(minute)
    }

    ts_index = pd.DatetimeIndex(bars["ts"])
    decision_idx = ts_index.get_indexer(pd.DatetimeIndex(decision_ts))
    if (decision_idx < 0).any():
        raise ValueError("decision timestamp not found in bars")
    decision_minutes = minute[decision_idx]

    cost = 1e-4 * cost_bps
    means = np.empty(n_resamples)
    for r in range(n_resamples):
        rets = np.empty(len(decision_minutes))
        for j, m in enumerate(decision_minutes):
            i = int(rng.choice(buckets[int(m)]))
            entry = opens[i + 1] * (1 + cost)
            exit_ = opens[i + 1 + horizon] * (1 - cost)
            rets[j] = exit_ / entry - 1.0
        means[r] = rets.mean()

    p = (1 + int(np.sum(means >= strategy_mean))) / (n_resamples + 1)
    return RandomBaselineResult(
        p_value=p, strategy_mean=strategy_mean, random_means=means,
        n_resamples=n_resamples, n_trades=len(decision_minutes),
    )


def buy_and_hold(bars: pd.DataFrame, cost_bps: float = 0.0) -> BuyHoldResult:
    """Hold the symbol over the whole period; one round trip of costs."""
    closes = bars["close"].to_numpy(dtype=np.float64)
    cost = 1e-4 * cost_bps
    entry = bars["open"].iloc[0] * (1 + cost)
    exit_ = closes[-1] * (1 - cost)

    daily = bars.groupby(bars["ts"].dt.tz_convert(NY).dt.date)["close"].last().to_numpy()
    daily_rets = np.diff(daily) / daily[:-1]

    return BuyHoldResult(
        total_return=float(exit_ / entry - 1.0),
        sharpe=stats.sharpe(daily_rets, TRADING_DAYS_PER_YEAR),
        max_drawdown=stats.max_drawdown(closes),
    )
