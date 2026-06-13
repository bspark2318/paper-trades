"""Candlestick anatomy: the OTHER folklore family.

Where templates match the price PATH over many bars (a double bottom is a double
bottom whatever the bars look like), candlestick patterns are defined entirely by
the OHLC ANATOMY of one to three bars — wick-to-body ratios, body colour, gaps.
A close-only representation cannot express "long lower wick, small body", so these
live in their own detectors that read raw OHLC directly.

Each detector is a hand-coded folklore rule returning whether the pattern is
present in the last n_bars candles (oldest -> newest). The anatomy thresholds
below are folklore-standard constants, deliberately NOT config knobs: tuning them
would multiply ledger entries for what is really one hypothesis. Whether any of
these actually predicts a forward move is, as always, the referee's call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from patterns.strategy.base import Direction

# Folklore-standard anatomy thresholds (fractions of a bar's high-low range / body).
LONG_WICK_RATIO = 2.0      # a "long" shadow is >= 2x the real body
SMALL_BODY_FRAC = 0.3      # a "small" body is <= 30% of the bar's range (the star)
STRONG_BODY_FRAC = 0.5     # a "strong" body is >= 50% of the bar's range


@dataclass(frozen=True)
class Candle:
    open: float
    high: float
    low: float
    close: float

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def rng(self) -> float:
        return self.high - self.low

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def bullish(self) -> bool:
        return self.close > self.open

    @property
    def bearish(self) -> bool:
        return self.close < self.open

    @property
    def mid(self) -> float:
        return (self.open + self.close) / 2.0


@dataclass(frozen=True)
class CandlePattern:
    name: str
    direction: Direction
    n_bars: int                                   # bars the rule inspects (1..3)
    detect: Callable[[list[Candle]], bool]        # receives exactly n_bars, oldest -> newest


# ---- single-bar ----

def _hammer(cs: list[Candle]) -> bool:
    """Small body near the top, long lower shadow — bullish reversal."""
    c = cs[-1]
    return (c.body > 0 and c.lower_wick >= LONG_WICK_RATIO * c.body
            and c.upper_wick <= c.body)


def _shooting_star(cs: list[Candle]) -> bool:
    """Small body near the bottom, long upper shadow — bearish reversal."""
    c = cs[-1]
    return (c.body > 0 and c.upper_wick >= LONG_WICK_RATIO * c.body
            and c.lower_wick <= c.body)


# ---- two-bar ----

def _bullish_engulfing(cs: list[Candle]) -> bool:
    p, c = cs[-2], cs[-1]
    return (p.bearish and c.bullish
            and c.close >= p.open and c.open <= p.close
            and c.body > p.body)


def _bearish_engulfing(cs: list[Candle]) -> bool:
    p, c = cs[-2], cs[-1]
    return (p.bullish and c.bearish
            and c.open >= p.close and c.close <= p.open
            and c.body > p.body)


def _piercing_line(cs: list[Candle]) -> bool:
    """Opens below the prior bearish bar, closes back above its midpoint (not a full engulf)."""
    p, c = cs[-2], cs[-1]
    return (p.bearish and c.bullish
            and c.open < p.close and c.close > p.mid and c.close < p.open)


def _dark_cloud_cover(cs: list[Candle]) -> bool:
    p, c = cs[-2], cs[-1]
    return (p.bullish and c.bearish
            and c.open > p.close and c.close < p.mid and c.close > p.open)


# ---- three-bar ----

def _morning_star(cs: list[Candle]) -> bool:
    """Strong down bar, a small-bodied star gapping below it, then a strong up bar
    closing into the first body — bullish reversal."""
    a, b, d = cs
    if not (a.bearish and a.rng > 0 and b.rng > 0):
        return False
    return (a.body >= STRONG_BODY_FRAC * a.rng
            and b.body <= SMALL_BODY_FRAC * b.rng
            and max(b.open, b.close) < a.close
            and d.bullish and d.close > a.mid)


def _evening_star(cs: list[Candle]) -> bool:
    a, b, d = cs
    if not (a.bullish and a.rng > 0 and b.rng > 0):
        return False
    return (a.body >= STRONG_BODY_FRAC * a.rng
            and b.body <= SMALL_BODY_FRAC * b.rng
            and min(b.open, b.close) > a.close
            and d.bearish and d.close < a.mid)


def _three_white_soldiers(cs: list[Candle]) -> bool:
    a, b, d = cs
    return (a.bullish and b.bullish and d.bullish
            and b.close > a.close and d.close > b.close
            and a.close > b.open > a.open and b.close > d.open > b.open)


def _three_black_crows(cs: list[Candle]) -> bool:
    a, b, d = cs
    return (a.bearish and b.bearish and d.bearish
            and b.close < a.close and d.close < b.close
            and a.close < b.open < a.open and b.close < d.open < b.open)


_CANDLES: tuple[CandlePattern, ...] = (
    CandlePattern("hammer", Direction.LONG, 1, _hammer),
    CandlePattern("shooting_star", Direction.SHORT, 1, _shooting_star),
    CandlePattern("bullish_engulfing", Direction.LONG, 2, _bullish_engulfing),
    CandlePattern("bearish_engulfing", Direction.SHORT, 2, _bearish_engulfing),
    CandlePattern("piercing_line", Direction.LONG, 2, _piercing_line),
    CandlePattern("dark_cloud_cover", Direction.SHORT, 2, _dark_cloud_cover),
    CandlePattern("morning_star", Direction.LONG, 3, _morning_star),
    CandlePattern("evening_star", Direction.SHORT, 3, _evening_star),
    CandlePattern("three_white_soldiers", Direction.LONG, 3, _three_white_soldiers),
    CandlePattern("three_black_crows", Direction.SHORT, 3, _three_black_crows),
)

CANDLES: dict[str, CandlePattern] = {p.name: p for p in _CANDLES}


def load_candles(names: tuple[str, ...]) -> list[CandlePattern]:
    if not names:
        raise ValueError("candles source needs at least one pattern in candle_patterns")
    unknown = [n for n in names if n not in CANDLES]
    if unknown:
        raise ValueError(f"Unknown candle patterns {unknown}; available: {sorted(CANDLES)}")
    return [CANDLES[n] for n in names]
