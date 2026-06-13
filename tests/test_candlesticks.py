import dataclasses

import numpy as np
import pandas as pd
import pytest

from patterns.config import Config
from patterns.strategy.base import Direction, make_source
from patterns.strategy.candlesticks import CANDLES, Candle, load_candles

NY = "America/New_York"


def C(o, h, l, c) -> Candle:
    return Candle(float(o), float(h), float(l), float(c))


def fires(name: str, *candles: Candle) -> bool:
    return CANDLES[name].detect(list(candles))


# ---------- anatomy detectors (no context) ----------

def test_hammer():
    assert fires("hammer", C(10.00, 10.06, 9.00, 10.05))      # long lower wick, tiny upper
    assert not fires("hammer", C(10.0, 10.5, 9.5, 10.4))      # big body, no wick


def test_shooting_star():
    assert fires("shooting_star", C(10.00, 11.00, 9.94, 9.95))  # long upper wick
    assert not fires("shooting_star", C(10.0, 10.06, 9.0, 10.05))


def test_bullish_engulfing():
    prev, cur = C(10.0, 10.05, 9.5, 9.6), C(9.55, 10.2, 9.5, 10.1)
    assert fires("bullish_engulfing", prev, cur)
    assert not fires("bullish_engulfing", cur, prev)           # order matters


def test_bearish_engulfing():
    prev, cur = C(9.6, 10.1, 9.55, 10.0), C(10.1, 10.15, 9.4, 9.5)
    assert fires("bearish_engulfing", prev, cur)


def test_piercing_line():
    prev, cur = C(10.0, 10.05, 9.4, 9.5), C(9.45, 9.9, 9.4, 9.85)
    assert fires("piercing_line", prev, cur)


def test_dark_cloud_cover():
    prev, cur = C(9.0, 9.6, 8.9, 9.5), C(9.55, 9.6, 9.1, 9.20)
    assert fires("dark_cloud_cover", prev, cur)


def test_morning_star():
    a, b, d = C(10.0, 10.1, 9.0, 9.1), C(8.95, 9.0, 8.9, 8.96), C(8.97, 9.8, 8.95, 9.6)
    assert fires("morning_star", a, b, d)


def test_evening_star():
    a, b, d = C(9.0, 10.1, 8.9, 10.0), C(10.05, 10.1, 10.0, 10.06), C(10.0, 10.05, 9.2, 9.4)
    assert fires("evening_star", a, b, d)


def test_three_white_soldiers():
    a, b, d = C(10.0, 10.6, 9.9, 10.5), C(10.3, 11.1, 10.2, 11.0), C(10.8, 11.6, 10.7, 11.5)
    assert fires("three_white_soldiers", a, b, d)


def test_three_black_crows():
    a, b, d = C(11.5, 11.6, 10.9, 11.0), C(11.2, 11.3, 10.4, 10.5), C(10.7, 10.8, 9.9, 10.0)
    assert fires("three_black_crows", a, b, d)


# ---------- source + context gate ----------

def candle_cfg(**over) -> Config:
    base = Config(signal_source="candles", candle_patterns=tuple(CANDLES),
                  candle_trend_lookback=10, min_history_bars=0, cost_bps=0.0, horizon=3)
    return dataclasses.replace(base, **over)


def session_df(rows: list[tuple], date: str = "2024-03-04") -> pd.DataFrame:
    ts = pd.date_range(f"{date} 09:30", periods=len(rows), freq="1min", tz=NY).tz_convert("UTC")
    o, h, l, c = (list(col) for col in zip(*rows))
    return pd.DataFrame({"ts": ts, "open": o, "high": h, "low": l, "close": c,
                         "volume": [1000.0] * len(rows)})


def trend_then_hammer(direction: str) -> pd.DataFrame:
    """12 filler bars trending `direction`, then a hammer bar."""
    step = -0.8 if direction == "down" else 0.8
    rows = []
    price = 100.0
    for _ in range(12):
        o = price
        price += step
        c = price
        rows.append((o, max(o, c) + 0.05, min(o, c) - 0.05, c))
    base = price
    rows.append((base, base + 0.06, base - 1.0, base + 0.05))   # hammer
    return session_df(rows)


def test_hammer_fires_after_downtrend():
    src = make_source(candle_cfg())
    bars = trend_then_hammer("down")
    src.prepare(bars)
    sig = src.signal_at(pd.Timestamp(bars["ts"].iloc[-1]))
    assert sig.direction is Direction.LONG
    assert sig.diagnostics["pattern"] == "hammer"


def test_hammer_suppressed_in_uptrend_by_context():
    src = make_source(candle_cfg())
    bars = trend_then_hammer("up")         # same hammer, wrong context
    src.prepare(bars)
    sig = src.signal_at(pd.Timestamp(bars["ts"].iloc[-1]))
    assert sig.direction is Direction.NO_TRADE
    assert sig.diagnostics["reason"] == "no_pattern"


def test_lookback_zero_is_pure_anatomy():
    src = make_source(candle_cfg(candle_trend_lookback=0))
    bars = trend_then_hammer("up")         # uptrend, but context disabled
    src.prepare(bars)
    sig = src.signal_at(pd.Timestamp(bars["ts"].iloc[-1]))
    assert sig.direction is Direction.LONG


def test_conflicting_candles_abstain():
    # hammer (LONG) and dark_cloud_cover (SHORT) both fire on this 2-bar pair;
    # with context off they survive the gate and must cancel to NO_TRADE.
    bars = session_df([(9.0, 9.6, 8.9, 9.5), (9.55, 9.6, 8.0, 9.20)])
    src = make_source(candle_cfg(candle_patterns=("hammer", "dark_cloud_cover"),
                                 candle_trend_lookback=0, enable_shorts=True))
    src.prepare(bars)
    sig = src.signal_at(pd.Timestamp(bars["ts"].iloc[-1]))
    assert sig.direction is Direction.NO_TRADE
    assert sig.diagnostics["reason"] == "conflict"
    assert set(sig.diagnostics["matched"]) == {"hammer", "dark_cloud_cover"}


def test_bearish_candle_needs_shorts_enabled():
    rows = []
    price = 100.0
    for _ in range(12):                    # uptrend into a shooting star
        o = price
        price += 0.8
        rows.append((o, max(o, price) + 0.05, min(o, price) - 0.05, price))
    base = price
    rows.append((base, base + 1.0, base - 0.06, base - 0.05))   # shooting star
    bars = session_df(rows)
    asof = pd.Timestamp(bars["ts"].iloc[-1])

    flat = make_source(candle_cfg(enable_shorts=False))
    flat.prepare(bars)
    assert flat.signal_at(asof).direction is Direction.NO_TRADE
    assert flat.signal_at(asof).diagnostics["reason"] == "shorts_disabled"

    shorting = make_source(candle_cfg(enable_shorts=True))
    shorting.prepare(bars)
    assert shorting.signal_at(asof).direction is Direction.SHORT


def test_multibar_pattern_does_not_cross_session():
    # a 3-bar morning star split across two sessions must not fire
    day1 = session_df([(10.0, 10.1, 9.0, 9.1)], "2024-03-04")
    day2 = session_df([(8.95, 9.0, 8.9, 8.96), (8.97, 9.8, 8.95, 9.6)], "2024-03-05")
    bars = pd.concat([day1, day2], ignore_index=True)
    src = make_source(candle_cfg(candle_patterns=("morning_star",), candle_trend_lookback=0))
    src.prepare(bars)
    sig = src.signal_at(pd.Timestamp(bars["ts"].iloc[-1]))
    assert sig.direction is Direction.NO_TRADE          # star's first bar is in the prior session


# ---------- guards + identity ----------

def test_unknown_candle_rejected():
    with pytest.raises(ValueError, match="Unknown candle patterns"):
        load_candles(("not_a_candle",))


def test_empty_candle_list_rejected():
    with pytest.raises(ValueError, match="at least one pattern"):
        load_candles(())


def test_trend_lookback_is_an_identity_field():
    assert candle_cfg(candle_trend_lookback=5).config_hash != candle_cfg(candle_trend_lookback=10).config_hash


def test_template_knobs_do_not_affect_candle_hash():
    a = candle_cfg(template_threshold=3.5, k=50)
    b = candle_cfg(template_threshold=9.9, k=999)
    assert a.config_hash == b.config_hash
