import dataclasses

import numpy as np
import pandas as pd
import pytest

from patterns.config import SOURCE_IDENTITY_FIELDS, Config
from patterns.strategy import SOURCES, make_source
from patterns.strategy.base import LONG, NO_TRADE
from tests.conftest import make_multi_session_bars
from tests.test_engine import plant_motif

W, H = 5, 3
MOTIF = np.array([0.004, -0.003, 0.005, -0.002, 0.006])


def small_cfg(**over) -> Config:
    base = Config(window=W, horizon=H, k=5, dedup_gap=2, min_matches=3,
                  min_history_bars=0, p_threshold=0.65, t_multiplier=1.5)
    return dataclasses.replace(base, **over)


def force_rise(bars: pd.DataFrame, after: int, horizon: int, pct: float = 0.01) -> pd.DataFrame:
    """Make the H bars after global index `after` a steady climb."""
    closes = bars["close"].to_numpy().copy()
    step = (1 + pct) ** (1 / horizon)
    for j in range(1, horizon + 1):
        closes[after + j] = closes[after + j - 1] * step
    out = bars.copy()
    out["close"] = closes
    return out


def bullish_history():
    """Plant MOTIF in several past sessions, each followed by a strong rise;
    plant it again (unresolved) at the end as the query."""
    dates = [f"2024-03-{d:02d}" for d in (4, 5, 6, 7, 8)]
    bars = make_multi_session_bars(dates, n_bars=40)
    for at in (20, 60, 100, 140):          # sessions 1-4
        bars = plant_motif(bars, at=at, motif=MOTIF)
        bars = force_rise(bars, after=at, horizon=H)
    query_at = 180                          # session 5
    bars = plant_motif(bars, at=query_at, motif=MOTIF)
    return bars, pd.Timestamp(bars["ts"].iloc[query_at])


def test_registry_and_factory():
    assert "knn_shape" in SOURCES
    src = make_source(small_cfg())
    assert src.name == "knn_shape"
    with pytest.raises(KeyError):
        make_source(dataclasses.replace(Config(), signal_source="nope"))


def test_identity_declaration_matches_config():
    for name, cls in SOURCES.items():
        assert tuple(cls.identity_fields) == SOURCE_IDENTITY_FIELDS[name]


def test_bullish_pattern_emits_long():
    bars, asof = bullish_history()
    src = make_source(small_cfg())
    src.prepare(bars)
    sig = src.signal_at(asof)
    assert sig.direction == LONG
    d = sig.diagnostics
    assert d["n"] >= 3 and d["pct_positive"] >= 0.65 and d["mean"] > 0


def test_no_trade_when_too_few_matches():
    bars, asof = bullish_history()
    src = make_source(small_cfg(min_matches=10_000))
    src.prepare(bars)
    sig = src.signal_at(asof)
    assert sig.direction == NO_TRADE
    assert sig.diagnostics["reason"] == "too_few_matches"


def test_no_trade_on_noise():
    bars = make_multi_session_bars([f"2024-03-{d:02d}" for d in (4, 5, 6)], n_bars=40)
    src = make_source(small_cfg(p_threshold=0.95))   # noise won't hit 95% positive
    src.prepare(bars)
    sig = src.signal_at(pd.Timestamp(bars["ts"].iloc[-1]))
    assert sig.direction == NO_TRADE


def test_signal_uses_only_past_evidence():
    """Diagnostics' candidate pool at an early query must be small (only prior bars)."""
    bars, _ = bullish_history()
    src = make_source(small_cfg(min_matches=1))
    src.prepare(bars)
    early = src.signal_at(pd.Timestamp(bars["ts"].iloc[50]))
    late = src.signal_at(pd.Timestamp(bars["ts"].iloc[190]))
    assert late.diagnostics["n_candidates"] > early.diagnostics["n_candidates"]
