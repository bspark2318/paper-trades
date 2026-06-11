"""Shared fixtures: synthetic minute-bar sessions and a temp database."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from patterns import db as dbm

NY = "America/New_York"


def make_session_bars(date: str, n_bars: int = 390, start_price: float = 100.0, seed: int = 0) -> pd.DataFrame:
    """One synthetic regular session of minute bars beginning 09:30 New York."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(f"{date} 09:30", tz=NY)
    ts = pd.date_range(start, periods=n_bars, freq="1min").tz_convert("UTC")
    rets = rng.normal(0, 0.0005, n_bars)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]])
    high = np.maximum(open_, close) * 1.0001
    low = np.minimum(open_, close) * 0.9999
    return pd.DataFrame(
        {"ts": ts, "open": open_, "high": high, "low": low, "close": close,
         "volume": rng.integers(1000, 5000, n_bars).astype(float)}
    )


def make_multi_session_bars(dates: list[str], n_bars: int | dict = 390, start_price: float = 100.0,
                            seed: int = 0) -> pd.DataFrame:
    frames = []
    price = start_price
    for i, date in enumerate(dates):
        n = n_bars[date] if isinstance(n_bars, dict) else n_bars
        df = make_session_bars(date, n_bars=n, start_price=price, seed=seed + i)
        price = float(df["close"].iloc[-1])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def conn(tmp_path):
    connection = dbm.connect(tmp_path / "test.db")
    yield connection
    connection.close()
