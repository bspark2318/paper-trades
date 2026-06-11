"""Minute-bar acquisition from Alpaca's historical data API.

Keys come from the environment only (ALPACA_API_KEY / ALPACA_SECRET_KEY).
Fetching is paginated by alpaca-py internally; we request regular-hours bars
and convert to a plain DataFrame so nothing downstream depends on alpaca types.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd

EARLIEST = datetime(2016, 1, 1, tzinfo=timezone.utc)


class MissingKeysError(RuntimeError):
    pass


def _client():
    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise MissingKeysError(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY not set. "
            "Create a free paper account at https://app.alpaca.markets and export both."
        )
    from alpaca.data.historical import StockHistoricalDataClient

    return StockHistoricalDataClient(key, secret)


def fetch_minute_bars(symbol: str, start: datetime | None, end: datetime | None = None) -> pd.DataFrame:
    """Fetch 1-minute bars [start, end] UTC. Returns columns ts/open/high/low/close/volume."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start or EARLIEST,
        end=end,
        adjustment="all",
    )
    bars = _client().get_stock_bars(request)
    df = bars.df
    if df.empty:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.reset_index()
    df = df.rename(columns={"timestamp": "ts"})
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df[["ts", "open", "high", "low", "close", "volume"]].sort_values("ts").reset_index(drop=True)
