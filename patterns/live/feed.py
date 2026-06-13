"""DbBarFeed: the live loop's view of closed bars, backed by the SQLite cache.

Each call refreshes the symbol from Alpaca (incremental, idempotent) then returns
the full stored history up to the last bar that is *fully closed* at asof. A
1-minute bar timestamped at its start T closes at T+1m; it is eligible only once
asof has passed T+1m, so the forming minute is never handed to a signal source.
"""

from __future__ import annotations

import sqlite3

import pandas as pd

from patterns.data import store


class DbBarFeed:
    def __init__(self, conn: sqlite3.Connection, symbol: str, *, refresh: bool = True):
        self.conn = conn
        self.symbol = symbol
        self.refresh = refresh

    def closed_bars(self, asof: pd.Timestamp) -> pd.DataFrame:
        if self.refresh:
            store.refresh(self.conn, self.symbol)
        bars = store.load_bars(self.conn, self.symbol)
        if bars.empty:
            return bars
        cutoff = pd.Timestamp(asof).tz_convert("UTC").floor("min") - pd.Timedelta(minutes=1)
        closed: pd.DataFrame = bars[bars["ts"] <= cutoff].reset_index(drop=True)
        return closed
