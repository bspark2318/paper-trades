"""Bar storage, session derivation, and incremental refresh.

Sessions are derived from the bars themselves (first/last regular-hours bar per
New-York date), which handles half-days without an external calendar. Only
regular-hours bars (09:30–16:00 ET) are stored: pre/post-market prints are a
different liquidity regime and would pollute shape matching.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pandas as pd

from patterns import db as dbm
from patterns.data import alpaca_data

NY = "America/New_York"


def filter_regular_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Keep bars whose start time is within 09:30 <= t < 16:00 New York time."""
    if df.empty:
        return df
    local = df["ts"].dt.tz_convert(NY)
    minutes = local.dt.hour * 60 + local.dt.minute
    rth: pd.DataFrame = df[(minutes >= 9 * 60 + 30) & (minutes < 16 * 60)]
    return rth.reset_index(drop=True)


def upsert_bars(conn: sqlite3.Connection, symbol: str, df: pd.DataFrame, source: str = "alpaca") -> int:
    if df.empty:
        return 0
    now = dbm.utcnow()
    rows = [
        (symbol, ts.isoformat(), o, h, l, c, v, source, now)
        for ts, o, h, l, c, v in df[["ts", "open", "high", "low", "close", "volume"]].itertuples(index=False)
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO bars (symbol, ts, open, high, low, close, volume, source, ingested_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.execute(
        "INSERT INTO ingestions (symbol, source, start_ts, end_ts, n_rows, fetched_at) VALUES (?, ?, ?, ?, ?, ?)",
        (symbol, source, df["ts"].min().isoformat(), df["ts"].max().isoformat(), len(df), now),
    )
    conn.commit()
    return len(rows)


def rebuild_sessions(conn: sqlite3.Connection, symbol: str) -> int:
    """Recompute the sessions table for a symbol from stored bars."""
    df = load_bars(conn, symbol)
    conn.execute("DELETE FROM sessions WHERE symbol = ?", (symbol,))
    if df.empty:
        conn.commit()
        return 0
    local_date = df["ts"].dt.tz_convert(NY).dt.date.astype(str)
    grouped = df.groupby(local_date)["ts"]
    rows = [
        (symbol, date, g.min().isoformat(), g.max().isoformat(), len(g))
        for date, g in grouped
    ]
    conn.executemany(
        "INSERT INTO sessions (symbol, date, first_ts, last_ts, n_bars) VALUES (?, ?, ?, ?, ?)", rows
    )
    conn.commit()
    return len(rows)


def load_bars(conn: sqlite3.Connection, symbol: str, end_ts: str | None = None) -> pd.DataFrame:
    """All stored bars for a symbol, time-ordered. end_ts (ISO) bounds inclusively."""
    query = "SELECT ts, open, high, low, close, volume FROM bars WHERE symbol = ?"
    params: list = [symbol]
    if end_ts is not None:
        query += " AND ts <= ?"
        params.append(end_ts)
    query += " ORDER BY ts"
    df = pd.read_sql_query(query, conn, params=params)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def load_sessions(conn: sqlite3.Connection, symbol: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT date, first_ts, last_ts, n_bars FROM sessions WHERE symbol = ? ORDER BY date",
        conn,
        params=[symbol],
    )
    for col in ("first_ts", "last_ts"):
        df[col] = pd.to_datetime(df[col], utc=True)
    return df


def last_bar_ts(conn: sqlite3.Connection, symbol: str) -> datetime | None:
    row = conn.execute("SELECT MAX(ts) FROM bars WHERE symbol = ?", (symbol,)).fetchone()
    return datetime.fromisoformat(row[0]) if row and row[0] else None


def refresh(conn: sqlite3.Connection, symbol: str) -> dict:
    """Idempotent incremental sync: fetch from last stored bar (minus a 1-day
    overlap re-fetched harmlessly via INSERT OR REPLACE) to now."""
    last = last_bar_ts(conn, symbol)
    start = (last - timedelta(days=1)) if last else None
    raw = alpaca_data.fetch_minute_bars(symbol, start=start)
    rth = filter_regular_hours(raw)
    n = upsert_bars(conn, symbol, rth)
    n_sessions = rebuild_sessions(conn, symbol)
    return {"symbol": symbol, "fetched": len(raw), "stored_rth": n, "sessions": n_sessions}


def coverage(conn: sqlite3.Connection, symbol: str) -> dict:
    sessions = load_sessions(conn, symbol)
    n_bars = conn.execute("SELECT COUNT(*) FROM bars WHERE symbol = ?", (symbol,)).fetchone()[0]
    if sessions.empty:
        return {"symbol": symbol, "bars": 0, "sessions": 0}
    short_days = int((sessions["n_bars"] < 380).sum())
    return {
        "symbol": symbol,
        "bars": n_bars,
        "sessions": len(sessions),
        "first_date": sessions["date"].iloc[0],
        "last_date": sessions["date"].iloc[-1],
        "short_or_gappy_sessions": short_days,
    }
