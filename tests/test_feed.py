"""The live no-lookahead guard: DbBarFeed must never hand a signal source the
currently-forming minute. A 1-minute bar stamped at start T closes at T+1m, so it
is eligible only once asof has passed T+1m."""

import pandas as pd

from patterns import db as dbm
from patterns.live.feed import DbBarFeed

UTC = "UTC"


def seed_bars(conn, symbol, start="2024-03-04 14:30", n=4):
    ts = pd.date_range(start, periods=n, freq="1min", tz=UTC)
    now = dbm.utcnow()
    conn.executemany(
        "INSERT OR REPLACE INTO bars (symbol, ts, open, high, low, close, volume, source, ingested_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, 'alpaca', ?)",
        [(symbol, t.isoformat(), 100.0, 100.5, 99.5, 100.0, 1000.0, now) for t in ts],
    )
    conn.commit()
    return ts


def feed(conn, symbol="QQQ"):
    return DbBarFeed(conn, symbol, refresh=False)   # refresh=False → no network


def test_forming_minute_is_excluded():
    conn = dbm.connect(":memory:")
    ts = seed_bars(conn, "QQQ", n=4)               # bars at :30 :31 :32 :33
    f = feed(conn)
    # mid-way through the :32 bar (14:32:20): :32 is still forming, last closed is :31
    out = f.closed_bars(pd.Timestamp("2024-03-04 14:32:20", tz=UTC))
    assert list(out["ts"]) == [ts[0], ts[1]]       # :30 and :31 only


def test_bar_becomes_visible_one_minute_after_its_start():
    conn = dbm.connect(":memory:")
    ts = seed_bars(conn, "QQQ", n=4)
    f = feed(conn)
    T = ts[1]                                        # the 14:31 bar
    # not yet closed at 14:31:30 (still forming)
    before = f.closed_bars(T + pd.Timedelta(seconds=30))
    assert T not in list(before["ts"])
    # closed exactly at 14:32:00
    at_close = f.closed_bars(T + pd.Timedelta(minutes=1))
    assert T in list(at_close["ts"])


def test_all_bars_visible_after_session():
    conn = dbm.connect(":memory:")
    ts = seed_bars(conn, "QQQ", n=4)
    f = feed(conn)
    out = f.closed_bars(pd.Timestamp("2024-03-04 18:00", tz=UTC))
    assert list(out["ts"]) == list(ts)


def test_empty_store_returns_empty():
    conn = dbm.connect(":memory:")
    out = feed(conn).closed_bars(pd.Timestamp("2024-03-04 18:00", tz=UTC))
    assert out.empty
