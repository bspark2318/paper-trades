import pandas as pd

from patterns.data import store
from tests.conftest import make_multi_session_bars, make_session_bars


def test_upsert_idempotent(conn):
    bars = make_session_bars("2024-03-04")
    store.upsert_bars(conn, "QQQ", bars)
    store.upsert_bars(conn, "QQQ", bars)  # re-run: no duplicates
    assert len(store.load_bars(conn, "QQQ")) == 390
    # but provenance recorded both ingestions
    assert conn.execute("SELECT COUNT(*) FROM ingestions").fetchone()[0] == 2


def test_regular_hours_filter():
    bars = make_session_bars("2024-03-04")
    premarket = bars.copy()
    premarket["ts"] = premarket["ts"] - pd.Timedelta(hours=3)  # 06:30 NY start
    combined = pd.concat([premarket, bars], ignore_index=True)
    rth = store.filter_regular_hours(combined)
    local = rth["ts"].dt.tz_convert(store.NY)
    assert (local.dt.hour * 60 + local.dt.minute).min() >= 570
    assert (local.dt.hour * 60 + local.dt.minute).max() < 960


def test_session_derivation_handles_half_days(conn):
    bars = make_multi_session_bars(["2024-07-02", "2024-07-03", "2024-07-05"],
                                   n_bars={"2024-07-02": 390, "2024-07-03": 210, "2024-07-05": 390})
    store.upsert_bars(conn, "QQQ", bars)
    store.rebuild_sessions(conn, "QQQ")
    sessions = store.load_sessions(conn, "QQQ")
    assert list(sessions["n_bars"]) == [390, 210, 390]
    half = sessions[sessions["date"] == "2024-07-03"].iloc[0]
    assert half["last_ts"].tz_convert(store.NY).strftime("%H:%M") == "12:59"


def test_load_bars_end_ts_bound(conn):
    bars = make_multi_session_bars(["2024-03-04", "2024-03-05"])
    store.upsert_bars(conn, "QQQ", bars)
    cutoff = bars["ts"].iloc[389].isoformat()  # end of first session
    assert len(store.load_bars(conn, "QQQ", end_ts=cutoff)) == 390


def test_coverage(conn):
    bars = make_multi_session_bars(["2024-03-04", "2024-03-05"], n_bars={"2024-03-04": 390, "2024-03-05": 200})
    store.upsert_bars(conn, "QQQ", bars)
    store.rebuild_sessions(conn, "QQQ")
    cov = store.coverage(conn, "QQQ")
    assert cov["bars"] == 590
    assert cov["sessions"] == 2
    assert cov["short_or_gappy_sessions"] == 1
