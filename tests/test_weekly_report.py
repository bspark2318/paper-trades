import pandas as pd

from patterns import db as dbm
from patterns.config import Config
from patterns.live.weekly_report import build_weekly, render_markdown, write_report

NY = "America/New_York"


def cfg(**over) -> Config:
    import dataclasses
    return dataclasses.replace(Config(signal_source="candles"), **over)


def memconn():
    return dbm.connect(":memory:")


def add_filled_order(conn, intent, side, qty, price, filled_at, symbol="QQQ"):
    conn.execute(
        """INSERT INTO orders (symbol, side, qty, intent, broker_order_id, status,
                               filled_qty, filled_avg_price, submitted_at, filled_at)
           VALUES (?, ?, ?, ?, ?, 'filled', ?, ?, ?, ?)""",
        (symbol, side, qty, intent, f"o{filled_at}", qty, price,
         filled_at.isoformat(), filled_at.isoformat()),
    )
    conn.commit()


def add_snapshot(conn, ts, equity):
    conn.execute("INSERT INTO position_snapshots (ts, account_equity) VALUES (?, ?)",
                 (ts.isoformat(), equity))
    conn.commit()


def add_bars(conn, symbol, ts_list, closes):
    now = dbm.utcnow()
    conn.executemany(
        "INSERT OR REPLACE INTO bars (symbol, ts, open, high, low, close, volume, source, ingested_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, 'alpaca', ?)",
        [(symbol, t.isoformat(), c, c, c, c, 1000.0, now) for t, c in zip(ts_list, closes)],
    )
    conn.commit()


def add_backtest_metrics(conn, c, mean_net):
    dbm.register_config(conn, c.config_hash, c.identity_json())
    rid = dbm.start_run(conn, "backtest", c.config_hash)
    dbm.finish_run(conn, rid, status="ok", metrics={"mean_net_ret": mean_net, "n_trades": 5})


def test_empty_week_renders_clean():
    conn = memconn()
    c = cfg()
    now = pd.Timestamp("2026-06-12 20:00", tz="UTC")
    rep = build_weekly(conn, c, now)
    assert rep.n_trades == 0
    md = render_markdown(conn, c, rep)
    assert "No closed trades this week" in md


def test_paired_trades_and_metrics():
    conn = memconn()
    c = cfg()
    now = pd.Timestamp("2026-06-12 20:00", tz="UTC")
    t0 = now - pd.Timedelta(days=2)
    # one winning round-trip: buy 50 @ 100, sell 50 @ 101
    add_filled_order(conn, "entry", "buy", 50, 100.0, t0)
    add_filled_order(conn, "time_stop", "sell", 50, 101.0, t0 + pd.Timedelta(minutes=15))
    # one losing round-trip: buy 50 @ 101, sell 50 @ 100.5
    add_filled_order(conn, "entry", "buy", 50, 101.0, t0 + pd.Timedelta(hours=1))
    add_filled_order(conn, "force_flat", "sell", 50, 100.5, t0 + pd.Timedelta(hours=1, minutes=20))

    rep = build_weekly(conn, c, now)
    assert rep.n_trades == 2
    assert rep.hit_rate == 0.5
    # pnl: 50*(101-100) + 50*(100.5-101) = 50 - 25 = 25
    assert abs(rep.total_pnl - 25.0) < 1e-9


def test_old_trades_excluded_from_window():
    conn = memconn()
    c = cfg()
    now = pd.Timestamp("2026-06-12 20:00", tz="UTC")
    old = now - pd.Timedelta(days=30)
    add_filled_order(conn, "entry", "buy", 10, 100.0, old)
    add_filled_order(conn, "time_stop", "sell", 10, 105.0, old + pd.Timedelta(minutes=15))
    rep = build_weekly(conn, c, now)
    assert rep.n_trades == 0


def test_divergence_against_backtest():
    conn = memconn()
    c = cfg()
    now = pd.Timestamp("2026-06-12 20:00", tz="UTC")
    t0 = now - pd.Timedelta(days=1)
    add_filled_order(conn, "entry", "buy", 50, 100.0, t0)
    add_filled_order(conn, "time_stop", "sell", 50, 100.5, t0 + pd.Timedelta(minutes=15))  # +0.5%
    add_backtest_metrics(conn, c, mean_net=0.01)   # backtest expected +1%

    rep = build_weekly(conn, c, now)
    assert rep.backtest_mean == 0.01
    md = render_markdown(conn, c, rep)
    assert "Live vs backtest" in md
    assert "running **worse**" in md            # live +0.5% < backtest +1%


def test_write_report_emits_files(tmp_path):
    conn = memconn()
    c = cfg(reports_dir=str(tmp_path))
    now = pd.Timestamp("2026-06-12 20:00", tz="UTC")
    t0 = now - pd.Timedelta(days=1)
    add_filled_order(conn, "entry", "buy", 50, 100.0, t0)
    add_filled_order(conn, "time_stop", "sell", 50, 101.0, t0 + pd.Timedelta(minutes=15))
    # snapshots + bars so the equity/benchmark PNG is produced
    snap_ts = pd.date_range(t0, now, freq="1h", tz="UTC")
    add_snapshot(conn, snap_ts[0], 100_000.0)
    for i, ts in enumerate(snap_ts[1:], 1):
        add_snapshot(conn, ts, 100_000.0 + i * 5)
    add_bars(conn, "QQQ", list(snap_ts), [100.0 + i * 0.01 for i in range(len(snap_ts))])

    rep = write_report(conn, c, now)
    from pathlib import Path
    assert rep.md_path is not None and Path(rep.md_path).exists()
    assert rep.png_path is not None and Path(rep.png_path).exists()
    assert "Weekly paper-trading report" in Path(rep.md_path).read_text()
