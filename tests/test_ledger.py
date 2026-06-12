import sqlite3
from pathlib import Path

from typer.testing import CliRunner

from patterns import db as dbm
from patterns.cli import app
from patterns.data import store
from patterns.validate.ledger import alpha, ledger_rows
from tests.test_walkforward import bullish_bars

runner = CliRunner()


def test_n_counts_distinct_hashes_including_crashed(conn):
    assert dbm.n_configs_tried(conn) == 0
    for h in ("aaa", "bbb", "ccc"):
        dbm.register_config(conn, h, "{}")
    dbm.start_run(conn, "backtest", "aaa")              # crashed: never finished
    assert dbm.n_configs_tried(conn) == 1
    rid = dbm.start_run(conn, "backtest", "aaa")        # same hypothesis again
    dbm.finish_run(conn, rid, "ok")
    assert dbm.n_configs_tried(conn) == 1
    dbm.start_run(conn, "backtest", "bbb")              # new hypothesis
    assert dbm.n_configs_tried(conn) == 2
    dbm.start_run(conn, "match", "ccc")                 # non-backtest runs don't count
    assert dbm.n_configs_tried(conn) == 2


def test_alpha_shrinks_with_n():
    assert alpha(0) == 0.05
    assert alpha(1) == 0.05
    assert alpha(10) == 0.005


def write_test_setup(tmp_path) -> tuple[Path, Path]:
    bars = bullish_bars(n_sessions=6, n_bars=40)
    db_path = tmp_path / "bt.db"
    conn = dbm.connect(db_path)
    store.upsert_bars(conn, "QQQ", bars)
    store.rebuild_sessions(conn, "QQQ")
    conn.close()
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"db_path: {db_path}\nreports_dir: {tmp_path / 'reports'}\n"
        "window: 5\nhorizon: 3\nk: 5\ndedup_gap: 2\nmin_matches: 3\n"
        "min_history_bars: 50\np_threshold: 0.65\ncost_bps: 0.0\n"
        "split_date: '2024-12-31'\n"
    )
    return cfg, db_path


def test_backtest_cli_end_to_end(tmp_path):
    cfg, db_path = write_test_setup(tmp_path)
    result = runner.invoke(app, ["backtest", "--config", str(cfg), "--resamples", "50"])
    assert result.exit_code == 0, result.output
    assert "trades:" in result.output
    assert "vs random" in result.output
    assert "ledger: N = 1" in result.output
    assert "upper bounds" in result.output
    assert list(Path(tmp_path / "reports").glob("equity_*.png"))

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    runs = conn.execute("SELECT * FROM runs WHERE run_type='backtest'").fetchall()
    assert len(runs) == 1 and runs[0]["status"] == "ok"
    n_trades = conn.execute("SELECT COUNT(*) FROM backtest_trades").fetchone()[0]
    assert n_trades >= 1


def test_ledger_counts_variants_not_reruns(tmp_path):
    cfg, db_path = write_test_setup(tmp_path)
    assert runner.invoke(app, ["backtest", "--config", str(cfg), "--resamples", "20"]).exit_code == 0
    assert runner.invoke(app, ["backtest", "--config", str(cfg), "--resamples", "20"]).exit_code == 0
    out = runner.invoke(app, ["ledger", "--config", str(cfg)])
    assert "configs tried: 1" in out.output            # rerun = same hypothesis

    assert runner.invoke(
        app, ["backtest", "--config", str(cfg), "--resamples", "20", "--set", "k=4"]
    ).exit_code == 0
    out = runner.invoke(app, ["ledger", "--config", str(cfg)])
    assert "configs tried: 2" in out.output            # variant = new hypothesis

    conn = dbm.connect(db_path)
    rows = ledger_rows(conn)
    assert len(rows) == 2
    assert {r.n_runs for r in rows} == {1, 2}
