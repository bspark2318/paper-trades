from pathlib import Path

from typer.testing import CliRunner

from patterns.cli import app
from patterns.data import store
from tests.conftest import make_multi_session_bars

runner = CliRunner()


def test_match_cli_end_to_end(tmp_path, conn, monkeypatch):
    dates = [f"2024-03-{d:02d}" for d in (4, 5, 6, 7, 8)]
    bars = make_multi_session_bars(dates, n_bars=60)
    store.upsert_bars(conn, "QQQ", bars)
    store.rebuild_sessions(conn, "QQQ")
    db_path = tmp_path / "cli.db"
    # conn fixture lives in its own tmp db; write a config pointing there
    import sqlite3

    dst = sqlite3.connect(db_path)
    conn.backup(dst)
    dst.close()

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"db_path: {db_path}\nreports_dir: {tmp_path / 'reports'}\n"
        "window: 5\nhorizon: 3\nk: 5\ndedup_gap: 2\nmin_history_bars: 0\n"
    )
    asof = bars["ts"].iloc[-1].isoformat()
    result = runner.invoke(app, ["match", "--asof", asof, "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "fwd return: mean" in result.output
    assert "TEST SET EVALUATED 0 TIMES" in result.output
    pngs = list(Path(tmp_path / "reports").glob("match_QQQ_*.png"))
    assert len(pngs) == 2


def test_match_cli_no_data(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"db_path: {tmp_path / 'empty.db'}\n")
    result = runner.invoke(app, ["match", "--asof", "2024-03-04 12:00", "--config", str(cfg)])
    assert result.exit_code == 1
    assert "no bars stored" in result.output
