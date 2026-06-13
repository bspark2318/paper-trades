import sqlite3

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from patterns import db as dbm
from patterns.cli import app
from patterns.data import store
from patterns.validate.evaluate import Verdict, decide, spend_evaluation
from tests.test_strategy import MOTIF, force_rise
from tests.conftest import make_multi_session_bars

runner = CliRunner()

GOOD_TRAIN = {"n_trades": 20, "mean_net_ret": 0.004, "hit_rate": 0.7, "sharpe": 2.0}
GOOD_TEST = {"n_trades": 8, "mean_net_ret": 0.003, "hit_rate": 0.65, "sharpe": 1.5}


# ---------- verdict logic ----------

def test_survives_when_all_criteria_pass():
    v = decide(GOOD_TRAIN, GOOD_TEST, p_random=0.001, a=0.05)
    assert v.survived and v.reasons == []


@pytest.mark.parametrize("test_override,p,a,expected_fragment", [
    ({"n_trades": 0}, 0.001, 0.05, "no trades"),
    ({"mean_net_ret": -0.001}, 0.001, 0.05, "not positive"),
    ({}, 0.04, 0.005, "does not beat random"),            # fails CORRECTED alpha
    ({"mean_net_ret": 0.0005}, 0.001, 0.05, "inconsistency"),  # < 0.25 x train
    ({"sharpe": -0.5}, 0.001, 0.05, "sign flip"),
])
def test_each_criterion_rejects(test_override, p, a, expected_fragment):
    v = decide(GOOD_TRAIN, {**GOOD_TEST, **test_override}, p_random=p, a=a)
    assert not v.survived
    assert any(expected_fragment in r for r in v.reasons), v.reasons


def test_rejection_lists_every_failure():
    v = decide(GOOD_TRAIN, {**GOOD_TEST, "mean_net_ret": -0.01, "sharpe": -1.0},
               p_random=0.9, a=0.005)
    assert len(v.reasons) >= 3


# ---------- gate mechanics ----------

def test_crash_spends_evaluation(conn):
    dbm.register_config(conn, "abc", "{}")
    run_id = dbm.start_run(conn, "evaluate", "abc")
    spend_evaluation(conn, "abc", run_id)
    # simulate crash: no record_verdict ever happens
    assert dbm.n_test_evaluations(conn) == 1
    row = conn.execute("SELECT verdict FROM test_evaluations").fetchone()
    assert row["verdict"] == "CRASHED"
    assert "TEST SET EVALUATED 1 TIME" in dbm.report_banner(conn)


# ---------- CLI end-to-end ----------

def split_bars(train_sessions: int, test_sessions: int, plant_test: bool) -> pd.DataFrame:
    """Planted bullish pattern in train; optionally in test too."""
    from tests.test_engine import plant_motif

    n_bars = 40
    total = train_sessions + test_sessions
    dates = [f"2024-03-{d:02d}" for d in range(4, 4 + total)]
    bars = make_multi_session_bars(dates, n_bars=n_bars)
    for s in range(total):
        if s < train_sessions or plant_test:
            # varying minute: edge must come from the SHAPE, not the clock,
            # or the TOD-matched random baseline rightly absorbs it
            at = s * n_bars + 8 + (5 * s) % 24
            bars = plant_motif(bars, at=at, motif=MOTIF)
            bars = force_rise(bars, after=at, horizon=3)
    from tests.conftest import rebuild_ohlc_from_closes

    return rebuild_ohlc_from_closes(bars)


def setup_db(tmp_path, plant_test: bool):
    bars = split_bars(train_sessions=6, test_sessions=3, plant_test=plant_test)
    db_path = tmp_path / "ev.db"
    conn = dbm.connect(db_path)
    store.upsert_bars(conn, "QQQ", bars)
    store.rebuild_sessions(conn, "QQQ")
    conn.close()
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"db_path: {db_path}\nreports_dir: {tmp_path / 'reports'}\n"
        "window: 5\nhorizon: 3\nk: 5\ndedup_gap: 2\nmin_matches: 3\n"
        "min_history_bars: 50\np_threshold: 0.85\ncost_bps: 0.0\n"  # 0.85: 5/5 needed, cuts noise LONGs
        "split_date: '2024-03-09'\n"      # 6 train sessions (03-04..03-09), 3 test
    )
    return cfg, db_path


def get_hash(cfg) -> str:
    out = runner.invoke(app, ["config", "--config", str(cfg)])
    return out.output.split("config_hash:")[1].strip()


def test_evaluate_cli_survives_on_persistent_pattern(tmp_path):
    cfg, db_path = setup_db(tmp_path, plant_test=True)
    assert runner.invoke(app, ["backtest", "--config", str(cfg), "--resamples", "50"]).exit_code == 0
    h = get_hash(cfg)
    result = runner.invoke(app, ["evaluate", h, "--yes", "--resamples", "50", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "VERDICT: SURVIVED" in result.output
    assert "TEST SET EVALUATED 1 TIME" in result.output

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT verdict FROM test_evaluations").fetchone()
    assert row["verdict"] == "SURVIVED"


def test_evaluate_cli_rejects_when_pattern_dies(tmp_path):
    cfg, db_path = setup_db(tmp_path, plant_test=False)
    assert runner.invoke(app, ["backtest", "--config", str(cfg), "--resamples", "50"]).exit_code == 0
    h = get_hash(cfg)
    result = runner.invoke(app, ["evaluate", h, "--yes", "--resamples", "50", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "VERDICT: REJECTED" in result.output
    assert "failed criteria:" in result.output


def test_evaluate_refuses_on_wrong_retype(tmp_path):
    cfg, db_path = setup_db(tmp_path, plant_test=True)
    assert runner.invoke(app, ["backtest", "--config", str(cfg), "--resamples", "20"]).exit_code == 0
    h = get_hash(cfg)
    result = runner.invoke(app, ["evaluate", h, "--config", str(cfg)], input="wrong-hash\n")
    assert result.exit_code == 1
    assert "aborted, nothing spent" in result.output
    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM test_evaluations").fetchone()[0] == 0


def test_evaluate_unknown_hash_refused(tmp_path):
    cfg, _ = setup_db(tmp_path, plant_test=True)
    result = runner.invoke(app, ["evaluate", "deadbeef0000", "--yes", "--config", str(cfg)])
    assert result.exit_code == 1
    assert "unknown config hash" in result.output


def test_second_evaluation_increments_counter(tmp_path):
    cfg, db_path = setup_db(tmp_path, plant_test=True)
    assert runner.invoke(app, ["backtest", "--config", str(cfg), "--resamples", "20"]).exit_code == 0
    h = get_hash(cfg)
    runner.invoke(app, ["evaluate", h, "--yes", "--resamples", "20", "--config", str(cfg)])
    out = runner.invoke(app, ["evaluate", h, "--yes", "--resamples", "20", "--config", str(cfg)])
    assert "TEST SET EVALUATED 2 TIMES" in out.output    # shame is visible
