"""SQLite state: bars, sessions, configs, runs, matches, trades, signals, orders.

Two anti-self-deception mechanisms live at the schema level:
- configs/runs: every backtested config hash is recorded forever (Bonferroni N).
- test_evaluations: one row per test-set evaluation, inserted and COMMITTED
  before any computation happens — a crashed or aborted evaluation still counts.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    symbol      TEXT NOT NULL,
    ts          TEXT NOT NULL,          -- bar start, ISO-8601 UTC
    open        REAL NOT NULL,
    high        REAL NOT NULL,
    low         REAL NOT NULL,
    close       REAL NOT NULL,
    volume      REAL NOT NULL,
    source      TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    PRIMARY KEY (symbol, ts, source)
);

CREATE TABLE IF NOT EXISTS ingestions (
    id         INTEGER PRIMARY KEY,
    symbol     TEXT NOT NULL,
    source     TEXT NOT NULL,
    start_ts   TEXT,
    end_ts     TEXT,
    n_rows     INTEGER NOT NULL,
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    symbol   TEXT NOT NULL,
    date     TEXT NOT NULL,             -- session date, America/New_York
    first_ts TEXT NOT NULL,
    last_ts  TEXT NOT NULL,
    n_bars   INTEGER NOT NULL,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS configs (
    config_hash TEXT PRIMARY KEY,
    config_json TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id           INTEGER PRIMARY KEY,
    run_type     TEXT NOT NULL,         -- match|backtest|evaluate|trade|report
    config_hash  TEXT REFERENCES configs(config_hash),
    seed         INTEGER,
    asof         TEXT,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    status       TEXT NOT NULL DEFAULT 'running',
    metrics_json TEXT
);

CREATE TABLE IF NOT EXISTS matches (
    id               INTEGER PRIMARY KEY,
    run_id           INTEGER NOT NULL REFERENCES runs(id),
    symbol           TEXT NOT NULL,
    query_ts         TEXT NOT NULL,
    match_ts         TEXT NOT NULL,
    rank             INTEGER NOT NULL,
    distance         REAL NOT NULL,
    fwd_return       REAL NOT NULL,
    kept_after_dedup INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS backtest_trades (
    id           INTEGER PRIMARY KEY,
    run_id       INTEGER NOT NULL REFERENCES runs(id),
    symbol       TEXT NOT NULL,
    side         TEXT NOT NULL,
    signal_ts    TEXT NOT NULL,
    entry_ts     TEXT,
    exit_ts      TEXT,
    entry_price  REAL,
    exit_price   REAL,
    gross_return REAL,
    net_return   REAL
);

CREATE TABLE IF NOT EXISTS test_evaluations (
    id           INTEGER PRIMARY KEY,
    config_hash  TEXT NOT NULL,
    run_id       INTEGER REFERENCES runs(id),
    invoked_at   TEXT NOT NULL,
    verdict      TEXT NOT NULL DEFAULT 'CRASHED',  -- updated only on completion
    metrics_json TEXT
);

CREATE TABLE IF NOT EXISTS signals (
    id              INTEGER PRIMARY KEY,
    run_id          INTEGER REFERENCES runs(id),
    config_hash     TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    signal_ts       TEXT NOT NULL,
    side            TEXT NOT NULL,
    n_matches       INTEGER NOT NULL,
    pct_positive    REAL NOT NULL,
    mean_fwd_return REAL NOT NULL,
    threshold_t     REAL NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending|executed|skipped|expired
    created_at      TEXT NOT NULL,
    UNIQUE (config_hash, symbol, signal_ts)
);

CREATE TABLE IF NOT EXISTS orders (
    id               INTEGER PRIMARY KEY,
    signal_id        INTEGER REFERENCES signals(id),
    run_id           INTEGER REFERENCES runs(id),
    symbol           TEXT NOT NULL,
    side             TEXT NOT NULL,
    qty              REAL NOT NULL,
    order_type       TEXT NOT NULL DEFAULT 'market',
    intent           TEXT NOT NULL,     -- entry|time_stop_exit|force_flat
    broker_order_id  TEXT UNIQUE,
    status           TEXT NOT NULL DEFAULT 'submitted',
    filled_qty       REAL,
    filled_avg_price REAL,
    submitted_at     TEXT NOT NULL,
    filled_at        TEXT
);

CREATE TABLE IF NOT EXISTS position_snapshots (
    id             INTEGER PRIMARY KEY,
    ts             TEXT NOT NULL,
    symbol         TEXT,
    qty            REAL,
    avg_entry      REAL,
    market_value   REAL,
    unrealized_pl  REAL,
    account_equity REAL NOT NULL
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    return conn


def register_config(conn: sqlite3.Connection, config_hash: str, config_json: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO configs (config_hash, config_json, created_at) VALUES (?, ?, ?)",
        (config_hash, config_json, utcnow()),
    )
    conn.commit()


def start_run(
    conn: sqlite3.Connection,
    run_type: str,
    config_hash: str | None = None,
    seed: int | None = None,
    asof: str | None = None,
) -> int:
    cur = conn.execute(
        "INSERT INTO runs (run_type, config_hash, seed, asof, started_at) VALUES (?, ?, ?, ?, ?)",
        (run_type, config_hash, seed, asof, utcnow()),
    )
    conn.commit()
    rowid = cur.lastrowid
    assert rowid is not None
    return rowid


def finish_run(conn: sqlite3.Connection, run_id: int, status: str = "ok", metrics: dict | None = None) -> None:
    conn.execute(
        "UPDATE runs SET finished_at = ?, status = ?, metrics_json = ? WHERE id = ?",
        (utcnow(), status, json.dumps(metrics) if metrics is not None else None, run_id),
    )
    conn.commit()


def n_configs_tried(conn: sqlite3.Connection) -> int:
    """Bonferroni N: distinct config hashes with at least one completed train backtest."""
    row = conn.execute(
        "SELECT COUNT(DISTINCT config_hash) FROM runs WHERE run_type = 'backtest' AND status = 'ok'"
    ).fetchone()
    return int(row[0])


def n_test_evaluations(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM test_evaluations").fetchone()[0])


def report_banner(conn: sqlite3.Connection) -> str:
    """Mandatory header for every report: how often the test set has been touched."""
    n_evals = n_test_evaluations(conn)
    n_cfg = n_configs_tried(conn)
    alpha = 0.05 / max(n_cfg, 1)
    return (
        f"configs tried: {n_cfg} | survivor threshold p < {alpha:.2e} (Bonferroni) | "
        f"TEST SET EVALUATED {n_evals} TIME{'S' if n_evals != 1 else ''}"
    )
