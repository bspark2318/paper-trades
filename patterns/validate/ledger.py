"""Multiple-testing ledger: every backtested hypothesis is on the record forever.

N = distinct config hashes ever attempted (db.n_configs_tried). The survivor
threshold for any single config is alpha = 0.05 / N — ten ideas tried means
each must be ten times more convincing. There is no API to remove a row.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from patterns import db as dbm
from patterns.validate.walkforward import TradeRecord


def alpha(n_configs: int) -> float:
    return 0.05 / max(n_configs, 1)


@dataclass(frozen=True)
class LedgerRow:
    config_hash: str
    n_runs: int
    last_run_at: str
    n_trades: int | None
    mean_net_ret: float | None
    p_random: float | None
    candidate: bool         # train-level: p_random < alpha AND positive mean (NOT a verdict)


def save_trades(conn: sqlite3.Connection, run_id: int, symbol: str,
                trades: list[TradeRecord]) -> None:
    conn.executemany(
        """INSERT INTO backtest_trades
           (run_id, symbol, side, signal_ts, entry_ts, exit_ts, entry_price, exit_price,
            gross_return, net_return)
           VALUES (?, ?, 'long', ?, ?, ?, ?, ?, ?, ?)""",
        [
            (run_id, symbol, t.entry_ts.isoformat(), t.entry_ts.isoformat(),
             t.exit_ts.isoformat(), t.entry_price, t.exit_price, t.net_ret, t.net_ret)
            for t in trades
        ],
    )
    conn.commit()


def ledger_rows(conn: sqlite3.Connection) -> list[LedgerRow]:
    n = dbm.n_configs_tried(conn)
    a = alpha(n)
    rows = []
    for r in conn.execute(
        """SELECT config_hash, COUNT(*) AS n_runs, MAX(started_at) AS last_run_at
           FROM runs WHERE run_type = 'backtest' GROUP BY config_hash
           ORDER BY last_run_at DESC"""
    ).fetchall():
        latest = conn.execute(
            """SELECT metrics_json FROM runs
               WHERE run_type = 'backtest' AND config_hash = ? AND status = 'ok'
                     AND metrics_json IS NOT NULL
               ORDER BY started_at DESC LIMIT 1""",
            (r["config_hash"],),
        ).fetchone()
        metrics = json.loads(latest["metrics_json"]) if latest else {}
        p = metrics.get("p_random")
        mean = metrics.get("mean_net_ret")
        rows.append(LedgerRow(
            config_hash=r["config_hash"],
            n_runs=r["n_runs"],
            last_run_at=r["last_run_at"],
            n_trades=metrics.get("n_trades"),
            mean_net_ret=mean,
            p_random=p,
            candidate=(p is not None and mean is not None and p < a and mean > 0),
        ))
    return rows
