"""Durable record of what the live loop saw and did: signals, orders, snapshots.

The db is the audit trail, NOT the source of truth for current holdings — that is
always the live broker (see trade_loop). The journal exists so `status` and the
weekly report can reconstruct history, and so the loop can recover the entry time
of a position it is already holding after a restart.
"""

from __future__ import annotations

import sqlite3

import pandas as pd

from patterns import db as dbm
from patterns.broker.protocols import Account, Order, OrderSide, OrderStatus, Position
from patterns.strategy.base import Signal


def _num(diag: dict, key: str) -> float:
    """Diagnostics are source-specific and loosely typed; pull a number or 0.0."""
    v = diag.get(key)
    return float(v) if isinstance(v, (int, float)) else 0.0


def record_signal(conn: sqlite3.Connection, run_id: int | None, config_hash: str, sig: Signal) -> int:
    """Insert the entry signal (idempotent on config/symbol/ts). Returns its row id."""
    diag = dict(sig.diagnostics)
    n_matches = int(_num(diag, "n_matched") or _num(diag, "n_matches"))
    conn.execute(
        """INSERT OR IGNORE INTO signals
           (run_id, config_hash, symbol, signal_ts, side, n_matches, pct_positive,
            mean_fwd_return, threshold_t, status, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
        (run_id, config_hash, sig.symbol, sig.asof.isoformat(), str(sig.direction),
         n_matches, _num(diag, "pct_positive"),
         _num(diag, "mean_fwd_return"), _num(diag, "threshold_t"),
         dbm.utcnow()),
    )
    row = conn.execute(
        "SELECT id FROM signals WHERE config_hash = ? AND symbol = ? AND signal_ts = ?",
        (config_hash, sig.symbol, sig.asof.isoformat()),
    ).fetchone()
    conn.commit()
    return int(row[0])


def record_order(conn: sqlite3.Connection, run_id: int | None, signal_id: int | None,
                 symbol: str, qty: float, side: OrderSide, intent: str,
                 broker_order_id: str, submitted_at: pd.Timestamp) -> None:
    conn.execute(
        """INSERT OR IGNORE INTO orders
           (signal_id, run_id, symbol, side, qty, intent, broker_order_id, status, submitted_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, 'submitted', ?)""",
        (signal_id, run_id, symbol, str(side), qty, intent, broker_order_id,
         submitted_at.isoformat()),
    )
    conn.commit()


def sync_order(conn: sqlite3.Connection, o: Order) -> None:
    """Reflect a broker order's current status into the journal."""
    conn.execute(
        """UPDATE orders SET status = ?, filled_qty = ?, filled_avg_price = ?, filled_at = ?
           WHERE broker_order_id = ?""",
        (str(o.status),
         o.qty if o.status is OrderStatus.FILLED else None,
         o.fill_price,
         o.filled_at.isoformat() if o.filled_at is not None else None,
         o.id),
    )
    if o.status is OrderStatus.FILLED:
        conn.execute(
            "UPDATE signals SET status = 'executed' WHERE id = "
            "(SELECT signal_id FROM orders WHERE broker_order_id = ?)",
            (o.id,),
        )
    conn.commit()


def pending_order_ids(conn: sqlite3.Connection, symbol: str) -> list[str]:
    rows = conn.execute(
        "SELECT broker_order_id FROM orders WHERE symbol = ? AND status = 'submitted'"
        " AND broker_order_id IS NOT NULL",
        (symbol,),
    ).fetchall()
    return [str(r[0]) for r in rows]


def open_entry_signal_ts(conn: sqlite3.Connection, config_hash: str, symbol: str) -> pd.Timestamp | None:
    """The signal_ts of the most recent recorded entry for this config+symbol.

    Used after a restart to recover when the currently-held position was entered,
    so the time-stop can be computed. Returns None if the db has no entry on record
    (a position adopted from the broker with no local trace — managed to force-flat)."""
    row = conn.execute(
        """SELECT s.signal_ts FROM orders o JOIN signals s ON o.signal_id = s.id
           WHERE o.intent = 'entry' AND o.symbol = ? AND s.config_hash = ?
           ORDER BY o.submitted_at DESC LIMIT 1""",
        (symbol, config_hash),
    ).fetchone()
    return pd.Timestamp(row[0]) if row else None


def snapshot(conn: sqlite3.Connection, ts: pd.Timestamp, account: Account,
             position: Position | None, last_price: float | None) -> None:
    if position is None:
        conn.execute(
            "INSERT INTO position_snapshots (ts, account_equity) VALUES (?, ?)",
            (ts.isoformat(), account.equity),
        )
    else:
        mv = position.qty * last_price if last_price is not None else None
        upl = (position.qty * (last_price - position.avg_entry_price)
               if last_price is not None else None)
        conn.execute(
            """INSERT INTO position_snapshots
               (ts, symbol, qty, avg_entry, market_value, unrealized_pl, account_equity)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (ts.isoformat(), position.symbol, position.qty, position.avg_entry_price,
             mv, upl, account.equity),
        )
    conn.commit()
