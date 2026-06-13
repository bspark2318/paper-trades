"""`patterns status`: what the paper account holds now and how the loop has done.

Reads live holdings from the broker (source of truth) and the order/snapshot
journal from the db, and frames P&L against QQQ buy-and-hold — the benchmark the
whole project measures itself against.
"""

from __future__ import annotations

import sqlite3

from patterns.config import Config
from patterns.live.trade_loop import LiveBroker


def render_status(conn: sqlite3.Connection, broker: LiveBroker, cfg: Config) -> str:
    from patterns import db as dbm

    sym = cfg.symbols[0]
    lines: list[str] = [dbm.report_banner(conn)]
    survivor = dbm.is_survivor(conn, cfg.config_hash)
    lines.append(
        f"\nconfig {cfg.config_hash} | source {cfg.signal_source} | "
        f"{'SURVIVOR — armed' if survivor else 'NOT a survivor — loop will refuse to trade'}"
    )

    account = broker.get_account()
    lines.append(f"paper equity: ${account.equity:,.2f} | cash ${account.cash:,.2f}")

    positions = [p for p in broker.get_positions() if p.symbol == sym]
    if not positions:
        lines.append(f"position: flat ({sym})")
    else:
        for p in positions:
            lines.append(f"position: {p.qty:g} {p.symbol} @ avg ${p.avg_entry_price:,.4f}")

    open_orders = [o for o in broker.get_open_orders() if o.symbol == sym]
    if open_orders:
        lines.append(f"open orders: {len(open_orders)}")
        for o in open_orders:
            lines.append(f"  {o.side} {o.qty:g} {o.symbol} [{o.status}]")

    # realized P&L from filled exit orders recorded in the journal
    row = conn.execute(
        """SELECT COUNT(*) AS n,
                  SUM(CASE WHEN intent IN ('time_stop','time_stop_exit','force_flat')
                           THEN 1 ELSE 0 END) AS exits
           FROM orders WHERE symbol = ? AND status = 'filled'""",
        (sym,),
    ).fetchone()
    n_filled = int(row["n"] or 0)
    n_exits = int(row["exits"] or 0)
    lines.append(f"journal: {n_filled} filled orders, {n_exits} closed trades")

    last = conn.execute(
        "SELECT ts, account_equity FROM position_snapshots ORDER BY ts DESC LIMIT 1"
    ).fetchone()
    if last is not None:
        lines.append(f"last snapshot: {last['ts']} | equity ${last['account_equity']:,.2f}")

    lines.append("NOTE: paper fills are optimistic — read all P&L as an upper bound.")
    return "\n".join(lines)
