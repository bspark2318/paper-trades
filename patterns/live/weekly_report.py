"""`patterns report weekly`: a markdown + PNG digest of the live paper account.

Built entirely from the journal the trade loop writes (orders, position_snapshots)
— no network. Its honest centerpiece is the live-vs-backtest divergence: this
week's realized per-trade return next to the backtest mean for the same config,
so paper-vs-reality drift is visible rather than buried. Every number carries the
paper-fill upper-bound caveat and the test-set evaluation banner.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from patterns import db as dbm
from patterns.config import Config
from patterns.data import store
from patterns.validate import evaluate as ev

NY = "America/New_York"


@dataclass(frozen=True)
class LiveTrade:
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    qty: float
    entry_price: float
    exit_price: float
    net_ret: float
    pnl: float
    exit_reason: str


@dataclass
class WeeklyReport:
    start: pd.Timestamp
    end: pd.Timestamp
    trades: list[LiveTrade]
    n_trades: int = 0
    hit_rate: float = float("nan")
    mean_net_ret: float = float("nan")
    total_pnl: float = 0.0
    backtest_mean: float | None = None
    equity_start: float | None = None
    equity_end: float | None = None
    bh_return: float | None = None
    png_path: str | None = None
    md_path: str | None = None
    extra: dict = field(default_factory=dict)


def _paired_trades(conn: sqlite3.Connection, symbol: str,
                   start: pd.Timestamp, end: pd.Timestamp) -> list[LiveTrade]:
    """Walk filled orders in time order, pairing each exit with the entry before
    it (one position at a time). Keep trades whose EXIT falls inside the window."""
    rows = conn.execute(
        """SELECT intent, qty, filled_avg_price, filled_at FROM orders
           WHERE symbol = ? AND status = 'filled' AND filled_at IS NOT NULL
           ORDER BY filled_at""",
        (symbol,),
    ).fetchall()
    trades: list[LiveTrade] = []
    open_entry: sqlite3.Row | None = None
    for r in rows:
        if r["intent"] == "entry":
            open_entry = r
        elif open_entry is not None:
            ep = float(open_entry["filled_avg_price"])
            xp = float(r["filled_avg_price"])
            exit_ts = pd.Timestamp(r["filled_at"])
            qty = float(open_entry["qty"])
            trades.append(LiveTrade(
                entry_ts=pd.Timestamp(open_entry["filled_at"]), exit_ts=exit_ts,
                qty=qty, entry_price=ep, exit_price=xp,
                net_ret=xp / ep - 1.0, pnl=qty * (xp - ep), exit_reason=str(r["intent"]),
            ))
            open_entry = None
    return [t for t in trades if start <= t.exit_ts <= end]


def _equity_curve(conn: sqlite3.Connection, start: pd.Timestamp,
                  end: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    rows = conn.execute(
        """SELECT ts, account_equity FROM position_snapshots
           WHERE ts >= ? AND ts <= ? ORDER BY ts""",
        (start.isoformat(), end.isoformat()),
    ).fetchall()
    ts = np.array([pd.Timestamp(r["ts"]) for r in rows])
    eq = np.array([float(r["account_equity"]) for r in rows])
    return ts, eq


def build_weekly(conn: sqlite3.Connection, cfg: Config, now: pd.Timestamp,
                 window_days: int = 7) -> WeeklyReport:
    sym = cfg.symbols[0]
    end = pd.Timestamp(now).tz_convert("UTC") if pd.Timestamp(now).tzinfo else pd.Timestamp(now, tz="UTC")
    start = end - pd.Timedelta(days=window_days)

    trades = _paired_trades(conn, sym, start, end)
    rep = WeeklyReport(start=start, end=end, trades=trades)

    if trades:
        rets = np.array([t.net_ret for t in trades])
        rep.n_trades = len(trades)
        rep.hit_rate = float(np.mean(rets > 0))
        rep.mean_net_ret = float(np.mean(rets))
        rep.total_pnl = float(sum(t.pnl for t in trades))

    bt = ev.latest_train_metrics(conn, cfg.config_hash)
    if bt is not None:
        rep.backtest_mean = bt.get("mean_net_ret")

    ts, eq = _equity_curve(conn, start, end)
    if len(eq):
        rep.equity_start = float(eq[0])
        rep.equity_end = float(eq[-1])
        rep.extra["equity_ts"] = ts
        rep.extra["equity"] = eq
        rep.bh_return = _buy_and_hold_return(conn, sym, ts[0], ts[-1])
    return rep


def _buy_and_hold_return(conn: sqlite3.Connection, symbol: str,
                         t0: pd.Timestamp, t1: pd.Timestamp) -> float | None:
    bars = store.load_bars(conn, symbol)
    if bars.empty:
        return None
    seg = bars[(bars["ts"] >= pd.Timestamp(t0)) & (bars["ts"] <= pd.Timestamp(t1))]
    if len(seg) < 2:
        return None
    return float(seg["close"].iloc[-1] / seg["close"].iloc[0] - 1.0)


def render_markdown(conn: sqlite3.Connection, cfg: Config, rep: WeeklyReport) -> str:
    survivor = dbm.is_survivor(conn, cfg.config_hash)
    L: list[str] = []
    L.append(f"# Weekly paper-trading report — {cfg.symbols[0]}")
    L.append("")
    L.append(f"_{rep.start:%Y-%m-%d} → {rep.end:%Y-%m-%d} UTC_")
    L.append("")
    L.append(f"> {dbm.report_banner(conn)}")
    L.append(">")
    L.append("> **Paper fills are optimistic — read all P&L as an upper bound.**")
    L.append("")
    L.append(f"- config: `{cfg.config_hash}` ({cfg.signal_source}) — "
             f"{'**survivor, armed**' if survivor else '**not a survivor** (loop refuses to trade)'}")
    L.append("")

    if rep.n_trades == 0:
        L.append("## No closed trades this week")
        L.append("")
        L.append("The loop opened no positions that closed inside the window "
                 "(it may have been flat, idle outside RTH, or holding).")
    else:
        L.append("## This week")
        L.append("")
        L.append("| metric | live |")
        L.append("| --- | --- |")
        L.append(f"| closed trades | {rep.n_trades} |")
        L.append(f"| hit rate | {rep.hit_rate:.0%} |")
        L.append(f"| mean net / trade | {rep.mean_net_ret:+.4%} |")
        L.append(f"| realized P&L | ${rep.total_pnl:+,.2f} |")
        if rep.equity_start is not None and rep.equity_end is not None:
            L.append(f"| equity | ${rep.equity_start:,.2f} → ${rep.equity_end:,.2f} |")
        if rep.bh_return is not None:
            L.append(f"| {cfg.symbols[0]} buy-and-hold (same window) | {rep.bh_return:+.4%} |")
        L.append("")
        L.append("## Live vs backtest")
        L.append("")
        if rep.backtest_mean is None:
            L.append("No backtest on record for this config — run `patterns backtest` to compare.")
        else:
            drift = rep.mean_net_ret - rep.backtest_mean
            verdict = ("live is running **worse** than backtest" if drift < 0
                       else "live is **in line with / better** than backtest")
            L.append(f"- backtest mean net/trade: {rep.backtest_mean:+.4%}")
            L.append(f"- live mean net/trade: {rep.mean_net_ret:+.4%}")
            L.append(f"- divergence: {drift:+.4%} — {verdict}.")
            L.append("")
            L.append("_Divergence is expected: the backtest assumed idealized fills. "
                     "Persistent negative drift is the signal that the edge does not survive contact._")
        L.append("")
        L.append("## Closed trades")
        L.append("")
        L.append("| entry | exit | qty | net | reason |")
        L.append("| --- | --- | --- | --- | --- |")
        for t in rep.trades:
            L.append(f"| {t.entry_ts:%m-%d %H:%M} | {t.exit_ts:%m-%d %H:%M} | "
                     f"{t.qty:g} | {t.net_ret:+.3%} | {t.exit_reason} |")

    if rep.png_path:
        L.append("")
        L.append(f"![equity]({Path(rep.png_path).name})")
    L.append("")
    return "\n".join(L)


def write_report(conn: sqlite3.Connection, cfg: Config, now: pd.Timestamp,
                 window_days: int = 7) -> WeeklyReport:
    from patterns import plotting

    rep = build_weekly(conn, cfg, now, window_days)
    reports = Path(cfg.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    stamp = f"{rep.end:%Y%m%d}"

    if "equity" in rep.extra:
        bench = None
        bh = _bench_curve(conn, cfg.symbols[0], rep.extra["equity_ts"], rep.extra["equity"])
        if bh is not None:
            bench = bh
        rep.png_path = str(plotting.plot_weekly_equity(
            rep.extra["equity_ts"], rep.extra["equity"], bench,
            reports / f"weekly_{cfg.config_hash}_{stamp}.png",
            title=f"paper equity vs {cfg.symbols[0]} buy-and-hold — week to {rep.end:%Y-%m-%d}",
        ))

    md = render_markdown(conn, cfg, rep)
    md_path = reports / f"weekly_{cfg.config_hash}_{stamp}.md"
    md_path.write_text(md)
    rep.md_path = str(md_path)
    return rep


def _bench_curve(conn: sqlite3.Connection, symbol: str, eq_ts: np.ndarray,
                 eq: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """QQQ buy-and-hold rebased to the live equity's starting value, sampled at the
    same timestamps as the equity snapshots."""
    bars = store.load_bars(conn, symbol)
    if bars.empty or len(eq) == 0:
        return None
    seg = bars[(bars["ts"] >= pd.Timestamp(eq_ts[0])) & (bars["ts"] <= pd.Timestamp(eq_ts[-1]))]
    if len(seg) < 2:
        return None
    start_px = float(seg["close"].iloc[0])
    bench_ts = seg["ts"].to_numpy()
    bench_eq = float(eq[0]) * seg["close"].to_numpy(dtype=np.float64) / start_px
    return bench_ts, bench_eq
