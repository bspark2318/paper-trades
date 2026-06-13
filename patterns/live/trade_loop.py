"""The intraday paper-trading loop — the first code that places live orders.

It runs unattended during regular hours and is built to be safe to kill and
restart at any instant. The design, agreed in review:

1. Sizing: each entry is `position_size` (5%) of *live* equity, whole shares,
   one position at a time.
2. Two exits, market orders, whichever comes first:
   - time-stop: sell `horizon` bars after entry — the exact hold the backtest
     measured, so live stays honest to what the referee proved;
   - force-flat: hard sell `force_flat_minutes_before_close` before the close,
     guaranteeing nothing is ever held overnight.
3. Source of truth is the LIVE BROKER, not the db. Every step re-reads positions
   and open orders from the broker and believes them. A position held with no
   local entry record (crash between fill and write) is adopted and managed to
   its exit rather than ignored or doubled.
4. Cadence: act once per *closed* bar (never the forming minute — that is
   lookahead in live too), during RTH only, and only for a config that has
   passed the evaluate gate (a survivor).

The whole state machine lives in step(); run() just calls it on a timer. Tests
drive step() directly against a mocked broker and a scripted clock — no network,
no sleeps.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from patterns.broker.alpaca import MarketClock
from patterns.broker.protocols import Account, Order, OrderSide, Position
from patterns.config import Config
from patterns.live import journal
from patterns.strategy.base import Direction, Signal, make_source


@runtime_checkable
class LiveBroker(Protocol):
    """Broker protocol plus the live-only reads the loop's reconcile needs."""

    def get_account(self) -> Account: ...
    def get_positions(self) -> list[Position]: ...
    def get_open_orders(self) -> list[Order]: ...
    def get_order(self, order_id: str) -> Order: ...
    def submit_order(self, symbol: str, qty: float, side: OrderSide) -> str: ...
    def get_clock(self) -> MarketClock: ...


@runtime_checkable
class BarFeed(Protocol):
    def closed_bars(self, asof: pd.Timestamp) -> pd.DataFrame:
        """Bars (history + today) up to and including the last fully-CLOSED bar at
        asof. Must never include the currently-forming minute."""
        ...


@dataclass(frozen=True)
class StepResult:
    action: str             # entry | hold | exit:time_stop | exit:force_flat | no_trade:... | ...
    detail: str = ""


class NotASurvivorError(RuntimeError):
    pass


class TradeLoop:
    def __init__(self, cfg: Config, broker: LiveBroker, feed: BarFeed,
                 conn: sqlite3.Connection, *, run_id: int | None = None,
                 require_survivor: bool = True):
        if require_survivor:
            from patterns import db as dbm
            if not dbm.is_survivor(conn, cfg.config_hash):
                raise NotASurvivorError(
                    f"config {cfg.config_hash} has not passed the evaluate gate — "
                    f"the loop trades survivors only. Run `patterns evaluate` first."
                )
        self.cfg = cfg
        self.broker = broker
        self.feed = feed
        self.conn = conn
        self.run_id = run_id
        self.symbol = cfg.symbols[0]
        self.source = make_source(cfg)
        self._last_decided_ts: pd.Timestamp | None = None

    # ---- one iteration ----

    def step(self) -> StepResult:
        clock = self.broker.get_clock()
        self._sync_pending()

        held = self._held_qty()
        account = self.broker.get_account()
        bars = self.feed.closed_bars(clock.timestamp)
        last_price = float(bars["close"].iloc[-1]) if not bars.empty else None
        self._snapshot(clock.timestamp, account, held, last_price)

        if not clock.is_open:
            return StepResult("market_closed")
        if bars.empty:
            return StepResult("no_bars")

        last_ts = pd.Timestamp(bars["ts"].iloc[-1])
        mins_to_close = (clock.next_close - clock.timestamp).total_seconds() / 60.0
        force_flat_now = mins_to_close <= self.cfg.force_flat_minutes_before_close

        if held > 0:
            return self._manage_open(bars, last_ts, held, force_flat_now, clock)
        return self._maybe_enter(bars, last_ts, mins_to_close, force_flat_now, account, last_price, clock)

    # ---- holding: time-stop or force-flat ----

    def _manage_open(self, bars: pd.DataFrame, last_ts: pd.Timestamp, held: float,
                     force_flat_now: bool, clock: MarketClock) -> StepResult:
        if self._pending_for(OrderSide.SELL):
            return StepResult("exit_pending")
        if force_flat_now:
            self._submit_exit(held, "force_flat", clock)
            return StepResult("exit:force_flat")
        entry_ts = journal.open_entry_signal_ts(self.conn, self.cfg.config_hash, self.symbol)
        if entry_ts is None:
            # adopted position with no local entry record: we cannot place the
            # time-stop honestly, so hold it to the force-flat safety net.
            return StepResult("hold", "adopted_no_entry_record")
        if self._bars_since(bars, entry_ts) >= self.cfg.horizon:
            self._submit_exit(held, "time_stop", clock)
            return StepResult("exit:time_stop")
        return StepResult("hold")

    # ---- flat: maybe enter ----

    def _maybe_enter(self, bars: pd.DataFrame, last_ts: pd.Timestamp, mins_to_close: float,
                     force_flat_now: bool, account: Account, last_price: float | None,
                     clock: MarketClock) -> StepResult:
        if self._pending_for(OrderSide.BUY):
            return StepResult("entry_pending")
        # one decision per closed bar
        if self._last_decided_ts is not None and last_ts <= self._last_decided_ts:
            return StepResult("already_decided")
        # no room left to run a full horizon before the force-flat: don't open
        if mins_to_close <= self.cfg.force_flat_minutes_before_close + self.cfg.horizon:
            return StepResult("too_close_to_enter")

        self._last_decided_ts = last_ts
        self.source.prepare(bars)
        sig = self.source.signal_at(last_ts)
        if sig.direction is not Direction.LONG:        # long-only execution in v1
            return StepResult(f"no_trade:{sig.direction}", str(sig.diagnostics.get("reason", "")))

        if last_price is None or last_price <= 0:
            return StepResult("no_price")
        qty = float(np.floor(account.equity * self.cfg.position_size / last_price))
        if qty <= 0:
            return StepResult("size_zero", f"equity {account.equity:.2f} too small for one share")
        self._submit_entry(sig, qty, clock)
        return StepResult("entry", f"{qty:g} @ ~{last_price:.2f}")

    # ---- broker reconcile helpers ----

    def _held_qty(self) -> float:
        return sum(p.qty for p in self.broker.get_positions() if p.symbol == self.symbol)

    def _pending_for(self, side: OrderSide) -> bool:
        return any(o.symbol == self.symbol and o.side is side for o in self.broker.get_open_orders())

    def _bars_since(self, bars: pd.DataFrame, entry_ts: pd.Timestamp) -> int:
        ts = bars["ts"].to_numpy()
        entry = np.datetime64(pd.Timestamp(entry_ts).tz_convert("UTC").tz_localize(None))
        utc = pd.to_datetime(bars["ts"], utc=True).dt.tz_localize(None).to_numpy()
        return int((utc > entry).sum())

    # ---- order placement + journaling ----

    def _submit_entry(self, sig: Signal, qty: float, clock: MarketClock) -> None:
        sig_id = journal.record_signal(self.conn, self.run_id, self.cfg.config_hash, sig)
        oid = self.broker.submit_order(self.symbol, qty, OrderSide.BUY)
        journal.record_order(self.conn, self.run_id, sig_id, self.symbol, qty,
                             OrderSide.BUY, "entry", oid, clock.timestamp)

    def _submit_exit(self, qty: float, intent: str, clock: MarketClock) -> None:
        oid = self.broker.submit_order(self.symbol, qty, OrderSide.SELL)
        journal.record_order(self.conn, self.run_id, None, self.symbol, qty,
                             OrderSide.SELL, intent, oid, clock.timestamp)

    def _sync_pending(self) -> None:
        for oid in journal.pending_order_ids(self.conn, self.symbol):
            try:
                journal.sync_order(self.conn, self.broker.get_order(oid))
            except KeyError:
                continue

    def _snapshot(self, ts: pd.Timestamp, account: Account, held: float,
                  last_price: float | None) -> None:
        pos: Position | None = None
        if held > 0:
            live = [p for p in self.broker.get_positions() if p.symbol == self.symbol]
            pos = live[0] if live else None
        journal.snapshot(self.conn, ts, account, pos, last_price)

    # ---- driver ----

    def run(self, *, max_iterations: int | None = None,
            sleep: Callable[[float], None] = time.sleep,
            poll_seconds: float = 60.0) -> None:
        """Poll on a timer. During RTH: act each minute. When closed: sleep longer.
        max_iterations bounds the loop for tests/dry-runs (None = forever)."""
        i = 0
        while max_iterations is None or i < max_iterations:
            result = self.step()
            i += 1
            clock = self.broker.get_clock()
            sleep(poll_seconds if clock.is_open else min(poll_seconds * 10, 600.0))
