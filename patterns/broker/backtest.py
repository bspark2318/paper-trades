"""BacktestAdapter: a minute-clock fill simulator implementing the Broker protocol.

Fill model (deliberately simple, biased against the strategy):
- market orders submitted during bar t fill at bar t+1's OPEN — you can never
  transact at a price you were still watching when you decided;
- costs: cost_bps per side, buys fill above the open, sells below;
- force-flat safety net: a position still open at a session's last bar is
  liquidated at that bar's close (the always-flat-overnight invariant holds
  even if the driving loop forgets).

Long-only in v1: sells are capped at the open position; shorting is rejected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from patterns.broker.protocols import Account, Order, OrderSide, OrderStatus, Position

NY = "America/New_York"


class BacktestAdapter:
    def __init__(self, bars: pd.DataFrame, cash: float = 100_000.0, cost_bps: float = 5.0):
        if bars.empty:
            raise ValueError("BacktestAdapter needs bars")
        self.symbol = "SIM"
        self._ts = bars["ts"].dt.tz_convert("UTC").to_numpy()
        self._open = bars["open"].to_numpy(dtype=np.float64)
        self._close = bars["close"].to_numpy(dtype=np.float64)
        session = pd.Series(bars["ts"].dt.tz_convert(NY).dt.date)
        self._last_of_session = (session != session.shift(-1)).to_numpy()
        self._i = 0
        self.cost_bps = cost_bps
        self._cash = cash
        self._qty = 0.0
        self._avg_entry = 0.0
        self._orders: dict[str, Order] = {}
        self._pending: list[str] = []
        self._n = 0

    # ---- clock ----

    @property
    def index(self) -> int:
        """Current bar index — the walk-forward aligns its bookkeeping to this."""
        return self._i

    @property
    def n_bars(self) -> int:
        return len(self._ts)

    def now(self) -> pd.Timestamp:
        return pd.Timestamp(self._ts[self._i])

    def advance(self) -> bool:
        """Move the clock one bar forward. Returns False at end of data.
        Order of events entering bar t+1: pending fills at t+1's open happen
        first; then, if t+1 is the session's last bar, force-flat at its close."""
        if self._i + 1 >= len(self._ts):
            return False
        self._i += 1
        for order_id in self._pending:
            self._fill(order_id, price=self._open[self._i])
        self._pending = []
        if self._last_of_session[self._i] and self._qty > 0:
            self._liquidate(price=self._close[self._i], reason="force_flat")
        return True

    def advance_to(self, ts: pd.Timestamp) -> None:
        target = np.datetime64(pd.Timestamp(ts).tz_convert("UTC").tz_localize(None))
        while pd.Timestamp(self._ts[self._i]).tz_localize(None) < target:
            if not self.advance():
                break

    # ---- Broker protocol ----

    def get_account(self) -> Account:
        equity = self._cash + self._qty * self._close[self._i]
        return Account(equity=equity, cash=self._cash)

    def get_positions(self) -> list[Position]:
        if self._qty == 0:
            return []
        return [Position(symbol=self.symbol, qty=self._qty, avg_entry_price=self._avg_entry)]

    def get_open_orders(self) -> list[Order]:
        """Orders submitted but not yet filled — the pending fills awaiting the next bar."""
        return [self._orders[oid] for oid in self._pending]

    def submit_order(self, symbol: str, qty: float, side: OrderSide) -> str:
        order_id = f"bt-{self._n}"
        self._n += 1
        if qty <= 0:
            order = self._make(order_id, symbol, qty, side, OrderStatus.REJECTED, reason="qty<=0")
        elif side is OrderSide.SELL and qty > self._qty:
            order = self._make(order_id, symbol, qty, side, OrderStatus.REJECTED, reason="no_shorting")
        else:
            order = self._make(order_id, symbol, qty, side, OrderStatus.SUBMITTED)
            self._pending.append(order_id)
        self._orders[order_id] = order
        return order_id

    def get_order(self, order_id: str) -> Order:
        return self._orders[order_id]

    # ---- internals ----

    def _make(self, order_id: str, symbol: str, qty: float, side: OrderSide,
              status: OrderStatus, reason: str = "") -> Order:
        return Order(id=order_id, symbol=symbol, qty=qty, side=side, status=status,
                     submitted_at=self.now(), reason=reason)

    def _cost_mult(self, side: OrderSide) -> float:
        bps = 1e-4 * self.cost_bps
        return 1.0 + bps if side is OrderSide.BUY else 1.0 - bps

    def _fill(self, order_id: str, price: float) -> None:
        o = self._orders[order_id]
        fill_price = price * self._cost_mult(o.side)
        if o.side is OrderSide.BUY:
            notional = o.qty * fill_price
            new_qty = self._qty + o.qty
            self._avg_entry = (self._avg_entry * self._qty + fill_price * o.qty) / new_qty
            self._qty = new_qty
            self._cash -= notional
        else:
            self._qty -= o.qty
            self._cash += o.qty * fill_price
            if self._qty == 0:
                self._avg_entry = 0.0
        self._orders[order_id] = Order(
            id=o.id, symbol=o.symbol, qty=o.qty, side=o.side, status=OrderStatus.FILLED,
            submitted_at=o.submitted_at, filled_at=self.now(), fill_price=fill_price,
            filled_qty=o.qty, reason=o.reason,
        )

    def _liquidate(self, price: float, reason: str) -> None:
        order_id = f"bt-{self._n}"
        self._n += 1
        o = self._make(order_id, self.symbol, self._qty, OrderSide.SELL,
                       OrderStatus.SUBMITTED, reason=reason)
        self._orders[order_id] = o
        self._fill(order_id, price=price)
