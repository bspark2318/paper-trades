"""AlpacaAdapter: the Broker protocol backed by a live Alpaca PAPER account.

Three guarantees make live paper results commensurable with the backtest and
keep this safe to run unattended:

- Paper-only by construction. The client is always built with paper=True and the
  constructor refuses paper=False outright — there is no code path to the live
  trading endpoint. (Live API keys simply won't authenticate against paper-api.)
- Keys from the environment only (ALPACA_API_KEY / ALPACA_SECRET_KEY); nothing is
  ever read from disk or args.
- Long-only in v1, enforced locally before anything reaches Alpaca: a sell larger
  than the held position, or any non-positive quantity, is rejected without a
  network call — identical semantics to BacktestAdapter.

Alpaca's SDK types never escape this module: every return value is one of our own
frozen dataclasses, so strategy/validation/live code stays broker-agnostic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

from patterns.broker.protocols import Account, Order, OrderSide, OrderStatus, Position


class MissingKeysError(RuntimeError):
    pass


@dataclass(frozen=True)
class MarketClock:
    """Snapshot of the exchange clock from Alpaca — the live loop's source of truth
    for RTH boundaries (BacktestAdapter has its own minute clock instead)."""

    is_open: bool
    timestamp: pd.Timestamp
    next_open: pd.Timestamp
    next_close: pd.Timestamp


def _require_keys() -> tuple[str, str]:
    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise MissingKeysError(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY not set. "
            "Create a free paper account at https://app.alpaca.markets and export both."
        )
    return key, secret


def _build_client() -> Any:
    key, secret = _require_keys()
    from alpaca.trading.client import TradingClient

    return TradingClient(key, secret, paper=True)


class AlpacaAdapter:
    def __init__(self, client: Any | None = None, *, paper: bool = True):
        if not paper:
            raise ValueError(
                "AlpacaAdapter is paper-only: live trading is intentionally unsupported"
            )
        # Injected client is for tests/mocks; otherwise build a real paper client.
        self._client: Any = client if client is not None else _build_client()
        self._rejected: dict[str, Order] = {}   # locally-refused orders (never sent)
        self._n_local = 0

    # ---- market clock ----

    def get_clock(self) -> MarketClock:
        c = self._client.get_clock()
        return MarketClock(
            is_open=bool(c.is_open),
            timestamp=pd.Timestamp(c.timestamp),
            next_open=pd.Timestamp(c.next_open),
            next_close=pd.Timestamp(c.next_close),
        )

    # ---- Broker protocol ----

    def get_account(self) -> Account:
        a = self._client.get_account()
        return Account(equity=float(a.equity), cash=float(a.cash))

    def get_positions(self) -> list[Position]:
        out: list[Position] = []
        for p in self._client.get_all_positions():
            qty = float(p.qty)
            if qty <= 0:        # long-only v1: ignore any short leg defensively
                continue
            out.append(Position(symbol=str(p.symbol), qty=qty,
                                avg_entry_price=float(p.avg_entry_price)))
        return out

    def submit_order(self, symbol: str, qty: float, side: OrderSide) -> str:
        if qty <= 0:
            return self._reject(symbol, qty, side, "qty<=0")
        if side is OrderSide.SELL:
            held = sum(p.qty for p in self.get_positions() if p.symbol == symbol)
            if qty > held:
                return self._reject(symbol, qty, side, "no_shorting")

        from alpaca.trading.enums import OrderSide as AlpacaSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=AlpacaSide.BUY if side is OrderSide.BUY else AlpacaSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        return str(self._client.submit_order(req).id)

    def get_order(self, order_id: str) -> Order:
        if order_id in self._rejected:
            return self._rejected[order_id]
        return _to_order(self._client.get_order_by_id(order_id))

    # ---- internals ----

    def _reject(self, symbol: str, qty: float, side: OrderSide, reason: str) -> str:
        order_id = f"local-reject-{self._n_local}"
        self._n_local += 1
        self._rejected[order_id] = Order(
            id=order_id, symbol=symbol, qty=qty, side=side,
            status=OrderStatus.REJECTED, submitted_at=pd.Timestamp.now("UTC"), reason=reason,
        )
        return order_id


# Alpaca order statuses collapse to our three: filled, terminal-dead, or in-flight.
_DEAD = {"rejected", "canceled", "cancelled", "expired", "done_for_day", "suspended"}


def _to_status(raw: Any) -> OrderStatus:
    s = str(getattr(raw, "value", raw)).lower()
    if s == "filled":
        return OrderStatus.FILLED
    if s in _DEAD:
        return OrderStatus.REJECTED
    return OrderStatus.SUBMITTED


def _to_side(raw: Any) -> OrderSide:
    return OrderSide.SELL if str(getattr(raw, "value", raw)).lower() == "sell" else OrderSide.BUY


def _to_order(o: Any) -> Order:
    fill_price = float(o.filled_avg_price) if getattr(o, "filled_avg_price", None) else None
    filled_at = pd.Timestamp(o.filled_at) if getattr(o, "filled_at", None) else None
    return Order(
        id=str(o.id),
        symbol=str(o.symbol),
        qty=float(o.qty),
        side=_to_side(o.side),
        status=_to_status(o.status),
        submitted_at=pd.Timestamp(o.submitted_at),
        filled_at=filled_at,
        fill_price=fill_price,
    )
