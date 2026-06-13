"""Broker boundary: strategy and validation code may import ONLY this module
from the broker package. BacktestAdapter and AlpacaAdapter are interchangeable
behind these protocols — that interchangeability is what makes backtest
results commensurable with live paper results.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

import pandas as pd


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(StrEnum):
    SUBMITTED = "submitted"
    FILLED = "filled"
    REJECTED = "rejected"


@dataclass(frozen=True)
class Account:
    equity: float
    cash: float


@dataclass(frozen=True)
class Position:
    symbol: str
    qty: float              # long-only in v1: always > 0 while open
    avg_entry_price: float


@dataclass(frozen=True)
class Order:
    id: str
    symbol: str
    qty: float
    side: OrderSide
    status: OrderStatus
    submitted_at: pd.Timestamp
    filled_at: pd.Timestamp | None = None
    fill_price: float | None = None
    filled_qty: float | None = None     # actual quantity filled (may be < qty on partials)
    reason: str = ""        # set when REJECTED, or "force_flat" on safety-net exits


@runtime_checkable
class Broker(Protocol):
    def get_account(self) -> Account: ...

    def get_positions(self) -> list[Position]: ...

    def get_open_orders(self) -> list[Order]: ...

    def submit_order(self, symbol: str, qty: float, side: OrderSide) -> str: ...

    def get_order(self, order_id: str) -> Order: ...
