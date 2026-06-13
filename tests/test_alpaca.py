"""AlpacaAdapter against a fake trading client — no network, no keys.

Covers the three guarantees: paper-only construction, env-only keys, and the
long-only local guard; plus faithful translation of Alpaca's types into our
frozen dataclasses.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from patterns.broker.alpaca import AlpacaAdapter, MissingKeysError
from patterns.broker.protocols import Broker, OrderSide, OrderStatus


class FakeTradingClient:
    """Implements only the methods AlpacaAdapter calls. Records submissions so a
    test can assert that locally-rejected orders never reach the wire."""

    def __init__(self, account=None, positions=None, clock=None, orders=None, open_orders=None):
        self._account = account
        self._positions = positions or []
        self._clock = clock
        self._orders = orders or {}
        self._open_orders = open_orders or []
        self.submitted: list = []
        self.order_queries: list = []
        self._next_id = 100

    def get_account(self):
        return self._account

    def get_all_positions(self):
        return self._positions

    def get_clock(self):
        return self._clock

    def submit_order(self, req):
        self.submitted.append(req)
        oid = str(self._next_id)
        self._next_id += 1
        o = SimpleNamespace(
            id=oid, symbol=req.symbol, qty=req.qty, side=req.side, status="filled",
            filled_avg_price="101.5", filled_at=pd.Timestamp("2026-06-12 14:31", tz="UTC"),
            submitted_at=pd.Timestamp("2026-06-12 14:30", tz="UTC"),
        )
        self._orders[oid] = o
        return o

    def get_order_by_id(self, oid):
        return self._orders[oid]

    def get_orders(self, req):
        self.order_queries.append(req)
        return self._open_orders


def _account(equity="100000", cash="50000"):
    return SimpleNamespace(equity=equity, cash=cash)


def _position(symbol="QQQ", qty="10", avg="500.0"):
    return SimpleNamespace(symbol=symbol, qty=qty, avg_entry_price=avg)


def adapter(**kw) -> AlpacaAdapter:
    return AlpacaAdapter(client=FakeTradingClient(**kw))


# ---------- construction guards ----------

def test_refuses_live_trading():
    with pytest.raises(ValueError, match="paper-only"):
        AlpacaAdapter(client=FakeTradingClient(), paper=False)


def test_missing_keys_raises(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    with pytest.raises(MissingKeysError):
        AlpacaAdapter()          # no injected client → tries to build from env


def test_satisfies_broker_protocol():
    assert isinstance(adapter(account=_account()), Broker)


# ---------- reads ----------

def test_get_account_maps_floats():
    a = adapter(account=_account(equity="123456.78", cash="9000.5")).get_account()
    assert a.equity == pytest.approx(123456.78)
    assert a.cash == pytest.approx(9000.5)


def test_get_positions_maps_and_filters_nonlong():
    a = adapter(positions=[_position("QQQ", "10", "500"),
                           _position("SPY", "0", "400"),       # flat → dropped
                           _position("IWM", "-3", "200")])     # short leg → dropped
    pos = a.get_positions()
    assert [p.symbol for p in pos] == ["QQQ"]
    assert pos[0].qty == 10.0 and pos[0].avg_entry_price == 500.0


def test_get_clock_maps():
    clk = SimpleNamespace(
        is_open=True,
        timestamp=pd.Timestamp("2026-06-12 14:30", tz="UTC"),
        next_open=pd.Timestamp("2026-06-15 13:30", tz="UTC"),
        next_close=pd.Timestamp("2026-06-12 20:00", tz="UTC"),
    )
    c = adapter(clock=clk).get_clock()
    assert c.is_open is True
    assert c.next_close == pd.Timestamp("2026-06-12 20:00", tz="UTC")


# ---------- orders ----------

def test_buy_reaches_the_wire_and_maps_filled():
    fake = FakeTradingClient(account=_account(), positions=[])
    a = AlpacaAdapter(client=fake)
    oid = a.submit_order("QQQ", 5, OrderSide.BUY)
    assert len(fake.submitted) == 1                  # actually sent
    assert str(fake.submitted[0].side.value).lower() == "buy"
    o = a.get_order(oid)
    assert o.status is OrderStatus.FILLED
    assert o.side is OrderSide.BUY
    assert o.fill_price == pytest.approx(101.5)


def test_sell_within_position_reaches_the_wire():
    fake = FakeTradingClient(account=_account(), positions=[_position("QQQ", "10")])
    a = AlpacaAdapter(client=fake)
    a.submit_order("QQQ", 10, OrderSide.SELL)
    assert len(fake.submitted) == 1


def test_sell_exceeding_position_rejected_without_network():
    fake = FakeTradingClient(account=_account(), positions=[_position("QQQ", "10")])
    a = AlpacaAdapter(client=fake)
    oid = a.submit_order("QQQ", 20, OrderSide.SELL)
    assert fake.submitted == []                      # never sent
    o = a.get_order(oid)
    assert o.status is OrderStatus.REJECTED
    assert o.reason == "no_shorting"


def test_nonpositive_qty_rejected_without_network():
    fake = FakeTradingClient(account=_account(), positions=[])
    a = AlpacaAdapter(client=fake)
    oid = a.submit_order("QQQ", 0, OrderSide.BUY)
    assert fake.submitted == []
    assert a.get_order(oid).reason == "qty<=0"


def test_get_open_orders_maps_and_queries_open_only():
    o1 = SimpleNamespace(
        id="200", symbol="QQQ", qty="5", side="buy", status="new",
        filled_avg_price=None, filled_at=None, filled_qty="0",
        submitted_at=pd.Timestamp("2026-06-12 14:30", tz="UTC"),
    )
    fake = FakeTradingClient(open_orders=[o1])
    a = AlpacaAdapter(client=fake)
    out = a.get_open_orders()
    assert len(fake.order_queries) == 1                          # asked the broker
    assert str(fake.order_queries[0].status).lower().endswith("open")
    assert len(out) == 1
    assert out[0].symbol == "QQQ" and out[0].side is OrderSide.BUY
    assert out[0].status is OrderStatus.SUBMITTED               # "new" → in-flight


def test_get_open_orders_empty():
    a = AlpacaAdapter(client=FakeTradingClient(open_orders=[]))
    assert a.get_open_orders() == []


def test_filled_qty_is_mapped():
    o = SimpleNamespace(
        id="x", symbol="QQQ", qty="10", side="buy", status="filled",
        filled_avg_price="100.0", filled_qty="7",                # partial fill
        filled_at=pd.Timestamp("2026-06-12 14:31", tz="UTC"),
        submitted_at=pd.Timestamp("2026-06-12 14:30", tz="UTC"),
    )
    a = AlpacaAdapter(client=FakeTradingClient(orders={"x": o}))
    assert a.get_order("x").filled_qty == pytest.approx(7.0)


@pytest.mark.parametrize("raw,expected", [
    ("filled", OrderStatus.FILLED),
    ("new", OrderStatus.SUBMITTED),
    ("accepted", OrderStatus.SUBMITTED),
    ("partially_filled", OrderStatus.SUBMITTED),
    ("rejected", OrderStatus.REJECTED),
    ("canceled", OrderStatus.REJECTED),
    ("expired", OrderStatus.REJECTED),
])
def test_order_status_mapping(raw, expected):
    o = SimpleNamespace(
        id="x", symbol="QQQ", qty="5", side="buy", status=raw,
        filled_avg_price="100" if raw == "filled" else None,
        filled_at=pd.Timestamp("2026-06-12 14:31", tz="UTC") if raw == "filled" else None,
        submitted_at=pd.Timestamp("2026-06-12 14:30", tz="UTC"),
    )
    a = AlpacaAdapter(client=FakeTradingClient(orders={"x": o}))
    assert a.get_order("x").status is expected
