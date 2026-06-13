import numpy as np
import pandas as pd
import pytest

from patterns.broker import Broker, OrderSide, OrderStatus
from patterns.broker.backtest import BacktestAdapter
from tests.conftest import make_multi_session_bars, make_session_bars


def adapter(n_sessions: int = 1, cost_bps: float = 0.0, n_bars: int = 20) -> BacktestAdapter:
    dates = [f"2024-03-{4 + i:02d}" for i in range(n_sessions)]
    bars = make_multi_session_bars(dates, n_bars=n_bars)
    return BacktestAdapter(bars, cash=100_000.0, cost_bps=cost_bps)


def test_satisfies_broker_protocol():
    assert isinstance(adapter(), Broker)


def test_market_order_fills_at_next_bar_open():
    bt = adapter()
    oid = bt.submit_order("SIM", 10, OrderSide.BUY)
    assert bt.get_order(oid).status is OrderStatus.SUBMITTED   # nothing yet
    bt.advance()
    o = bt.get_order(oid)
    assert o.status is OrderStatus.FILLED
    assert o.fill_price == pytest.approx(bt._open[bt._i])      # next bar's open, not prior close
    assert o.filled_at == bt.now()


def test_costs_charged_both_sides():
    bt = adapter(cost_bps=10.0)            # 10 bps each way
    oid = bt.submit_order("SIM", 10, OrderSide.BUY)
    bt.advance()
    buy = bt.get_order(oid)
    assert buy.fill_price == pytest.approx(bt._open[bt._i] * 1.001)
    oid2 = bt.submit_order("SIM", 10, OrderSide.SELL)
    bt.advance()
    sell = bt.get_order(oid2)
    assert sell.fill_price == pytest.approx(bt._open[bt._i] * 0.999)


def test_force_flat_at_session_close():
    bt = adapter(n_sessions=2, n_bars=10)
    bt.submit_order("SIM", 10, OrderSide.BUY)
    bt.advance()
    assert bt.get_positions()
    while not bt._last_of_session[bt._i]:
        bt.advance()
    assert bt.get_positions() == []        # liquidated on the session's last bar
    flat_orders = [o for o in bt._orders.values() if o.reason == "force_flat"]
    assert len(flat_orders) == 1
    assert flat_orders[0].status is OrderStatus.FILLED


def test_no_shorting_rejected():
    bt = adapter()
    oid = bt.submit_order("SIM", 5, OrderSide.SELL)
    o = bt.get_order(oid)
    assert o.status is OrderStatus.REJECTED
    assert o.reason == "no_shorting"
    bt.advance()
    assert bt.get_order(oid).status is OrderStatus.REJECTED    # never fills later


def test_equity_accounting_round_trip():
    bt = adapter(cost_bps=0.0)
    start = bt.get_account().equity
    oid = bt.submit_order("SIM", 100, OrderSide.BUY)
    bt.advance()
    entry = bt.get_order(oid).fill_price
    oid2 = bt.submit_order("SIM", 100, OrderSide.SELL)
    bt.advance()
    exit_ = bt.get_order(oid2).fill_price
    assert bt.get_positions() == []
    assert bt.get_account().equity == pytest.approx(start + 100 * (exit_ - entry))


def test_round_trip_with_costs_loses_spread():
    bars = make_session_bars("2024-03-04", n_bars=20)
    bars["open"] = 100.0
    bars["close"] = 100.0   # flat market: only costs move equity
    bt = BacktestAdapter(bars, cash=100_000.0, cost_bps=10.0)
    start = bt.get_account().equity
    bt.submit_order("SIM", 100, OrderSide.BUY)
    bt.advance()
    bt.submit_order("SIM", 100, OrderSide.SELL)
    bt.advance()
    lost = start - bt.get_account().equity
    assert lost == pytest.approx(100 * 100.0 * 0.001 * 2)      # 10bps x 2 sides


def test_advance_to_timestamp():
    bt = adapter(n_sessions=2, n_bars=10)
    target = pd.Timestamp(bt._ts[15])
    bt.advance_to(target)
    assert bt.now() == target
