import dataclasses

import pandas as pd
import pytest

from patterns import db as dbm
from patterns.broker.alpaca import MarketClock
from patterns.broker.protocols import Account, Order, OrderSide, OrderStatus, Position
from patterns.config import Config
from patterns.live import journal
from patterns.live.trade_loop import NotASurvivorError, TradeLoop
from patterns.strategy.base import Direction, Signal

NY = "America/New_York"


# ---------- fakes ----------

class FakeBroker:
    """In-memory Broker + clock. Market orders fill instantly unless fill=False."""

    def __init__(self, clock: MarketClock, equity: float = 100_000.0, price: float = 100.0):
        self.clock = clock
        self._equity = equity
        self._cash = equity
        self._positions: dict[str, Position] = {}
        self._orders: dict[str, Order] = {}
        self._open: list[str] = []
        self._n = 0
        self.price = price
        self.fill = True

    def get_clock(self) -> MarketClock:
        return self.clock

    def get_account(self) -> Account:
        return Account(equity=self._equity, cash=self._cash)

    def get_positions(self) -> list[Position]:
        return list(self._positions.values())

    def get_open_orders(self) -> list[Order]:
        return [self._orders[o] for o in self._open]

    def get_order(self, order_id: str) -> Order:
        return self._orders[order_id]

    def submit_order(self, symbol: str, qty: float, side: OrderSide) -> str:
        oid = f"f{self._n}"
        self._n += 1
        if self.fill:
            if side is OrderSide.BUY:
                self._positions[symbol] = Position(symbol, qty, self.price)
                self._cash -= qty * self.price
            else:
                self._positions.pop(symbol, None)
                self._cash += qty * self.price
            self._orders[oid] = Order(oid, symbol, qty, side, OrderStatus.FILLED,
                                      submitted_at=self.clock.timestamp,
                                      filled_at=self.clock.timestamp, fill_price=self.price)
        else:
            self._orders[oid] = Order(oid, symbol, qty, side, OrderStatus.SUBMITTED,
                                      submitted_at=self.clock.timestamp)
            self._open.append(oid)
        return oid

    # test helper: seed a held position as if entered earlier
    def plant_position(self, symbol: str, qty: float, avg: float) -> None:
        self._positions[symbol] = Position(symbol, qty, avg)


class StubFeed:
    def __init__(self, bars: pd.DataFrame):
        self.bars = bars

    def closed_bars(self, asof: pd.Timestamp) -> pd.DataFrame:
        return self.bars


class StubSource:
    def __init__(self, direction: Direction, reason: str = "rule"):
        self.direction = direction
        self.reason = reason
        self.prepared = False

    def prepare(self, bars: pd.DataFrame) -> None:
        self.prepared = True

    def signal_at(self, asof: pd.Timestamp) -> Signal:
        return Signal(asof=pd.Timestamp(asof), symbol="QQQ", direction=self.direction,
                      diagnostics={"reason": self.reason})


# ---------- helpers ----------

def session_bars(n: int, date: str = "2024-03-04", price: float = 100.0) -> pd.DataFrame:
    ts = pd.date_range(f"{date} 09:30", periods=n, freq="1min", tz=NY).tz_convert("UTC")
    return pd.DataFrame({"ts": ts, "open": price, "high": price + 0.5,
                         "low": price - 0.5, "close": price, "volume": 1000.0})


def clock(now: pd.Timestamp, minutes_to_close: float, is_open: bool = True) -> MarketClock:
    return MarketClock(is_open=is_open, timestamp=now,
                       next_open=now + pd.Timedelta(hours=20),
                       next_close=now + pd.Timedelta(minutes=minutes_to_close))


def cfg(**over) -> Config:
    base = Config(signal_source="candles", horizon=3, position_size=0.05,
                  force_flat_minutes_before_close=5)
    return dataclasses.replace(base, **over)


def memconn():
    return dbm.connect(":memory:")


def make_survivor(conn, c: Config) -> None:
    conn.execute(
        "INSERT INTO test_evaluations (config_hash, invoked_at, verdict) VALUES (?, ?, 'SURVIVED')",
        (c.config_hash, dbm.utcnow()),
    )
    conn.commit()


def build_loop(c: Config, broker: FakeBroker, bars: pd.DataFrame, conn,
               direction: Direction = Direction.LONG, require_survivor: bool = False) -> TradeLoop:
    loop = TradeLoop(c, broker, StubFeed(bars), conn, require_survivor=require_survivor)
    loop.source = StubSource(direction)            # type: ignore[assignment]
    return loop


# ---------- survivor gate ----------

def test_loop_refuses_non_survivor():
    conn = memconn()
    c = cfg()
    with pytest.raises(NotASurvivorError):
        TradeLoop(c, FakeBroker(clock(pd.Timestamp("2024-03-04 15:00", tz=NY), 60)),
                  StubFeed(session_bars(20)), conn, require_survivor=True)


def test_loop_arms_for_survivor():
    conn = memconn()
    c = cfg()
    make_survivor(conn, c)
    loop = TradeLoop(c, FakeBroker(clock(pd.Timestamp("2024-03-04 15:00", tz=NY), 60)),
                     StubFeed(session_bars(20)), conn, require_survivor=True)
    assert loop.symbol == "QQQ"


# ---------- entry: sizing + one-at-a-time ----------

def test_enters_long_and_sizes_5pct_whole_shares():
    conn = memconn()
    c = cfg()
    bars = session_bars(20)
    now = pd.Timestamp(bars["ts"].iloc[-1]) + pd.Timedelta(minutes=1)
    broker = FakeBroker(clock(now, 120), equity=100_000.0, price=100.0)
    loop = build_loop(c, broker, bars, conn, Direction.LONG)

    res = loop.step()
    assert res.action == "entry"
    pos = broker.get_positions()[0]
    assert pos.qty == 50          # floor(100000 * 0.05 / 100)


def test_does_not_open_second_position():
    conn = memconn()
    c = cfg()
    bars = session_bars(20)
    now = pd.Timestamp(bars["ts"].iloc[-1]) + pd.Timedelta(minutes=1)
    broker = FakeBroker(clock(now, 120), price=100.0)
    loop = build_loop(c, broker, bars, conn, Direction.LONG)

    loop.step()                                   # entry
    res = loop.step()                             # now holding
    assert res.action == "hold"
    assert len(broker.get_positions()) == 1


def test_no_trade_signal_does_not_enter():
    conn = memconn()
    c = cfg()
    bars = session_bars(20)
    now = pd.Timestamp(bars["ts"].iloc[-1]) + pd.Timedelta(minutes=1)
    broker = FakeBroker(clock(now, 120))
    loop = build_loop(c, broker, bars, conn, Direction.NO_TRADE)
    res = loop.step()
    assert res.action.startswith("no_trade")
    assert broker.get_positions() == []


def test_pending_entry_blocks_resubmit():
    conn = memconn()
    c = cfg()
    bars = session_bars(20)
    now = pd.Timestamp(bars["ts"].iloc[-1]) + pd.Timedelta(minutes=1)
    broker = FakeBroker(clock(now, 120))
    broker.fill = False                           # entry stays open/unfilled
    loop = build_loop(c, broker, bars, conn, Direction.LONG)
    assert loop.step().action == "entry"
    # still flat (unfilled) but an open BUY exists → must not submit again
    loop._last_decided_ts = None                  # allow a fresh decision this bar
    assert loop.step().action == "entry_pending"


# ---------- exits ----------

def test_time_stop_exit_after_horizon():
    conn = memconn()
    c = cfg(horizon=3)
    bars = session_bars(20)
    entry_ts = pd.Timestamp(bars["ts"].iloc[10])
    # seed the journal as if we entered at bar 10
    sig = Signal(asof=entry_ts, symbol="QQQ", direction=Direction.LONG, diagnostics={"reason": "rule"})
    sid = journal.record_signal(conn, None, c.config_hash, sig)
    journal.record_order(conn, None, sid, "QQQ", 50, OrderSide.BUY, "entry", "x1", entry_ts)

    now = pd.Timestamp(bars["ts"].iloc[-1]) + pd.Timedelta(minutes=1)   # many bars later
    broker = FakeBroker(clock(now, 120), price=100.0)
    broker.plant_position("QQQ", 50, 100.0)
    loop = build_loop(c, broker, bars, conn, Direction.LONG)

    res = loop.step()
    assert res.action == "exit:time_stop"
    assert broker.get_positions() == []


def test_no_exit_before_horizon():
    conn = memconn()
    c = cfg(horizon=3)
    bars = session_bars(20)
    entry_ts = pd.Timestamp(bars["ts"].iloc[-2])   # only 1 bar elapsed at the last bar
    sig = Signal(asof=entry_ts, symbol="QQQ", direction=Direction.LONG, diagnostics={"reason": "rule"})
    sid = journal.record_signal(conn, None, c.config_hash, sig)
    journal.record_order(conn, None, sid, "QQQ", 50, OrderSide.BUY, "entry", "x1", entry_ts)

    now = pd.Timestamp(bars["ts"].iloc[-1]) + pd.Timedelta(minutes=1)
    broker = FakeBroker(clock(now, 120), price=100.0)
    broker.plant_position("QQQ", 50, 100.0)
    loop = build_loop(c, broker, bars, conn, Direction.LONG)

    assert loop.step().action == "hold"
    assert len(broker.get_positions()) == 1


def test_exit_pending_blocks_resubmit():
    conn = memconn()
    c = cfg(horizon=3)
    bars = session_bars(20)
    now = pd.Timestamp(bars["ts"].iloc[-1]) + pd.Timedelta(minutes=1)
    broker = FakeBroker(clock(now, 120), price=100.0)
    broker.plant_position("QQQ", 50, 100.0)
    broker.fill = False
    broker.submit_order("QQQ", 50, OrderSide.SELL)     # an exit already in flight
    loop = build_loop(c, broker, bars, conn, Direction.LONG)
    res = loop.step()
    assert res.action == "exit_pending"
    # no second sell submitted (still exactly one open order)
    assert len(broker.get_open_orders()) == 1


def test_journal_sync_marks_order_and_signal_executed():
    conn = memconn()
    c = cfg()
    ts = pd.Timestamp("2024-03-04 15:00", tz="UTC")
    sig = Signal(asof=ts, symbol="QQQ", direction=Direction.LONG, diagnostics={"reason": "rule"})
    sid = journal.record_signal(conn, None, c.config_hash, sig)
    journal.record_order(conn, None, sid, "QQQ", 50, OrderSide.BUY, "entry", "b1", ts)

    filled = Order("b1", "QQQ", 50, OrderSide.BUY, OrderStatus.FILLED, submitted_at=ts,
                   filled_at=ts, fill_price=100.0, filled_qty=48)   # partial fill
    journal.sync_order(conn, filled)

    o = conn.execute("SELECT status, filled_qty FROM orders WHERE broker_order_id='b1'").fetchone()
    assert o["status"] == "filled" and o["filled_qty"] == 48        # real fill qty, not ordered 50
    s = conn.execute("SELECT status FROM signals WHERE id = ?", (sid,)).fetchone()
    assert s["status"] == "executed"


def test_size_zero_when_equity_too_small():
    conn = memconn()
    c = cfg()
    bars = session_bars(20, price=100.0)
    now = pd.Timestamp(bars["ts"].iloc[-1]) + pd.Timedelta(minutes=1)
    broker = FakeBroker(clock(now, 120), equity=10.0, price=100.0)   # 10*0.05/100 < 1 share
    loop = build_loop(c, broker, bars, conn, Direction.LONG)
    res = loop.step()
    assert res.action == "size_zero"
    assert broker.get_positions() == []


def test_no_price_when_close_nonpositive():
    conn = memconn()
    c = cfg()
    bars = session_bars(20, price=0.0)                 # close == 0
    now = pd.Timestamp(bars["ts"].iloc[-1]) + pd.Timedelta(minutes=1)
    broker = FakeBroker(clock(now, 120), price=100.0)
    loop = build_loop(c, broker, bars, conn, Direction.LONG)
    assert loop.step().action == "no_price"
    assert broker.get_positions() == []


def test_no_bars_while_open_is_noop():
    conn = memconn()
    c = cfg()
    empty = session_bars(0)
    now = pd.Timestamp("2024-03-04 15:00", tz=NY)
    broker = FakeBroker(clock(now, 120))
    loop = build_loop(c, broker, empty, conn, Direction.LONG)
    assert loop.step().action == "no_bars"
    assert broker.get_positions() == []


def test_force_flat_overrides_and_sells():
    conn = memconn()
    c = cfg(horizon=3, force_flat_minutes_before_close=5)
    bars = session_bars(20)
    now = pd.Timestamp(bars["ts"].iloc[-1]) + pd.Timedelta(minutes=1)
    broker = FakeBroker(clock(now, 3), price=100.0)   # 3 min to close < 5
    broker.plant_position("QQQ", 50, 100.0)
    loop = build_loop(c, broker, bars, conn, Direction.LONG)

    res = loop.step()
    assert res.action == "exit:force_flat"
    assert broker.get_positions() == []


def test_no_entry_too_close_to_close():
    conn = memconn()
    c = cfg(horizon=15, force_flat_minutes_before_close=5)
    bars = session_bars(20)
    now = pd.Timestamp(bars["ts"].iloc[-1]) + pd.Timedelta(minutes=1)
    broker = FakeBroker(clock(now, 10), price=100.0)   # 10 <= 5 + 15
    loop = build_loop(c, broker, bars, conn, Direction.LONG)
    assert loop.step().action == "too_close_to_enter"
    assert broker.get_positions() == []


# ---------- restart / idempotency ----------

def test_adopted_position_without_record_holds_to_force_flat():
    conn = memconn()
    c = cfg(horizon=3)
    bars = session_bars(20)
    now = pd.Timestamp(bars["ts"].iloc[-1]) + pd.Timedelta(minutes=1)
    broker = FakeBroker(clock(now, 120), price=100.0)   # plenty of time to close
    broker.plant_position("QQQ", 50, 100.0)             # held, but no journal entry
    loop = build_loop(c, broker, bars, conn, Direction.LONG)

    res = loop.step()
    assert res.action == "hold"
    assert res.detail == "adopted_no_entry_record"
    assert len(broker.get_positions()) == 1            # not doubled, not dropped


def test_adopted_position_force_flats_near_close():
    conn = memconn()
    c = cfg(horizon=3)
    bars = session_bars(20)
    now = pd.Timestamp(bars["ts"].iloc[-1]) + pd.Timedelta(minutes=1)
    broker = FakeBroker(clock(now, 2), price=100.0)
    broker.plant_position("QQQ", 50, 100.0)
    loop = build_loop(c, broker, bars, conn, Direction.LONG)
    assert loop.step().action == "exit:force_flat"


# ---------- closed market / cadence ----------

def test_market_closed_is_noop():
    conn = memconn()
    c = cfg()
    bars = session_bars(20)
    now = pd.Timestamp("2024-03-04 21:30", tz=NY)
    broker = FakeBroker(clock(now, 0, is_open=False))
    loop = build_loop(c, broker, bars, conn, Direction.LONG)
    assert loop.step().action == "market_closed"
    assert broker.get_positions() == []


def test_one_decision_per_closed_bar():
    conn = memconn()
    c = cfg()
    bars = session_bars(20)
    now = pd.Timestamp(bars["ts"].iloc[-1]) + pd.Timedelta(minutes=1)
    broker = FakeBroker(clock(now, 120))
    loop = build_loop(c, broker, bars, conn, Direction.NO_TRADE)
    loop.step()                                   # decides on the last bar
    assert loop.step().action == "already_decided"   # same bar, no second decision


def test_run_stops_after_max_iterations():
    conn = memconn()
    c = cfg()
    bars = session_bars(20)
    now = pd.Timestamp(bars["ts"].iloc[-1]) + pd.Timedelta(minutes=1)
    broker = FakeBroker(clock(now, 120))
    loop = build_loop(c, broker, bars, conn, Direction.NO_TRADE)
    calls = {"n": 0}
    loop.run(max_iterations=3, sleep=lambda s: calls.__setitem__("n", calls["n"] + 1))
    assert calls["n"] == 3
