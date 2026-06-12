"""Walk-forward backtest: drive a SignalSource through history, one bar at a time.

Honesty rules enforced here:
- decisions at bar t, fills at bar t+1's open (BacktestAdapter);
- entries only when flat (one position), only after min_history_bars warm-up,
  and only when the full horizon fits in the remaining session — the validated
  hypothesis is "hold H bars", so we never enter a trade that can't be it;
- exit is a time stop: sell submitted so it fills exactly H bars after entry.

The source sees the full bars at prepare() time but answers signal_at(t) under
its own no-lookahead mask; the perturbation property test in tests/ proves the
combination leaks nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from patterns.broker.backtest import BacktestAdapter
from patterns.broker.protocols import OrderSide, OrderStatus
from patterns.config import Config
from patterns.strategy.base import Direction, Signal, make_source
from patterns.validate import stats

NY = "America/New_York"
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class TradeRecord:
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_price: float          # cost-adjusted fill
    exit_price: float
    qty: float
    net_ret: float              # exit/entry - 1, costs included in both fills
    exit_reason: str            # "time_stop" | "force_flat"


@dataclass
class WalkForwardResult:
    trades: list[TradeRecord]
    signals: list[Signal]       # every query made, including NO_TRADEs
    equity_ts: np.ndarray
    equity: np.ndarray
    metrics: dict = field(default_factory=dict)


def _bars_left_in_session(bars: pd.DataFrame) -> np.ndarray:
    """bars_left[i] = bars after i in i's session."""
    session = pd.Series(bars["ts"].dt.tz_convert(NY).dt.date)
    last_pos = session.groupby(session).transform(lambda s: s.index[-1]).to_numpy()
    return (last_pos - np.arange(len(bars))).astype(np.int64)


def run_walkforward(cfg: Config, bars: pd.DataFrame, cash: float = 100_000.0,
                    min_query_idx: int | None = None) -> WalkForwardResult:
    """min_query_idx restricts queries to bars at/after that index (still subject
    to min_history_bars) — the evaluate gate uses it to trade the test period
    only, while the matcher keeps the full prior history as evidence."""
    source = make_source(cfg)
    source.prepare(bars)
    bt = BacktestAdapter(bars, cash=cash, cost_bps=cfg.cost_bps)
    bars_left = _bars_left_in_session(bars)
    ts = bars["ts"].to_numpy()
    first_query = max(cfg.min_history_bars, min_query_idx or 0)

    signals: list[Signal] = []
    trades: list[TradeRecord] = []
    equity = np.empty(len(bars))

    entry_order_id: str | None = None
    exit_order_id: str | None = None
    exit_due: int = -1          # bar index whose open should be the exit fill

    while True:
        i = bt.index
        equity[i] = bt.get_account().equity

        holding = bool(bt.get_positions())
        if holding and exit_order_id is None and i == exit_due - 1:
            pos = bt.get_positions()[0]
            exit_order_id = bt.submit_order(pos.symbol, pos.qty, OrderSide.SELL)
        elif (
            not holding
            and entry_order_id is None
            and i >= first_query
            and i % cfg.query_stride == 0
            and bars_left[i] >= cfg.horizon + 1     # entry at i+1, exit at i+1+H, same session
        ):
            sig = source.signal_at(pd.Timestamp(ts[i]))
            signals.append(sig)
            if sig.direction is Direction.LONG:
                account = bt.get_account()
                qty = account.equity * cfg.position_size / bt._close[i]
                entry_order_id = bt.submit_order(cfg.symbols[0], qty, OrderSide.BUY)

        if not bt.advance():
            break

        if entry_order_id is not None:
            entry = bt.get_order(entry_order_id)
            if entry.status is OrderStatus.FILLED and exit_due < 0:
                exit_due = bt.index + cfg.horizon
            elif entry.status is OrderStatus.REJECTED:
                entry_order_id = None

        if entry_order_id is not None and not bt.get_positions() and exit_due >= 0:
            # position is gone: either our time stop filled or the safety net fired
            entry = bt.get_order(entry_order_id)
            if exit_order_id is not None and bt.get_order(exit_order_id).status is OrderStatus.FILLED:
                exit_o, reason = bt.get_order(exit_order_id), "time_stop"
            else:
                flats = [o for o in bt._orders.values()
                         if o.reason == "force_flat" and o.status is OrderStatus.FILLED]
                exit_o, reason = flats[-1], "force_flat"
            assert entry.fill_price is not None and exit_o.fill_price is not None
            trades.append(TradeRecord(
                entry_ts=entry.filled_at, exit_ts=exit_o.filled_at,  # type: ignore[arg-type]
                entry_price=entry.fill_price, exit_price=exit_o.fill_price,
                qty=entry.qty, net_ret=exit_o.fill_price / entry.fill_price - 1.0,
                exit_reason=reason,
            ))
            entry_order_id = exit_order_id = None
            exit_due = -1

    result = WalkForwardResult(
        trades=trades, signals=signals,
        equity_ts=ts, equity=equity,
    )
    result.metrics = _metrics(result, bars, cash)
    return result


def _metrics(r: WalkForwardResult, bars: pd.DataFrame, cash: float) -> dict:
    rets = np.array([t.net_ret for t in r.trades])
    n_sessions = bars["ts"].dt.tz_convert(NY).dt.date.nunique()
    years = max(n_sessions / TRADING_DAYS_PER_YEAR, 1e-9)
    return {
        "n_signals": len(r.signals),
        "n_long_signals": sum(1 for s in r.signals if s.direction is Direction.LONG),
        "n_trades": len(r.trades),
        "n_force_flat": sum(1 for t in r.trades if t.exit_reason == "force_flat"),
        "hit_rate": stats.hit_rate(rets),
        "mean_net_ret": float(np.mean(rets)) if len(rets) else float("nan"),
        "total_return": float(r.equity[-1] / cash - 1.0),
        "sharpe": stats.sharpe(rets, periods_per_year=len(rets) / years),
        "max_drawdown": stats.max_drawdown(r.equity),
        "n_sessions": int(n_sessions),
    }
