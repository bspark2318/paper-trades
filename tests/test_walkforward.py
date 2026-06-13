import dataclasses

import numpy as np
import pandas as pd
import pytest

from patterns.config import Config
from patterns.strategy.base import Direction
from patterns.validate import stats
from patterns.validate.walkforward import run_walkforward
from tests.conftest import make_multi_session_bars
from tests.test_strategy import MOTIF, force_rise

W, H = 5, 3


def wf_cfg(**over) -> Config:
    base = Config(window=W, horizon=H, k=5, dedup_gap=2, min_matches=3,
                  min_history_bars=50, p_threshold=0.65, t_multiplier=1.5,
                  query_stride=1, cost_bps=0.0)
    return dataclasses.replace(base, **over)


def bullish_bars(n_sessions: int = 6, n_bars: int = 40) -> pd.DataFrame:
    from tests.test_engine import plant_motif

    from tests.conftest import rebuild_ohlc_from_closes

    dates = [f"2024-03-{d:02d}" for d in range(4, 4 + n_sessions)]
    bars = make_multi_session_bars(dates, n_bars=n_bars)
    for s in range(n_sessions):
        at = s * n_bars + 20
        bars = plant_motif(bars, at=at, motif=MOTIF)
        bars = force_rise(bars, after=at, horizon=H)
    return rebuild_ohlc_from_closes(bars)


# ---------- stats ----------

def test_sharpe_hand_computed():
    r = np.array([0.01, 0.02, -0.01, 0.03])
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert stats.sharpe(r, 252) == pytest.approx(expected)
    assert np.isnan(stats.sharpe(np.array([0.01]), 252))         # too few
    assert np.isnan(stats.sharpe(np.array([0.01, 0.01]), 252))   # zero variance


def test_max_drawdown_hand_computed():
    eq = np.array([100.0, 120.0, 90.0, 110.0, 80.0])
    assert stats.max_drawdown(eq) == pytest.approx((120 - 80) / 120)
    assert stats.max_drawdown(np.array([100.0, 110.0, 120.0])) == 0.0


# ---------- walk-forward ----------

def test_walkforward_takes_trades_on_planted_pattern():
    res = run_walkforward(wf_cfg(), bullish_bars())
    assert res.metrics["n_trades"] >= 1
    assert res.metrics["n_force_flat"] == 0           # entry guard leaves room for the horizon
    for t in res.trades:
        assert t.exit_reason == "time_stop"
        assert t.exit_ts > t.entry_ts


def test_trades_never_overlap():
    res = run_walkforward(wf_cfg(), bullish_bars())
    for prev, nxt in zip(res.trades, res.trades[1:]):
        assert nxt.entry_ts >= prev.exit_ts


def test_no_signals_before_warmup():
    bars = bullish_bars()
    res = run_walkforward(wf_cfg(min_history_bars=100), bars)
    first_allowed = pd.Timestamp(bars["ts"].iloc[100])
    assert all(s.asof >= first_allowed for s in res.signals)


def test_costs_reduce_returns():
    bars = bullish_bars()
    free = run_walkforward(wf_cfg(cost_bps=0.0), bars)
    costly = run_walkforward(wf_cfg(cost_bps=20.0), bars)
    assert costly.metrics["n_trades"] == free.metrics["n_trades"] >= 1
    assert costly.metrics["mean_net_ret"] < free.metrics["mean_net_ret"]


def test_perturbing_future_does_not_change_past_signals():
    """THE no-lookahead property test: change everything after T, nothing
    before T may move — signals, directions, diagnostics."""
    bars = bullish_bars()
    cutoff_idx = len(bars) // 2
    cutoff_ts = pd.Timestamp(bars["ts"].iloc[cutoff_idx])

    perturbed = bars.copy()
    rng = np.random.default_rng(99)
    scale = 1 + rng.normal(0, 0.01, len(bars) - cutoff_idx)
    for col in ("open", "high", "low", "close"):
        perturbed.loc[cutoff_idx:, col] = perturbed.loc[cutoff_idx:, col].to_numpy() * scale

    cfg = wf_cfg()
    a = run_walkforward(cfg, bars)
    b = run_walkforward(cfg, perturbed)

    sig_a = [s for s in a.signals if s.asof < cutoff_ts]
    sig_b = [s for s in b.signals if s.asof < cutoff_ts]
    assert len(sig_a) == len(sig_b) >= 1
    for x, y in zip(sig_a, sig_b):
        assert x.asof == y.asof
        assert x.direction == y.direction
        assert x.diagnostics == y.diagnostics

    trades_a = [t for t in a.trades if t.exit_ts < cutoff_ts]
    trades_b = [t for t in b.trades if t.exit_ts < cutoff_ts]
    assert trades_a == trades_b


def test_long_signals_lead_to_trades():
    res = run_walkforward(wf_cfg(), bullish_bars())
    # every LONG signal while flat became a trade (one-position rule may skip none here
    # because queries only happen when flat)
    assert res.metrics["n_trades"] == res.metrics["n_long_signals"]
