import numpy as np
import pandas as pd
import pytest

from patterns.validate.baselines import buy_and_hold, random_baseline
from tests.conftest import make_multi_session_bars, make_session_bars

H = 3


def bars_and_decisions(n_sessions: int = 4, n_bars: int = 40):
    dates = [f"2024-03-{d:02d}" for d in range(4, 4 + n_sessions)]
    bars = make_multi_session_bars(dates, n_bars=n_bars)
    # decisions at bar 10 and 20 of the last two sessions
    idx = [(n_sessions - 2) * n_bars + 10, (n_sessions - 1) * n_bars + 20]
    return bars, [pd.Timestamp(bars["ts"].iloc[i]) for i in idx]


def test_seeded_reproducibility():
    bars, dec = bars_and_decisions()
    a = random_baseline(bars, dec, 0.001, H, 5.0, 0, n_resamples=50, seed=7)
    b = random_baseline(bars, dec, 0.001, H, 5.0, 0, n_resamples=50, seed=7)
    np.testing.assert_array_equal(a.random_means, b.random_means)
    assert a.p_value == b.p_value
    c = random_baseline(bars, dec, 0.001, H, 5.0, 0, n_resamples=50, seed=8)
    assert not np.array_equal(a.random_means, c.random_means)


def jump_bars(n_sessions: int = 3, n_bars: int = 40, jump_at: int = 21) -> pd.DataFrame:
    """Flat at 100 all day except a +3 jump over bars [jump_at, jump_at+2],
    same minute every session. Only entries at that minute catch the move."""
    frames = []
    for s in range(n_sessions):
        start = pd.Timestamp(f"2024-03-{4 + s:02d} 09:30", tz="America/New_York")
        ts = pd.date_range(start, periods=n_bars, freq="1min").tz_convert("UTC")
        close = np.full(n_bars, 100.0)
        close[jump_at] = 101.0
        close[jump_at + 1] = 102.0
        close[jump_at + 2 :] = 103.0
        open_ = np.concatenate([[100.0], close[:-1]])
        frames.append(pd.DataFrame({
            "ts": ts, "open": open_, "high": close, "low": open_,
            "close": close, "volume": 1000.0,
        }))
    return pd.concat(frames, ignore_index=True)


def test_time_of_day_matching():
    """Random entries must be drawn from the strategy's decision minutes —
    decisions at the jump minute always catch the jump, decisions at a flat
    minute never do. A non-TOD-matched sampler would mix the two."""
    bars = jump_bars()
    n_bars = 40
    at_jump = [pd.Timestamp(bars["ts"].iloc[s * n_bars + 20]) for s in (1, 2)]
    at_flat = [pd.Timestamp(bars["ts"].iloc[s * n_bars + 5]) for s in (1, 2)]

    jump = random_baseline(bars, at_jump, 0.0, H, 0.0, 0, n_resamples=30, seed=1)
    # decision bar 20 → entry open[21]=100, exit open[24]=103 in every session
    np.testing.assert_allclose(jump.random_means, 0.03)

    flat = random_baseline(bars, at_flat, 0.0, H, 0.0, 0, n_resamples=30, seed=1)
    np.testing.assert_allclose(flat.random_means, 0.0)


def test_p_value_bounds_and_addone():
    bars, dec = bars_and_decisions()
    # impossible strategy mean → no resample beats it → p = 1/(n+1), never 0
    hi = random_baseline(bars, dec, 10.0, H, 5.0, 0, n_resamples=49, seed=3)
    assert hi.p_value == pytest.approx(1 / 50)
    # hopeless strategy mean → all resamples beat it → p = 1
    lo = random_baseline(bars, dec, -10.0, H, 5.0, 0, n_resamples=49, seed=3)
    assert lo.p_value == 1.0


def test_costs_lower_random_means():
    bars, dec = bars_and_decisions()
    free = random_baseline(bars, dec, 0.0, H, 0.0, 0, n_resamples=50, seed=5)
    costly = random_baseline(bars, dec, 0.0, H, 50.0, 0, n_resamples=50, seed=5)
    assert costly.random_means.mean() < free.random_means.mean()


def test_buy_and_hold_hand_computed():
    bars = make_session_bars("2024-03-04", n_bars=20)
    bh = buy_and_hold(bars, cost_bps=0.0)
    expected = bars["close"].iloc[-1] / bars["open"].iloc[0] - 1.0
    assert bh.total_return == pytest.approx(expected)
    assert bh.max_drawdown >= 0.0


def test_buy_and_hold_costs_bite():
    bars = make_session_bars("2024-03-04", n_bars=20)
    assert buy_and_hold(bars, cost_bps=10.0).total_return < buy_and_hold(bars, 0.0).total_return
