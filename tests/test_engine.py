import numpy as np
import pandas as pd
import pytest

from patterns.engine import matcher
from patterns.engine.dedup import dedup_matches
from patterns.engine.normalize import normalize
from patterns.engine.windows import build_windows
from tests.conftest import make_multi_session_bars, make_session_bars

W, H = 5, 3


def small_bars(n_sessions=2, n_bars=30):
    return make_multi_session_bars([f"2024-03-{4 + i:02d}" for i in range(n_sessions)], n_bars=n_bars)


# ---------- windows ----------

def test_window_alignment_hand_computed():
    bars = make_session_bars("2024-03-04", n_bars=12)
    ws = build_windows(bars, window=W, horizon=H)
    closes = bars["close"].to_numpy()
    # 12 bars -> 11 returns -> 7 windows ending at bars 5..11
    assert ws.n_windows == 7
    assert list(ws.end_idx) == list(range(5, 12))
    # forward return of first window: close[5+3]/close[5]-1
    assert ws.fwd_ret[0] == pytest.approx(closes[8] / closes[5] - 1)
    # windows ending at bars 9,10,11 have horizon beyond bar 11 -> NaN
    assert np.isnan(ws.fwd_ret[-3:]).all()
    assert not np.isnan(ws.fwd_ret[:-3]).any()


def test_windows_never_cross_sessions():
    bars = small_bars(n_sessions=2, n_bars=30)
    ws = build_windows(bars, window=W, horizon=H)
    label = bars["ts"].dt.tz_convert("America/New_York").dt.date.to_numpy()
    for row in range(ws.n_windows):
        e = ws.end_idx[row]
        # window spans closes [e-W, e]: all in one session
        assert label[e - W] == label[e]
        # forward return, when defined, stays in that session too
        if not np.isnan(ws.fwd_ret[row]):
            assert label[e + H] == label[e]


def test_fwd_nan_at_session_close_boundary():
    bars = small_bars(n_sessions=2, n_bars=30)
    ws = build_windows(bars, window=W, horizon=H)
    per_session_nan = 0
    for row in range(ws.n_windows):
        if np.isnan(ws.fwd_ret[row]):
            per_session_nan += 1
    assert per_session_nan == 2 * H  # last H windows of each session


# ---------- normalization ----------

def test_price_level_invariance():
    bars = make_session_bars("2024-03-04", n_bars=50)
    scaled = bars.copy()
    for col in ("open", "high", "low", "close"):
        scaled[col] = scaled[col] * 100.0
    z1, _ = normalize("logret_zscore", np.diff(np.log(bars["close"].to_numpy()))[None, :])
    z2, _ = normalize("logret_zscore", np.diff(np.log(scaled["close"].to_numpy()))[None, :])
    np.testing.assert_allclose(z1, z2, rtol=1e-5)


def test_volatility_invariance_zscore_only():
    rng = np.random.default_rng(0)
    shape = rng.normal(0, 1, 20)
    calm, wild = (0.001 * shape)[None, :], (0.05 * shape)[None, :]
    z_calm, _ = normalize("logret_zscore", calm)
    z_wild, _ = normalize("logret_zscore", wild)
    np.testing.assert_allclose(z_calm, z_wild, rtol=1e-4)
    r_calm, _ = normalize("logret_raw", calm)
    r_wild, _ = normalize("logret_raw", wild)
    assert not np.allclose(r_calm, r_wild)


def test_zero_variance_window_marked_invalid():
    X = np.vstack([np.zeros(10), np.random.default_rng(0).normal(size=10)])
    _, valid = normalize("logret_zscore", X)
    assert list(valid) == [False, True]


# ---------- matcher ----------

def plant_motif(bars: pd.DataFrame, at: int, motif: np.ndarray) -> pd.DataFrame:
    """Overwrite closes so the W returns ending at bar `at` equal `motif`."""
    closes = bars["close"].to_numpy().copy()
    for j, r in enumerate(motif):
        idx = at - len(motif) + 1 + j
        closes[idx] = closes[idx - 1] * np.exp(r)
    out = bars.copy()
    out["close"] = closes
    return out


def test_planted_motif_retrieved_rank_one():
    bars = make_multi_session_bars([f"2024-03-{d:02d}" for d in (4, 5, 6, 7, 8)], n_bars=60)
    motif = np.array([0.004, -0.003, 0.005, -0.002, 0.006])
    bars = plant_motif(bars, at=30, motif=motif)            # historical occurrence (session 1)
    bars = plant_motif(bars, at=260, motif=motif * 2.0)     # query: same shape, double scale (session 5)
    ws = build_windows(bars, window=W, horizon=H)
    out = matcher.query(ws, pd.Timestamp(bars["ts"].iloc[260]), k=10, dedup_gap=2)
    assert out.match_ts[0] == pd.Timestamp(bars["ts"].iloc[30])
    assert out.distance[0] == pytest.approx(0.0, abs=1e-3)


def test_distance_matches_brute_force():
    bars = small_bars(n_sessions=3, n_bars=40)
    ws = build_windows(bars, window=W, horizon=H)
    q_row = ws.n_windows - 1
    out = matcher.query(ws, pd.Timestamp(ws.end_ts[q_row]), k=5, dedup_gap=1)
    mask = matcher.eligible_mask(ws, q_row)
    brute = np.linalg.norm(ws.Z[mask] - ws.Z[q_row], axis=1)
    assert out.distance[0] == pytest.approx(np.sort(brute)[0], rel=1e-5)


def test_no_lookahead_future_match_excluded():
    dates = [f"2024-03-{d:02d}" for d in (4, 5, 6, 7, 8)]
    bars = make_multi_session_bars(dates, n_bars=60)
    motif = np.array([0.004, -0.003, 0.005, -0.002, 0.006])
    bars = plant_motif(bars, at=120, motif=motif)   # query bar (session 3)
    bars = plant_motif(bars, at=200, motif=motif)   # perfect match in the FUTURE (session 4)
    ws = build_windows(bars, window=W, horizon=H)
    out = matcher.query(ws, pd.Timestamp(bars["ts"].iloc[120]), k=10, dedup_gap=2)
    future_ts = pd.Timestamp(bars["ts"].iloc[200])
    assert future_ts not in out.match_ts


def test_no_lookahead_inequality_holds_for_all_matches():
    bars = small_bars(n_sessions=3, n_bars=50)
    ws = build_windows(bars, window=W, horizon=H)
    q_row = ws.n_windows - 1
    q_end = ws.end_idx[q_row]
    out = matcher.query(ws, pd.Timestamp(ws.end_ts[q_row]), k=20, dedup_gap=1)
    ts_to_end = {pd.Timestamp(t, tz="UTC"): e for t, e in zip(ws.end_ts, ws.end_idx)}
    for t in out.match_ts:
        assert ts_to_end[t] + H <= q_end - W


def test_match_too_close_to_query_excluded():
    """A window just before the query whose OUTCOME overlaps the query window is illegal."""
    bars = make_session_bars("2024-03-04", n_bars=60)
    ws = build_windows(bars, window=W, horizon=H)
    q_row = ws.n_windows - 1
    q_end = ws.end_idx[q_row]
    mask = matcher.eligible_mask(ws, q_row)
    illegal = (ws.end_idx + H > q_end - W)
    assert not (mask & illegal).any()


# ---------- dedup ----------

def test_dedup_collapses_cluster_to_best():
    ends = np.array([100, 101, 99, 300, 102])  # best-first order
    kept = dedup_matches(ends, gap=10)
    assert list(ends[kept]) == [100, 300]


def test_dedup_exact_gap_boundary():
    assert list(dedup_matches(np.array([100, 110]), gap=10)) == [0]      # 10 apart: same event
    assert list(dedup_matches(np.array([100, 111]), gap=10)) == [0, 1]   # 11 apart: distinct


def test_dedup_limit():
    ends = np.arange(0, 1000, 50)
    assert len(dedup_matches(ends, gap=10, limit=5)) == 5
