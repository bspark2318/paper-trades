"""Session-aware window construction.

A window is W consecutive 1-minute log-returns that lie entirely inside one
regular session; its forward return covers the H bars after the window's end,
also required to fit in the same session (else NaN — unusable as evidence).
Windows never see the overnight gap.

Global bar indices count RTH bars only, in time order across all sessions.
They are the coordinate system for both no-lookahead eligibility and dedup.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from patterns.engine.normalize import normalize

NY = "America/New_York"


@dataclass
class WindowSet:
    Z: np.ndarray            # (M, W) normalized float32
    end_idx: np.ndarray      # (M,) global bar index of each window's last bar
    end_ts: np.ndarray       # (M,) UTC timestamps of each window's last bar
    fwd_ret: np.ndarray      # (M,) H-bar forward return from window end; NaN if it leaves the session
    valid: np.ndarray        # (M,) row is normalizable
    window: int
    horizon: int
    bar_ts: np.ndarray       # (N,) all bar timestamps (global index → ts)
    closes: np.ndarray       # (N,) all closes

    @property
    def n_windows(self) -> int:
        return len(self.Z)

    def row_for_ts(self, asof: pd.Timestamp) -> int:
        """Row whose window ends at the latest bar <= asof. Raises if none."""
        asof = pd.Timestamp(asof)
        if asof.tzinfo is not None:
            asof = asof.tz_convert("UTC").tz_localize(None)
        pos = np.searchsorted(self.end_ts, np.datetime64(asof), side="right") - 1
        if pos < 0:
            raise ValueError(f"No window ends at or before {asof}")
        return int(pos)


def build_windows(bars: pd.DataFrame, window: int, horizon: int,
                  normalization: str = "logret_zscore") -> WindowSet:
    """bars: time-ordered RTH bars (ts UTC, close). Sessions inferred from NY dates."""
    # Internally timestamps are UTC-naive datetime64 (numpy-friendly);
    # the matcher converts back to tz-aware UTC at its API boundary.
    ts = bars["ts"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
    closes = bars["close"].to_numpy(dtype=np.float64)
    session_label = bars["ts"].dt.tz_convert(NY).dt.date.to_numpy()

    rows, end_idx, fwd = [], [], []
    start = 0
    n = len(bars)
    for i in range(1, n + 1):
        if i == n or session_label[i] != session_label[start]:
            _collect_session(closes, start, i, window, horizon, rows, end_idx, fwd)
            start = i

    if rows:
        X = np.vstack(rows)
        end_idx_arr = np.asarray(end_idx, dtype=np.int64)
        fwd_arr = np.asarray(fwd, dtype=np.float64)
    else:
        X = np.empty((0, window))
        end_idx_arr = np.empty(0, dtype=np.int64)
        fwd_arr = np.empty(0)

    Z, valid = normalize(normalization, X)
    return WindowSet(
        Z=Z,
        end_idx=end_idx_arr,
        end_ts=ts[end_idx_arr] if len(end_idx_arr) else np.empty(0, dtype="datetime64[ns]"),
        fwd_ret=fwd_arr,
        valid=valid,
        window=window,
        horizon=horizon,
        bar_ts=ts,
        closes=closes,
    )


def _collect_session(closes, lo, hi, window, horizon, rows, end_idx, fwd):
    """Append all windows of one session [lo, hi) to the accumulators."""
    n = hi - lo
    if n - 1 < window:  # need W returns, i.e. W+1 closes
        return
    r = np.diff(np.log(closes[lo:hi]))                       # (n-1,)
    X = np.lib.stride_tricks.sliding_window_view(r, window)  # (n-window, window)
    # row j covers returns r[j .. j+window-1] → ends at local close j+window
    local_end = np.arange(window, n)
    rows.append(X.copy())
    end_idx.extend(lo + local_end)
    # forward return only if the full horizon stays inside this session
    fwd_ok = local_end + horizon <= n - 1
    fwd_vals = np.full(len(local_end), np.nan)
    ok_end = local_end[fwd_ok]
    fwd_vals[fwd_ok] = closes[lo + ok_end + horizon] / closes[lo + ok_end] - 1.0
    fwd.extend(fwd_vals)
