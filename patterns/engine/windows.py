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
from jaxtyping import Bool, Float, Int, Shaped

from patterns.engine.normalize import normalize

NY = "America/New_York"


@dataclass
class WindowSet:
    """M windows over N bars; D = window * features-per-bar values per shape."""

    Z: Float[np.ndarray, "M D"]        # normalized float32 shape matrix
    end_idx: Int[np.ndarray, " M"]     # global bar index of each window's last bar
    end_ts: Shaped[np.ndarray, " M"]   # UTC-naive datetime64 of each window's last bar
    fwd_ret: Float[np.ndarray, " M"]   # H-bar forward return; NaN if it leaves the session
    valid: Bool[np.ndarray, " M"]      # row is normalizable
    window: int
    horizon: int
    bar_ts: Shaped[np.ndarray, " N"]   # all bar timestamps (global index → ts)
    closes: Float[np.ndarray, " N"]    # all closes

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


FEATURES_PER_BAR = {"close": 1, "ohlc": 4}


def build_windows(bars: pd.DataFrame, window: int, horizon: int,
                  normalization: str = "logret_zscore", features: str = "close") -> WindowSet:
    """bars: time-ordered RTH bars (ts UTC, ohlc). Sessions inferred from NY dates.

    features="close": each bar contributes one log-return → W values per window.
    features="ohlc": each bar contributes log(o/h/l/c vs previous close) → 4W values;
    wicks and bodies become part of the shape.
    """
    if features not in FEATURES_PER_BAR:
        raise ValueError(f"Unknown features {features!r}; available: {sorted(FEATURES_PER_BAR)}")
    # Internally timestamps are UTC-naive datetime64 (numpy-friendly);
    # the matcher converts back to tz-aware UTC at its API boundary.
    ts = bars["ts"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
    ohlc = {col: bars[col].to_numpy(dtype=np.float64) for col in ("open", "high", "low", "close")}
    closes = ohlc["close"]
    session_label = bars["ts"].dt.tz_convert(NY).dt.date.to_numpy()

    rows: list[np.ndarray] = []
    end_idx: list[int] = []
    fwd: list[float] = []
    start = 0
    n = len(bars)
    for i in range(1, n + 1):
        if i == n or session_label[i] != session_label[start]:
            _collect_session(ohlc, start, i, window, horizon, features, rows, end_idx, fwd)
            start = i

    if rows:
        X = np.vstack(rows)
        end_idx_arr = np.asarray(end_idx, dtype=np.int64)
        fwd_arr = np.asarray(fwd, dtype=np.float64)
    else:
        X = np.empty((0, window * FEATURES_PER_BAR[features]))
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


def _collect_session(
    ohlc: dict[str, np.ndarray],
    lo: int,
    hi: int,
    window: int,
    horizon: int,
    features: str,
    rows: list[np.ndarray],
    end_idx: list[int],
    fwd: list[float],
) -> None:
    """Append all windows of one session [lo, hi) to the accumulators."""
    closes = ohlc["close"]
    n = hi - lo
    if n - 1 < window:  # need W returns, i.e. W+1 closes
        return
    prev_close = closes[lo:hi - 1]
    if features == "close":
        F = np.log(closes[lo + 1:hi] / prev_close)[:, None]   # (n-1, 1)
    else:  # ohlc: each bar located relative to the previous close
        F = np.stack(
            [np.log(ohlc[c][lo + 1:hi] / prev_close) for c in ("open", "high", "low", "close")],
            axis=1,
        )                                                     # (n-1, 4)
    # all windows of `window` consecutive bars; flatten bar-major → (n-window, window*nf)
    X = np.lib.stride_tricks.sliding_window_view(F, window, axis=0)  # (n-window, nf, window)
    X = X.transpose(0, 2, 1).reshape(X.shape[0], -1)
    # row j covers bars j+1 .. j+window → ends at local close j+window
    local_end = np.arange(window, n)
    rows.append(np.ascontiguousarray(X))
    end_idx.extend(lo + local_end)
    # forward return only if the full horizon stays inside this session
    fwd_ok = local_end + horizon <= n - 1
    fwd_vals = np.full(len(local_end), np.nan)
    ok_end = local_end[fwd_ok]
    fwd_vals[fwd_ok] = closes[lo + ok_end + horizon] / closes[lo + ok_end] - 1.0
    fwd.extend(fwd_vals)
