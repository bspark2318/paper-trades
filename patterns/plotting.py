"""Match visualizations: query-vs-matches overlay and forward-return histogram.

All prices are rebased to 1.0 at the window start so different epochs overlay;
x-axis is bars relative to the window end (decision point at 0).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from patterns.engine.matcher import MatchOutcome
from patterns.engine.windows import WindowSet


def plot_match_overlay(ws: WindowSet, outcome: MatchOutcome, path: str | Path, top: int = 10) -> Path:
    """Query window (black) over its top matches; matches continue past 0 into
    their known futures — the grey fan after the line is the evidence."""
    W, H = ws.window, ws.horizon
    q_row = ws.row_for_ts(outcome.query_ts)
    q_end = int(ws.end_idx[q_row])

    fig, ax = plt.subplots(figsize=(10, 6))
    x_full = np.arange(-W, H + 1)
    for ts in outcome.match_ts[:top]:
        e = int(ws.end_idx[ws.row_for_ts(ts)])
        seg = ws.closes[e - W : e + H + 1] / ws.closes[e - W]
        ax.plot(x_full, seg, alpha=0.45, lw=1.0, color="steelblue")

    q_seg = ws.closes[q_end - W : q_end + 1] / ws.closes[q_end - W]
    ax.plot(np.arange(-W, 1), q_seg, color="black", lw=2.5, label="query (now)")

    ax.axvline(0, ls="--", color="gray", lw=1)
    ax.set_xlabel(f"bars relative to decision point (window={W}, horizon={H})")
    ax.set_ylabel("price rebased to window start")
    ax.set_title(f"{outcome.query_ts:%Y-%m-%d %H:%M} UTC — top {min(top, outcome.n)} of {outcome.n} matches")
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_equity(equity_ts: np.ndarray, equity: np.ndarray, path: str | Path,
                title: str = "walk-forward equity") -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity_ts, equity, lw=1.2, color="steelblue")
    ax.set_ylabel("equity ($)")
    ax.set_title(title)
    fig.autofmt_xdate()
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_fwd_histogram(outcome: MatchOutcome, path: str | Path) -> Path:
    """Distribution of what happened next, across all kept matches."""
    fwd_pct = outcome.fwd_ret * 100
    mean = float(np.mean(fwd_pct))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(fwd_pct, bins=min(30, max(5, outcome.n // 2)), color="steelblue", alpha=0.8)
    ax.axvline(0, color="gray", lw=1)
    ax.axvline(mean, color="firebrick", lw=2, label=f"mean {mean:+.3f}%")
    ax.set_xlabel(f"forward return over horizon (%)  [n={outcome.n}]")
    ax.set_ylabel("matches")
    ax.set_title(f"{outcome.query_ts:%Y-%m-%d %H:%M} UTC — forward-return distribution")
    ax.legend()
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
