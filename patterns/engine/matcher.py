"""k-NN matcher over a WindowSet.

No-lookahead eligibility for a query window ending at global bar q:
a candidate ending at e may be used only if its entire forward-return
period was observable before the query window began:

    e + H <= q - W

Combined with the session rule (fwd_ret is NaN unless the horizon fits
inside the candidate's own session), this guarantees no information from
the query's present or future ever enters the match set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from patterns.engine.dedup import dedup_matches
from patterns.engine.windows import WindowSet

CANDIDATE_BUFFER_FACTOR = 5  # take 5k nearest pre-dedup so dedup losses don't starve k


def _utc(t) -> pd.Timestamp:
    return pd.Timestamp(t, tz="UTC")


@dataclass
class MatchOutcome:
    query_ts: pd.Timestamp
    match_ts: list          # kept matches, best first
    distance: np.ndarray
    fwd_ret: np.ndarray
    n_candidates: int       # eligible windows before top-k/dedup

    @property
    def n(self) -> int:
        return len(self.match_ts)

    def stats(self) -> dict:
        if self.n == 0:
            return {"n": 0, "mean": np.nan, "median": np.nan, "pct_positive": np.nan}
        return {
            "n": self.n,
            "mean": float(np.mean(self.fwd_ret)),
            "median": float(np.median(self.fwd_ret)),
            "pct_positive": float(np.mean(self.fwd_ret > 0)),
        }


def eligible_mask(ws: WindowSet, q_row: int) -> np.ndarray:
    """Candidates legal for a query at row q_row (no lookahead, usable fwd, normalizable)."""
    q_end = ws.end_idx[q_row]
    return (
        ws.valid
        & ~np.isnan(ws.fwd_ret)
        & (ws.end_idx + ws.horizon <= q_end - ws.window)
    )


def query(ws: WindowSet, asof: pd.Timestamp, k: int, dedup_gap: int) -> MatchOutcome:
    q_row = ws.row_for_ts(asof)
    if not ws.valid[q_row]:
        raise ValueError(f"Query window at {asof} is degenerate (zero variance)")
    mask = eligible_mask(ws, q_row)
    cand = np.flatnonzero(mask)
    if len(cand) == 0:
        return MatchOutcome(_utc(ws.end_ts[q_row]), [], np.empty(0), np.empty(0), 0)

    diff = ws.Z[cand] - ws.Z[q_row]
    dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))

    buffer = min(len(cand), CANDIDATE_BUFFER_FACTOR * k)
    nearest = np.argpartition(dist, buffer - 1)[:buffer]
    nearest = nearest[np.argsort(dist[nearest], kind="stable")]

    kept = dedup_matches(ws.end_idx[cand[nearest]], gap=dedup_gap, limit=k)
    sel = cand[nearest[kept]]
    return MatchOutcome(
        query_ts=_utc(ws.end_ts[q_row]),
        match_ts=[_utc(t) for t in ws.end_ts[sel]],
        distance=dist[nearest[kept]],
        fwd_ret=ws.fwd_ret[sel],
        n_candidates=int(len(cand)),
    )
