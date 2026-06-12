"""Overlap dedup: near-coincident windows are one market event, not
independent evidence. Greedy best-first — of any cluster of overlapping
matches, exactly the closest survives."""

from __future__ import annotations

import numpy as np


def dedup_matches(end_indices: np.ndarray, gap: int, limit: int | None = None) -> list[int]:
    """end_indices: candidate global end-indices, already sorted best-first.
    Returns positions (into the input) of kept candidates; a candidate is
    rejected if within `gap` bars of any already-kept one."""
    kept_pos: list[int] = []
    kept_end: list[int] = []
    for pos, e in enumerate(end_indices):
        if all(abs(int(e) - ke) > gap for ke in kept_end):
            kept_pos.append(pos)
            kept_end.append(int(e))
            if limit is not None and len(kept_pos) >= limit:
                break
    return kept_pos
