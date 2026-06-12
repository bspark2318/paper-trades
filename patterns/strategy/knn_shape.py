"""knn_shape: the k-NN shape-matching hypothesis as a SignalSource.

Rule: among the deduped k nearest historical look-alikes,
LONG  iff pct_positive >= p_threshold AND mean_fwd >= t_multiplier * uncond,
where uncond is the unconditional mean H-bar forward return over exactly the
candidates eligible at `asof` (the no-lookahead mask doubles as a
point-in-time expanding mean — no future bars ever enter the threshold).
Shorts, when enabled, are the same logic mirrored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from patterns.config import Config
from patterns.engine import matcher
from patterns.engine.windows import build_windows
from patterns.strategy.base import LONG, NO_TRADE, SHORT, Signal, register_source


@register_source
class KnnShape:
    name = "knn_shape"
    identity_fields = (
        "window",
        "k",
        "dedup_gap",
        "p_threshold",
        "t_multiplier",
        "min_matches",
        "features",
        "normalization",
    )

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.symbol = cfg.symbols[0]
        self.ws = None

    def prepare(self, bars: pd.DataFrame) -> None:
        self.ws = build_windows(
            bars,
            window=self.cfg.window,
            horizon=self.cfg.horizon,
            normalization=self.cfg.normalization,
            features=self.cfg.features,
        )

    def signal_at(self, asof: pd.Timestamp) -> Signal:
        cfg = self.cfg
        out = matcher.query(self.ws, asof, k=cfg.k, dedup_gap=cfg.dedup_gap)
        stats = out.stats()

        if out.n < cfg.min_matches:
            return self._signal(asof, NO_TRADE, stats, reason="too_few_matches")

        q_row = self.ws.row_for_ts(asof)
        eligible = matcher.eligible_mask(self.ws, q_row)
        uncond = float(np.mean(self.ws.fwd_ret[eligible]))
        threshold = cfg.t_multiplier * uncond

        if stats["pct_positive"] >= cfg.p_threshold and stats["mean"] >= threshold:
            direction = LONG
        elif (
            cfg.enable_shorts
            and (1.0 - stats["pct_positive"]) >= cfg.p_threshold
            and stats["mean"] <= -threshold
        ):
            direction = SHORT
        else:
            direction = NO_TRADE

        return self._signal(
            asof, direction, stats,
            reason="rule", uncond_mean=uncond, threshold=threshold,
            n_candidates=out.n_candidates,
        )

    def _signal(self, asof, direction, stats, **extra) -> Signal:
        return Signal(
            asof=pd.Timestamp(asof),
            symbol=self.symbol,
            direction=direction,
            diagnostics={**stats, **extra},
        )
